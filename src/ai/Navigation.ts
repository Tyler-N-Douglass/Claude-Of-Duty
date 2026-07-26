/**
 * Navigation — a walkability grid baked from physics queries, A* with a binary
 * heap, string-pulled path smoothing, and Dijkstra flow fields for squads.
 *
 * The grid is single-layer by design: the playable space is a street level with
 * interiors on the ground storey, so one height sample per column is both
 * sufficient and an order of magnitude cheaper than a layered mesh. Everything
 * is typed arrays and index arithmetic; a full 84x84m map at 0.5m is ~28k cells,
 * which A* chews through in well under a millisecond.
 */
import * as THREE from 'three';
import { RayMask, type PhysicsWorld } from '../core/Contracts';

export const NAV_CELL = 0.5;

/** Cell flag bits. */
const F_WALK = 1;
const F_SAMPLED = 2;

/** Vertical difference between neighbouring cells that still counts as walkable. */
const MAX_STEP = 0.44;
/** Agent capsule radius used when baking clearance. */
const AGENT_RADIUS = 0.34;
/** Ray height used for the headroom test. */
const HEADROOM = 1.72;

/** Height the ground probe starts from. Above the ground storey, below its ceiling. */
const PROBE_Y = 2.2;
const PROBE_DROP = 5.5;

const _p = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();

/** Diagonal cost. */
const SQRT2 = Math.SQRT2;

// ---------------------------------------------------------------------------
// Binary heap over cell indices, keyed by f-score. Reused across queries.
// ---------------------------------------------------------------------------

class IndexHeap {
  private keys: Float32Array;
  private vals: Int32Array;
  private n = 0;

  constructor(capacity: number) {
    this.keys = new Float32Array(capacity);
    this.vals = new Int32Array(capacity);
  }

  get size(): number {
    return this.n;
  }

  clear(): void {
    this.n = 0;
  }

  push(value: number, key: number): void {
    if (this.n >= this.vals.length) this.grow();
    let i = this.n++;
    this.keys[i] = key;
    this.vals[i] = value;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.keys[parent] <= this.keys[i]) break;
      this.swap(i, parent);
      i = parent;
    }
  }

  pop(): number {
    if (this.n === 0) return -1;
    const top = this.vals[0];
    this.n--;
    if (this.n > 0) {
      this.keys[0] = this.keys[this.n];
      this.vals[0] = this.vals[this.n];
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < this.n && this.keys[l] < this.keys[m]) m = l;
        if (r < this.n && this.keys[r] < this.keys[m]) m = r;
        if (m === i) break;
        this.swap(i, m);
        i = m;
      }
    }
    return top;
  }

  private swap(i: number, j: number): void {
    const k = this.keys[i];
    this.keys[i] = this.keys[j];
    this.keys[j] = k;
    const v = this.vals[i];
    this.vals[i] = this.vals[j];
    this.vals[j] = v;
  }

  private grow(): void {
    const keys = new Float32Array(this.keys.length * 2);
    keys.set(this.keys);
    const vals = new Int32Array(this.vals.length * 2);
    vals.set(this.vals);
    this.keys = keys;
    this.vals = vals;
  }
}

// ---------------------------------------------------------------------------

/**
 * A Dijkstra distance field over the grid with a gradient at every cell. Squads
 * share one field per objective, so ten agents heading for the player cost one
 * flood fill rather than ten searches — and, because each agent reads a smooth
 * gradient rather than a shared polyline, they spread out instead of stacking.
 */
export class FlowField {
  readonly dist: Float32Array;
  readonly dirX: Float32Array;
  readonly dirZ: Float32Array;
  /** Cell the field converges on, -1 when empty. */
  goal = -1;
  /** World position the field was built for; used to decide when to rebuild. */
  readonly goalPosition = new THREE.Vector3();
  /** Seconds since the last rebuild. */
  age = 0;

  constructor(private readonly grid: NavGrid) {
    const n = grid.cols * grid.rows;
    this.dist = new Float32Array(n);
    this.dirX = new Float32Array(n);
    this.dirZ = new Float32Array(n);
    this.dist.fill(Infinity);
  }

  /** True when the field can steer from `at`. */
  valid(at: THREE.Vector3): boolean {
    if (this.goal < 0) return false;
    const i = this.grid.cellAt(at.x, at.z);
    return i >= 0 && Number.isFinite(this.dist[i]);
  }

  /**
   * Steering direction at a world point, written into `out` (xz, y=0). Returns
   * false when the point is off the field, in which case `out` is untouched.
   */
  sample(x: number, z: number, out: THREE.Vector3): boolean {
    const g = this.grid;
    const i = g.cellAt(x, z);
    if (i < 0 || !Number.isFinite(this.dist[i])) return false;
    // Bilinear blend of the four surrounding gradients: sampling the raw cell
    // direction makes agents visibly snap between the eight compass headings.
    const fx = (x - g.originX) / g.cell - 0.5;
    const fz = (z - g.originZ) / g.cell - 0.5;
    const cx = Math.floor(fx);
    const cz = Math.floor(fz);
    const tx = fx - cx;
    const tz = fz - cz;
    let sx = 0;
    let sz = 0;
    let w = 0;
    for (let dz = 0; dz <= 1; dz++) {
      for (let dx = 0; dx <= 1; dx++) {
        const nx = cx + dx;
        const nz = cz + dz;
        if (nx < 0 || nz < 0 || nx >= g.cols || nz >= g.rows) continue;
        const ni = nz * g.cols + nx;
        if (!Number.isFinite(this.dist[ni])) continue;
        const weight = (dx ? tx : 1 - tx) * (dz ? tz : 1 - tz);
        if (weight <= 0) continue;
        sx += this.dirX[ni] * weight;
        sz += this.dirZ[ni] * weight;
        w += weight;
      }
    }
    if (w <= 1e-4) {
      sx = this.dirX[i];
      sz = this.dirZ[i];
    }
    const len = Math.hypot(sx, sz);
    if (len < 1e-4) {
      out.set(this.dirX[i], 0, this.dirZ[i]);
      return out.lengthSq() > 1e-6;
    }
    out.set(sx / len, 0, sz / len);
    return true;
  }

  /** Path distance to the goal in metres, Infinity when unreachable. */
  costAt(x: number, z: number): number {
    const i = this.grid.cellAt(x, z);
    return i < 0 ? Infinity : this.dist[i];
  }

  build(target: THREE.Vector3): boolean {
    const g = this.grid;
    const start = g.nearestWalkableCell(target.x, target.z, 6);
    if (start < 0) return false;
    this.goal = start;
    this.goalPosition.copy(target);
    this.age = 0;

    const dist = this.dist;
    dist.fill(Infinity);
    const heap = g.heap;
    heap.clear();
    dist[start] = 0;
    heap.push(start, 0);

    const cols = g.cols;
    const rows = g.rows;
    const flags = g.flags;
    const cost = g.cost;
    const cell = g.cell;

    while (heap.size > 0) {
      const cur = heap.pop();
      const cd = dist[cur];
      const cxi = cur % cols;
      const czi = (cur / cols) | 0;
      for (let k = 0; k < 8; k++) {
        const nx = cxi + NEIGHBOUR_X[k];
        const nz = czi + NEIGHBOUR_Z[k];
        if (nx < 0 || nz < 0 || nx >= cols || nz >= rows) continue;
        const ni = nz * cols + nx;
        if ((flags[ni] & F_WALK) === 0) continue;
        if (k >= 4 && !g.diagonalOpen(cxi, czi, nx, nz)) continue;
        const step = (k >= 4 ? SQRT2 : 1) * cell * cost[ni];
        const nd = cd + step;
        if (nd + 1e-4 >= dist[ni]) continue;
        dist[ni] = nd;
        heap.push(ni, nd);
      }
    }

    // Gradient pass: point each cell at its cheapest neighbour.
    for (let z = 0; z < rows; z++) {
      for (let x = 0; x < cols; x++) {
        const i = z * cols + x;
        if (!Number.isFinite(dist[i])) {
          this.dirX[i] = 0;
          this.dirZ[i] = 0;
          continue;
        }
        let best = dist[i];
        let bx = 0;
        let bz = 0;
        for (let k = 0; k < 8; k++) {
          const nx = x + NEIGHBOUR_X[k];
          const nz = z + NEIGHBOUR_Z[k];
          if (nx < 0 || nz < 0 || nx >= cols || nz >= rows) continue;
          const ni = nz * cols + nx;
          const d = dist[ni];
          if (d < best) {
            best = d;
            bx = NEIGHBOUR_X[k];
            bz = NEIGHBOUR_Z[k];
          }
        }
        const len = Math.hypot(bx, bz);
        this.dirX[i] = len > 0 ? bx / len : 0;
        this.dirZ[i] = len > 0 ? bz / len : 0;
      }
    }
    return true;
  }
}

const NEIGHBOUR_X = [1, -1, 0, 0, 1, 1, -1, -1];
const NEIGHBOUR_Z = [0, 0, 1, -1, 1, -1, 1, -1];

// ---------------------------------------------------------------------------

export class NavGrid {
  cell = NAV_CELL;
  cols = 0;
  rows = 0;
  originX = 0;
  originZ = 0;
  ready = false;

  height = new Float32Array(0);
  flags = new Uint8Array(0);
  /** Distance to the nearest non-walkable cell, metres, capped at 4. */
  clear = new Float32Array(0);
  /** Traversal cost multiplier: >1 near walls so agents run down the middle. */
  cost = new Float32Array(0);

  /** Shared with FlowField to avoid a second allocation. */
  heap = new IndexHeap(1);

  private gScore = new Float32Array(0);
  private stamp = new Int32Array(0);
  private cameFrom = new Int32Array(0);
  private closed = new Uint8Array(0);
  private searchId = 0;
  private readonly cellPath: number[] = [];

  /** Cells that are walkable, for random point sampling. */
  private walkList = new Int32Array(0);
  private walkCount = 0;
  private rngState = 0x9e3779b9;

  // =========================================================================
  // Bake
  // =========================================================================

  /**
   * Samples the world into a walkability grid. Two rays per column: one down to
   * find the floor, one up to reject anything without standing headroom. Slope
   * and step limits then come out of the height field for free.
   */
  build(physics: PhysicsWorld, bounds: THREE.Box3, cell = NAV_CELL): void {
    this.cell = cell;
    this.originX = Math.floor(bounds.min.x / cell) * cell;
    this.originZ = Math.floor(bounds.min.z / cell) * cell;
    this.cols = Math.max(2, Math.ceil((bounds.max.x - this.originX) / cell) + 1);
    this.rows = Math.max(2, Math.ceil((bounds.max.z - this.originZ) / cell) + 1);
    const n = this.cols * this.rows;

    this.height = new Float32Array(n);
    this.flags = new Uint8Array(n);
    this.clear = new Float32Array(n);
    this.cost = new Float32Array(n);
    this.gScore = new Float32Array(n);
    this.stamp = new Int32Array(n);
    this.cameFrom = new Int32Array(n);
    this.closed = new Uint8Array(n);
    this.heap = new IndexHeap(Math.max(64, n >> 2));
    this.cost.fill(1);

    const probeTop = Math.min(PROBE_Y, bounds.max.y - 0.5);

    for (let z = 0; z < this.rows; z++) {
      const wz = this.originZ + z * cell;
      for (let x = 0; x < this.cols; x++) {
        const i = z * this.cols + x;
        const wx = this.originX + x * cell;
        _p.set(wx, probeTop, wz);
        const h = physics.groundHeight(_p, PROBE_DROP);
        if (h === null) {
          this.height[i] = NaN;
          continue;
        }
        this.height[i] = h;
        this.flags[i] = F_SAMPLED;
        _p.set(wx, h + 0.12, wz);
        const blocked = physics.raycast(_p, _up, HEADROOM, RayMask.Solid);
        if (blocked === null) this.flags[i] |= F_WALK;
      }
    }

    this.applySlopeLimit();
    this.bakeClearance();
    this.bakeCost();
    this.collectWalkable();
    this.ready = this.walkCount > 16;
  }

  /** Rejects cells whose neighbours are a cliff or a wall face away. */
  private applySlopeLimit(): void {
    const { cols, rows, flags, height } = this;
    const drop = new Uint8Array(cols * rows);
    for (let z = 0; z < rows; z++) {
      for (let x = 0; x < cols; x++) {
        const i = z * cols + x;
        if ((flags[i] & F_WALK) === 0) continue;
        const h = height[i];
        let open = 0;
        for (let k = 0; k < 4; k++) {
          const nx = x + NEIGHBOUR_X[k];
          const nz = z + NEIGHBOUR_Z[k];
          if (nx < 0 || nz < 0 || nx >= cols || nz >= rows) continue;
          const ni = nz * cols + nx;
          if ((flags[ni] & F_SAMPLED) === 0) continue;
          if (Math.abs(height[ni] - h) <= MAX_STEP) open++;
        }
        // An island with no traversable neighbour is a ledge, a windowsill or a
        // crate top: standable, but not somewhere a path may route through.
        if (open === 0) drop[i] = 1;
      }
    }
    for (let i = 0; i < drop.length; i++) if (drop[i]) flags[i] &= ~F_WALK;
  }

  /** Two-pass chamfer distance transform from the blocked cells. */
  private bakeClearance(): void {
    const { cols, rows, flags, clear, cell } = this;
    const near = cell;
    const diag = cell * SQRT2;
    const MAXD = 4;
    for (let i = 0; i < clear.length; i++) clear[i] = (flags[i] & F_WALK) === 0 ? 0 : MAXD;

    for (let z = 0; z < rows; z++) {
      for (let x = 0; x < cols; x++) {
        const i = z * cols + x;
        let d = clear[i];
        if (d === 0) continue;
        if (x > 0) d = Math.min(d, clear[i - 1] + near);
        if (z > 0) d = Math.min(d, clear[i - cols] + near);
        if (x > 0 && z > 0) d = Math.min(d, clear[i - cols - 1] + diag);
        if (x < cols - 1 && z > 0) d = Math.min(d, clear[i - cols + 1] + diag);
        clear[i] = d;
      }
    }
    for (let z = rows - 1; z >= 0; z--) {
      for (let x = cols - 1; x >= 0; x--) {
        const i = z * cols + x;
        let d = clear[i];
        if (d === 0) continue;
        if (x < cols - 1) d = Math.min(d, clear[i + 1] + near);
        if (z < rows - 1) d = Math.min(d, clear[i + cols] + near);
        if (x < cols - 1 && z < rows - 1) d = Math.min(d, clear[i + cols + 1] + diag);
        if (x > 0 && z < rows - 1) d = Math.min(d, clear[i + cols - 1] + diag);
        clear[i] = d;
      }
    }

    // A cell an agent physically cannot stand in is not walkable.
    for (let i = 0; i < clear.length; i++) {
      if ((flags[i] & F_WALK) && clear[i] < AGENT_RADIUS * 0.62) flags[i] &= ~F_WALK;
    }
  }

  private bakeCost(): void {
    const { clear, cost, flags, height, cols, rows } = this;
    for (let z = 0; z < rows; z++) {
      for (let x = 0; x < cols; x++) {
        const i = z * cols + x;
        if ((flags[i] & F_WALK) === 0) {
          cost[i] = 1;
          continue;
        }
        // Wall-hugging costs more; so does anything with a height change, which
        // keeps routes off kerbs and rubble when a flat line exists.
        const c = clear[i];
        let f = c < 0.75 ? 1 + (0.75 - c) * 2.2 : 1;
        let slope = 0;
        for (let k = 0; k < 4; k++) {
          const nx = x + NEIGHBOUR_X[k];
          const nz = z + NEIGHBOUR_Z[k];
          if (nx < 0 || nz < 0 || nx >= cols || nz >= rows) continue;
          const ni = nz * cols + nx;
          if ((flags[ni] & F_WALK) === 0) continue;
          slope = Math.max(slope, Math.abs(height[ni] - height[i]));
        }
        f += slope * 1.6;
        cost[i] = f;
      }
    }
  }

  private collectWalkable(): void {
    const list: number[] = [];
    for (let i = 0; i < this.flags.length; i++) if (this.flags[i] & F_WALK) list.push(i);
    this.walkList = Int32Array.from(list);
    this.walkCount = list.length;
  }

  // =========================================================================
  // Queries
  // =========================================================================

  cellAt(x: number, z: number): number {
    const cx = Math.round((x - this.originX) / this.cell);
    const cz = Math.round((z - this.originZ) / this.cell);
    if (cx < 0 || cz < 0 || cx >= this.cols || cz >= this.rows) return -1;
    return cz * this.cols + cx;
  }

  cellCentre(i: number, out: THREE.Vector3): THREE.Vector3 {
    const cx = i % this.cols;
    const cz = (i / this.cols) | 0;
    return out.set(this.originX + cx * this.cell, this.height[i], this.originZ + cz * this.cell);
  }

  isWalkable(x: number, z: number): boolean {
    const i = this.cellAt(x, z);
    return i >= 0 && (this.flags[i] & F_WALK) !== 0;
  }

  clearanceAt(x: number, z: number): number {
    const i = this.cellAt(x, z);
    return i < 0 ? 0 : this.clear[i];
  }

  /**
   * Bilinearly filtered floor height. Cheaper and far more stable frame to frame
   * than a physics ray, which matters because foot IK samples it every frame for
   * every visible soldier.
   */
  groundAt(x: number, z: number): number | null {
    const fx = (x - this.originX) / this.cell;
    const fz = (z - this.originZ) / this.cell;
    const x0 = Math.floor(fx);
    const z0 = Math.floor(fz);
    if (x0 < 0 || z0 < 0 || x0 >= this.cols - 1 || z0 >= this.rows - 1) return null;
    const tx = fx - x0;
    const tz = fz - z0;
    const i00 = z0 * this.cols + x0;
    const h00 = this.height[i00];
    const h10 = this.height[i00 + 1];
    const h01 = this.height[i00 + this.cols];
    const h11 = this.height[i00 + this.cols + 1];
    if (!Number.isFinite(h00) || !Number.isFinite(h10) || !Number.isFinite(h01) || !Number.isFinite(h11)) {
      return Number.isFinite(h00) ? h00 : null;
    }
    const a = h00 + (h10 - h00) * tx;
    const b = h01 + (h11 - h01) * tx;
    return a + (b - a) * tz;
  }

  /** Spiral search outward for a standable cell. Returns -1 if none in range. */
  nearestWalkableCell(x: number, z: number, maxRadiusCells = 8): number {
    const cx = Math.round((x - this.originX) / this.cell);
    const cz = Math.round((z - this.originZ) / this.cell);
    for (let r = 0; r <= maxRadiusCells; r++) {
      let best = -1;
      let bestD = Infinity;
      for (let dz = -r; dz <= r; dz++) {
        for (let dx = -r; dx <= r; dx++) {
          if (r > 0 && Math.max(Math.abs(dx), Math.abs(dz)) !== r) continue;
          const nx = cx + dx;
          const nz = cz + dz;
          if (nx < 0 || nz < 0 || nx >= this.cols || nz >= this.rows) continue;
          const ni = nz * this.cols + nx;
          if ((this.flags[ni] & F_WALK) === 0) continue;
          const d = dx * dx + dz * dz;
          if (d < bestD) {
            bestD = d;
            best = ni;
          }
        }
      }
      if (best >= 0) return best;
    }
    return -1;
  }

  /** Snaps a world point onto the navigable surface. Returns false if off-mesh. */
  snap(pos: THREE.Vector3, out: THREE.Vector3, maxRadiusCells = 8): boolean {
    const i = this.nearestWalkableCell(pos.x, pos.z, maxRadiusCells);
    if (i < 0) return false;
    this.cellCentre(i, out);
    return true;
  }

  /** Deterministic per-grid RNG so layouts replay identically. */
  private rand(): number {
    this.rngState = (this.rngState * 1664525 + 1013904223) >>> 0;
    return this.rngState / 4294967296;
  }

  randomPoint(out: THREE.Vector3): boolean {
    if (this.walkCount === 0) return false;
    for (let attempt = 0; attempt < 8; attempt++) {
      const i = this.walkList[(this.rand() * this.walkCount) | 0];
      if (this.clear[i] < 0.7) continue;
      this.cellCentre(i, out);
      return true;
    }
    this.cellCentre(this.walkList[(this.rand() * this.walkCount) | 0], out);
    return true;
  }

  /** A walkable point within `radius` of `origin`, biased away from walls. */
  randomPointNear(origin: THREE.Vector3, radius: number, out: THREE.Vector3): boolean {
    const cells = Math.max(1, Math.round(radius / this.cell));
    const cx = Math.round((origin.x - this.originX) / this.cell);
    const cz = Math.round((origin.z - this.originZ) / this.cell);
    let best = -1;
    let bestScore = -Infinity;
    for (let attempt = 0; attempt < 24; attempt++) {
      const nx = cx + ((this.rand() * (cells * 2 + 1)) | 0) - cells;
      const nz = cz + ((this.rand() * (cells * 2 + 1)) | 0) - cells;
      if (nx < 0 || nz < 0 || nx >= this.cols || nz >= this.rows) continue;
      const ni = nz * this.cols + nx;
      if ((this.flags[ni] & F_WALK) === 0) continue;
      const score = this.clear[ni] + this.rand() * 0.4;
      if (score > bestScore) {
        bestScore = score;
        best = ni;
      }
    }
    if (best < 0) return false;
    this.cellCentre(best, out);
    return true;
  }

  /** Both orthogonal cells of a diagonal move must be open (no corner cutting). */
  diagonalOpen(x0: number, z0: number, x1: number, z1: number): boolean {
    const a = z0 * this.cols + x1;
    const b = z1 * this.cols + x0;
    return (this.flags[a] & F_WALK) !== 0 && (this.flags[b] & F_WALK) !== 0;
  }

  /**
   * Supercover line walk between two world points. This is the line-of-sight
   * test the string puller uses: it is a walkability test, not a bullet test,
   * so it rejects shortcuts through a doorway's frame that a physics ray would
   * happily allow at chest height.
   */
  lineWalkable(ax: number, az: number, bx: number, bz: number, minClear = AGENT_RADIUS * 0.8): boolean {
    const inv = 1 / this.cell;
    let x = Math.round((ax - this.originX) * inv);
    let z = Math.round((az - this.originZ) * inv);
    const ex = Math.round((bx - this.originX) * inv);
    const ez = Math.round((bz - this.originZ) * inv);
    if (x < 0 || z < 0 || x >= this.cols || z >= this.rows) return false;
    if (ex < 0 || ez < 0 || ex >= this.cols || ez >= this.rows) return false;

    let dx = Math.abs(ex - x);
    let dz = Math.abs(ez - z);
    const sx = x < ex ? 1 : -1;
    const sz = z < ez ? 1 : -1;
    let err = dx - dz;
    let guard = dx + dz + 2;
    let lastH = this.height[z * this.cols + x];

    while (guard-- > 0) {
      const i = z * this.cols + x;
      if ((this.flags[i] & F_WALK) === 0) return false;
      if (this.clear[i] < minClear) return false;
      const h = this.height[i];
      if (Math.abs(h - lastH) > MAX_STEP) return false;
      lastH = h;
      if (x === ex && z === ez) return true;
      const e2 = err * 2;
      let moved = false;
      if (e2 > -dz) {
        err -= dz;
        x += sx;
        moved = true;
      }
      if (e2 < dx) {
        err += dx;
        z += sz;
        if (moved) {
          // Diagonal step: both shoulder cells must be clear too.
          const ca = z * this.cols + (x - sx);
          const cb = (z - sz) * this.cols + x;
          if ((this.flags[ca] & F_WALK) === 0 || (this.flags[cb] & F_WALK) === 0) return false;
        }
      }
      if (x < 0 || z < 0 || x >= this.cols || z >= this.rows) return false;
    }
    return false;
  }

  // =========================================================================
  // A*
  // =========================================================================

  /**
   * Finds a smoothed path from `from` to `to`, appending world points to `out`.
   * `out` is cleared first. The start point is not included; the first entry is
   * the first corner to steer at.
   */
  findPath(from: THREE.Vector3, to: THREE.Vector3, out: THREE.Vector3[]): boolean {
    out.length = 0;
    if (!this.ready) return false;
    const start = this.nearestWalkableCell(from.x, from.z, 8);
    let goal = this.nearestWalkableCell(to.x, to.z, 8);
    if (start < 0 || goal < 0) return false;
    if (start === goal) {
      out.push(new THREE.Vector3(to.x, this.height[goal], to.z));
      return true;
    }

    if (!this.search(start, goal)) return false;
    this.buildCellPath(start, goal);
    this.stringPull(from, to, out);
    return out.length > 0;
  }

  private search(start: number, goal: number): boolean {
    const id = ++this.searchId;
    const { cols, rows, flags, cost, cell, gScore, stamp, cameFrom, closed, heap } = this;
    heap.clear();

    const gx = goal % cols;
    const gz = (goal / cols) | 0;
    const heuristic = (i: number): number => {
      const x = i % cols;
      const z = (i / cols) | 0;
      const dx = Math.abs(x - gx);
      const dz = Math.abs(z - gz);
      // Octile distance: admissible on an 8-connected grid and much tighter
      // than Euclidean, which roughly halves the expanded node count.
      return (Math.max(dx, dz) + (SQRT2 - 1) * Math.min(dx, dz)) * cell;
    };

    stamp[start] = id;
    gScore[start] = 0;
    cameFrom[start] = -1;
    closed[start] = 0;
    heap.push(start, heuristic(start));

    // Hard bound so a pathological query can never stall the frame.
    let expansions = 0;
    const budget = Math.min(20000, cols * rows);

    while (heap.size > 0) {
      const cur = heap.pop();
      if (cur === goal) return true;
      if (closed[cur] === id) continue;
      closed[cur] = id;
      if (++expansions > budget) return false;

      const cxi = cur % cols;
      const czi = (cur / cols) | 0;
      const cg = gScore[cur];

      for (let k = 0; k < 8; k++) {
        const nx = cxi + NEIGHBOUR_X[k];
        const nz = czi + NEIGHBOUR_Z[k];
        if (nx < 0 || nz < 0 || nx >= cols || nz >= rows) continue;
        const ni = nz * cols + nx;
        if ((flags[ni] & F_WALK) === 0) continue;
        if (k >= 4 && !this.diagonalOpen(cxi, czi, nx, nz)) continue;
        if (Math.abs(this.height[ni] - this.height[cur]) > MAX_STEP) continue;
        const step = (k >= 4 ? SQRT2 : 1) * cell * cost[ni];
        const tentative = cg + step;
        if (stamp[ni] === id && tentative + 1e-4 >= gScore[ni]) continue;
        stamp[ni] = id;
        gScore[ni] = tentative;
        cameFrom[ni] = cur;
        heap.push(ni, tentative + heuristic(ni));
      }
    }
    return false;
  }

  private buildCellPath(start: number, goal: number): void {
    const path = this.cellPath;
    path.length = 0;
    let c = goal;
    let guard = this.cols * this.rows;
    while (c >= 0 && guard-- > 0) {
      path.push(c);
      if (c === start) break;
      c = this.cameFrom[c];
    }
    path.reverse();
  }

  /**
   * String pulling: walk forward from the current anchor as far as the last
   * point still reachable in a straight walkable line, then anchor there. Turns
   * a staircase of grid cells into the two or three corners a human would take.
   */
  private stringPull(from: THREE.Vector3, to: THREE.Vector3, out: THREE.Vector3[]): void {
    const path = this.cellPath;
    if (path.length === 0) return;

    let anchorX = from.x;
    let anchorZ = from.z;
    let i = 0;
    let guard = path.length + 4;

    while (i < path.length && guard-- > 0) {
      let furthest = i;
      for (let j = path.length - 1; j > i; j--) {
        this.cellCentre(path[j], _a);
        if (this.lineWalkable(anchorX, anchorZ, _a.x, _a.z)) {
          furthest = j;
          break;
        }
      }
      this.cellCentre(path[furthest], _a);
      out.push(new THREE.Vector3(_a.x, _a.y, _a.z));
      anchorX = _a.x;
      anchorZ = _a.z;
      if (furthest >= path.length - 1) break;
      i = furthest + 1;
    }

    // Replace the terminal corner with the true destination when it is directly
    // reachable, so agents finish on the cover point rather than its cell centre.
    const last = out[out.length - 1];
    if (last && this.lineWalkable(last.x, last.z, to.x, to.z)) {
      const h = this.groundAt(to.x, to.z);
      out.push(new THREE.Vector3(to.x, h ?? last.y, to.z));
    }

    // Merge corners that are nearly collinear; they only make agents wobble.
    for (let k = out.length - 2; k >= 1; k--) {
      const prev = out[k - 1];
      const cur = out[k];
      const next = out[k + 1];
      const ax = cur.x - prev.x;
      const az = cur.z - prev.z;
      const bx = next.x - cur.x;
      const bz = next.z - cur.z;
      const la = Math.hypot(ax, az);
      const lb = Math.hypot(bx, bz);
      if (la < 1e-4 || lb < 1e-4) {
        out.splice(k, 1);
        continue;
      }
      const dot = (ax * bx + az * bz) / (la * lb);
      if (dot > 0.995 && this.lineWalkable(prev.x, prev.z, next.x, next.z)) out.splice(k, 1);
    }
  }

  // =========================================================================

  /** Cheap "is this spot sensible to stand on" test used by cover selection. */
  standable(pos: THREE.Vector3, minClearance = AGENT_RADIUS): boolean {
    const i = this.cellAt(pos.x, pos.z);
    if (i < 0 || (this.flags[i] & F_WALK) === 0) return false;
    if (this.clear[i] < minClearance) return false;
    return Math.abs(this.height[i] - pos.y) < 1.2;
  }

  /**
   * Pushes a position away from nearby geometry using the clearance gradient.
   * Used as a last-resort unstick when an agent's steering wedges it on a
   * corner the grid says is fine but the collision capsule does not.
   */
  pushOffWalls(pos: THREE.Vector3, out: THREE.Vector3): boolean {
    const i = this.cellAt(pos.x, pos.z);
    if (i < 0) return false;
    if (this.clear[i] >= AGENT_RADIUS + 0.1) return false;
    const x = i % this.cols;
    const z = (i / this.cols) | 0;
    let gx = 0;
    let gz = 0;
    for (let k = 0; k < 8; k++) {
      const nx = x + NEIGHBOUR_X[k];
      const nz = z + NEIGHBOUR_Z[k];
      if (nx < 0 || nz < 0 || nx >= this.cols || nz >= this.rows) continue;
      const ni = nz * this.cols + nx;
      const w = this.clear[ni] - this.clear[i];
      gx += NEIGHBOUR_X[k] * w;
      gz += NEIGHBOUR_Z[k] * w;
    }
    const len = Math.hypot(gx, gz);
    if (len < 1e-4) return false;
    out.set(gx / len, 0, gz / len);
    return true;
  }

  dispose(): void {
    this.height = new Float32Array(0);
    this.flags = new Uint8Array(0);
    this.clear = new Float32Array(0);
    this.cost = new Float32Array(0);
    this.gScore = new Float32Array(0);
    this.stamp = new Int32Array(0);
    this.cameFrom = new Int32Array(0);
    this.closed = new Uint8Array(0);
    this.walkList = new Int32Array(0);
    this.walkCount = 0;
    this.cellPath.length = 0;
    this.ready = false;
  }
}

// ---------------------------------------------------------------------------
// Path following
// ---------------------------------------------------------------------------

/**
 * Holds a path and produces a steering target. Owned per agent; the repath
 * stagger lives in the AI system so only a couple of agents search per frame.
 */
export class PathFollower {
  readonly points: THREE.Vector3[] = [];
  index = 0;
  /** Seconds since this path was computed. */
  age = 0;
  /** World point the path was built toward. */
  readonly destination = new THREE.Vector3();
  valid = false;

  set(points: THREE.Vector3[], destination: THREE.Vector3): void {
    this.points.length = 0;
    for (let i = 0; i < points.length; i++) this.points.push(points[i]);
    this.index = 0;
    this.age = 0;
    this.destination.copy(destination);
    this.valid = this.points.length > 0;
  }

  clear(): void {
    this.points.length = 0;
    this.index = 0;
    this.valid = false;
  }

  get finished(): boolean {
    return !this.valid || this.index >= this.points.length;
  }

  get remaining(): number {
    return this.valid ? this.points.length - this.index : 0;
  }

  /**
   * Advances past reached corners and writes the current steering target.
   * `arrive` is the corner acceptance radius; the final point uses a tighter one
   * so agents actually reach their cover slot.
   */
  target(from: THREE.Vector3, out: THREE.Vector3, arrive = 0.55): boolean {
    while (this.index < this.points.length) {
      const p = this.points[this.index];
      const last = this.index === this.points.length - 1;
      const r = last ? 0.32 : arrive;
      const dx = p.x - from.x;
      const dz = p.z - from.z;
      if (dx * dx + dz * dz > r * r) {
        out.copy(p);
        return true;
      }
      this.index++;
    }
    return false;
  }

  /** Straight-line distance left along the remaining corners. */
  lengthFrom(from: THREE.Vector3): number {
    if (!this.valid || this.index >= this.points.length) return 0;
    let total = 0;
    _b.copy(from);
    for (let i = this.index; i < this.points.length; i++) {
      const p = this.points[i];
      total += Math.hypot(p.x - _b.x, p.z - _b.z);
      _b.copy(p);
    }
    return total;
  }
}
