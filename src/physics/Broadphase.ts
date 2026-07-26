/**
 * Static-world broadphase: a uniform grid over a flat world-space triangle soup.
 *
 * Level geometry is extracted from three meshes exactly once (at addStatic time)
 * into `Float32Array`s, then binned into a bounded 3D grid. Rays walk the grid
 * with a 3D DDA (Amanatides & Woo) so a hitscan shot touches a few dozen
 * triangles instead of the whole scene graph; capsule/sphere queries pull the
 * cells overlapping an AABB. Nothing here allocates once the grid is built.
 */
import * as THREE from 'three';
import type { SurfaceKind } from '../core/Contracts';

// ---------------------------------------------------------------------------
// Surface enum packing (one byte per triangle instead of a string reference)
// ---------------------------------------------------------------------------

export const SURFACE_LIST: readonly SurfaceKind[] = [
  'concrete', 'metal', 'wood', 'dirt', 'sand', 'glass',
  'water', 'flesh', 'foliage', 'plaster', 'rubber', 'fabric',
];

const SURFACE_TO_INDEX = new Map<SurfaceKind, number>();
for (let i = 0; i < SURFACE_LIST.length; i++) SURFACE_TO_INDEX.set(SURFACE_LIST[i], i);

export function surfaceIndex(s: SurfaceKind): number {
  const i = SURFACE_TO_INDEX.get(s);
  return i === undefined ? 0 : i;
}

export function surfaceFromIndex(i: number): SurfaceKind {
  const s = SURFACE_LIST[i];
  return s === undefined ? 'concrete' : s;
}

// ---------------------------------------------------------------------------
// Triangle extraction
// ---------------------------------------------------------------------------

const _m4 = /*@__PURE__*/ new THREE.Matrix4();
const _v3 = /*@__PURE__*/ new THREE.Vector3();

/** Growable float sink used only while building; never touched at query time. */
class TriBuffer {
  private data = new Float32Array(9 * 256);
  private used = 0;

  push(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number,
  ): void {
    if (this.used + 9 > this.data.length) {
      const next = new Float32Array(Math.max(this.data.length * 2, this.used + 9));
      next.set(this.data.subarray(0, this.used));
      this.data = next;
    }
    const d = this.data;
    let i = this.used;
    d[i++] = ax; d[i++] = ay; d[i++] = az;
    d[i++] = bx; d[i++] = by; d[i++] = bz;
    d[i++] = cx; d[i++] = cy; d[i++] = cz;
    this.used = i;
  }

  get triCount(): number {
    return (this.used / 9) | 0;
  }

  /** Copies out to an exactly sized array so the grid never holds slack memory. */
  finish(): Float32Array {
    return this.data.slice(0, this.used);
  }
}

/** Triangles smaller than this in area are dropped: they can only produce NaNs. */
const MIN_TRI_AREA2 = 1e-12;

function isMesh(o: THREE.Object3D): o is THREE.Mesh {
  return (o as THREE.Mesh).isMesh === true;
}

function emitGeometry(
  geo: THREE.BufferGeometry,
  matrix: THREE.Matrix4,
  out: TriBuffer,
  budget: number,
): number {
  const pos = geo.getAttribute('position');
  if (!pos || pos.itemSize < 3) return 0;

  const vcount = pos.count;
  // Transform each unique vertex once; indexed geometry reuses them 2-6x.
  const wp = new Float32Array(vcount * 3);
  for (let i = 0; i < vcount; i++) {
    _v3.fromBufferAttribute(pos, i).applyMatrix4(matrix);
    wp[i * 3] = _v3.x;
    wp[i * 3 + 1] = _v3.y;
    wp[i * 3 + 2] = _v3.z;
  }

  const index = geo.getIndex();
  const total = index ? index.count : vcount;
  const start = Math.max(0, geo.drawRange.start | 0);
  let count = geo.drawRange.count;
  if (!Number.isFinite(count)) count = total;
  count = Math.min(count, total - start);
  if (count < 3) return 0;

  let emitted = 0;
  for (let i = 0; i + 2 < count && emitted < budget; i += 3) {
    const i0 = index ? index.getX(start + i) : start + i;
    const i1 = index ? index.getX(start + i + 1) : start + i + 1;
    const i2 = index ? index.getX(start + i + 2) : start + i + 2;
    if (i0 >= vcount || i1 >= vcount || i2 >= vcount) continue;

    const ax = wp[i0 * 3], ay = wp[i0 * 3 + 1], az = wp[i0 * 3 + 2];
    const bx = wp[i1 * 3], by = wp[i1 * 3 + 1], bz = wp[i1 * 3 + 2];
    const cx = wp[i2 * 3], cy = wp[i2 * 3 + 1], cz = wp[i2 * 3 + 2];

    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    if (nx * nx + ny * ny + nz * nz < MIN_TRI_AREA2) continue;
    if (!Number.isFinite(ax + ay + az + bx + by + bz + cx + cy + cz)) continue;

    out.push(ax, ay, az, bx, by, bz, cx, cy, cz);
    emitted++;
  }
  return emitted;
}

/**
 * Walks a subtree and bakes every mesh into world-space triangles.
 * `userData.noCollide === true` prunes the whole subtree, which gives the level
 * builder an escape hatch for decorative geometry (grass cards, decals, glow
 * quads) that must never eat a bullet.
 */
export function extractTriangles(root: THREE.Object3D, maxTriangles = 400_000): Float32Array {
  root.updateWorldMatrix(true, true);
  const out = new TriBuffer();
  let budget = maxTriangles;

  const visit = (obj: THREE.Object3D): void => {
    if (budget <= 0) return;
    if (obj.userData.noCollide === true) return;
    if (isMesh(obj) && (obj as THREE.SkinnedMesh).isSkinnedMesh !== true) {
      const geo = obj.geometry;
      if (geo && geo.isBufferGeometry) {
        const inst = obj as THREE.InstancedMesh;
        if (inst.isInstancedMesh === true) {
          const n = Math.min(inst.count, inst.instanceMatrix.count);
          for (let i = 0; i < n && budget > 0; i++) {
            inst.getMatrixAt(i, _m4);
            _m4.premultiply(inst.matrixWorld);
            budget -= emitGeometry(geo, _m4, out, budget);
          }
        } else {
          budget -= emitGeometry(geo, obj.matrixWorld, out, budget);
        }
      }
    }
    const kids = obj.children;
    for (let i = 0; i < kids.length; i++) visit(kids[i]);
  };

  visit(root);
  return out.finish();
}

// ---------------------------------------------------------------------------
// The grid
// ---------------------------------------------------------------------------

export interface StaticEntry {
  id: number;
  root: THREE.Object3D;
  surface: SurfaceKind;
  /** RayMask bits this entry answers to. */
  mask: number;
  /** World-space triangle soup, 9 floats per triangle. */
  verts: Float32Array;
  triCount: number;
}

/** Visitor for ray traversal. Returns the new distance limit; <= 0 aborts. */
export type RayVisitor = (tri: number) => number;
export type AabbVisitor = (tri: number) => void;

const BASE_CELL = 2.0;
/** ~8MB of Int32 for the cell table at the top end; plenty for an MP-sized map. */
const MAX_CELLS = 2_000_000;
/** A triangle spanning more cells than this is cheaper to test unconditionally. */
const MAX_TRI_CELLS = 1024;

export class TriangleGrid {
  private readonly entries: StaticEntry[] = [];
  private dirty = true;

  triCount = 0;
  /** 9 floats per triangle, world space. */
  verts = new Float32Array(0);
  /** 3 floats per triangle, unit geometric normal. */
  normals = new Float32Array(0);
  triEntry = new Int32Array(0);
  triMask = new Int32Array(0);
  triSurface = new Uint8Array(0);

  private triStamp = new Int32Array(0);
  private stamp = 0;

  /**
   * Hands out the next visited-stamp. Wrapping past Int32 would make stale marks
   * read as fresh and silently drop triangles from a query, so the counter is
   * recycled long before it can.
   */
  private nextStamp(): number {
    if (++this.stamp >= 0x40000000) {
      this.triStamp.fill(0);
      this.stamp = 1;
    }
    return this.stamp;
  }

  private cellStart = new Int32Array(1);
  private cellItems = new Int32Array(0);
  private oversized = new Int32Array(0);

  private cs = BASE_CELL;
  private inv = 1 / BASE_CELL;
  private nx = 1; private ny = 1; private nz = 1;
  private minX = 0; private minY = 0; private minZ = 0;
  private maxX = 0; private maxY = 0; private maxZ = 0;

  /** Lowest point of the baked world; a scalar so hot paths stay allocation free. */
  get worldMinY(): number {
    this.ensureBuilt();
    return this.minY;
  }

  /** Copies the world AABB into `out` as [minX,minY,minZ,maxX,maxY,maxZ]. */
  getBounds(out: Float64Array): Float64Array {
    this.ensureBuilt();
    out[0] = this.minX; out[1] = this.minY; out[2] = this.minZ;
    out[3] = this.maxX; out[4] = this.maxY; out[5] = this.maxZ;
    return out;
  }

  add(entry: StaticEntry): void {
    if (entry.triCount <= 0) return;
    this.entries.push(entry);
    this.dirty = true;
  }

  removeById(id: number): boolean {
    let removed = false;
    for (let i = this.entries.length - 1; i >= 0; i--) {
      if (this.entries[i].id === id) {
        this.entries.splice(i, 1);
        removed = true;
      }
    }
    if (removed) this.dirty = true;
    return removed;
  }

  entryOf(tri: number): StaticEntry | null {
    const e = this.entries[this.triEntry[tri]];
    return e === undefined ? null : e;
  }

  clear(): void {
    this.entries.length = 0;
    this.dirty = true;
    this.triCount = 0;
    this.verts = new Float32Array(0);
    this.normals = new Float32Array(0);
    this.triEntry = new Int32Array(0);
    this.triMask = new Int32Array(0);
    this.triSurface = new Uint8Array(0);
    this.triStamp = new Int32Array(0);
    this.cellStart = new Int32Array(1);
    this.cellItems = new Int32Array(0);
    this.oversized = new Int32Array(0);
  }

  markDirty(): void {
    this.dirty = true;
  }

  ensureBuilt(): void {
    if (this.dirty) this.build();
  }

  // -------------------------------------------------------------------------

  private build(): void {
    this.dirty = false;

    let total = 0;
    for (const e of this.entries) total += e.triCount;
    this.triCount = total;

    if (total === 0) {
      this.verts = new Float32Array(0);
      this.normals = new Float32Array(0);
      this.triEntry = new Int32Array(0);
      this.triMask = new Int32Array(0);
      this.triSurface = new Uint8Array(0);
      this.triStamp = new Int32Array(0);
      this.cellStart = new Int32Array(1);
      this.cellItems = new Int32Array(0);
      this.oversized = new Int32Array(0);
      this.nx = this.ny = this.nz = 1;
      this.minX = this.minY = this.minZ = 0;
      this.maxX = this.maxY = this.maxZ = 0;
      return;
    }

    const verts = new Float32Array(total * 9);
    const normals = new Float32Array(total * 3);
    const triEntry = new Int32Array(total);
    const triMask = new Int32Array(total);
    const triSurface = new Uint8Array(total);

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    let ti = 0;
    for (let ei = 0; ei < this.entries.length; ei++) {
      const e = this.entries[ei];
      const src = e.verts;
      const n = e.triCount;
      verts.set(src.subarray(0, n * 9), ti * 9);
      const si = surfaceIndex(e.surface);
      for (let k = 0; k < n; k++) {
        const o = (ti + k) * 9;
        const ax = verts[o], ay = verts[o + 1], az = verts[o + 2];
        const bx = verts[o + 3], by = verts[o + 4], bz = verts[o + 5];
        const cx = verts[o + 6], cy = verts[o + 7], cz = verts[o + 8];

        const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        let nx = e1y * e2z - e1z * e2y;
        let ny = e1z * e2x - e1x * e2z;
        let nz = e1x * e2y - e1y * e2x;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        nx /= len; ny /= len; nz /= len;
        const no = (ti + k) * 3;
        normals[no] = nx; normals[no + 1] = ny; normals[no + 2] = nz;

        triEntry[ti + k] = ei;
        triMask[ti + k] = e.mask;
        triSurface[ti + k] = si;

        if (ax < minX) minX = ax; if (ax > maxX) maxX = ax;
        if (bx < minX) minX = bx; if (bx > maxX) maxX = bx;
        if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
        if (ay < minY) minY = ay; if (ay > maxY) maxY = ay;
        if (by < minY) minY = by; if (by > maxY) maxY = by;
        if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
        if (az < minZ) minZ = az; if (az > maxZ) maxZ = az;
        if (bz < minZ) minZ = bz; if (bz > maxZ) maxZ = bz;
        if (cz < minZ) minZ = cz; if (cz > maxZ) maxZ = cz;
      }
      ti += n;
    }

    // Pad so surface-flush geometry is never exactly on the grid boundary.
    const pad = BASE_CELL;
    minX -= pad; minY -= pad; minZ -= pad;
    maxX += pad; maxY += pad; maxZ += pad;

    this.verts = verts;
    this.normals = normals;
    this.triEntry = triEntry;
    this.triMask = triMask;
    this.triSurface = triSurface;
    this.triStamp = new Int32Array(total);
    this.stamp = 0;
    this.minX = minX; this.minY = minY; this.minZ = minZ;
    this.maxX = maxX; this.maxY = maxY; this.maxZ = maxZ;

    const spanX = maxX - minX, spanY = maxY - minY, spanZ = maxZ - minZ;
    let cs = BASE_CELL;
    let nx = 1, ny = 1, nz = 1;
    for (let guard = 0; guard < 32; guard++) {
      nx = Math.max(1, Math.ceil(spanX / cs));
      ny = Math.max(1, Math.ceil(spanY / cs));
      nz = Math.max(1, Math.ceil(spanZ / cs));
      if (nx * ny * nz <= MAX_CELLS) break;
      cs *= 1.5;
    }
    this.cs = cs;
    this.inv = 1 / cs;
    this.nx = nx; this.ny = ny; this.nz = nz;

    const ncells = nx * ny * nz;
    const counts = new Int32Array(ncells + 1);
    const oversizedList: number[] = [];

    // Pass 1: count per-cell occupancy from each triangle's AABB.
    for (let t = 0; t < total; t++) {
      const o = t * 9;
      const x0 = Math.min(verts[o], verts[o + 3], verts[o + 6]);
      const x1 = Math.max(verts[o], verts[o + 3], verts[o + 6]);
      const y0 = Math.min(verts[o + 1], verts[o + 4], verts[o + 7]);
      const y1 = Math.max(verts[o + 1], verts[o + 4], verts[o + 7]);
      const z0 = Math.min(verts[o + 2], verts[o + 5], verts[o + 8]);
      const z1 = Math.max(verts[o + 2], verts[o + 5], verts[o + 8]);

      const cx0 = clampi(((x0 - minX) * this.inv) | 0, 0, nx - 1);
      const cx1 = clampi(((x1 - minX) * this.inv) | 0, 0, nx - 1);
      const cy0 = clampi(((y0 - minY) * this.inv) | 0, 0, ny - 1);
      const cy1 = clampi(((y1 - minY) * this.inv) | 0, 0, ny - 1);
      const cz0 = clampi(((z0 - minZ) * this.inv) | 0, 0, nz - 1);
      const cz1 = clampi(((z1 - minZ) * this.inv) | 0, 0, nz - 1);

      const span = (cx1 - cx0 + 1) * (cy1 - cy0 + 1) * (cz1 - cz0 + 1);
      if (span > MAX_TRI_CELLS) {
        oversizedList.push(t);
        continue;
      }
      for (let z = cz0; z <= cz1; z++) {
        for (let y = cy0; y <= cy1; y++) {
          const base = (z * ny + y) * nx;
          for (let x = cx0; x <= cx1; x++) counts[base + x + 1]++;
        }
      }
    }

    for (let i = 0; i < ncells; i++) counts[i + 1] += counts[i];
    const items = new Int32Array(counts[ncells]);
    const cursor = new Int32Array(ncells);
    cursor.set(counts.subarray(0, ncells));

    // Pass 2: scatter.
    for (let t = 0; t < total; t++) {
      const o = t * 9;
      const x0 = Math.min(verts[o], verts[o + 3], verts[o + 6]);
      const x1 = Math.max(verts[o], verts[o + 3], verts[o + 6]);
      const y0 = Math.min(verts[o + 1], verts[o + 4], verts[o + 7]);
      const y1 = Math.max(verts[o + 1], verts[o + 4], verts[o + 7]);
      const z0 = Math.min(verts[o + 2], verts[o + 5], verts[o + 8]);
      const z1 = Math.max(verts[o + 2], verts[o + 5], verts[o + 8]);

      const cx0 = clampi(((x0 - minX) * this.inv) | 0, 0, nx - 1);
      const cx1 = clampi(((x1 - minX) * this.inv) | 0, 0, nx - 1);
      const cy0 = clampi(((y0 - minY) * this.inv) | 0, 0, ny - 1);
      const cy1 = clampi(((y1 - minY) * this.inv) | 0, 0, ny - 1);
      const cz0 = clampi(((z0 - minZ) * this.inv) | 0, 0, nz - 1);
      const cz1 = clampi(((z1 - minZ) * this.inv) | 0, 0, nz - 1);

      if ((cx1 - cx0 + 1) * (cy1 - cy0 + 1) * (cz1 - cz0 + 1) > MAX_TRI_CELLS) continue;
      for (let z = cz0; z <= cz1; z++) {
        for (let y = cy0; y <= cy1; y++) {
          const base = (z * ny + y) * nx;
          for (let x = cx0; x <= cx1; x++) items[cursor[base + x]++] = t;
        }
      }
    }

    this.cellStart = counts;
    this.cellItems = items;
    this.oversized = Int32Array.from(oversizedList);
  }

  // -------------------------------------------------------------------------
  // Queries
  // -------------------------------------------------------------------------

  /**
   * Amanatides–Woo DDA. `dir` must be normalized. The visitor is called at most
   * once per triangle per traversal (stamp de-dup) and returns the distance the
   * traversal should now respect, so a closest-hit query prunes itself.
   */
  traverseRay(
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    maxDist: number,
    visit: RayVisitor,
  ): void {
    this.ensureBuilt();
    if (this.triCount === 0) return;
    if (dx === 0 && dy === 0 && dz === 0) return;

    const stamp = this.nextStamp();
    const triStamp = this.triStamp;
    let limit = maxDist;

    const over = this.oversized;
    for (let i = 0; i < over.length; i++) {
      const t = over[i];
      if (triStamp[t] === stamp) continue;
      triStamp[t] = stamp;
      limit = visit(t);
      if (limit <= 0) return;
    }

    // Clip against the grid AABB (triangles cannot exist outside it).
    let t0 = 0;
    let t1 = limit;
    if (dx !== 0) {
      const ia = (this.minX - ox) / dx;
      const ib = (this.maxX - ox) / dx;
      const lo = ia < ib ? ia : ib;
      const hi = ia < ib ? ib : ia;
      if (lo > t0) t0 = lo;
      if (hi < t1) t1 = hi;
    } else if (ox < this.minX || ox > this.maxX) return;
    if (dy !== 0) {
      const ia = (this.minY - oy) / dy;
      const ib = (this.maxY - oy) / dy;
      const lo = ia < ib ? ia : ib;
      const hi = ia < ib ? ib : ia;
      if (lo > t0) t0 = lo;
      if (hi < t1) t1 = hi;
    } else if (oy < this.minY || oy > this.maxY) return;
    if (dz !== 0) {
      const ia = (this.minZ - oz) / dz;
      const ib = (this.maxZ - oz) / dz;
      const lo = ia < ib ? ia : ib;
      const hi = ia < ib ? ib : ia;
      if (lo > t0) t0 = lo;
      if (hi < t1) t1 = hi;
    } else if (oz < this.minZ || oz > this.maxZ) return;
    if (t0 > t1) return;

    const nx = this.nx, ny = this.ny, nz = this.nz;
    const inv = this.inv, cs = this.cs;
    const entry = t0 + 1e-5;
    let cx = clampi((((ox + dx * entry) - this.minX) * inv) | 0, 0, nx - 1);
    let cy = clampi((((oy + dy * entry) - this.minY) * inv) | 0, 0, ny - 1);
    let cz = clampi((((oz + dz * entry) - this.minZ) * inv) | 0, 0, nz - 1);

    const stepX = dx > 0 ? 1 : dx < 0 ? -1 : 0;
    const stepY = dy > 0 ? 1 : dy < 0 ? -1 : 0;
    const stepZ = dz > 0 ? 1 : dz < 0 ? -1 : 0;

    let tMaxX = Infinity, tDeltaX = Infinity;
    if (stepX !== 0) {
      const bound = this.minX + (cx + (stepX > 0 ? 1 : 0)) * cs;
      tMaxX = (bound - ox) / dx;
      tDeltaX = cs / Math.abs(dx);
    }
    let tMaxY = Infinity, tDeltaY = Infinity;
    if (stepY !== 0) {
      const bound = this.minY + (cy + (stepY > 0 ? 1 : 0)) * cs;
      tMaxY = (bound - oy) / dy;
      tDeltaY = cs / Math.abs(dy);
    }
    let tMaxZ = Infinity, tDeltaZ = Infinity;
    if (stepZ !== 0) {
      const bound = this.minZ + (cz + (stepZ > 0 ? 1 : 0)) * cs;
      tMaxZ = (bound - oz) / dz;
      tDeltaZ = cs / Math.abs(dz);
    }

    const cellStart = this.cellStart;
    const cellItems = this.cellItems;
    let tCell = t0;

    for (;;) {
      const ci = (cz * ny + cy) * nx + cx;
      const s = cellStart[ci];
      const e = cellStart[ci + 1];
      for (let k = s; k < e; k++) {
        const t = cellItems[k];
        if (triStamp[t] === stamp) continue;
        triStamp[t] = stamp;
        limit = visit(t);
        if (limit <= 0) return;
      }

      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) {
          tCell = tMaxX; cx += stepX; if (cx < 0 || cx >= nx) return; tMaxX += tDeltaX;
        } else {
          tCell = tMaxZ; cz += stepZ; if (cz < 0 || cz >= nz) return; tMaxZ += tDeltaZ;
        }
      } else {
        if (tMaxY < tMaxZ) {
          tCell = tMaxY; cy += stepY; if (cy < 0 || cy >= ny) return; tMaxY += tDeltaY;
        } else {
          tCell = tMaxZ; cz += stepZ; if (cz < 0 || cz >= nz) return; tMaxZ += tDeltaZ;
        }
      }
      if (tCell > limit || tCell > t1) return;
    }
  }

  /** Visits every triangle whose cell overlaps the world-space box. */
  queryAABB(
    minx: number, miny: number, minz: number,
    maxx: number, maxy: number, maxz: number,
    visit: AabbVisitor,
  ): void {
    this.ensureBuilt();
    if (this.triCount === 0) return;

    const stamp = this.nextStamp();
    const triStamp = this.triStamp;

    const over = this.oversized;
    for (let i = 0; i < over.length; i++) {
      const t = over[i];
      if (triStamp[t] === stamp) continue;
      triStamp[t] = stamp;
      visit(t);
    }

    if (maxx < this.minX || minx > this.maxX) return;
    if (maxy < this.minY || miny > this.maxY) return;
    if (maxz < this.minZ || minz > this.maxZ) return;

    const nx = this.nx, ny = this.ny, nz = this.nz, inv = this.inv;
    const cx0 = clampi(((minx - this.minX) * inv) | 0, 0, nx - 1);
    const cx1 = clampi(((maxx - this.minX) * inv) | 0, 0, nx - 1);
    const cy0 = clampi(((miny - this.minY) * inv) | 0, 0, ny - 1);
    const cy1 = clampi(((maxy - this.minY) * inv) | 0, 0, ny - 1);
    const cz0 = clampi(((minz - this.minZ) * inv) | 0, 0, nz - 1);
    const cz1 = clampi(((maxz - this.minZ) * inv) | 0, 0, nz - 1);

    const cellStart = this.cellStart;
    const cellItems = this.cellItems;
    for (let z = cz0; z <= cz1; z++) {
      for (let y = cy0; y <= cy1; y++) {
        const base = (z * ny + y) * nx;
        for (let x = cx0; x <= cx1; x++) {
          const ci = base + x;
          const s = cellStart[ci];
          const e = cellStart[ci + 1];
          for (let k = s; k < e; k++) {
            const t = cellItems[k];
            if (triStamp[t] === stamp) continue;
            triStamp[t] = stamp;
            visit(t);
          }
        }
      }
    }
  }
}

function clampi(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// ---------------------------------------------------------------------------
// Primitive intersection maths (allocation free; results in module scratch)
// ---------------------------------------------------------------------------

/**
 * Möller–Trumbore, two-sided. Returns the ray parameter or -1 on miss.
 * The 1e-7 barycentric slack closes the hairline cracks between adjacent
 * triangles that would otherwise leak bullets through welded level geometry.
 */
export function rayTriangle(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): number {
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
  const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;

  const px = dy * e2z - dz * e2y;
  const py = dz * e2x - dx * e2z;
  const pz = dx * e2y - dy * e2x;

  const det = e1x * px + e1y * py + e1z * pz;
  if (det > -1e-12 && det < 1e-12) return -1;
  const invDet = 1 / det;

  const tx = ox - ax, ty = oy - ay, tz = oz - az;
  const u = (tx * px + ty * py + tz * pz) * invDet;
  if (u < -1e-7 || u > 1 + 1e-7) return -1;

  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;

  const v = (dx * qx + dy * qy + dz * qz) * invDet;
  if (v < -1e-7 || u + v > 1 + 1e-7) return -1;

  return (e2x * qx + e2y * qy + e2z * qz) * invDet;
}

/** Closest point on triangle ABC to P (Ericson, Real-Time Collision Detection). */
export function closestPointTriangle(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
  out: Float64Array, o: number,
): void {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;

  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) { out[o] = ax; out[o + 1] = ay; out[o + 2] = az; return; }

  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) { out[o] = bx; out[o + 1] = by; out[o + 2] = bz; return; }

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    out[o] = ax + abx * v; out[o + 1] = ay + aby * v; out[o + 2] = az + abz * v;
    return;
  }

  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) { out[o] = cx; out[o + 1] = cy; out[o + 2] = cz; return; }

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    out[o] = ax + acx * w; out[o + 1] = ay + acy * w; out[o + 2] = az + acz * w;
    return;
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    out[o] = bx + (cx - bx) * w; out[o + 1] = by + (cy - by) * w; out[o + 2] = bz + (cz - bz) * w;
    return;
  }

  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  out[o] = ax + abx * v + acx * w;
  out[o + 1] = ay + aby * v + acy * w;
  out[o + 2] = az + abz * v + acz * w;
}

const _ss = new Float64Array(6);

/** Closest points between segments P1Q1 and P2Q2; returns squared distance. */
function closestSegmentSegment(
  p1x: number, p1y: number, p1z: number,
  q1x: number, q1y: number, q1z: number,
  p2x: number, p2y: number, p2z: number,
  q2x: number, q2y: number, q2z: number,
): number {
  const d1x = q1x - p1x, d1y = q1y - p1y, d1z = q1z - p1z;
  const d2x = q2x - p2x, d2y = q2y - p2y, d2z = q2z - p2z;
  const rx = p1x - p2x, ry = p1y - p2y, rz = p1z - p2z;
  const a = d1x * d1x + d1y * d1y + d1z * d1z;
  const e = d2x * d2x + d2y * d2y + d2z * d2z;
  const f = d2x * rx + d2y * ry + d2z * rz;

  let s: number;
  let t: number;
  const EPS = 1e-12;
  if (a <= EPS && e <= EPS) {
    s = 0; t = 0;
  } else if (a <= EPS) {
    s = 0;
    t = clamp01(f / e);
  } else {
    const c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= EPS) {
      t = 0;
      s = clamp01(-c / a);
    } else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z;
      const denom = a * e - b * b;
      s = denom !== 0 ? clamp01((b * f - c * e) / denom) : 0;
      t = (b * s + f) / e;
      if (t < 0) { t = 0; s = clamp01(-c / a); }
      else if (t > 1) { t = 1; s = clamp01((b - c) / a); }
    }
  }

  const c1x = p1x + d1x * s, c1y = p1y + d1y * s, c1z = p1z + d1z * s;
  const c2x = p2x + d2x * t, c2y = p2y + d2y * t, c2z = p2z + d2z * t;
  _ss[0] = c1x; _ss[1] = c1y; _ss[2] = c1z;
  _ss[3] = c2x; _ss[4] = c2y; _ss[5] = c2z;
  const dx = c1x - c2x, dy = c1y - c2y, dz = c1z - c2z;
  return dx * dx + dy * dy + dz * dz;
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Closest point on the segment, written by `closestSegmentTriangle`. */
export const CST_SEG = new Float64Array(3);
/** Closest point on the triangle, written by `closestSegmentTriangle`. */
export const CST_TRI = new Float64Array(3);

const _cpA = new Float64Array(3);

/**
 * Squared distance between segment PQ and triangle ABC, writing the witness
 * points into CST_SEG / CST_TRI. Returns 0 when the segment pierces the
 * triangle, in which case both witnesses are the pierce point.
 */
export function closestSegmentTriangle(
  px: number, py: number, pz: number,
  qx: number, qy: number, qz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): number {
  // Piercing case first: the witness maths below cannot see through the face.
  const sx = qx - px, sy = qy - py, sz = qz - pz;
  const segLen2 = sx * sx + sy * sy + sz * sz;
  if (segLen2 > 1e-16) {
    const t = rayTriangle(px, py, pz, sx, sy, sz, ax, ay, az, bx, by, bz, cx, cy, cz);
    if (t >= 0 && t <= 1) {
      const hx = px + sx * t, hy = py + sy * t, hz = pz + sz * t;
      CST_SEG[0] = hx; CST_SEG[1] = hy; CST_SEG[2] = hz;
      CST_TRI[0] = hx; CST_TRI[1] = hy; CST_TRI[2] = hz;
      return 0;
    }
  }

  let best = Infinity;

  closestPointTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz, _cpA, 0);
  let dx = px - _cpA[0], dy = py - _cpA[1], dz = pz - _cpA[2];
  let d2 = dx * dx + dy * dy + dz * dz;
  if (d2 < best) {
    best = d2;
    CST_SEG[0] = px; CST_SEG[1] = py; CST_SEG[2] = pz;
    CST_TRI[0] = _cpA[0]; CST_TRI[1] = _cpA[1]; CST_TRI[2] = _cpA[2];
  }

  closestPointTriangle(qx, qy, qz, ax, ay, az, bx, by, bz, cx, cy, cz, _cpA, 0);
  dx = qx - _cpA[0]; dy = qy - _cpA[1]; dz = qz - _cpA[2];
  d2 = dx * dx + dy * dy + dz * dz;
  if (d2 < best) {
    best = d2;
    CST_SEG[0] = qx; CST_SEG[1] = qy; CST_SEG[2] = qz;
    CST_TRI[0] = _cpA[0]; CST_TRI[1] = _cpA[1]; CST_TRI[2] = _cpA[2];
  }

  d2 = closestSegmentSegment(px, py, pz, qx, qy, qz, ax, ay, az, bx, by, bz);
  if (d2 < best) {
    best = d2;
    CST_SEG[0] = _ss[0]; CST_SEG[1] = _ss[1]; CST_SEG[2] = _ss[2];
    CST_TRI[0] = _ss[3]; CST_TRI[1] = _ss[4]; CST_TRI[2] = _ss[5];
  }
  d2 = closestSegmentSegment(px, py, pz, qx, qy, qz, bx, by, bz, cx, cy, cz);
  if (d2 < best) {
    best = d2;
    CST_SEG[0] = _ss[0]; CST_SEG[1] = _ss[1]; CST_SEG[2] = _ss[2];
    CST_TRI[0] = _ss[3]; CST_TRI[1] = _ss[4]; CST_TRI[2] = _ss[5];
  }
  d2 = closestSegmentSegment(px, py, pz, qx, qy, qz, cx, cy, cz, ax, ay, az);
  if (d2 < best) {
    best = d2;
    CST_SEG[0] = _ss[0]; CST_SEG[1] = _ss[1]; CST_SEG[2] = _ss[2];
    CST_TRI[0] = _ss[3]; CST_TRI[1] = _ss[4]; CST_TRI[2] = _ss[5];
  }

  return best;
}
