/**
 * The map: a war-torn Mediterranean town square, roughly 90x90m.
 *
 * LAYOUT — three lanes, in the Crash/Backlot mould:
 *
 *                      N1  (archway)          z = -38
 *        W3 ┃                  ┃  E4 ┃
 *           ┃   W1 (interior)  ┃ E2  ┃ A1     the east alley network
 *   west    ┃ ┌──────────┐     ┃ ware┃ ↕
 *   alley   ┃ │ shop/stair│  ▓▓ SQUARE ▓▓     fountain, car, stalls
 *     ↕     ┃ └──────────┘     ┃ E1  ┃ A3
 *           ┃   W2 (arcade)    ┃ cafe┃ ↕
 *                      S1  (archway)          z = +38
 *
 *  - MID:   the main street on x ∈ [-6.8, 7] opening into the square. Long
 *           sightline, broken at three points by the fountain, a burnt-out car
 *           and a barrier checkpoint, so it is interesting but not a rail.
 *  - WEST:  the interior route — open shop front on the square, back corridor,
 *           dog-legged stairwell, first-floor balcony over the square, second
 *           floor with a collapsed slab, out onto bay A's roof terrace. South
 *           of it, W2's arcade is a covered colonnade you can run the length of.
 *  - EAST:  an alley network — A2 crosses east from the street, A1 runs north
 *           behind the warehouse and A3 south behind the cafe, offset from each
 *           other so neither is a straight shot. The warehouse breach is a
 *           third way in, and its mezzanine overlooks the square.
 *
 * Everything solid registers a *box* with the physics world, never the visual
 * mesh. Visual geometry carries every chamfer and greeble it needs; collision
 * stays at ~13k triangles so bullet and capsule queries are cheap and nothing
 * ever snags on a 2cm bevel.
 */
import * as THREE from 'three';
import type { GameContext, FrameTime, System, SurfaceKind } from '../core/Contracts';
import { MaterialLibrary, type SurfaceLook } from './Materials';
import {
  ColliderSet,
  ContactField,
  GeoBuilder,
  OcclusionBaker,
  Rng,
  archway,
  bevelBox,
  breachedWall,
  cableGeo,
  copingProfile,
  corniceProfile,
  cylinderGeo,
  doorGeo,
  drapedSheet,
  extrudeProfile,
  finalizeGeometry,
  greebleFace,
  kerbProfile,
  matrixOf,
  paintAerial,
  paintRunoff,
  paintSphere,
  panelledWall,
  pipeRun,
  plainBox,
  railingGeo,
  rubbleCone,
  scaleUv,
  shutterGeo,
  sillProfile,
  spallPatch,
  stairsGeo,
  tintGeometry,
  wallWithOpenings,
  windowFrameGeo,
  worldPlanarUv,
  type BoxSpec,
  type Opening,
} from './Geometry';
import { MaterialVault, PropSystem, buildBurntCar, buildLaundryLine, type PropKind } from './Props';

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------

interface SurfaceDef {
  look: SurfaceLook;
  /** Triplanar tiles per metre. */
  tiles: number;
  color?: number;
  seed?: number;
  roughness?: number;
  metalness?: number;
  emissive?: number;
  emissiveIntensity?: number;
  side?: THREE.Side;
  /** Overrides the look's default surface kind for footsteps and impacts. */
  kind?: SurfaceKind;
}

/**
 * A deliberately narrow palette: five plaster tints in the same warm family,
 * one cool accent, and everything else grey-brown. Colour variety in a town
 * comes from *value* and *wear*, not hue.
 */
const SURFACES = {
  sand: { look: 'sand', tiles: 0.42, seed: 1 },
  // The asphalt map is already authored at a road's real 0.09-0.13 reflectance.
  // Tinting it down another two thirds left the street darker than the sky's
  // specular reflection on it, which is what made shaded tarmac render navy.
  road: { look: 'asphalt', tiles: 0.34, seed: 2, color: 0xd9d3c7 },
  pavement: { look: 'concrete_floor', tiles: 0.46, seed: 3, color: 0xbdb4a1 },
  paving: { look: 'tile_floor', tiles: 0.5, seed: 4, color: 0xb5aa93 },
  dirt: { look: 'dirt_gravel', tiles: 0.55, seed: 5 },

  // One family, five values. Every tint sits within about 12 degrees of hue of
  // the next; what separates the buildings is how light they are and how much
  // the weather has taken off them. A war-torn town does not have a mint
  // building next to a salmon building — it has one limestone quarry, one
  // render mix, and sixty years of sun.
  plasterCream: { look: 'plaster_painted', tiles: 0.44, seed: 11, color: 0xcdc0a8 },
  plasterOchre: { look: 'plaster_painted', tiles: 0.44, seed: 12, color: 0xb8a381 },
  plasterWhite: { look: 'plaster_painted', tiles: 0.44, seed: 13, color: 0xd6cebc },
  plasterBlue: { look: 'plaster_painted', tiles: 0.44, seed: 14, color: 0xa9a898 },
  plasterRose: { look: 'plaster_painted', tiles: 0.44, seed: 15, color: 0xb59a89 },

  // Set back behind every opening on a building you cannot enter. Without it a
  // window is a hole onto the inside of an unlit box, which reads as a bug.
  // Not black: a room that has any bounce at all in it is warm and very dark,
  // and a warm very-dark reads as depth where 0x000000 reads as a hole in the
  // frame buffer.
  interiorDark: { look: 'plaster_painted', tiles: 0.7, seed: 16, color: 0x2e2418, roughness: 1.25 },

  stone: { look: 'concrete_wall', tiles: 0.42, seed: 21, color: 0xbdb09a },
  concrete: { look: 'concrete_wall', tiles: 0.44, seed: 22, color: 0xa9a49a },
  // Untinted, the brick map renders as a bright salmon rectangle against pale
  // limestone render and every spall patch reads as a sticker. A brick that has
  // been weathering behind a coat of render for fifty years is dark and grey.
  brick: { look: 'brick', tiles: 0.5, seed: 23, color: 0xac8a70 },
  rubble: { look: 'rubble', tiles: 0.65, seed: 24 },

  wood: { look: 'wood_plank', tiles: 1.0, seed: 31, color: 0x9c7c52 },
  woodDark: { look: 'wood_plank', tiles: 1.1, seed: 32, color: 0x6c5236 },
  metalRust: { look: 'rusted_metal', tiles: 1.0, seed: 41, color: 0x8a6a52 },
  metalPaint: { look: 'painted_metal', tiles: 1.0, seed: 42, color: 0xa8a396 },
  corrugated: { look: 'corrugated_metal', tiles: 0.7, seed: 43, color: 0x9a9186 },
  glass: { look: 'glass_dirty', tiles: 1.0, seed: 51 },
  fabric: { look: 'fabric_canvas', tiles: 1.2, seed: 61, color: 0xc2ac86 },

  signRed: { look: 'painted_metal', tiles: 1.4, seed: 71, color: 0x8e2f26 },
  signBlue: { look: 'painted_metal', tiles: 1.4, seed: 72, color: 0x2f4f6e },
  // The one lit element on the map: a tube sign over the cafe. Kept dim so the
  // bloom gets a tight core rather than a haze.
  signLit: {
    look: 'painted_metal', tiles: 1.6, seed: 73, color: 0x2a2622,
    emissive: 0xffb45a, emissiveIntensity: 2.4,
  },
} satisfies Record<string, SurfaceDef>;

type MatKey = keyof typeof SURFACES;

const surfaceDef = (m: MatKey): SurfaceDef => SURFACES[m];

// ---------------------------------------------------------------------------
// Face placement
// ---------------------------------------------------------------------------

type Axis = 'x' | 'z';

/**
 * Maps "a wall face at world plane `at`, running along `axis` from `from` to
 * `to`, looking `outward`" onto local wall space. Every facade, its trim, its
 * shutters and its balconies are positioned through this so nothing has to
 * re-derive a rotation by hand.
 */
class Face {
  readonly yaw: number;
  readonly centre: number;
  readonly length: number;
  /** Sign converting a world run-coordinate into wall-local X. */
  readonly runSign: number;

  constructor(
    readonly axis: Axis,
    readonly from: number,
    readonly to: number,
    readonly at: number,
    readonly outward: 1 | -1,
    readonly base: number,
  ) {
    this.centre = (from + to) * 0.5;
    this.length = Math.abs(to - from);
    this.yaw = axis === 'z' ? (outward > 0 ? Math.PI * 0.5 : -Math.PI * 0.5) : outward > 0 ? 0 : Math.PI;
    this.runSign = axis === 'z' ? -outward : outward;
  }

  /** Wall-local X for a world coordinate along the run. */
  local(run: number): number {
    return (run - this.centre) * this.runSign;
  }

  /** World position: `run` along the face, `y` above the base, `depth` outward. */
  world(run: number, y: number, depth: number, out = new THREE.Vector3()): THREE.Vector3 {
    if (this.axis === 'z') out.set(this.at + this.outward * depth, this.base + y, run);
    else out.set(run, this.base + y, this.at + this.outward * depth);
    return out;
  }

  /** Placement matrix for an object whose local +Z should face outward. */
  matrix(run: number, y: number, depth: number, extraYaw = 0, scale = 1): THREE.Matrix4 {
    const p = this.world(run, y, depth);
    return matrixOf(p.x, p.y, p.z, this.yaw + extraYaw, 0, 0, scale);
  }

  /** World run-coordinate for a wall-local X. Inverse of `local`. */
  run(localX: number): number {
    return this.centre + localX * this.runSign;
  }
}

/**
 * Wall-local X for a point on an elevation, without constructing the Face.
 * Used when authoring a shell's extra openings, which are declared before the
 * shell builds its own faces.
 */
function faceLocal(axis: Axis, from: number, to: number, outward: 1 | -1, run: number): number {
  return (run - (from + to) * 0.5) * (axis === 'z' ? -outward : outward);
}

// ---------------------------------------------------------------------------
// Lighting host (structural — never imports the lighting subsystem)
// ---------------------------------------------------------------------------

interface LightingLike extends System {
  addPointLight(
    position: THREE.Vector3,
    color: THREE.ColorRepresentation,
    intensity: number,
    range: number,
    castShadow?: boolean,
  ): THREE.PointLight;
  removeLight(light: THREE.Light): void;
}

// ---------------------------------------------------------------------------
// Public data
// ---------------------------------------------------------------------------

export interface SpawnPoint {
  position: THREE.Vector3;
  yaw: number;
}

export interface CoverPoint {
  position: THREE.Vector3;
  normal: THREE.Vector3;
}

interface FloorRect {
  x0: number;
  z0: number;
  x1: number;
  z1: number;
  y: number;
}

interface Bucket {
  key: string;
  mat: MatKey;
  builder: GeoBuilder;
}

const FLOOR_H = 3.5;
const ROAD_W0 = -6.8;
const ROAD_W1 = 7.0;

const _wv = new THREE.Vector3();

// ---------------------------------------------------------------------------

export class LevelSystem implements System {
  readonly name = 'level';

  private ctx: GameContext | null = null;
  private materials: MaterialLibrary | null = null;
  private vault: MaterialVault | null = null;
  private props: PropSystem | null = null;
  private collider = new ColliderSet();

  private readonly root = new THREE.Group();
  private readonly buckets = new Map<string, Bucket>();
  private readonly meshes: THREE.Mesh[] = [];
  private readonly collisionMeshes: THREE.Mesh[] = [];
  private readonly lights: THREE.Light[] = [];
  private readonly disposables: (THREE.BufferGeometry | THREE.Material | THREE.Texture)[] = [];

  private readonly floors: FloorRect[] = [];
  private readonly spawns: SpawnPoint[] = [];
  private readonly cover: CoverPoint[] = [];
  private readonly navBounds = new THREE.Box3(
    new THREE.Vector3(-42, -2, -42),
    new THREE.Vector3(42, 26, 42),
  );

  private waterNormal: THREE.Texture | null = null;

  /**
   * Blast points. Applied to every merged bucket at commit time as a vertex
   * halo, so soot runs across whatever happens to be near it — road, kerb,
   * wall base, rubble — instead of stopping at an object boundary the way a
   * decal does.
   */
  private readonly scorch: { p: THREE.Vector3; r: number; s: number }[] = [];

  // =========================================================================
  // Lifecycle
  // =========================================================================

  init(ctx: GameContext): void {
    this.ctx = ctx;
    this.root.name = 'level';
    ctx.scene.add(this.root);

    this.materials = new MaterialLibrary(ctx.renderer, ctx.quality);
    this.vault = new MaterialVault(this.materials, ctx.quality);
    this.props = new PropSystem(this.vault, ctx.quality, 0x51ee7);

    this.buildTerrain();
    this.buildPerimeter();
    this.buildWestBlocks();
    this.buildEastBlocks();
    this.buildEndCaps();
    this.buildSquare();
    this.buildUtilities();
    this.buildScatter();
    this.commit(ctx);
    this.buildLights(ctx);
    this.buildSpawns();
    this.buildCover();
  }

  update(time: FrameTime, ctx: GameContext): void {
    this.props?.update(time.elapsed);
    if (this.waterNormal) {
      // Two-rate scroll: the surface drifts, the ripples run faster across it.
      this.waterNormal.offset.x = (time.elapsed * 0.021) % 1;
      this.waterNormal.offset.y = (time.elapsed * 0.037) % 1;
    }
  }

  dispose(): void {
    const ctx = this.ctx;
    for (const m of this.collisionMeshes) {
      ctx?.physics.removeStatic(m);
      m.geometry.dispose();
    }
    this.collisionMeshes.length = 0;

    const lighting = ctx?.system<LightingLike>('lighting');
    for (const l of this.lights) lighting?.removeLight(l);
    this.lights.length = 0;

    this.root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh && !(mesh as THREE.InstancedMesh).isInstancedMesh) mesh.geometry.dispose();
    });
    this.meshes.length = 0;
    for (const d of this.disposables) d.dispose();
    this.disposables.length = 0;

    for (const b of this.buckets.values()) b.builder.dispose();
    this.buckets.clear();

    this.props?.dispose();
    this.vault?.dispose();
    this.materials?.dispose();
    this.props = null;
    this.vault = null;
    this.materials = null;

    this.root.clear();
    this.root.parent?.remove(this.root);
    this.ctx = null;
  }

  // =========================================================================
  // Public queries
  // =========================================================================

  getSpawnPoints(): SpawnPoint[] {
    return this.spawns.map((s) => ({ position: s.position.clone(), yaw: s.yaw }));
  }

  getCoverPoints(): CoverPoint[] {
    return this.cover.map((c) => ({ position: c.position.clone(), normal: c.normal.clone() }));
  }

  getNavBounds(): THREE.Box3 {
    return this.navBounds.clone();
  }

  /**
   * Walkable height under a point. `below` picks the highest floor at or under
   * a reference height, which is what lets a query inside a building find the
   * shop floor instead of the roof three storeys above it.
   */
  groundHeight(x: number, z: number, below = Infinity): number {
    let best = 0;
    for (const f of this.floors) {
      if (x < f.x0 || x > f.x1 || z < f.z0 || z > f.z1) continue;
      if (f.y > below + 1e-3) continue;
      if (f.y > best) best = f.y;
    }
    return best;
  }

  // =========================================================================
  // Bucketed geometry
  // =========================================================================

  private bucket(zone: string, mat: MatKey): GeoBuilder {
    const key = `${zone}/${mat}`;
    let b = this.buckets.get(key);
    if (!b) {
      b = { key, mat, builder: new GeoBuilder() };
      this.buckets.set(key, b);
    }
    return b.builder;
  }

  private add(zone: string, mat: MatKey, geo: THREE.BufferGeometry, matrix?: THREE.Matrix4): void {
    this.bucket(zone, mat).add(geo, matrix);
  }

  private solid(surface: SurfaceKind, boxes: readonly BoxSpec[], matrix?: THREE.Matrix4): void {
    this.collider.addBoxes(surface, boxes, matrix);
  }

  private kindOf(mat: MatKey): SurfaceKind {
    const def = surfaceDef(mat);
    return def.kind ?? (this.vault as MaterialVault).surfaceOf(def.look);
  }

  // =========================================================================
  // Terrain
  // =========================================================================

  /**
   * Tessellated ground. The subdivision is not decorative: vertex-baked AO and
   * ground grime need vertices to write into, and a two-triangle plane cannot
   * darken where it meets a wall.
   */
  private groundGrid(
    x0: number,
    z0: number,
    x1: number,
    z1: number,
    y: number,
    step: number,
    seed: number,
    undulate = 0.018,
  ): THREE.BufferGeometry {
    const w = x1 - x0;
    const d = z1 - z0;
    const nx = Math.max(1, Math.round(w / step));
    const nz = Math.max(1, Math.round(d / step));
    const geo = new THREE.PlaneGeometry(w, d, nx, nz);
    geo.rotateX(-Math.PI * 0.5);
    geo.translate((x0 + x1) * 0.5, y, (z0 + z1) * 0.5);
    if (undulate > 0) {
      const rng = new Rng(seed);
      const pos = geo.getAttribute('position') as THREE.BufferAttribute;
      for (let i = 0; i < pos.count; i++) {
        const px = pos.getX(i);
        const pz = pos.getZ(i);
        // Deterministic value noise, two octaves; keeps the surface from being
        // a perfect optical plane without breaking the walkable height.
        const n =
          Math.sin(px * 0.37 + pz * 0.23) * 0.6 +
          Math.sin(px * 1.13 - pz * 0.91) * 0.3 +
          rng.jitter(0.25);
        pos.setY(i, y + n * undulate);
      }
      pos.needsUpdate = true;
      geo.computeVertexNormals();
    }
    return finalizeGeometry(geo);
  }

  /**
   * Ground is layered by height, not by draw order: sand sits 6cm below the
   * road, dirt 1cm above it, paving 2cm. Each layer's undulation amplitude is
   * smaller than the gap to the layer above, which is what keeps a hand-placed
   * stack of overlapping surfaces from ever z-fighting.
   */
  private buildTerrain(): void {
    // Base ground, wider than the play space so no edge is ever visible.
    this.add('ground', 'sand', this.groundGrid(-70, -70, 70, 70, -0.06, 4.0, 1, 0.03));
    this.collider.addBox('sand', { cx: 0, cy: -0.56, cz: 0, sx: 170, sy: 1, sz: 170 });
    this.floors.push({ x0: -70, z0: -70, x1: 70, z1: 70, y: -0.05 });

    // Road: south leg, square, north leg. It runs on through both archways so
    // the vista reads as one continuous street rather than stopping at a wall.
    // The 0.7m step is not decoration: the baked contact shadow along every
    // kerb and wall base is a per-vertex term, and a 1.6m grid cannot resolve a
    // 40cm dark line. Ground triangles are the cheapest in the map.
    this.add('mid', 'road', this.groundGrid(ROAD_W0, -41, ROAD_W1, 41, 0, 0.85, 2, 0.012));
    this.collider.addBox('concrete', {
      cx: (ROAD_W0 + ROAD_W1) * 0.5, cy: -0.18, cz: 0, sx: ROAD_W1 - ROAD_W0, sy: 0.36, sz: 82,
    });
    this.floors.push({ x0: ROAD_W0, z0: -41, x1: ROAD_W1, z1: 41, y: 0 });

    // Square paving on the east side of the road.
    this.add('mid', 'paving', this.groundGrid(ROAD_W1, -26, 16, -4, 0.02, 0.85, 3, 0.01));
    this.collider.addBox('concrete', { cx: 11.5, cy: -0.16, cz: -15, sx: 9, sy: 0.36, sz: 22 });
    this.floors.push({ x0: ROAD_W1, z0: -26, x1: 16, z1: -4, y: 0.02 });

    // Pavement strip in front of the cafe, with a kerb.
    this.add('east', 'pavement', this.groundGrid(ROAD_W1, -4, 9.5, 26, 0.15, 0.8, 4, 0.008));
    this.collider.addBox('concrete', { cx: 8.25, cy: 0.0, cz: 11, sx: 2.5, sy: 0.3, sz: 30 });
    this.floors.push({ x0: ROAD_W1, z0: -4, x1: 9.5, z1: 26, y: 0.15 });

    const kerb = extrudeProfile(kerbProfile(0.15, 0.17), 30, { bevel: 0.01 });
    kerb.rotateY(Math.PI);
    this.add('east', 'pavement', kerb, matrixOf(ROAD_W1, 0, 11, 0));

    // Alley, courtyard and forecourt floors — compacted dirt, not asphalt.
    for (const r of [
      { x0: -30, z0: -32, x1: -26, z1: 32, s: 6 },
      { x0: 30, z0: -30, x1: 34, z1: -6, s: 7 },
      { x0: 9.5, z0: -4, x1: 40, z1: 2, s: 8 },
      { x0: 22, z0: 2, x1: 26, z1: 20, s: 9 },
      { x0: 9.5, z0: 20, x1: 26, z1: 26, s: 10 },
      { x0: -26, z0: 2, x1: -6.9, z1: 6, s: 11 },
      // Between the road and the arcade colonnade.
      { x0: -10.7, z0: 5.6, x1: ROAD_W0, z1: 30.4, s: 12 },
      // North and south of the square, either side of the road.
      { x0: ROAD_W1, z0: -34, x1: 16.2, z1: -26, s: 13 },
      { x0: ROAD_W1, z0: 26, x1: 26, z1: 34, s: 14 },
      { x0: -10.7, z0: 30, x1: ROAD_W0, z1: 34, s: 15 },
      { x0: -26, z0: -32, x1: -6.9, z1: -26, s: 16 },
      { x0: 16, z0: -34, x1: 44, z1: -30, s: 17 },
    ]) {
      this.add('ground', 'dirt', this.groundGrid(r.x0, r.z0, r.x1, r.z1, 0.01, 1.0, r.s, 0.02));
      this.collider.addBox('dirt', {
        cx: (r.x0 + r.x1) * 0.5, cy: -0.17, cz: (r.z0 + r.z1) * 0.5,
        sx: r.x1 - r.x0, sy: 0.36, sz: r.z1 - r.z0,
      });
      this.floors.push({ x0: r.x0, z0: r.z0, x1: r.x1, z1: r.z1, y: 0.01 });
    }
  }

  // =========================================================================
  // Facades
  // =========================================================================

  /**
   * Window openings for one elevation. Bays are irregular by design: a fixed
   * pitch with a couple of skipped, widened or shifted bays per storey is what
   * separates a street from a spreadsheet.
   */
  private windowRows(
    face: Face,
    storeys: number,
    rng: Rng,
    opts: { bay?: number; groundSkip?: boolean; balconyRow?: number } = {},
  ): { openings: Opening[]; runs: { run: number; y: number; w: number; h: number; storey: number }[] } {
    const bay = opts.bay ?? 2.85;
    const usable = face.length - 1.4;
    const count = Math.max(1, Math.round(usable / bay));
    const pitch = usable / count;
    const openings: Opening[] = [];
    const runs: { run: number; y: number; w: number; h: number; storey: number }[] = [];

    for (let s = 0; s < storeys; s++) {
      if (s === 0 && opts.groundSkip) continue;
      const floorY = s * FLOOR_H;
      for (let i = 0; i < count; i++) {
        if (rng.chance(0.1)) continue;
        const run = face.from + (face.to - face.from) * ((i + 0.5) / count);
        let w = 0.92 + rng.range(-0.06, 0.34);
        let h = 1.5 + rng.range(-0.1, 0.24);
        let sill = floorY + 0.95 + rng.jitter(0.06);
        if (opts.balconyRow === s && rng.chance(0.45)) {
          // A full-height balcony door instead of a window.
          w = 0.94;
          h = 2.16;
          sill = floorY + 0.06;
        }
        if (pitch < w + 0.5) continue;
        openings.push({ x: face.local(run), y: sill, w, h });
        runs.push({ run, y: sill, w, h, storey: s });
      }
    }
    return { openings, runs };
  }

  /**
   * Builds one elevation: slab with holes, plinth, string course, cornice, then
   * joinery, shutters, runoff staining and pipework in the openings.
   */
  private facade(o: {
    zone: string;
    mat: MatKey;
    face: Face;
    height: number;
    thickness?: number;
    storeys?: number;
    extraOpenings?: Opening[];
    plinth?: boolean;
    cornice?: boolean;
    detail?: boolean;
    greeble?: number;
    seed: number;
    groundSkip?: boolean;
    balconyRow?: number;
    windows?: boolean;
    twoLeaf?: boolean;
    /** Shell holes punched through this elevation, in world run-coordinates. */
    breaches?: { run: number; y: number; r: number; seed: number }[];
    /** False when the openings lead into a real, built interior. */
    backing?: boolean;
  }): { runs: { run: number; y: number; w: number; h: number; storey: number }[] } {
    const face = o.face;
    const rng = new Rng(o.seed);
    const t = o.thickness ?? 0.42;
    const storeys = o.storeys ?? Math.max(1, Math.floor(o.height / FLOOR_H));

    const win = o.windows === false
      ? { openings: [] as Opening[], runs: [] }
      : this.windowRows(face, storeys, rng, { groundSkip: o.groundSkip, balconyRow: o.balconyRow });
    const breachHoles: Opening[] = (o.breaches ?? []).map((b) => ({
      x: face.local(b.run),
      y: Math.max(0.25, b.y - b.r),
      w: b.r * 2.1,
      h: b.r * 1.8,
      raw: true,
    }));
    const openings = [...win.openings, ...(o.extraOpenings ?? []), ...breachHoles];

    const wall = wallWithOpenings(face.length, o.height, t, {
      openings,
      plinth: o.plinth === false ? 0 : 0.26,
      bandY: storeys > 1 ? FLOOR_H - 0.18 : undefined,
      twoLeaf: o.twoLeaf !== false,
      // A 10cm rebate reads as a hole in card. 18cm plus the 23cm leaf step is
      // half a metre of masonry you can see the thickness of from any angle.
      reveal: 0.18,
      bevel: 0.022,
    });

    // Runoff below every sill, and grime pooling in the plinth reveal.
    for (const w of win.runs) {
      paintRunoff(wall.geometry, face.local(w.run), t * 0.5, w.y - 0.09, Math.min(2.6, w.y - 0.2), w.w * 0.62, 0x5d5142, 0.72);
    }

    const m = face.matrix(face.centre, 0, -t * 0.5);
    this.add(o.zone, o.mat, wall.geometry, m);
    this.solid(this.kindOf(o.mat), wall.boxes, m);

    // Fake room behind each opening: a dark panel plus a side return, so a
    // grazing view still shows depth rather than a flat black rectangle.
    if (o.backing !== false) {
      for (const op of openings) {
        const run = face.run(op.x);
        // Deeper rooms behind wider openings, so a shop front does not read
        // as a 55cm cupboard.
        const depth = -(t * 0.5 + 0.5 + Math.min(1.7, op.w * 0.38));
        const back = plainBox(op.w + 0.5, op.h + 0.5, 0.1);
        this.add(o.zone, 'interiorDark', back, face.matrix(run, op.y + op.h * 0.5, depth));
        const roomD = Math.abs(depth) - t * 0.5;
        const side = plainBox(0.1, op.h + 0.5, roomD);
        for (const s of [-1, 1]) {
          this.add(
            o.zone,
            'interiorDark',
            side.clone(),
            face.matrix(run + face.runSign * s * (op.w * 0.5 + 0.22), op.y + op.h * 0.5, depth + roomD * 0.5),
          );
        }
        side.dispose();
        const soffit = plainBox(op.w + 0.5, 0.1, roomD);
        this.add(o.zone, 'interiorDark', soffit, face.matrix(run, op.y + op.h + 0.2, depth + roomD * 0.5));
      }
    }

    // Render fallen off in patches, exposing the brick. Weighted low on the
    // wall and around openings, which is where damp and blast actually take it.
    if (o.detail !== false) {
      const patches = 2 + (rng.chance(0.5) ? 1 : 0);
      for (let i = 0; i < patches; i++) {
        const near = win.runs.length > 0 && rng.chance(0.55) ? rng.pick(win.runs) : null;
        const run = near ? near.run + rng.jitter(1.1) : face.from + rng.next() * (face.to - face.from);
        const y = near
          ? Math.max(0.4, near.y + rng.jitter(1.3))
          : 0.3 + Math.pow(rng.next(), 1.7) * (o.height - 0.9);
        const p = spallPatch(rng.range(0.5, 1.35), rng.range(0.4, 0.95), 0.02, o.seed * 31 + i * 17);
        this.add(o.zone, 'brick', p, face.matrix(run, y, t * 0.5 + 0.004));
      }
    }

    if (o.greeble && o.greeble > 0) {
      const g = greebleFace(face.length - 0.6, o.height - 0.5, o.greeble, o.seed + 91, 0.05);
      this.add(o.zone, o.mat, g, face.matrix(face.centre, 0.25, 0.001));
    }

    if (o.cornice !== false) {
      const c = extrudeProfile(corniceProfile(0.3, 0.36), face.length + 0.16, { bevel: 0.01 });
      c.rotateY(-Math.PI * 0.5);
      this.add(o.zone, 'stone', c, face.matrix(face.centre, o.height, 0));
    }

    if (o.breaches) for (const b of o.breaches) this.shellHole(o.zone, face, t, b);

    if (o.detail !== false) this.decorate(o.zone, face, win.runs, t, rng);
    return { runs: win.runs };
  }

  /**
   * Dresses a raw rectangular opening into a shell hole.
   *
   * The opening itself is punched by `wallWithOpenings` so you can genuinely
   * see and shoot through it. What makes it read as damage rather than as a
   * missing window is the rim: broken lumps of masonry left round the edge at
   * random depths, render blown off in a wide irregular halo around it, rebar
   * standing out of the bottom lip, soot fanning up the wall, and the material
   * lying on the pavement underneath in the direction it was thrown.
   */
  private shellHole(
    zone: string,
    face: Face,
    thickness: number,
    b: { run: number; y: number; r: number; seed: number },
  ): void {
    const rng = new Rng(b.seed);
    const halfT = thickness * 0.5;
    // The punched opening is a rectangle; the rim has to sit on *its* perimeter.
    // Ringing an ellipse round a rectangular hole puts half the lumps inside the
    // void, where they hang in mid-air with nothing behind them — which is
    // exactly what a swarm of floating cubes in front of a wall looks like.
    const hw = b.r * 1.05;
    const hh = b.r * 0.9;

    const n = 14;
    for (let i = 0; i < n; i++) {
      const t = (i + rng.range(0.15, 0.85)) / n;
      // Walk the rectangle perimeter, then push each lump a little outward into
      // the solid wall so it always has masonry behind it.
      const u = t * 4;
      let ex: number;
      let ey: number;
      if (u < 1) { ex = -hw + 2 * hw * u; ey = -hh; }
      else if (u < 2) { ex = hw; ey = -hh + 2 * hh * (u - 1); }
      else if (u < 3) { ex = hw - 2 * hw * (u - 2); ey = hh; }
      else { ex = -hw; ey = hh - 2 * hh * (u - 3); }
      const push = rng.range(0.02, 0.18);
      ex += Math.sign(ex) * push * (Math.abs(ex) > hw * 0.85 ? 1 : 0.3);
      ey += Math.sign(ey) * push * (Math.abs(ey) > hh * 0.85 ? 1 : 0.3);

      const s = rng.range(0.15, 0.36);
      const g = bevelBox(s * rng.range(0.9, 1.8), s * rng.range(0.8, 1.5), thickness * rng.range(0.85, 1.15), s * 0.16);
      // Kept close to the wall plane: these are lumps of the wall that did not
      // come away, not rocks stuck to it.
      g.rotateY(rng.jitter(0.22));
      g.rotateZ(rng.jitter(0.3));
      this.add(
        zone,
        'concrete',
        g,
        face.matrix(face.run(face.local(b.run) + ex), b.y + ey, rng.range(-0.04, 0.05)),
      );
    }

    // Render blown off in a halo, heaviest below where the blast washed down.
    for (let i = 0; i < 3; i++) {
      const a = rng.range(0, Math.PI * 2);
      const d = b.r * rng.range(1.1, 2.1);
      const p = spallPatch(rng.range(0.6, 1.4), rng.range(0.5, 1.1), 0.022, b.seed * 17 + i);
      this.add(zone, 'brick', p, face.matrix(
        face.run(face.local(b.run) + Math.cos(a) * d),
        Math.max(0.5, b.y + Math.sin(a) * d * 0.8 - b.r * 0.3),
        halfT + 0.004,
      ));
    }

    // Rebar out of the bottom lip. Two tufts, short: this is reinforcement
    // sticking out of a broken slab edge, not scrub growing out of the wall.
    for (let i = 0; i < 2; i++) {
      const w = face.world(face.run(face.local(b.run) + rng.jitter(hw * 0.55)), b.y - hh * 0.85, halfT + 0.08);
      this.props?.place('rebar_tuft', w.x, w.y, w.z, face.yaw + rng.jitter(0.5), rng.range(0.55, 0.85));
    }

    // Soot up the wall, and what came out of it on the ground below.
    const centre = face.world(b.run, b.y, halfT);
    this.scorch.push({ p: new THREE.Vector3(centre.x, centre.y + b.r * 0.55, centre.z), r: b.r * 3.6, s: 0.55 });
    const foot = face.world(b.run, 0, halfT + b.r * 0.9);
    const out = face.world(b.run, 0, halfT + 1);
    const base = face.world(b.run, 0, halfT);
    this.debrisCone(
      zone,
      foot.x,
      this.groundHeight(foot.x, foot.z) + 0.01,
      foot.z,
      b.r * 1.5,
      b.r * 0.32,
      b.seed + 3,
      out.x - base.x,
      out.z - base.z,
    );
  }

  /** Joinery, shutters, glass, balconies, downpipes. */
  private decorate(
    zone: string,
    face: Face,
    runs: { run: number; y: number; w: number; h: number; storey: number }[],
    thickness: number,
    rng: Rng,
  ): void {
    for (const w of runs) {
      const setBack = thickness * 0.42;
      const roll = rng.next();

      if (roll < 0.30) {
        // Shuttered, one leaf usually ajar — and one in six hanging off its
        // bottom hinge, which is the single detail that says "nobody has lived
        // here for two years".
        const sw = w.w * 0.5 - 0.01;
        for (const side of [-1, 1]) {
          const hanging = side > 0 && rng.chance(0.17);
          const open = hanging
            ? rng.range(0.9, 1.5)
            : side > 0 && rng.chance(0.42)
              ? rng.range(0.5, 1.25)
              : rng.range(0.0, 0.05);
          const g = shutterGeo(sw, w.h - 0.03);
          g.translate(side * sw * 0.5, 0, 0);
          if (hanging) {
            // Swing about the *bottom* corner as well, so it lolls forward.
            g.translate(0, (w.h - 0.03) * 0.5, 0);
            g.rotateZ(side * rng.range(0.25, 0.6));
            g.translate(0, -(w.h - 0.03) * 0.5, 0);
          }
          g.rotateY(-side * open);
          g.translate(-side * sw * 0.5, 0, 0);
          const run = w.run + face.runSign * side * sw * 0.5;
          this.add(zone, 'woodDark', g, face.matrix(run, w.y + w.h * 0.5, setBack * 0.4));
        }
      } else if (roll < 0.60) {
        // Glazed, mostly broken.
        const frame = windowFrameGeo(w.w - 0.05, w.h - 0.05, 0.07);
        this.add(zone, 'woodDark', frame, face.matrix(w.run, w.y + w.h * 0.5, -setBack));
        if (rng.chance(0.45)) {
          const pane = new THREE.PlaneGeometry(w.w - 0.13, w.h - 0.13);
          this.add(zone, 'glass', finalizeGeometry(pane), face.matrix(w.run, w.y + w.h * 0.5, -setBack - 0.005));
        } else {
          // Shards clinging to the frame.
          const shards: THREE.BufferGeometry[] = [];
          for (let i = 0; i < 5; i++) {
            const sw = rng.range(0.08, 0.3);
            const sh = rng.range(0.08, 0.34);
            const p = new THREE.PlaneGeometry(sw, sh);
            p.translate(rng.jitter((w.w - sw) * 0.45), (w.h - sh) * 0.45 * rng.sign(), 0);
            shards.push(finalizeGeometry(p));
          }
          for (const s of shards) this.add(zone, 'glass', s, face.matrix(w.run, w.y + w.h * 0.5, -setBack - 0.005));
        }
      } else if (roll < 0.76) {
        // Boarded up: planks nailed across the reveal at whatever angle came to
        // hand, with gaps you can see the dark through.
        const n = rng.int(3, 4);
        for (let i = 0; i < n; i++) {
          const pw = w.w + rng.range(0.06, 0.3);
          const ph = rng.range(0.15, 0.26);
          const py = -w.h * 0.42 + (w.h * 0.84 * (i + 0.5)) / n + rng.jitter(0.05);
          const g = bevelBox(pw, ph, 0.032, 0.006);
          g.rotateZ(rng.jitter(0.09));
          this.add(zone, 'wood', g, face.matrix(w.run + face.runSign * rng.jitter(0.06), w.y + w.h * 0.5 + py, setBack * 0.2));
        }
        if (rng.chance(0.3)) {
          const brace = bevelBox(Math.hypot(w.w, w.h) * 0.92, 0.19, 0.03, 0.006);
          brace.rotateZ(Math.atan2(w.h, w.w) * rng.sign());
          this.add(zone, 'wood', brace, face.matrix(w.run, w.y + w.h * 0.5, setBack * 0.2 + 0.03));
        }
      }
      // The remainder are simply empty holes — a bombed street has plenty.

      if (w.storey > 0 && w.h > 2.0 && rng.chance(0.16)) {
        // Collapsed balcony: the slab has sheared at the wall and dropped on
        // its brackets, the railing has gone over with it, and the whole thing
        // is hanging. One of these per street is worth more than a hundred
        // intact ones.
        const tilt = rng.range(0.5, 0.95);
        const slab = bevelBox(w.w + 0.9, 0.14, 1.05, 0.02);
        const sm = face.matrix(w.run, w.y - 0.12 - Math.sin(tilt) * 0.5, 0.42, 0);
        sm.multiply(matrixOf(0, 0, 0, 0, tilt, rng.jitter(0.12)));
        this.add(zone, 'stone', slab, sm);
        this.solid('concrete', [{ cx: 0, cy: 0, cz: 0, sx: w.w + 0.9, sy: 0.2, sz: 0.9 }], sm);
        const rail = railingGeo(w.w + 0.7, 0.98, { balusterSpacing: 0.14 });
        const rm = face.matrix(w.run + face.runSign * rng.jitter(0.2), w.y - 0.3 - Math.sin(tilt) * 0.95, 0.85, 0);
        rm.multiply(matrixOf(0, 0, 0, 0, tilt + rng.range(0.1, 0.5), rng.jitter(0.25)));
        this.add(zone, 'metalRust', rail, rm);
        for (const sx of [-1, 1]) {
          const bracket = bevelBox(0.1, 0.42, 0.6, 0.012);
          bracket.rotateX(rng.range(0.6, 1.3));
          this.add(zone, 'stone', bracket, face.matrix(w.run + sx * (w.w * 0.5 + 0.2), w.y - 0.4, 0.3));
        }
        // What fell off it, on the pavement directly below.
        const foot = face.world(w.run, 0, 1.0);
        this.debrisCone(
          zone, foot.x, this.groundHeight(foot.x, foot.z) + 0.01, foot.z,
          1.5, 0.34, Math.round(w.run * 71) + 5,
          face.world(w.run, 0, 1).x - face.world(w.run, 0, 0).x,
          face.world(w.run, 0, 1).z - face.world(w.run, 0, 0).z,
        );
      } else if (w.storey > 0 && w.h > 2.0) {
        // Balcony under the full-height openings.
        const slab = bevelBox(w.w + 0.9, 0.14, 1.05, 0.02);
        this.add(zone, 'stone', slab, face.matrix(w.run, w.y - 0.07, 0.5));
        this.solid('concrete', [{ cx: 0, cy: 0, cz: 0, sx: w.w + 0.9, sy: 0.14, sz: 1.05 }], face.matrix(w.run, w.y - 0.07, 0.5));
        const rail = railingGeo(w.w + 0.86, 0.98, { balusterSpacing: 0.14 });
        this.add(zone, 'metalRust', rail, face.matrix(w.run, w.y, 1.0));
        this.solid('metal', [{ cx: 0, cy: 0.5, cz: 0, sx: w.w + 0.86, sy: 1.0, sz: 0.08 }], face.matrix(w.run, w.y, 1.0));
        for (const sx of [-1, 1]) {
          const bracket = bevelBox(0.1, 0.42, 0.6, 0.012);
          bracket.rotateX(0.5);
          this.add(zone, 'stone', bracket, face.matrix(w.run + sx * (w.w * 0.5 + 0.2), w.y - 0.34, 0.36));
        }
      }
    }

    // Downpipes at the ends of the elevation, with a shoe at the bottom.
    const height = runs.length > 0 ? Math.max(...runs.map((r) => r.y + r.h)) + 1.2 : 6;
    for (const t of [0.06, 0.94]) {
      if (!rng.chance(0.72)) continue;
      const run = face.from + (face.to - face.from) * t;
      const pipe = pipeRun(height, 0.052, Math.max(2, Math.round(height / 2.2)));
      this.add(zone, 'metalRust', pipe, face.matrix(run, height * 0.5, 0.09));
      const shoe = cylinderGeo(0.052, 0.072, 0.3, 10);
      shoe.rotateX(0.5);
      this.add(zone, 'metalRust', shoe, face.matrix(run, 0.2, 0.15));
      // Damp patch where it drains: the grime bake reads the geometry, this
      // reads the story.
      this.props?.place('grass', ...this.spread(face, run, 0.24), 0, 0.85);
    }
  }

  /** Ground position a short way out from a face, as a place/scatter argument. */
  private spread(face: Face, run: number, depth: number): [number, number, number] {
    const p = face.world(run, 0, depth);
    return [p.x, this.groundHeight(p.x, p.z), p.z];
  }

  // =========================================================================
  // Building shells
  // =========================================================================

  private shell(o: {
    zone: string;
    mat: MatKey;
    x0: number;
    z0: number;
    x1: number;
    z1: number;
    base?: number;
    height: number;
    thickness?: number;
    seed: number;
    /** Elevations that get windows and joinery. */
    detailed?: Array<'n' | 's' | 'e' | 'w'>;
    /** Elevations left completely blank (party walls). */
    blank?: Array<'n' | 's' | 'e' | 'w'>;
    extra?: Partial<Record<'n' | 's' | 'e' | 'w', Opening[]>>;
    breaches?: Partial<Record<'n' | 's' | 'e' | 'w', { run: number; y: number; r: number; seed: number }[]>>;
    parapet?: number;
    noParapet?: Array<'n' | 's' | 'e' | 'w'>;
    roof?: boolean;
    balconyRow?: number;
    groundSkip?: boolean;
    /** True when the interior is really built, so openings need no fake room. */
    enterable?: boolean;
  }): void {
    const t = o.thickness ?? 0.42;
    const base = o.base ?? 0;
    const detailed = new Set(o.detailed ?? []);
    const blank = new Set(o.blank ?? []);

    const faces: Record<'n' | 's' | 'e' | 'w', Face> = {
      n: new Face('x', o.x0, o.x1, o.z0, -1, base),
      s: new Face('x', o.x0, o.x1, o.z1, 1, base),
      w: new Face('z', o.z0 + t, o.z1 - t, o.x0, -1, base),
      e: new Face('z', o.z0 + t, o.z1 - t, o.x1, 1, base),
    };

    let i = 0;
    for (const k of ['n', 's', 'w', 'e'] as const) {
      const face = faces[k];
      const det = detailed.has(k);
      this.facade({
        zone: o.zone,
        mat: o.mat,
        face,
        height: o.height,
        thickness: t,
        seed: o.seed + i * 137,
        extraOpenings: o.extra?.[k],
        breaches: o.breaches?.[k],
        windows: !blank.has(k),
        detail: det,
        twoLeaf: det,
        greeble: det ? 0 : 3,
        cornice: det,
        balconyRow: det ? o.balconyRow : undefined,
        groundSkip: o.groundSkip,
        backing: !o.enterable,
      });
      i++;
    }

    // Parapet + coping. A face can opt out where a roof terrace has to be
    // walked onto from the neighbouring block.
    const par = o.parapet ?? 0.95;
    const skipPar = new Set(o.noParapet ?? []);
    if (par > 0) {
      for (const k of ['n', 's', 'w', 'e'] as const) {
        if (skipPar.has(k)) continue;
        const face = faces[k];
        const len = k === 'n' || k === 's' ? o.x1 - o.x0 : o.z1 - o.z0;
        const wall = bevelBox(len, par, t * 0.62, 0.022);
        this.add(o.zone, o.mat, wall, face.matrix(face.centre, o.height + par * 0.5, -t * 0.3));
        this.solid(this.kindOf(o.mat), [{ cx: 0, cy: 0, cz: 0, sx: len, sy: par, sz: t * 0.62 }], face.matrix(face.centre, o.height + par * 0.5, -t * 0.3));
        const cop = extrudeProfile(copingProfile(t * 0.78, 0.1), len + 0.08, { bevel: 0.008 });
        cop.rotateY(-Math.PI * 0.5);
        this.add(o.zone, 'stone', cop, face.matrix(face.centre, o.height + par, -t * 0.3));
      }
    }

    if (o.roof !== false) {
      const roof = this.groundGrid(o.x0 + t, o.z0 + t, o.x1 - t, o.z1 - t, base + o.height, 1.5, o.seed + 7, 0.02);
      this.add(o.zone, 'concrete', roof);
      this.collider.addBox('concrete', {
        cx: (o.x0 + o.x1) * 0.5,
        cy: base + o.height - 0.16,
        cz: (o.z0 + o.z1) * 0.5,
        sx: o.x1 - o.x0,
        sy: 0.32,
        sz: o.z1 - o.z0,
      });
      this.floors.push({ x0: o.x0, z0: o.z0, x1: o.x1, z1: o.z1, y: base + o.height });
      this.roofClutter(o.zone, o.x0 + 1.4, o.z0 + 1.4, o.x1 - 1.4, o.z1 - 1.4, base + o.height, o.seed + 313);
    }
  }

  /** AC plant, dishes, vents and a rubble drift on every roof. */
  private roofClutter(zone: string, x0: number, z0: number, x1: number, z1: number, y: number, seed: number): void {
    const props = this.props;
    if (!props || x1 <= x0 || z1 <= z0) return;
    const rng = new Rng(seed);
    const area = (x1 - x0) * (z1 - z0);
    const n = Math.max(1, Math.round(area / 82));
    for (let i = 0; i < n; i++) {
      const x = rng.range(x0, x1);
      const z = rng.range(z0, z1);
      const roll = rng.next();
      if (roll < 0.42) props.placeSolid(this.collider, 'ac_unit', x, y, z, rng.range(0, Math.PI * 2), 1);
      else if (roll < 0.68) props.place('dish', x, y, z, rng.range(0, Math.PI * 2), rng.range(0.8, 1.15));
      else if (roll < 0.86) props.placeSolid(this.collider, 'barrel', x, y, z, rng.range(0, Math.PI * 2), 1);
      else props.place('rubble_chunk', x, y, z, rng.range(0, Math.PI * 2), rng.range(0.7, 1.3));
    }
    // Stair head. Every flat roof in the world has one and it is the cheapest
    // thing there is that breaks a roofline in silhouette — a 2.5m box on top
    // of a 10m block is what stops the block reading as an extruded rectangle.
    if ((x1 - x0) * (z1 - z0) > 46) {
      const sw = rng.range(2.2, 3.1);
      const sd = rng.range(2.0, 2.8);
      const sh = rng.range(2.3, 3.1);
      const sx = rng.range(x0 + sw * 0.5, x1 - sw * 0.5);
      const sz = rng.range(z0 + sd * 0.5, z1 - sd * 0.5);
      const box = bevelBox(sw, sh, sd, 0.03);
      this.add(zone, 'plasterWhite', box, matrixOf(sx, y + sh * 0.5, sz, rng.jitter(0.05)));
      this.collider.addBox('plaster', { cx: sx, cy: y + sh * 0.5, cz: sz, sx: sw, sy: sh, sz: sd });
      const cap = bevelBox(sw + 0.3, 0.16, sd + 0.3, 0.02);
      this.add(zone, 'stone', cap, matrixOf(sx, y + sh + 0.08, sz, 0));
      // A door out onto the roof, and a dark opening behind it.
      const dir = rng.chance(0.5) ? 1 : -1;
      this.add(zone, 'interiorDark', plainBox(1.05, 2.1, 0.12), matrixOf(sx, y + 1.05, sz + dir * (sd * 0.5 - 0.04), 0));
      this.add(zone, 'woodDark', doorGeo(1.0, 2.05), matrixOf(sx + 0.1, y + 1.02, sz + dir * (sd * 0.5 + 0.03), dir > 0 ? 0 : Math.PI));
      if (rng.chance(0.6)) {
        const vent = cylinderGeo(0.11, 0.11, 0.9, 8);
        this.add(zone, 'metalRust', vent, matrixOf(sx + sw * 0.3, y + sh + 0.5, sz, 0));
      }
    }

    // A cistern tank: reads instantly as a Mediterranean rooftop.
    if (rng.chance(0.65)) {
      const tank = cylinderGeo(0.62, 0.62, 1.05, 14);
      tank.translate(0, 0.52, 0);
      scaleUv(tank, 1.1);
      const legs = new GeoBuilder();
      legs.add(tank);
      for (let i = 0; i < 4; i++) {
        const a = (i / 4) * Math.PI * 2 + 0.78;
        const leg = bevelBox(0.07, 0.5, 0.07, 0.01);
        leg.translate(Math.cos(a) * 0.44, 0.25, Math.sin(a) * 0.44);
        legs.add(leg);
      }
      const g = legs.build();
      if (g) {
        const x = rng.range(x0, x1);
        const z = rng.range(z0, z1);
        this.add(zone, 'metalPaint', g, matrixOf(x, y + 0.5, z, rng.range(0, 6.28)));
        this.collider.addBox('metal', { cx: x, cy: y + 1.0, cz: z, sx: 1.24, sy: 1.05, sz: 1.24 });
      }
    }
  }

  // =========================================================================
  // West blocks — the interior route
  // =========================================================================

  private buildWestBlocks(): void {
    // ---- W1: three sub-masses, different heights and facade planes so the
    // elevation steps instead of running as one 28m wall.
    // Bay A is two storeys; its roof doubles as the terrace at the end of the
    // interior route, so its south parapet is omitted to let you step across
    // from bay B's second floor.
    this.shell({
      zone: 'west', mat: 'plasterCream', x0: -26, z0: -26, x1: -6.9, z1: -16,
      height: 7.1, seed: 1001, detailed: ['e'], blank: ['w', 'n', 's'], parapet: 0.9,
      noParapet: ['s'], enterable: true,
      extra: { e: [{ x: faceLocal('z', -26, -16, 1, -20.5), y: 0, w: 2.4, h: 2.6 }] },
    });
    this.shell({
      zone: 'west', mat: 'plasterOchre', x0: -26, z0: -16, x1: -6.8, z1: -6,
      height: 10.7, seed: 1002, detailed: ['e'], blank: ['w', 'n', 's'], parapet: 1.0, balconyRow: 1, enterable: true,
      groundSkip: true,
      // Straight up the hero sightline and across the skyline pose: the one
      // piece of heavy damage on the west terrace.
      breaches: { e: [{ run: -11.6, y: 6.6, r: 1.3, seed: 8801 }] },
      extra: {
        e: [{ x: faceLocal('z', -16, -6, 1, -11), y: 0, w: 6.4, h: 3.05 }],
        // Door from the second floor out onto bay A's terrace.
        n: [{ x: faceLocal('x', -26, -6.8, -1, -18), y: FLOOR_H * 2, w: 1.15, h: 2.2 }],
      },
    });
    this.shell({
      zone: 'west', mat: 'plasterWhite', x0: -26, z0: -6, x1: -7.1, z1: 2,
      height: 10.2, seed: 1003, detailed: ['e', 's'], blank: ['w', 'n'], parapet: 0.9, balconyRow: 1, enterable: true,
      breaches: { e: [{ run: -2.6, y: 7.3, r: 1.0, seed: 8802 }] },
      extra: { e: [{ x: faceLocal('z', -6, 2, 1, -2), y: 0, w: 1.3, h: 2.35 }] },
    });

    this.buildW1Interior();

    // ---- W2: arcade apartments. The main mass stops at the back wall of the
    // walkway; the colonnade and the storey it carries are built separately, so
    // the ground floor is genuinely open and the west flank is a route.
    this.shell({
      zone: 'west', mat: 'plasterRose', x0: -26, z0: 6, x1: -13.7, z1: 30,
      height: 7.2, seed: 1101, detailed: ['s'], blank: ['w', 'n'], parapet: 0.95,
      extra: {
        e: [
          { x: faceLocal('z', 6, 30, 1, 10.5), y: 0, w: 1.1, h: 2.25 },
          { x: faceLocal('z', 6, 30, 1, 18.0), y: 0, w: 1.1, h: 2.25 },
          { x: faceLocal('z', 6, 30, 1, 25.5), y: 0, w: 1.1, h: 2.25 },
        ],
      },
    });
    this.buildArcade();

    // ---- W3: the west perimeter block, closing the alley.
    this.shell({
      zone: 'west', mat: 'plasterCream', x0: -40, z0: -32, x1: -30, z1: 32,
      height: 13.5, seed: 1201, detailed: ['e'], blank: ['w', 'n', 's'], parapet: 1.1, roof: false,
    });
  }

  /**
   * W1 interior: shop → back corridor → stairwell → first floor → balcony over
   * the square → second floor with a collapsed slab → roof.
   */
  private buildW1Interior(): void {
    const props = this.props;
    if (!props) return;
    const rng = new Rng(0x1f0a);

    // The stairwell is a 5.0 x 4.6m void at x [-25.6,-20.6], z [-15.4,-10.8],
    // inside the three-storey bay so the flights can reach the second floor.
    const SW = { x0: -25.6, z0: -15.4, x1: -20.6, z1: -10.8 };

    // Ground floor: one slab under the whole footprint.
    this.interiorFloor('west', 'paving', -25.6, -25.6, -7.1, 1.6, 0.0);

    // First floor, in three strips around the void.
    this.interiorFloor('west', 'concrete', SW.x0, -25.6, SW.x1, SW.z0, FLOOR_H);
    this.interiorFloor('west', 'concrete', SW.x0, SW.z1, SW.x1, 1.6, FLOOR_H);
    this.interiorFloor('west', 'concrete', SW.x1, -25.6, -7.1, 1.6, FLOOR_H);

    // Second floor exists only over the three-storey bays (z >= -16), minus the
    // bite a shell took out of the slab: you can see and shoot down into the
    // shop through it.
    this.interiorFloor('west', 'concrete', SW.x0, SW.z1, SW.x1, 1.6, FLOOR_H * 2);
    this.interiorFloor('west', 'concrete', SW.x1, -16.0, -18.0, 1.6, FLOOR_H * 2);
    this.interiorFloor('west', 'concrete', -13.0, -16.0, -7.1, 1.6, FLOOR_H * 2);
    this.interiorFloor('west', 'concrete', -18.0, -16.0, -13.0, -13.0, FLOOR_H * 2);
    this.interiorFloor('west', 'concrete', -18.0, -9.0, -13.0, 1.6, FLOOR_H * 2);
    // The slab's broken edge: chunks still hanging on the reinforcement.
    for (const [x, z, w, d, yaw] of [
      [-17.6, -12.4, 1.3, 0.7, 0.3],
      [-13.4, -9.6, 0.9, 1.1, -0.5],
      [-15.5, -12.9, 1.6, 0.5, 0.15],
    ] as const) {
      const lip = bevelBox(w, 0.26, d, 0.02);
      this.add('west', 'concrete', lip, matrixOf(x, FLOOR_H * 2 - 0.14, z, yaw));
      this.props?.place('rebar_tuft', x, FLOOR_H * 2 - 0.3, z, yaw, 0.7);
    }
    this.debrisCone('west', -15.4, 0.0, -11.0, 3.4, 1.0, 812);

    // Stairs: four half-flights, dog-legged around a full-width landing at each
    // half level. The landings are the full 5m so each one lands flush against
    // the floor slab it serves.
    const UP_Z = -15.0;
    const DOWN_Z = -11.1;
    for (let f = 0; f < 4; f++) {
      const y = f * FLOOR_H * 0.5;
      const up = f % 2 === 0;
      const s = stairsGeo(1.5, FLOOR_H * 0.5, 2.6, 9);
      const m = up ? matrixOf(-24.6, y, UP_Z, 0) : matrixOf(-22.0, y, DOWN_Z, Math.PI);
      this.add('west', 'concrete', s.geometry, m);
      this.solid('concrete', s.boxes, m);
      const lz = up ? -11.6 : -14.55;
      const ld = up ? 1.6 : 1.7;
      const landing = bevelBox(5.0, 0.24, ld, 0.02);
      const lm = matrixOf(-23.1, y + FLOOR_H * 0.5 - 0.12, lz, 0);
      this.add('west', 'concrete', landing, lm);
      this.solid('concrete', [{ cx: 0, cy: 0, cz: 0, sx: 5.0, sy: 0.24, sz: ld }], lm);
      this.floors.push({ x0: -25.6, z0: lz - ld * 0.5, x1: -20.6, z1: lz + ld * 0.5, y: y + FLOOR_H * 0.5 });
    }
    // The only unguarded edge is the 1.3m slot between the two landings.
    for (const y of [FLOOR_H, FLOOR_H * 2]) {
      const r = railingGeo(1.25, 1.0, { balusterSpacing: 0.16 });
      this.add('west', 'metalRust', r, matrixOf(-23.3, y, -13.05, Math.PI * 0.5));
      this.collider.addBox('metal', { cx: -23.3, cy: y + 0.5, cz: -13.05, sx: 0.12, sy: 1.0, sz: 1.25 });
    }

    // Interior partitions. Doorways line up into a shootable corridor.
    // Doorways are placed so each floor has exactly one way through: the route
    // reads as a route rather than an open plan you can shortcut across.
    const partitions: Array<{ axis: 'x' | 'z'; from: number; to: number; at: number; y: number; door?: number }> = [
      { axis: 'z', from: -25.4, to: 1.4, at: -20.6, y: 0, door: -13 },
      { axis: 'z', from: -25.4, to: 1.4, at: -20.6, y: FLOOR_H, door: -18 },
      { axis: 'x', from: -20.4, to: -7.2, at: -14.6, y: 0, door: -12.4 },
      { axis: 'x', from: -20.4, to: -7.2, at: -5.6, y: FLOOR_H, door: -12.0 },
      { axis: 'x', from: -20.4, to: -7.2, at: -15.4, y: FLOOR_H, door: -10.5 },
      { axis: 'x', from: -17.8, to: -7.2, at: -6.5, y: FLOOR_H * 2, door: -11.0 },
    ];
    for (const p of partitions) {
      const len = p.to - p.from;
      const openings: Opening[] = [];
      if (p.door !== undefined) {
        // Partitions along Z are placed with yaw +90deg, which maps wall-local
        // +X onto world -Z; partitions along X keep local +X on world +X.
        const sign = p.axis === 'z' ? -1 : 1;
        openings.push({ x: (p.door - (p.from + p.to) * 0.5) * sign, y: 0, w: 1.05, h: 2.15 });
      }
      const w = wallWithOpenings(len, FLOOR_H - 0.24, 0.22, {
        openings, twoLeaf: false, sills: false, lintels: true, bevel: 0.014,
      });
      const m = p.axis === 'z'
        ? matrixOf(p.at, p.y, (p.from + p.to) * 0.5, Math.PI * 0.5)
        : matrixOf((p.from + p.to) * 0.5, p.y, p.at, 0);
      this.add('west', 'plasterWhite', w.geometry, m);
      this.solid('plaster', w.boxes, m);
    }

    // Shop fittings behind the open front.
    const counter = bevelBox(3.4, 0.95, 0.72, 0.02);
    this.add('west', 'wood', counter, matrixOf(-12.2, 0.48, -8.6, 0.06));
    this.collider.addBox('wood', { cx: -12.2, cy: 0.48, cz: -8.6, sx: 3.4, sy: 0.95, sz: 0.72 });
    for (let s = 0; s < 4; s++) {
      const shelf = bevelBox(0.42, 0.05, 3.6, 0.01);
      this.add('west', 'wood', shelf, matrixOf(-14.3, 0.55 + s * 0.62, -11.4, 0));
    }
    const shelfFrame = panelledWall(3.6, 2.5, 0.16, 3, 3, 771);
    this.add('west', 'woodDark', shelfFrame.geometry, matrixOf(-14.5, 0, -11.4, Math.PI * 0.5));
    this.collider.addBox('wood', { cx: -14.5, cy: 1.25, cz: -11.4, sx: 0.2, sy: 2.5, sz: 3.6 });

    for (const [x, z] of [[-16.5, -6.0], [-18.4, -3.2], [-10.6, -3.6]] as const) {
      this.props?.placeSolid(this.collider, 'crate_large', x, 0, z, rng.range(0, 3.14));
      if (rng.chance(0.6)) this.props?.placeSolid(this.collider, 'crate_small', x + rng.jitter(0.9), 0.86, z + rng.jitter(0.6), rng.range(0, 3.14));
    }
    for (const [x, z] of [[-23.4, -14.5], [-22.2, -10.0], [-17.0, -20.4]] as const) {
      this.props?.placeSolid(this.collider, 'barrel', x, 0, z, rng.range(0, 3.14));
    }

    // Balcony over the square, off the first-floor rooms.
    const slab = bevelBox(7.2, 0.2, 1.9, 0.024);
    this.add('west', 'stone', slab, matrixOf(-5.9, FLOOR_H - 0.1, -11.0, 0));
    this.collider.addBox('concrete', { cx: -5.9, cy: FLOOR_H - 0.1, cz: -11.0, sx: 7.2, sy: 0.2, sz: 1.9 });
    this.floors.push({ x0: -9.5, z0: -12, x1: -2.3, z1: -10, y: FLOOR_H });
    const rail = railingGeo(7.1, 1.02, { balusterSpacing: 0.15 });
    this.add('west', 'metalRust', rail, matrixOf(-2.4, FLOOR_H, -11.0, Math.PI * 0.5));
    this.collider.addBox('metal', { cx: -2.4, cy: FLOOR_H + 0.5, cz: -11.0, sx: 0.1, sy: 1.02, sz: 7.1 });
    for (const z of [-11.9, -10.1]) {
      const r = railingGeo(1.85, 1.02, { balusterSpacing: 0.15 });
      this.add('west', 'metalRust', r, matrixOf(-5.9, FLOOR_H, z, 0));
      this.collider.addBox('metal', { cx: -5.9, cy: FLOOR_H + 0.5, cz: z, sx: 1.85, sy: 1.02, sz: 0.1 });
    }
    for (const sx of [-1, 1]) {
      const brk = bevelBox(0.14, 0.5, 1.5, 0.014);
      brk.rotateX(0.55);
      this.add('west', 'stone', brk, matrixOf(-5.9 + sx * 3.2, FLOOR_H - 0.45, -10.6, 0));
    }

    // Laundry across the alley mouth, above head height.
    const cloth = this.props?.cloth();
    if (cloth && this.vault) {
      this.root.add(
        buildLaundryLine(this.vault, cloth, new THREE.Vector3(-7.4, 6.4, -18.5), new THREE.Vector3(-7.4, 6.9, -24.0), 0.45, 4, 91),
      );
    }
  }

  private interiorFloor(
    zone: string,
    mat: MatKey,
    x0: number,
    z0: number,
    x1: number,
    z1: number,
    y: number,
  ): void {
    if (x1 <= x0 || z1 <= z0) return;
    const g = this.groundGrid(x0, z0, x1, z1, y, 1.0, Math.round((x0 + z0) * 13) + 7, 0.008);
    this.add(zone, mat, g);
    if (y > 0.05) {
      const under = bevelBox(x1 - x0, 0.24, z1 - z0, 0.02);
      this.add(zone, 'concrete', under, matrixOf((x0 + x1) * 0.5, y - 0.17, (z0 + z1) * 0.5, 0));
      // Downstand beams. A ceiling that is one flat slab is the interior
      // equivalent of an untextured wall: nothing for the light to break over,
      // no scale reference, and no shadow anywhere in the top third of the
      // frame. Six boxes fix all three.
      // Only where there is actually a room under the slab; a few of these
      // "floors" are 12cm plinths and their beams would be underground.
      const spanX = x1 - x0 >= z1 - z0;
      const run = spanX ? z1 - z0 : x1 - x0;
      const n = y < 1.5 ? 0 : Math.max(1, Math.min(7, Math.round(run / 2.6)));
      for (let i = 1; i <= n; i++) {
        const t = i / (n + 1);
        const beam = spanX
          ? bevelBox(x1 - x0 - 0.1, 0.34, 0.28, 0.018)
          : bevelBox(0.28, 0.34, z1 - z0 - 0.1, 0.018);
        const bx = spanX ? (x0 + x1) * 0.5 : x0 + (x1 - x0) * t;
        const bz = spanX ? z0 + (z1 - z0) * t : (z0 + z1) * 0.5;
        this.add(zone, 'concrete', beam, matrixOf(bx, y - 0.46, bz, 0));
      }
    }
    this.collider.addBox('concrete', {
      cx: (x0 + x1) * 0.5,
      cy: y - 0.14,
      cz: (z0 + z1) * 0.5,
      sx: x1 - x0,
      sy: 0.28,
      sz: z1 - z0,
    });
    this.floors.push({ x0, z0, x1, z1, y });
  }

  /** W2's ground-floor colonnade: six arches, a covered walkway, a storey over. */
  private buildArcade(): void {
    const rng = new Rng(0x2c01);
    const z0 = 6.0;
    const z1 = 30.0;
    const bays = 6;
    const pitch = (z1 - z0) / bays;
    const archH = 4.3;
    const colX = -10.4;

    for (let i = 0; i < bays; i++) {
      const zc = z0 + pitch * (i + 0.5);
      const a = archway(pitch, archH, 0.62, pitch - 1.15, 2.5, 0.95, 0.02);
      const m = matrixOf(colX, 0, zc, Math.PI * 0.5);
      this.add('west', 'plasterRose', a.geometry, m);
      this.solid('plaster', a.boxes, m);
      // Impost band at the springing of each arch.
      const band = extrudeProfile(
        [new THREE.Vector2(0, 0.1), new THREE.Vector2(0.1, 0.08), new THREE.Vector2(0.12, -0.02), new THREE.Vector2(0, -0.06)],
        1.0,
        { bevel: 0.006 },
      );
      band.rotateY(-Math.PI * 0.5);
      for (const s of [-1, 1]) {
        this.add('west', 'stone', band.clone(), matrixOf(colX + 0.32, 2.5, zc + s * (pitch * 0.5 - 0.28), Math.PI * 0.5));
      }
      band.dispose();
    }

    // Storey carried over the walkway: its own elevation, based at the arcade
    // head so the window rows line up with the main block behind.
    const upper = new Face('z', z0, z1, colX, -1, archH);
    this.facade({
      zone: 'west', mat: 'plasterRose', face: upper, height: 7.2 - archH,
      thickness: 0.4, storeys: 1, seed: 1150, plinth: false, cornice: true,
      detail: true, balconyRow: 0, backing: true, twoLeaf: true,
    });
    const par = bevelBox(0.42, 0.95, z1 - z0, 0.022);
    this.add('west', 'plasterRose', par, matrixOf(colX - 0.2, 7.68, (z0 + z1) * 0.5, 0));
    this.collider.addBox('plaster', { cx: colX - 0.2, cy: 7.68, cz: (z0 + z1) * 0.5, sx: 0.42, sy: 0.95, sz: z1 - z0 });

    // Ceiling of the walkway / floor of the storey above, and the roof over it.
    const ceiling = bevelBox(3.5, 0.3, z1 - z0, 0.02);
    this.add('west', 'plasterRose', ceiling, matrixOf(-12.05, archH + 0.15, (z0 + z1) * 0.5, 0));
    this.collider.addBox('plaster', { cx: -12.05, cy: archH + 0.15, cz: (z0 + z1) * 0.5, sx: 3.5, sy: 0.3, sz: z1 - z0 });
    const roofSpan = this.groundGrid(-13.8, z0, colX, z1, 7.2, 1.5, 1155, 0.02);
    this.add('west', 'concrete', roofSpan);
    this.collider.addBox('concrete', { cx: -12.1, cy: 7.04, cz: (z0 + z1) * 0.5, sx: 3.4, sy: 0.32, sz: z1 - z0 });
    this.floors.push({ x0: -13.8, z0, x1: colX, z1, y: 7.2 });
    this.interiorFloor('west', 'paving', -13.7, z0 + 0.2, -10.5, z1 - 0.2, 0.12);

    // Shop doors along the back wall of the walkway.
    for (const zc of [10.5, 18.0, 25.5]) {
      const d = doorGeo(1.05, 2.2);
      this.add('west', 'woodDark', d, matrixOf(-13.62, 1.22, zc, Math.PI * 0.5));
    }
    for (const [zc, mat] of [[10.5, 'signRed'], [20.0, 'signBlue']] as const) {
      const board = bevelBox(2.6, 0.62, 0.1, 0.014);
      this.add('west', mat, board, matrixOf(-13.55, 2.72, zc, Math.PI * 0.5));
      for (const s of [-1, 1]) {
        const arm = cylinderGeo(0.018, 0.018, 0.4, 6);
        arm.rotateZ(Math.PI * 0.5);
        this.add('west', 'metalRust', arm, matrixOf(-13.72, 2.98, zc + s * 1.1, Math.PI * 0.5));
      }
    }

    // Laundry strung between the colonnade and the block behind it.
    const cloth = this.props?.cloth();
    if (cloth && this.vault) {
      for (const [za, zb] of [[9.0, 15.5], [17.5, 24.0]] as const) {
        this.root.add(
          buildLaundryLine(this.vault, cloth, new THREE.Vector3(-10.6, 3.9, za), new THREE.Vector3(-13.3, 4.05, zb), 0.42, 5, Math.round(za * 31)),
        );
      }
    }

    for (let i = 0; i < 5; i++) {
      this.props?.placeSolid(this.collider, rng.chance(0.5) ? 'crate_small' : 'barrel', rng.range(-13.2, -11.0), 0.12, rng.range(z0 + 1.5, z1 - 1.5), rng.range(0, 3.14));
    }
  }

  // =========================================================================
  // East blocks — the alley network
  // =========================================================================

  private buildEastBlocks(): void {
    // E1: the cafe. Terrace, awning and a first-floor balcony.
    this.shell({
      zone: 'east', mat: 'plasterBlue', x0: 9.5, z0: 2, x1: 22, z1: 20,
      height: 7.6, seed: 2001, detailed: ['w', 'n'], blank: ['e'], parapet: 0.9, balconyRow: 1,
      breaches: { w: [{ run: 13.2, y: 5.3, r: 1.15, seed: 8803 }] },
      extra: { w: [{ x: faceLocal('z', 2, 20, -1, 6.4), y: 0, w: 3.2, h: 2.8 }] },
    });
    this.buildCafeFront();

    // E2: the warehouse. One tall volume, a mezzanine, and a breach in the
    // west wall that reads all the way across the square.
    this.buildWarehouse();

    // E3: apartment slab behind the alley.
    this.shell({
      zone: 'east', mat: 'plasterCream', x0: 26, z0: 2, x1: 40, z1: 26,
      height: 11.2, seed: 2101, detailed: ['w'], blank: ['e', 's', 'n'], parapet: 1.05, balconyRow: 1,
    });

    // E4: perimeter block east of the alley.
    this.shell({
      zone: 'east', mat: 'plasterOchre', x0: 34, z0: -30, x1: 44, z1: -6,
      height: 15.5, seed: 2201, detailed: ['w'], blank: ['e', 'n', 's'], parapet: 1.2, roof: false,
    });

    // E6: closes the south connector.
    this.shell({
      zone: 'east', mat: 'plasterWhite', x0: 9.5, z0: 26, x1: 26, z1: 33,
      height: 8.4, seed: 2301, detailed: ['n'], blank: ['s', 'e', 'w'], parapet: 0.9,
    });
  }

  private buildCafeFront(): void {
    const rng = new Rng(0x3caf);
    // Awning over the terrace: sagging cloth on steel arms.
    const cloth = this.props?.cloth();
    const parts = new GeoBuilder();
    for (let i = 0; i < 7; i++) {
      const z = 3.4 + i * 2.3;
      const arm = cylinderGeo(0.028, 0.028, 2.5, 8);
      arm.rotateZ(Math.PI * 0.5);
      arm.rotateY(0.0);
      arm.rotateX(0.0);
      parts.add(arm, matrixOf(8.3, 3.15, z, 0, -0.16));
      const stay = cylinderGeo(0.02, 0.02, 1.5, 6);
      stay.rotateZ(0.9);
      parts.add(stay, matrixOf(8.85, 2.6, z, 0));
    }
    const armGeo = parts.build();
    if (armGeo) this.add('east', 'metalRust', armGeo);

    if (cloth) {
      const panels: THREE.BufferGeometry[] = [];
      for (let i = 0; i < 12; i++) {
        const z0 = 3.0 + i * 1.35;
        const z1 = z0 + 1.35;
        const g = new THREE.PlaneGeometry(2.55, z1 - z0, 2, 2);
        g.rotateX(-Math.PI * 0.5);
        g.rotateZ(0.17);
        g.translate(8.2, 3.02, (z0 + z1) * 0.5);
        const pos = g.getAttribute('position') as THREE.BufferAttribute;
        for (let v = 0; v < pos.count; v++) {
          pos.setY(v, pos.getY(v) - 0.07 * Math.sin(((pos.getZ(v) - z0) / 1.35) * Math.PI));
        }
        g.computeVertexNormals();
        panels.push(finalizeGeometry(g));
      }
      // Scalloped valance.
      for (let i = 0; i < 18; i++) {
        const z = 3.0 + i * 0.9 + 0.45;
        const s = bevelBox(0.86, 0.24, 0.012, 0.004);
        s.rotateY(Math.PI * 0.5);
        s.translate(7.0, 2.72, z);
        panels.push(s);
      }
      const merged = new GeoBuilder();
      for (const p of panels) merged.add(p);
      const g = merged.build();
      if (g) {
        tintGeometry(g, 0xbb4a3c);
        scaleUv(g, 1.2);
        const mesh = new THREE.Mesh(g, cloth);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData.noCollide = true;
        const depth = cloth.userData.clothDepth as THREE.MeshDepthMaterial | undefined;
        if (depth) mesh.customDepthMaterial = depth;
        this.root.add(mesh);
        this.meshes.push(mesh);
      }
    }

    // Signage over the door, and a projecting blade sign.
    const board = bevelBox(3.4, 0.78, 0.12, 0.016);
    this.add('east', 'signRed', board, matrixOf(9.42, 3.7, 6.4, -Math.PI * 0.5));
    const blade = bevelBox(0.11, 1.5, 0.9, 0.014);
    this.add('east', 'signBlue', blade, matrixOf(9.0, 4.9, 12.0, 0));
    // The one emissive element on the map: a failing tube along the blade.
    for (const dy of [-0.42, 0.42]) {
      const tube = cylinderGeo(0.026, 0.026, 0.74, 8);
      tube.rotateX(Math.PI * 0.5);
      this.add('east', 'signLit', tube, matrixOf(8.92, 4.9 + dy, 12.0, 0));
    }
    const bracket = cylinderGeo(0.022, 0.022, 0.8, 6);
    bracket.rotateZ(Math.PI * 0.5);
    this.add('east', 'metalRust', bracket, matrixOf(9.1, 5.55, 12.0, 0));

    // Terrace furniture and planters.
    for (let i = 0; i < 4; i++) {
      const z = 4.5 + i * 3.4;
      this.props?.placeSolid(this.collider, 'planter', 8.2, 0.15, z, rng.range(0, 3.14));
    }
    for (let i = 0; i < 3; i++) {
      this.props?.placeSolid(this.collider, 'crate_small', rng.range(7.6, 9.0), 0.15, rng.range(14, 19), rng.range(0, 3.14));
    }
  }

  private buildWarehouse(): void {
    const rng = new Rng(0x4a11);
    const x0 = 16;
    const x1 = 30;
    const z0 = -30;
    const z1 = -4;
    const h = 9.6;
    const t = 0.5;

    // Blank walls first, then the west elevation gets punched through.
    for (const [k, face] of [
      ['n', new Face('x', x0, x1, z0, -1, 0)],
      ['s', new Face('x', x0, x1, z1, 1, 0)],
      ['e', new Face('z', z0 + t, z1 - t, x1, 1, 0)],
    ] as const) {
      const highWindows: Opening[] = [];
      const len = face.length;
      const n = Math.max(2, Math.round(len / 4.2));
      for (let i = 0; i < n; i++) {
        if (rng.chance(0.18)) continue;
        const run = face.from + (face.to - face.from) * ((i + 0.5) / n);
        highWindows.push({ x: face.local(run), y: 6.1, w: 2.0, h: 1.5 });
      }
      const extra: Opening[] = [];
      if (k === 'n') extra.push({ x: face.local(23), y: 0, w: 4.2, h: 4.4 });
      const wall = wallWithOpenings(len, h, t, {
        openings: [...highWindows, ...extra],
        plinth: 0.3, twoLeaf: true, reveal: 0.12, bevel: 0.022, sills: true,
      });
      const m = face.matrix(face.centre, 0, -t * 0.5);
      this.add('east', 'concrete', wall.geometry, m);
      this.solid('concrete', wall.boxes, m);
      const g = greebleFace(len - 1, h - 1, 10, 5501 + len, 0.06);
      this.add('east', 'concrete', g, face.matrix(face.centre, 0.5, 0.001));
    }

    // West elevation: shell breach plus a roller shutter.
    const west = new Face('z', z0 + t, z1 - t, x0, -1, 0);
    const breach = breachedWall(
      west.length,
      h,
      t,
      [
        { x: west.local(-18), y: 2.3, r: 2.35, seed: 771 },
        { x: west.local(-25.5), y: 5.4, r: 1.15, seed: 772 },
      ],
      [
        { x: west.local(-9.2), y: 0, w: 4.6, h: 4.5 },
        { x: west.local(-27.5), y: 6.2, w: 2.0, h: 1.5 },
      ],
    );
    // Scorch haloes around each breach, applied in wall-local space before the
    // geometry is transformed into the merge bucket.
    paintSphere(breach.geometry, new THREE.Vector3(west.local(-18), 2.3, t * 0.5), 4.4, 0x2a241f, 0.62, 1.9);
    paintSphere(breach.geometry, new THREE.Vector3(west.local(-25.5), 5.4, t * 0.5), 2.4, 0x2a241f, 0.55, 1.9);
    const wm = west.matrix(west.centre, 0, -t * 0.5);
    this.add('east', 'concrete', breach.geometry, wm);
    this.solid('concrete', breach.boxes, wm);

    // Roller shutter, half open.
    const shutter = new GeoBuilder();
    for (let i = 0; i < 9; i++) {
      const s = bevelBox(4.4, 0.17, 0.05, 0.012);
      shutter.add(s, matrixOf(0, 2.3 + i * 0.185, 0, 0));
    }
    const sg = shutter.build();
    if (sg) {
      scaleUv(sg, 1.6);
      this.add('east', 'corrugated', sg, matrixOf(x0 - 0.18, 0, -9.2, Math.PI * 0.5));
      this.collider.addBox('metal', { cx: x0 - 0.18, cy: 3.1, cz: -9.2, sx: 0.1, sy: 1.7, sz: 4.4 });
    }

    // Corrugated roof on trusses, with two holes punched through.
    const roofY = h;
    for (let i = 0; i < 7; i++) {
      const z = z0 + 1.8 + i * 3.7;
      const truss = bevelBox(x1 - x0 - 0.9, 0.22, 0.18, 0.014);
      this.add('east', 'metalRust', truss, matrixOf((x0 + x1) * 0.5, roofY - 0.24, z, 0));
      for (let j = 0; j < 5; j++) {
        const web = bevelBox(0.09, 0.5, 0.09, 0.01);
        this.add('east', 'metalRust', web, matrixOf(x0 + 1.6 + j * 2.6, roofY - 0.55, z, 0, 0, 0.5));
      }
    }
    // The sheet is authored as a vertical panel with two holes blown through it
    // and then laid flat: rotateX(-90deg) maps the panel's height axis onto -Z,
    // so the far edge lands at z1 - 0.3 and the near edge at z0 + 0.3.
    const sheetSpan = z1 - z0 - 0.6;
    const zRef = z1 - 0.3;
    const roofSheet = wallWithOpenings(x1 - x0 - 0.6, sheetSpan, 0.12, {
      openings: [
        { x: 2.4, y: 6.5, w: 4.6, h: 5.0 },
        { x: -3.6, y: 17.0, w: 3.0, h: 3.2 },
      ],
      twoLeaf: false, sills: false, lintels: false, plinth: 0, bevel: 0.01,
    });
    roofSheet.geometry.rotateX(-Math.PI * 0.5);
    roofSheet.geometry.translate((x0 + x1) * 0.5, roofY, zRef);
    scaleUv(roofSheet.geometry, 0.9);
    this.add('east', 'corrugated', roofSheet.geometry);
    for (const b of roofSheet.boxes) {
      this.collider.addBox('metal', {
        cx: (x0 + x1) * 0.5 + b.cx,
        cy: roofY + 0.06,
        cz: zRef - b.cy,
        sx: b.sx,
        sy: 0.14,
        sz: b.sy,
      });
    }
    // Light shafts need something to fall on: the debris from each roof hole
    // sits directly under it.
    this.debrisCone('east', (x0 + x1) * 0.5 + 2.4, 0.03, zRef - 6.5 - 2.5, 2.8, 0.7, 940);
    this.debrisCone('east', (x0 + x1) * 0.5 - 3.6, 0.03, zRef - 17.0 - 1.6, 1.9, 0.5, 941);

    this.interiorFloor('east', 'concrete', x0 + t, z0 + t, x1 - t, z1 - t, 0.03);

    // Mezzanine with stairs — the height advantage over the square.
    const mezY = 4.3;
    this.interiorFloor('east', 'concrete', 23.4, z0 + t, x1 - t, -17.0, mezY);
    const rail = railingGeo(6.4, 1.05, { balusterSpacing: 0.17 });
    this.add('east', 'metalRust', rail, matrixOf(23.4, mezY, -23.3, Math.PI * 0.5));
    this.collider.addBox('metal', { cx: 23.4, cy: mezY + 0.5, cz: -23.3, sx: 0.1, sy: 1.05, sz: 6.4 });
    const rail2 = railingGeo(6.0, 1.05, { balusterSpacing: 0.17 });
    this.add('east', 'metalRust', rail2, matrixOf(26.6, mezY, -17.0, 0));
    this.collider.addBox('metal', { cx: 26.6, cy: mezY + 0.5, cz: -17.0, sx: 6.0, sy: 1.05, sz: 0.1 });
    const st = stairsGeo(1.4, mezY, 4.2, 13);
    const stm = matrixOf(22.4, 0.03, -28.4, Math.PI * 0.5);
    this.add('east', 'metalRust', st.geometry, stm);
    this.solid('metal', st.boxes, stm);

    // Contents.
    for (let i = 0; i < 9; i++) {
      const x = rng.range(x0 + 1.6, x1 - 1.6);
      const z = rng.range(z0 + 1.6, z1 - 2.4);
      this.props?.placeSolid(this.collider, rng.chance(0.55) ? 'crate_large' : 'pallet', x, 0.03, z, rng.range(0, 3.14));
    }
    for (let i = 0; i < 6; i++) {
      this.props?.placeSolid(this.collider, 'barrel', rng.range(x0 + 1.4, x1 - 1.4), 0.03, rng.range(z0 + 1.4, z1 - 1.4), rng.range(0, 3.14));
    }
    for (let i = 0; i < 4; i++) {
      this.props?.placeSolid(this.collider, 'tyre', rng.range(24, 29), 0.03 + i * 0.22, -26.5 + rng.jitter(0.1), rng.range(0, 3.14));
    }

    // Debris cones spilling *into* the room, in the direction the blast
    // travelled — and a smaller one back out onto the square, because a shell
    // that goes through a wall throws material both ways.
    this.debrisCone('east', 17.9, 0.03, -18, 3.2, 0.85, 900, 1, 0.12);
    this.debrisCone('east', 17.0, 0.03, -25.5, 1.9, 0.5, 901, 1, -0.2);
    this.debrisCone('mid', 15.1, 0.02, -18.0, 2.4, 0.42, 902, -1, 0.1);
  }

  /**
   * Rubble spilling from a breach, plus rebar, loose chunks and a scorch halo.
   *
   * `dx,dz` is the direction the blast travelled. Debris that has been thrown
   * is not a heap: it is a fan, densest at the wall, thinning and getting finer
   * downrange, with the odd big piece flung well clear. Everything on this map
   * that has been hit spills the way it was hit.
   */
  private debrisCone(
    zone: string,
    x: number,
    y: number,
    z: number,
    radius: number,
    height: number,
    seed: number,
    dx = 0,
    dz = 0,
  ): void {
    const rng0 = new Rng(seed);
    const yaw = dx === 0 && dz === 0 ? rng0.range(0, 6.28) : 0;
    // Soot radiating from the seat of it, biased downrange.
    const sl = Math.hypot(dx, dz);
    this.scorch.push({
      p: new THREE.Vector3(x + (sl > 1e-4 ? (dx / sl) * radius * 0.7 : 0), y + 0.1, z + (sl > 1e-4 ? (dz / sl) * radius * 0.7 : 0)),
      r: radius * 2.3,
      s: 0.52,
    });
    const cone = rubbleCone(radius, height, Math.round(9 + radius * 2.6), seed, {
      throwX: dx,
      throwZ: dz,
      reach: 2.4,
      spread: 0.8,
    });
    scaleUv(cone, 1.4);
    this.add(zone, 'rubble', cone, matrixOf(x, y, z, yaw));
    this.collider.addBox('concrete', { cx: x, cy: y + height * 0.28, cz: z, sx: radius * 1.2, sy: height * 0.55, sz: radius * 1.2 });

    const rng = new Rng(seed + 1);
    const throwLen = Math.hypot(dx, dz);
    const ux = throwLen > 1e-4 ? dx / throwLen : 0;
    const uz = throwLen > 1e-4 ? dz / throwLen : 0;
    for (let i = 0; i < 3; i++) {
      this.props?.place('rebar_tuft', x + rng.jitter(radius * 0.7), y + rng.range(0.08, 0.35), z + rng.jitter(radius * 0.7), rng.range(0, 6.28), rng.range(0.55, 0.9));
    }
    // Chunks strung out downrange, dust and paper further still.
    for (let i = 0; i < 5; i++) {
      const t = Math.pow(rng.next(), 0.6);
      const along = t * radius * 3.0;
      const lat = rng.jitter(radius * (0.5 + t * 0.9));
      const px = x + ux * along - uz * lat + (throwLen > 1e-4 ? 0 : rng.jitter(radius * 1.4));
      const pz = z + uz * along + ux * lat + (throwLen > 1e-4 ? 0 : rng.jitter(radius * 1.4));
      this.props?.place('rubble_chunk', px, y, pz, rng.range(0, 6.28), rng.range(0.55, 1.4) * (1 - t * 0.4));
      if (rng.chance(0.4)) this.props?.place('brick_shard', px + rng.jitter(0.7), y, pz + rng.jitter(0.7), rng.range(0, 6.28), rng.range(0.7, 1.3));
      if (rng.chance(0.3)) this.props?.place('gravel', px + rng.jitter(1.1), y, pz + rng.jitter(1.1), rng.range(0, 6.28), rng.range(0.8, 1.5));
    }
    for (let i = 0; i < 3; i++) {
      const t = 0.4 + rng.next() * 1.6;
      this.props?.place(
        'paper',
        x + ux * radius * 2.6 * t + rng.jitter(radius),
        y,
        z + uz * radius * 2.6 * t + rng.jitter(radius),
        rng.range(0, 6.28),
        rng.range(0.7, 1.2),
      );
    }
  }

  // =========================================================================
  // End caps and perimeter
  // =========================================================================

  private buildEndCaps(): void {
    // North: a bombed civic building with an archway through it.
    this.archBlock('north', 'plasterCream', -30, -42, 30, -34, 12.5, 0.5, 4.6, 5001);
    // South: the same idea, lower, offset arch so the two vistas differ.
    this.archBlock('south', 'plasterOchre', -30, 34, 30, 42, 11.0, -1.0, 4.2, 5002);

    // ---- Skyline. Both fixed vistas run the length of the street and end on
    // one of these blocks; left flat they are a 60m horizontal bar across the
    // top of the frame and the composition has nowhere to go. A stepped crown,
    // a section that has been blown off it, and one tower three times the
    // height of the street wall is what turns each vista into a picture.

    // North crown: a tall west wing, a bombed-out centre-east, low parapet
    // beyond it. The step reads from the hero pose 47m down the street.
    this.crownRun('north', 'plasterCream', -30, -6.5, -41.6, -34.4, 13.85, 3.5, 5101);
    this.crownRun('north', 'plasterCream', -6.5, 5.0, -41.4, -34.6, 13.85, 1.6, 5102);
    this.brokenCrest('north', 'plasterCream', 13.5, 24.0, -41.0, -35.0, 13.85, 3.2, 5103);
    this.crownRun('north', 'plasterCream', 24.0, 30, -41.6, -34.4, 13.85, 2.2, 5104);

    // Tower placement is a framing decision, not a plan decision. The hero pose
    // looks up a corridor whose only open sky is between the west block's edge
    // and the warehouse; anything outside that wedge is behind a facade at 15m
    // and will never be seen. So it stands just east of the arch, inside the
    // wedge, tall enough to clear the warehouse roof behind it.
    this.buildTower({
      zone: 'north', mat: 'plasterCream', x: 9.4, z: -38.2, size: 5.2,
      height: 21.5, seed: 5201, broken: true,
    });

    // South crown: lower and busier, with the campanile as the focal point of
    // the skyline pose.
    this.crownRun('south', 'plasterOchre', -30, -12.0, 34.4, 41.6, 12.35, 2.4, 5111);
    this.brokenCrest('south', 'plasterOchre', -12.0, 2.0, 35.0, 41.0, 12.35, 2.8, 5112);
    this.crownRun('south', 'plasterOchre', 2.0, 16.0, 34.6, 41.4, 12.35, 4.1, 5113);
    this.crownRun('south', 'plasterOchre', 16.0, 30, 34.4, 41.6, 12.35, 1.8, 5114);

    this.buildTower({
      zone: 'south', mat: 'plasterOchre', x: 8.0, z: 37.4, size: 5.0,
      height: 23.5, seed: 5202,
    });
  }

  /**
   * A setback storey on the crown of an end block: mass, coping, a couple of
   * openings so it does not read as a solid billboard, and a shadow line where
   * it steps back from the elevation below.
   */
  private crownRun(
    zone: string,
    mat: MatKey,
    x0: number,
    x1: number,
    z0: number,
    z1: number,
    baseY: number,
    height: number,
    seed: number,
  ): void {
    if (x1 - x0 < 1 || height < 0.5) return;
    const rng = new Rng(seed);
    const w = x1 - x0;
    const d = z1 - z0;
    const cx = (x0 + x1) * 0.5;
    const cz = (z0 + z1) * 0.5;

    const g = bevelBox(w, height, d, 0.035);
    this.add(zone, mat, g, matrixOf(cx, baseY + height * 0.5, cz, 0));
    this.collider.addBox('concrete', { cx, cy: baseY + height * 0.5, cz, sx: w, sy: height, sz: d });

    // Coping all the way round: one continuous highlight along the top edge is
    // what separates a parapet from an extruded rectangle.
    for (const [len, yaw, ox, oz] of [
      [w, 0, 0, -d * 0.5], [w, Math.PI, 0, d * 0.5],
      [d, Math.PI * 0.5, -w * 0.5, 0], [d, -Math.PI * 0.5, w * 0.5, 0],
    ] as const) {
      const cop = extrudeProfile(copingProfile(0.46, 0.12), len + 0.1, { bevel: 0.009 });
      cop.rotateY(-Math.PI * 0.5);
      this.add(zone, 'stone', cop, matrixOf(cx + ox, baseY + height, cz + oz, yaw));
    }

    // Openings on the street elevation only, and only if the run is tall enough
    // to carry them.
    if (height > 2.2) {
      const face = new Face('x', x0 + 0.6, x1 - 0.6, z0 < 0 ? z1 : z0, z0 < 0 ? 1 : -1, baseY);
      const n = Math.max(1, Math.round(face.length / 3.4));
      for (let i = 0; i < n; i++) {
        if (rng.chance(0.25)) continue;
        const run = face.from + (face.to - face.from) * ((i + 0.5) / n);
        const ow = rng.range(0.85, 1.15);
        const oh = Math.min(height - 1.0, rng.range(1.1, 1.5));
        const hole = bevelBox(ow, oh, 0.7, 0.02);
        this.add('ends', 'interiorDark', hole, face.matrix(run, height * 0.45, -0.22));
        const s = extrudeProfile(sillProfile(0.17, 0.09), ow + 0.26, { bevel: 0.008 });
        s.rotateY(-Math.PI * 0.5);
        this.add(zone, 'stone', s, face.matrix(run, height * 0.45 - oh * 0.5, 0.04));
      }
    }
  }

  /**
   * The same crown with a shell through it. A run of stumps at wildly different
   * heights, rebar standing out of the tallest, and the material that came off
   * it lying on the roof behind. Silhouette does the work here — a broken
   * skyline reads as damage from 50m in a way that no amount of soot decal on a
   * straight parapet ever will.
   */
  private brokenCrest(
    zone: string,
    mat: MatKey,
    x0: number,
    x1: number,
    z0: number,
    z1: number,
    baseY: number,
    height: number,
    seed: number,
  ): void {
    const rng = new Rng(seed);
    const d = z1 - z0;
    const cz = (z0 + z1) * 0.5;
    const span = x1 - x0;
    // Blast centre: everything near it is gone, the stumps grow back to full
    // height toward the edges.
    const blast = x0 + span * rng.range(0.35, 0.62);
    let x = x0;
    let i = 0;
    while (x < x1 - 0.05) {
      const w = Math.min(x1 - x, rng.range(0.7, 2.1));
      const cx = x + w * 0.5;
      const away = Math.min(1, Math.abs(cx - blast) / (span * 0.42));
      const h = height * (0.1 + Math.pow(away, 1.4) * 0.9) * rng.range(0.75, 1.12);
      if (h > 0.22) {
        const dd = d * rng.range(0.55, 1.0);
        const g = bevelBox(w * 1.06, h, dd, 0.03);
        this.add(zone, mat, g, matrixOf(cx, baseY + h * 0.5, cz + rng.jitter(d * 0.12), rng.jitter(0.03)));
        this.collider.addBox('concrete', { cx, cy: baseY + h * 0.5, cz, sx: w, sy: h, sz: dd });
        if (h > height * 0.5) {
          const cop = extrudeProfile(copingProfile(0.44, 0.11), w * 0.9, { bevel: 0.008 });
          cop.rotateY(-Math.PI * 0.5);
          this.add(zone, 'stone', cop, matrixOf(cx, baseY + h, cz + dd * 0.5 - d * 0.5, 0));
        }
        if (rng.chance(0.4)) {
          this.props?.place('rebar_tuft', cx, baseY + h - 0.08, cz + rng.jitter(d * 0.25), rng.range(0, 6.28), rng.range(0.6, 1.0));
        }
      }
      x += w;
      i++;
    }
    // What came off it: on the roof behind, and thrown down onto the street.
    this.debrisCone(zone, blast, baseY, cz + d * 0.1, 2.4, 0.7, seed + 5, 0, z0 < 0 ? -0.4 : 0.4);
    const front = z0 < 0 ? z1 + 2.4 : z0 - 2.4;
    this.debrisCone(zone, blast, 0.01, front, 3.0, 0.95, seed + 6, 0, z0 < 0 ? 1 : -1);
  }

  /**
   * A masonry tower: plinth, tapering shaft with slit windows and string
   * courses, an open belfry, a pyramid cap. `broken` takes the cap off and
   * leaves the top storey as a stump with its floor slab hanging.
   */
  private buildTower(o: {
    zone: string;
    mat: MatKey;
    x: number;
    z: number;
    size: number;
    height: number;
    seed: number;
    broken?: boolean;
  }): void {
    const rng = new Rng(o.seed);
    const s = o.size;
    const half = s * 0.5;
    const t = 0.5;
    const belfryH = s * 1.05;
    const shaftH = o.height - belfryH;

    // Plinth.
    const plinth = bevelBox(s + 0.7, 0.85, s + 0.7, 0.04);
    this.add(o.zone, 'stone', plinth, matrixOf(o.x, 0.42, o.z, 0));
    this.collider.addBox('concrete', { cx: o.x, cy: 0.42, cz: o.z, sx: s + 0.7, sy: 0.85, sz: s + 0.7 });

    // Shaft: four elevations, each with a vertical row of slits. They are
    // narrow on purpose — a tower reads by the ratio of solid to void, and a
    // tower with domestic windows in it reads as a thin building.
    const faces: Face[] = [
      new Face('x', o.x - half, o.x + half, o.z - half, -1, 0.8),
      new Face('x', o.x - half, o.x + half, o.z + half, 1, 0.8),
      new Face('z', o.z - half + t, o.z + half - t, o.x - half, -1, 0.8),
      new Face('z', o.z - half + t, o.z + half - t, o.x + half, 1, 0.8),
    ];
    for (let f = 0; f < faces.length; f++) {
      const face = faces[f];
      const openings: Opening[] = [];
      const levels = Math.max(2, Math.floor(shaftH / 4.6));
      for (let i = 0; i < levels; i++) {
        if (rng.chance(0.18)) continue;
        openings.push({ x: rng.jitter(0.25), y: 2.6 + i * (shaftH - 4.0) / levels, w: 0.62, h: 1.55 });
      }
      const wall = wallWithOpenings(face.length, shaftH - 0.8, t, {
        openings, twoLeaf: true, reveal: 0.14, bevel: 0.026, plinth: 0, sills: true, lintels: true,
      });
      const m = face.matrix(face.centre, 0, -t * 0.5);
      this.add(o.zone, o.mat, wall.geometry, m);
      this.solid(this.kindOf(o.mat), wall.boxes, m);
      // A dark reveal behind every slit; a tower with lit holes reads hollow.
      for (const op of openings) {
        this.add('ends', 'interiorDark', plainBox(op.w + 0.4, op.h + 0.4, 0.1), face.matrix(face.run(op.x), op.y + op.h * 0.5, -1.0));
      }
      // String courses.
      for (const frac of [0.34, 0.68]) {
        const band = extrudeProfile(
          [new THREE.Vector2(0, 0.11), new THREE.Vector2(0.13, 0.09), new THREE.Vector2(0.15, -0.02), new THREE.Vector2(0, -0.08)],
          face.length + 0.02,
          { bevel: 0.008 },
        );
        band.rotateY(-Math.PI * 0.5);
        this.add(o.zone, 'stone', band, face.matrix(face.centre, 0.8 + shaftH * frac, 0.0));
      }
      // Spalled render, heavier low down where the street has chewed at it.
    }
    this.collider.addBox('concrete', { cx: o.x, cy: 0.8 + (shaftH - 0.8) * 0.5, cz: o.z, sx: s, sy: shaftH - 0.8, sz: s });

    // Cornice under the belfry.
    for (const face of faces.slice(0, 2)) {
      const c = extrudeProfile(corniceProfile(0.36, 0.42), s + 0.6, { bevel: 0.012 });
      c.rotateY(-Math.PI * 0.5);
      this.add(o.zone, 'stone', c, face.matrix(face.centre, shaftH, 0.0));
    }
    for (const face of faces.slice(2)) {
      const c = extrudeProfile(corniceProfile(0.36, 0.42), s - t * 2 + 0.6, { bevel: 0.012 });
      c.rotateY(-Math.PI * 0.5);
      this.add(o.zone, 'stone', c, face.matrix(face.centre, shaftH, 0.0));
    }

    // Belfry: four arched openings, so the top is a lantern and not a lump.
    for (let f = 0; f < 4; f++) {
      const face = faces[f];
      const a = archway(face.length, belfryH, 0.42, s * 0.5, belfryH * 0.52, belfryH * 0.28, 0.02);
      const m = face.matrix(face.centre, shaftH + 0.05, -0.21);
      this.add(o.zone, o.mat, a.geometry, m);
      this.solid(this.kindOf(o.mat), a.boxes, m);
    }
    // Something inside the lantern to catch light, and a dark backing so it is
    // not a window onto the sky.
    const bellFrame = bevelBox(s * 0.72, 0.16, 0.16, 0.014);
    this.add(o.zone, 'stone', bellFrame, matrixOf(o.x, shaftH + belfryH * 0.82, o.z, 0));
    const bell = cylinderGeo(0.16, 0.42, 0.62, 10);
    this.add(o.zone, 'stone', bell, matrixOf(o.x, shaftH + belfryH * 0.5, o.z, 0));

    if (o.broken) {
      // Cap gone: a jagged crest and the material that came off it.
      const crest = new GeoBuilder();
      for (let i = 0; i < 9; i++) {
        const a = (i / 9) * Math.PI * 2;
        const h = rng.range(0.12, 0.75);
        const b = bevelBox(rng.range(0.5, 1.1), h, 0.42, 0.025);
        crest.add(b, matrixOf(
          o.x + Math.cos(a) * (half - 0.2),
          shaftH + belfryH + h * 0.5,
          o.z + Math.sin(a) * (half - 0.2),
          -a,
        ));
      }
      const cg = crest.build();
      if (cg) this.add(o.zone, o.mat, cg);
      for (let i = 0; i < 3; i++) {
        this.props?.place('rebar_tuft', o.x + rng.jitter(half * 0.8), shaftH + belfryH + 0.08, o.z + rng.jitter(half * 0.8), rng.range(0, 6.28), rng.range(0.7, 1.1));
      }
      this.debrisCone(o.zone, o.x + rng.jitter(1.4), 0.02, o.z + (o.z < 0 ? 3.4 : -3.4), 3.4, 1.0, o.seed + 71, 0, o.z < 0 ? 1 : -1);
    } else {
      // Shallow pyramid cap on a moulded band. Four facets, sharp ridge, and a
      // finial that gives the silhouette a point to end on.
      const band = bevelBox(s + 0.34, 0.34, s + 0.34, 0.03);
      this.add(o.zone, 'stone', band, matrixOf(o.x, shaftH + belfryH + 0.17, o.z, 0));
      const cap = cylinderGeo(0.02, (s + 0.34) * 0.72, s * 0.62, 4);
      this.add(o.zone, 'stone', cap, matrixOf(o.x, shaftH + belfryH + 0.34 + s * 0.31, o.z, Math.PI * 0.25));
      const finial = cylinderGeo(0.04, 0.14, 0.75, 6);
      this.add(o.zone, 'stone', finial, matrixOf(o.x, shaftH + belfryH + 0.34 + s * 0.62 + 0.3, o.z, 0));
    }

    this.collider.addBox('concrete', {
      cx: o.x, cy: shaftH + belfryH * 0.5, cz: o.z, sx: s, sy: belfryH, sz: s,
    });
  }

  private archBlock(
    zone: string,
    mat: MatKey,
    x0: number,
    z0: number,
    x1: number,
    z1: number,
    height: number,
    archX: number,
    archW: number,
    seed: number,
  ): void {
    const rng = new Rng(seed);
    const depth = z1 - z0;
    const t = 0.95;
    const north = z0 < 0;
    const faceZ = north ? z1 : z0;
    const backZ = north ? z0 : z1;
    const outward: 1 | -1 = north ? 1 : -1;
    const face = new Face('x', x0, x1, faceZ, outward, 0);
    const back = new Face('x', x0, x1, backZ, (north ? -1 : 1) as 1 | -1, 0);
    const archHalf = archW * 0.5;
    const passH = 4.6;

    const punch = (f: Face, windows: boolean): THREE.BufferGeometry => {
      const openings: Opening[] = [{ x: f.local(archX), y: 0, w: archW, h: passH }];
      if (windows) {
        const n = Math.round((x1 - x0) / 3.1);
        for (let i = 0; i < n; i++) {
          const run = x0 + ((i + 0.5) / n) * (x1 - x0);
          if (Math.abs(run - archX) < archHalf + 1.4) continue;
          for (let s = 0; s < Math.floor(height / FLOOR_H); s++) {
            if (rng.chance(0.16)) continue;
            openings.push({ x: f.local(run), y: s * FLOOR_H + 1.0, w: 1.0, h: 1.6 });
          }
        }
      }
      const w = wallWithOpenings(x1 - x0, height, t, {
        openings,
        plinth: 0.3,
        bandY: windows ? FLOOR_H - 0.2 : undefined,
        twoLeaf: windows,
        reveal: 0.12,
        bevel: 0.024,
        sills: windows,
      });
      const m = f.matrix(f.centre, 0, -t * 0.5);
      if (windows) {
        // A shell has taken a bite out of the parapet; soot fans down from it.
        paintSphere(w.geometry, new THREE.Vector3(f.local(cx), height * 0.82, t * 0.5), 6.5, 0x3a332c, 0.5, 2.0);
      }
      this.add(zone, mat, w.geometry, m);
      this.solid(this.kindOf(mat), w.boxes, m);
      return w.geometry;
    };

    const cx = x0 + (x1 - x0) * (north ? 0.82 : 0.2);
    punch(face, true);
    punch(back, false);

    // Solid mass between the two leaves, split around the passage.
    const inner0 = faceZ - outward * t;
    const inner1 = backZ + outward * t;
    const mz = (inner0 + inner1) * 0.5;
    const md = Math.abs(inner1 - inner0);
    for (const [bx0, bx1] of [
      [x0, archX - archHalf],
      [archX + archHalf, x1],
    ] as const) {
      if (bx1 - bx0 < 0.2) continue;
      const g = bevelBox(bx1 - bx0, height, md, 0.04);
      this.add(zone, mat, g, matrixOf((bx0 + bx1) * 0.5, height * 0.5, mz, 0));
      this.collider.addBox('concrete', {
        cx: (bx0 + bx1) * 0.5, cy: height * 0.5, cz: mz, sx: bx1 - bx0, sy: height, sz: md,
      });
    }
    const over = bevelBox(archW, height - passH, md, 0.04);
    this.add(zone, mat, over, matrixOf(archX, (height + passH) * 0.5, mz, 0));
    this.collider.addBox('concrete', {
      cx: archX, cy: (height + passH) * 0.5, cz: mz, sx: archW, sy: height - passH, sz: md,
    });

    // The arch heads sitting in the two rectangular openings.
    for (const f of [face, back]) {
      const head = archway(archW + 1.7, passH + 1.6, t * 0.98, archW, 3.3, 1.2, 0.022);
      const hm = f.matrix(archX, 0, -t * 0.5);
      this.add(zone, 'stone', head.geometry, hm);
      this.solid('concrete', head.boxes.filter((b) => b.cy > passH - 0.9 && b.cy < passH + 0.4), hm);
    }
    // Barrel soffit over the passage.
    const soffit = archway(archW + 0.02, passH + 1.5, md * 0.98, archW - 0.06, 3.3, 1.2, 0.02);
    this.add(zone, 'stone', soffit.geometry, matrixOf(archX, 0, mz, 0));

    const cor = extrudeProfile(corniceProfile(0.34, 0.4), x1 - x0 + 0.2, { bevel: 0.012 });
    cor.rotateY(-Math.PI * 0.5);
    this.add(zone, 'stone', cor, face.matrix(face.centre, height, 0));
    const par = bevelBox(x1 - x0, 1.35, depth * 0.9, 0.03);
    this.add(zone, mat, par, matrixOf((x0 + x1) * 0.5, height + 0.68, (z0 + z1) * 0.5, 0));
    this.collider.addBox('concrete', {
      cx: (x0 + x1) * 0.5, cy: height + 0.68, cz: (z0 + z1) * 0.5, sx: x1 - x0, sy: 1.35, sz: depth * 0.9,
    });

    // Rubble spilling toward the street from the damaged end, thrown the way
    // the building fell.
    this.debrisCone(zone, cx, 0, faceZ + outward * 1.9, 3.8, 1.25, seed + 33, outward * 0.25, outward);

    this.floors.push({ x0, z0, x1, z1, y: 0 });
  }

  private buildPerimeter(): void {
    const rng = new Rng(0x9e21);
    const bands: Array<[number, number, number, number, number]> = [
      [-52, -52, -40, 52, 16.5],
      [44, -52, 56, 52, 17.5],
      [-40, -52, 44, -42, 14.0],
      [-40, 42, 44, 54, 13.0],
    ];
    for (const [x0, z0, x1, z1, h] of bands) {
      // Broken into bays with varied heights so the skyline is not a bathtub.
      const alongX = x1 - x0 > z1 - z0;
      const steps = Math.max(3, Math.round((alongX ? x1 - x0 : z1 - z0) / 13));
      for (let i = 0; i < steps; i++) {
        const t0 = i / steps;
        const t1 = (i + 1) / steps;
        const bx0 = alongX ? x0 + (x1 - x0) * t0 : x0;
        const bx1 = alongX ? x0 + (x1 - x0) * t1 : x1;
        const bz0 = alongX ? z0 : z0 + (z1 - z0) * t0;
        const bz1 = alongX ? z1 : z0 + (z1 - z0) * t1;
        const bh = h * rng.range(0.72, 1.22);
        const g = bevelBox(bx1 - bx0, bh, bz1 - bz0, 0.05);
        this.add('perimeter', rng.pick(['plasterCream', 'plasterOchre', 'plasterWhite', 'stone'] as MatKey[]), g, matrixOf((bx0 + bx1) * 0.5, bh * 0.5, (bz0 + bz1) * 0.5, 0));
        this.collider.addBox('concrete', {
          cx: (bx0 + bx1) * 0.5, cy: bh * 0.5, cz: (bz0 + bz1) * 0.5,
          sx: bx1 - bx0, sy: bh, sz: bz1 - bz0,
        });
        const par = bevelBox(bx1 - bx0 + 0.3, 0.9, bz1 - bz0 + 0.3, 0.03);
        this.add('perimeter', 'stone', par, matrixOf((bx0 + bx1) * 0.5, bh + 0.4, (bz0 + bz1) * 0.5, 0));
      }
    }

    // Distant skyline: cheap massing beyond the play space, for the horizon.
    const sky = new GeoBuilder();
    for (let i = 0; i < 46; i++) {
      const a = rng.range(0, Math.PI * 2);
      const r = rng.range(72, 190);
      const w = rng.range(9, 26);
      const d = rng.range(9, 26);
      const hh = rng.range(7, 34) * (1 - r / 260);
      // 70m+ out and behind the fog: a chamfer here is smaller than a pixel
      // and costs 3.6x the triangles of the box it is on.
      const g = plainBox(w, hh, d);
      sky.add(g, matrixOf(Math.cos(a) * r, hh * 0.5, Math.sin(a) * r, rng.range(0, 1.57)));
      if (rng.chance(0.4)) {
        const cap = plainBox(w * 0.4, hh * 0.28, d * 0.4);
        sky.add(cap, matrixOf(Math.cos(a) * r + rng.jitter(w * 0.2), hh + hh * 0.14, Math.sin(a) * r + rng.jitter(d * 0.2), 0));
      }
    }
    const skyGeo = sky.build();
    if (skyGeo) {
      tintGeometry(skyGeo, 0xb9ad97);
      // Aerial perspective. Fog in the pipeline handles the value shift, but
      // the far massing also has to lose *chroma* and contrast or it sits in
      // the same plane as the street wall 12m away. Lifting the albedo toward
      // the horizon colour is what puts 200m of air in front of it.
      paintAerial(skyGeo, 0, 0, 60, 220, 0x9fb2c6, 0.82);
      const mesh = new THREE.Mesh(skyGeo, this.vault!.get('plaster_painted', { seed: 99, color: 0xc4b79f, roughness: 1.1 }));
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      mesh.userData.noCollide = true;
      mesh.name = 'skyline';
      this.root.add(mesh);
      this.meshes.push(mesh);
    }
  }

  // =========================================================================
  // The square
  // =========================================================================

  private buildSquare(): void {
    const rng = new Rng(0x5a71);
    this.buildFountain(5.0, -14.0);

    // Burnt-out car, nose into the kerb.
    const car = buildBurntCar(this.vault!, 0x0cad);
    const cm = matrixOf(-2.6, 0, -20.2, 0.42);
    car.object.applyMatrix4(cm);
    this.root.add(car.object);
    this.solid('metal', car.boxes, cm);
    this.debrisCone('mid', -2.6, 0, -20.2, 2.6, 0.22, 611);

    // A second wreck jammed in the alley mouth, angled to break the sightline.
    const car2 = buildBurntCar(this.vault!, 0x0cbe);
    const cm2 = matrixOf(30.4, 0, -9.4, 1.85);
    car2.object.applyMatrix4(cm2);
    this.root.add(car2.object);
    this.solid('metal', car2.boxes, cm2);

    // Market stalls along the square's paved edge and the pavement.
    for (const [x, z, yaw] of [
      [10.6, -8.6, 0.12], [13.2, -13.4, -0.35], [9.8, -19.2, 0.5], [8.4, 5.4, -0.1], [8.6, 11.6, 0.08],
    ] as const) {
      this.props?.placeSolid(this.collider, 'stall', x, this.groundHeight(x, z), z, yaw);
      for (let i = 0; i < 3; i++) {
        this.props?.placeSolid(
          this.collider,
          rng.chance(0.5) ? 'crate_small' : 'crate_large',
          x + rng.jitter(1.5),
          this.groundHeight(x, z),
          z + rng.jitter(1.2),
          rng.range(0, 3.14),
        );
      }
    }

    // Checkpoint: jersey barriers staggered across the road, sandbag position
    // behind them. This is the cover that keeps the main street honest.
    const barriers: Array<[number, number, number]> = [
      [-4.6, -2.0, 0.06], [-1.2, -3.4, -0.1], [2.4, -1.6, 0.14], [5.6, -3.2, 0.02],
      [-5.2, 12.0, 1.55], [1.5, 15.0, 0.0], [4.8, 8.0, 1.5],
      [12.0, -3.0, 1.52], [32.0, -8.5, 0.0], [31.8, -19.0, 1.5],
    ];
    for (const [x, z, yaw] of barriers) {
      this.props?.placeSolid(this.collider, 'jersey', x, this.groundHeight(x, z), z, yaw);
    }

    for (const [ox, oz, yaw, rows] of [
      [-5.0, -6.6, 0.1, 3], [12.9, -22.4, 1.6, 3], [-8.2, 20.0, 0.0, 2],
    ] as const) {
      this.sandbagEmplacement(ox, oz, yaw, rows, rng);
    }

    for (let i = 0; i < 4; i++) {
      const x = rng.range(9.5, 15);
      const z = rng.range(-25, -6);
      this.props?.placeSolid(this.collider, 'tyre', x, this.groundHeight(x, z), z, rng.range(0, 3.14));
      this.props?.placeSolid(this.collider, 'tyre', x + rng.jitter(0.08), this.groundHeight(x, z) + 0.22, z + rng.jitter(0.08), rng.range(0, 3.14));
      this.props?.placeSolid(this.collider, 'tyre', x + rng.jitter(0.1), this.groundHeight(x, z) + 0.44, z + rng.jitter(0.1), rng.range(0, 3.14));
    }

    // Tarpaulins thrown over stacked goods — the one soft silhouette in a
    // street otherwise made entirely of hard edges.
    for (const [x, z, w, d, yaw, seed] of [
      [11.4, -10.2, 2.3, 1.9, 0.3, 401],
      [10.2, -20.4, 2.0, 1.7, -0.5, 402],
      [8.9, 16.4, 1.8, 1.6, 0.9, 403],
      [20.4, -27.0, 2.6, 2.2, 0.15, 404],
    ] as const) {
      const y = this.groundHeight(x, z, 1.5) + 0.94;
      const sheet = drapedSheet(w, d, 0.42, seed);
      this.add('mid', 'fabric', sheet, matrixOf(x, y, z, yaw));
      this.collider.addBox('fabric', { cx: x, cy: y - 0.45, cz: z, sx: w * 0.8, sy: 0.9, sz: d * 0.8 });
      for (let i = 0; i < 2; i++) {
        this.props?.placeSolid(this.collider, 'crate_small', x + rng.jitter(0.5), this.groundHeight(x, z, 1.5), z + rng.jitter(0.5), rng.range(0, 3.14));
      }
    }

    this.collapsedSlabs();

    // Rubble drifts where the buildings have shed material. Each one spills
    // away from the wall that dropped it, not in a tidy ring around itself.
    for (const [x, z, r, h, s, dx, dz] of [
      [-6.2, -16.0, 2.6, 0.7, 701, 1, 0.2],
      [-6.6, -3.4, 2.0, 0.5, 702, 1, -0.3],
      [16.4, -6.2, 2.4, 0.6, 703, -1, 0.35],
      [-27.2, 6.0, 2.2, 0.55, 704, 1, 0.1],
      [21.0, 21.5, 2.0, 0.5, 705, -0.6, -1],
      [-16.0, -27.0, 2.6, 0.65, 706, 0.2, 1],
    ] as const) {
      this.debrisCone('mid', x, 0, z, r, h, s, dx, dz);
    }
  }

  /**
   * Utility poles and the wires between them.
   *
   * Both fixed vistas are a street corridor with a band of empty sky down the
   * middle of the frame. Catenaries crossing that band cost a few hundred
   * triangles and give the composition leading lines, a sense of the volume of
   * air over the street, and a silhouette element at a scale nothing else on
   * the map occupies. Every reference frame of a built-up engagement has them.
   */
  private buildUtilities(): void {
    const rng = new Rng(0x7017);

    // Zones here are chosen to land in buckets the map already has, so a pole
    // and a hundred metres of wire cost zero extra draw calls.
    interface Pole { x: number; z: number; h: number; lean: number; zone: string; }
    const poles: Pole[] = [
      { x: 8.7, z: -6.6, h: 8.6, lean: 0.035, zone: 'east' },
      { x: 10.4, z: -24.6, h: 9.3, lean: -0.05, zone: 'east' },
      { x: 7.85, z: 12.2, h: 8.1, lean: 0.06, zone: 'east' },
      { x: -8.9, z: 4.4, h: 7.6, lean: -0.03, zone: 'west' },
    ];

    for (const p of poles) {
      const y0 = this.groundHeight(p.x, p.z);
      const trunk = cylinderGeo(0.13, 0.19, p.h, 8);
      const m = matrixOf(p.x, y0 + p.h * 0.5, p.z, rng.range(0, 6.28), p.lean, p.lean * 0.6);
      this.add(p.zone, 'woodDark', trunk, m);
      this.collider.addBox('wood', { cx: p.x, cy: y0 + p.h * 0.5, cz: p.z, sx: 0.36, sy: p.h, sz: 0.36 });

      // Two crossarms with insulators, and a transformer can on the tallest.
      for (const [dy, len] of [[-0.55, 1.9], [-1.35, 1.45]] as const) {
        const yaw = rng.range(0, 3.14);
        const arm = bevelBox(len, 0.11, 0.13, 0.014);
        this.add(p.zone, 'woodDark', arm, matrixOf(p.x, y0 + p.h + dy, p.z, yaw));
        for (const s of [-1, 0, 1]) {
          if (s === 0 && rng.chance(0.5)) continue;
          const ins = cylinderGeo(0.045, 0.06, 0.13, 6);
          this.add(p.zone, 'stone', ins, matrixOf(
            p.x + Math.cos(yaw) * s * len * 0.42,
            y0 + p.h + dy + 0.11,
            p.z - Math.sin(yaw) * s * len * 0.42,
            0,
          ));
        }
      }
      if (rng.chance(0.6)) {
        const can = cylinderGeo(0.24, 0.24, 0.62, 10);
        this.add(p.zone, 'metalRust', can, matrixOf(p.x + 0.3, y0 + p.h - 2.4, p.z, 0));
      }
      // Everything collects at the base of a pole.
      this.props?.place('gravel', p.x + rng.jitter(0.6), y0, p.z + rng.jitter(0.6), rng.range(0, 6.28), 1.2);
      this.props?.place('weed', p.x + rng.jitter(0.45), y0, p.z + rng.jitter(0.45), rng.range(0, 6.28), 1.1);
    }

    // Spans. Doubled wires where the run is long, because one lonely wire reads
    // as a mistake and three read as a street.
    const spans: Array<[number, number, number, number, number, number, number]> = [
      // x0, y0, z0, x1, y1, z1, sag
      [8.7, 7.95, -6.6, -6.95, 6.9, -4.6, 0.95],
      [8.7, 8.2, -6.6, 10.4, 8.85, -24.6, 1.35],
      [10.4, 8.6, -24.6, 16.1, 8.2, -27.5, 0.55],
      [10.4, 8.05, -24.6, 6.6, 9.6, -34.2, 0.8],
      [7.85, 7.5, 12.2, -8.9, 7.1, 4.4, 1.25],
      [-8.9, 7.05, 4.4, -10.5, 6.5, 9.0, 0.4],
      [7.85, 7.2, 12.2, 9.45, 6.6, 19.4, 0.5],
      [8.7, 7.4, -6.6, 7.85, 7.6, 12.2, 1.5],
    ];
    const wires = new GeoBuilder();
    for (let i = 0; i < spans.length; i++) {
      const [ax, ay, az, bx, by, bz, sag] = spans[i];
      const n = Math.hypot(bx - ax, bz - az) > 17 ? 2 : 1;
      for (let k = 0; k < n; k++) {
        const off = (k - (n - 1) * 0.5) * 0.3;
        wires.add(
          cableGeo(
            new THREE.Vector3(ax, ay - k * 0.34, az + off),
            new THREE.Vector3(bx, by - k * 0.34, bz + off),
            sag * rng.range(0.85, 1.2),
            0.021,
            12,
          ),
        );
      }
    }
    const wg = wires.build();
    if (wg) this.add('east', 'metalRust', wg);
  }

  /**
   * A collapsed slab stack leaning off the warehouse's west wall.
   *
   * The hero pose looks straight down the street and everything in it is
   * vertical or horizontal. Two floor slabs that have come down and are leaning
   * at 50-odd degrees put a diagonal into the frame, throw a long raking
   * shadow across the paving, and give the eye a piece of readable damage at
   * mid-distance — which is what the squint test is asking for.
   */
  private collapsedSlabs(): void {
    const rng = new Rng(0x51ab);
    const wallX = 16.0;

    // Stub of the collapsed bay, broken off at head height.
    let x = wallX - 4.6;
    while (x < wallX - 0.4) {
      const w = Math.min(wallX - 0.4 - x, rng.range(0.6, 1.5));
      const h = rng.range(1.1, 3.4);
      const g = bevelBox(w * 1.05, h, 0.44, 0.026);
      this.add('east', 'concrete', g, matrixOf(x + w * 0.5, h * 0.5, -12.4, rng.jitter(0.03)));
      this.collider.addBox('concrete', { cx: x + w * 0.5, cy: h * 0.5, cz: -12.4, sx: w, sy: h, sz: 0.44 });
      if (rng.chance(0.5)) {
        const p = spallPatch(rng.range(0.4, 0.9), rng.range(0.4, 0.9), 0.022, 0x51ab + x * 7);
        this.add('east', 'brick', p, matrixOf(x + w * 0.5, h * 0.6, -12.63, 0));
      }
      if (rng.chance(0.45)) this.props?.place('rebar_tuft', x + w * 0.5, h - 0.1, -12.4, rng.range(0, 6.28), rng.range(0.8, 1.5));
      x += w;
    }

    // Two slabs off the same floor plate, leaning at different angles so they
    // read as having fallen rather than been placed.
    for (const [sx, sz, tilt, len, wide, yaw] of [
      [13.6, -15.2, 0.92, 6.4, 3.6, 0.18],
      [14.6, -18.4, 1.16, 5.2, 2.8, -0.26],
    ] as const) {
      const slab = bevelBox(wide, 0.28, len, 0.03);
      const m = matrixOf(sx, Math.sin(tilt) * len * 0.5 + 0.2, sz, yaw, tilt, 0);
      this.add('east', 'concrete', slab, m);
      this.solid('concrete', [{ cx: 0, cy: 0, cz: 0, sx: wide, sy: 0.28, sz: len }], m);
      // Rebar fringing the broken edge, sagging out of the underside.
      for (let i = 0; i < 3; i++) {
        const u = (i - 1) * wide * 0.26;
        this.props?.place(
          'rebar_tuft',
          sx + Math.cos(yaw) * u + Math.sin(yaw) * Math.cos(tilt) * len * 0.48,
          0.12 + Math.sin(tilt) * len * 0.02,
          sz - Math.sin(yaw) * u + Math.cos(yaw) * Math.cos(tilt) * len * 0.48,
          rng.range(0, 6.28),
          rng.range(0.6, 0.95),
        );
      }
    }

    // Everything that came down with them, spilling out into the square.
    this.debrisCone('east', 14.2, 0.02, -15.6, 3.2, 1.05, 0x51b1, -1, 0.25);
    this.debrisCone('east', 15.0, 0.02, -19.6, 2.2, 0.7, 0x51b2, -1, -0.4);
    for (let i = 0; i < 5; i++) {
      this.props?.placeSolid(this.collider, rng.chance(0.5) ? 'rubble_chunk' : 'barrel', rng.range(12.6, 15.4), 0.02, rng.range(-20.5, -13.0), rng.range(0, 3.14));
    }
  }

  private sandbagEmplacement(x: number, z: number, yaw: number, rows: number, rng: Rng): void {
    const props = this.props;
    if (!props) return;
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const y0 = this.groundHeight(x, z);
    for (let r = 0; r < rows; r++) {
      const n = 7 - r;
      for (let i = 0; i < n; i++) {
        const lx = (i - (n - 1) * 0.5) * 0.5 + (r % 2 ? 0.25 : 0) + rng.jitter(0.02);
        const lz = rng.jitter(0.03);
        const wx = x + lx * c - lz * s;
        const wz = z + lx * s + lz * c;
        props.placeSolid(this.collider, 'sandbag', wx, y0 + r * 0.19, wz, yaw + rng.jitter(0.06));
      }
    }
    // A short return wall on one flank, so it is a position and not a fence.
    for (let r = 0; r < rows; r++) {
      for (let i = 0; i < 3; i++) {
        const lz = 0.32 + i * 0.5;
        const lx = 1.7;
        const wx = x + lx * c - lz * s;
        const wz = z + lx * s + lz * c;
        props.placeSolid(this.collider, 'sandbag', wx, y0 + r * 0.19, wz, yaw + Math.PI * 0.5 + rng.jitter(0.06));
      }
    }
  }

  /** Octagonal fountain: stepped basin, moulded pedestal, animated water. */
  private buildFountain(x: number, z: number): void {
    const vault = this.vault;
    if (!vault) return;
    const R = 3.3;

    const basinProfile = [
      new THREE.Vector2(R, 0),
      new THREE.Vector2(R, 0.62),
      new THREE.Vector2(R - 0.16, 0.72),
      new THREE.Vector2(R - 0.34, 0.66),
      new THREE.Vector2(R - 0.42, 0.2),
      new THREE.Vector2(R - 0.5, 0.16),
      new THREE.Vector2(R - 0.5, 0),
    ];
    const place = matrixOf(x, 0, z, 0.17);
    const wall = new GeoBuilder();
    const sides = 8;
    for (let i = 0; i < sides; i++) {
      const a = (i / sides) * Math.PI * 2;
      const half = Math.tan(Math.PI / sides) * R;
      // The profile's +X is the outward radial direction and it sweeps along Z,
      // which is the tangent; a yaw of -a swings that pair onto the facet.
      const seg = extrudeProfile(basinProfile, half * 2 * 1.02, { bevel: 0.014 });
      wall.add(seg, matrixOf(0, 0, 0, -a));
      const bm = new THREE.Matrix4().multiplyMatrices(place, matrixOf(0, 0, 0, -a));
      this.collider.addBox('concrete', { cx: R - 0.25, cy: 0.36, cz: 0, sx: 0.5, sy: 0.72, sz: half * 2.04 }, bm);
    }
    // Step at the base.
    const step = cylinderGeo(R + 0.42, R + 0.55, 0.18, sides);
    wall.add(step, matrixOf(0, 0.09, 0, Math.PI / sides));
    // Pedestal.
    const ped = cylinderGeo(0.34, 0.5, 1.15, 10);
    wall.add(ped, matrixOf(0, 0.62, 0, 0));
    const bowl = cylinderGeo(0.86, 0.3, 0.28, 12);
    wall.add(bowl, matrixOf(0, 1.28, 0, 0));
    const finial = cylinderGeo(0.06, 0.16, 0.42, 8);
    wall.add(finial, matrixOf(0, 1.6, 0, 0));

    const geo = wall.build();
    if (geo) {
      scaleUv(geo, 0.8);
      this.add('mid', 'stone', geo, place);
    }
    this.collider.addBox('concrete', { cx: x, cy: 0.7, cz: z, sx: 1.1, sy: 1.4, sz: 1.1 });
    this.collider.addBox('concrete', { cx: x, cy: 0.09, cz: z, sx: (R + 0.5) * 2, sy: 0.18, sz: (R + 0.5) * 2 });

    // Water. A private clone of the dirty-glass look with its own normal map
    // instance, so scrolling it cannot disturb any other material.
    const base = vault.get('glass_dirty', { seed: 66 });
    const water = base.clone() as THREE.MeshPhysicalMaterial;
    water.vertexColors = true;
    water.color.setHex(0x3d6f66);
    water.roughness = 0.06;
    water.metalness = 0;
    water.transparent = true;
    water.opacity = 0.9;
    water.depthWrite = true;
    water.envMapIntensity = 2.2;
    water.side = THREE.FrontSide;
    if (base.normalMap) {
      const n = base.normalMap.clone();
      n.wrapS = THREE.RepeatWrapping;
      n.wrapT = THREE.RepeatWrapping;
      n.repeat.set(2.5, 2.5);
      n.needsUpdate = true;
      water.normalMap = n;
      water.normalScale.set(0.32, 0.32);
      this.waterNormal = n;
      this.disposables.push(n);
    }
    vault.own(water);

    const surface = new THREE.CylinderGeometry(R - 0.46, R - 0.46, 0.01, sides, 3);
    const wgeo = finalizeGeometry(surface);
    wgeo.translate(x, 0.5, z);
    const mesh = new THREE.Mesh(wgeo, water);
    mesh.receiveShadow = true;
    mesh.castShadow = false;
    mesh.userData.noCollide = true;
    mesh.name = 'fountain-water';
    this.root.add(mesh);
    this.meshes.push(mesh);
    this.collider.addBox('water', { cx: x, cy: 0.48, cz: z, sx: (R - 0.4) * 2, sy: 0.04, sz: (R - 0.4) * 2 });

    // Wet staining and algae on the basin, and weeds in the cracked step.
    const rng = new Rng(0x0f00);
    for (let i = 0; i < 14; i++) {
      const a = rng.range(0, Math.PI * 2);
      const r = R + rng.range(0.5, 1.5);
      this.props?.place('grass', x + Math.cos(a) * r, 0.0, z + Math.sin(a) * r, rng.range(0, 6.28), rng.range(0.7, 1.2));
    }
  }

  // =========================================================================
  // Scatter
  // =========================================================================

  private buildScatter(): void {
    const props = this.props;
    if (!props) return;

    const ground = (x: number, z: number) => ({ y: this.groundHeight(x, z), slope: 0 });

    // Gutters: debris hugs the base of every wall, which is where it actually
    // ends up. Bands are 1.8m deep strips along each facade line.
    const gutters: Array<[number, number, number, number]> = [
      [-6.9, -26, -4.6, 2], [7.0, -4, 9.6, 26], [16, -30, 18.4, -4],
      [-30, -32, -27.6, 32], [30, -30, 32.4, -6], [22, 2, 24.4, 20],
      [-26, -26, -23.6, 2], [9.5, -4, 40, -1.4], [-30, 32, 30, 33.8],
      [-10.4, 6, -8.0, 30], [26, 2, 28.4, 26], [9.5, 20, 26, 22.4],
    ];
    // Weighted toward the cheap kinds: gravel and paper carry the density, the
    // expensive chunks only punctuate it.
    const rubbleKinds: PropKind[] = [
      'gravel', 'gravel', 'gravel', 'brick_shard', 'brick_shard', 'paper', 'can', 'plank', 'rubble_chunk',
    ];
    for (const [x0, z0, x1, z1] of gutters) {
      props.scatter(rubbleKinds, x0, z0, x1, z1, 1.5, ground);
      props.scatter(['grass', 'weed', 'brush'], x0, z0, x1, z1, 1.5, ground, undefined, 1);
    }

    // Open ground: sparse gravel and dry weeds, denser away from the road.
    props.scatter(['gravel', 'brick_shard'], -30, -32, 34, 34, 3.4, ground, (x, z) => {
      if (x > ROAD_W0 - 0.6 && x < ROAD_W1 + 0.6 && z > -34 && z < 34) return false;
      return this.isOpen(x, z);
    });
    props.scatter(['grass', 'weed', 'shrub', 'brush'], -30, -32, 34, 34, 2.6, ground, (x, z) => {
      if (x > ROAD_W0 && x < ROAD_W1) return false;
      return this.isOpen(x, z);
    });

    // Weeds in the road cracks and along the kerb line.
    props.scatter(['grass', 'weed'], ROAD_W0, -34, ROAD_W1, 34, 4.2, ground, undefined, 0.75);

    // Alleys get the heaviest litter.
    for (const [x0, z0, x1, z1] of [
      [-30, -32, -26, 32], [30, -30, 34, -6], [22, 2, 26, 20], [9.5, -4, 40, 2],
    ] as const) {
      props.scatter(rubbleKinds, x0, z0, x1, z1, 1.35, ground);
      props.scatter(['grass', 'weed', 'brush', 'shrub'], x0, z0, x1, z1, 1.3, ground);
    }
  }

  /** True where scatter is allowed: outside every building footprint. */
  private isOpen(x: number, z: number): boolean {
    const blocked: Array<[number, number, number, number]> = [
      [-26, -26, -6.8, 2], [-26, 6, -13.7, 30], [-40, -32, -30, 32],
      [9.5, 2, 22, 20], [16, -30, 30, -4], [26, 2, 40, 26], [34, -30, 44, -6],
      [9.5, 26, 26, 33], [-30, -42, 30, -34], [-30, 34, 30, 42],
    ];
    for (const [x0, z0, x1, z1] of blocked) {
      if (x > x0 - 0.3 && x < x1 + 0.3 && z > z0 - 0.3 && z < z1 + 0.3) return false;
    }
    return true;
  }

  // =========================================================================
  // Commit
  // =========================================================================

  private commit(ctx: GameContext): void {
    const vault = this.vault;
    if (!vault) return;

    // Bake occlusion from the collision volumes before anything is merged into
    // its final material bucket.
    const baker = new OcclusionBaker(
      new THREE.Box3(new THREE.Vector3(-60, -2, -60), new THREE.Vector3(60, 30, 60)),
      0.6,
    );
    // Contact occlusion is baked separately and in 2D. The voxel baker cannot
    // resolve it — a wall's base and the road it stands on land in the same
    // 0.6m cell and get thrown away as self-occlusion, which is exactly why
    // nothing in this map looked like it was touching the ground.
    const contact = new ContactField(
      new THREE.Box3(new THREE.Vector3(-58, 0, -58), new THREE.Vector3(58, 0, 58)),
      0.35,
    );
    for (const b of this.collider.boxes) {
      baker.addBox(b.center, b.half);
      contact.addOccluder(b.center, b.half);
    }
    contact.finalize();

    for (const bucket of this.buckets.values()) {
      const geo = bucket.builder.build();
      if (!geo) continue;
      const def = surfaceDef(bucket.mat);
      const glass = def.look === 'glass_dirty';

      baker.shade(geo, {
        strength: glass ? 0.2 : 0.74,
        groundHeight: 0.7,
        groundStrength: glass ? 0.12 : 0.46,
        dirtColor: 0x6b5c46,
        seed: (def.seed ?? 1) * 7,
      });
      contact.shade(geo, {
        ground: glass ? 0.1 : bucket.mat === 'interiorDark' ? 0.15 : 0.56,
        overhead: glass ? 0.08 : 0.36,
        soffit: glass ? 0.06 : 0.32,
        tint: 0x3a3128,
        max: 0.82,
      });

      // Soot. Applied after the occlusion terms so a blast reads as burnt
      // rather than merely shaded.
      if (!glass) {
        geo.computeBoundingBox();
        const bb = geo.boundingBox;
        for (const sc of this.scorch) {
          if (bb && (
            sc.p.x + sc.r < bb.min.x || sc.p.x - sc.r > bb.max.x ||
            sc.p.y + sc.r < bb.min.y || sc.p.y - sc.r > bb.max.y ||
            sc.p.z + sc.r < bb.min.z || sc.p.z - sc.r > bb.max.z
          )) continue;
          paintSphere(geo, sc.p, sc.r, 0x2c2620, sc.s, 2.1);
        }
      }

      // Aerial perspective on the standing geometry. The perimeter ring and the
      // end blocks are 40-55m out; without a chroma lift they render at the
      // same contrast as the wall two metres from the muzzle and the frame has
      // no depth in it at all.
      paintAerial(geo, 0, 0, 20, 92, 0x9fb2c6, 0.42);

      if (!vault.triplanarEnabled) worldPlanarUv(geo, def.tiles);
      geo.computeBoundingSphere();
      geo.computeBoundingBox();

      const mat = vault.get(def.look, {
        color: def.color,
        seed: def.seed,
        roughness: def.roughness,
        metalness: def.metalness,
        emissive: def.emissive,
        emissiveIntensity: def.emissiveIntensity,
        side: def.side,
        triplanar: def.tiles,
      });

      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = `level:${bucket.key}`;
      // Glass and the unlit fake rooms behind window holes are pure cost in the
      // shadow pass and contribute nothing to it.
      // Ground planes are the lowest thing in the map and the distant massing
      // is outside every cascade that matters; neither can cast a shadow onto
      // anything, and both are large. Keeping them out of the depth passes is
      // free performance with no visible consequence.
      const zone = bucket.key.slice(0, bucket.key.indexOf('/'));
      const isFloor = bucket.mat === 'sand' || bucket.mat === 'road' || bucket.mat === 'dirt';
      mesh.castShadow =
        def.look !== 'glass_dirty' &&
        bucket.mat !== 'interiorDark' &&
        !isFloor &&
        zone !== 'perimeter';
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      mesh.userData.noCollide = true;
      this.root.add(mesh);
      this.meshes.push(mesh);
    }
    baker.dispose();

    // Props last, so their instances see the final ground heights.
    const propRoot = this.props?.build();
    if (propRoot) this.root.add(propRoot);

    // Collision: one invisible mesh per surface kind.
    const proxyMat = new THREE.MeshBasicMaterial({ visible: false });
    this.disposables.push(proxyMat);
    for (const solid of this.collider.build()) {
      const mesh = new THREE.Mesh(solid.geometry, proxyMat);
      mesh.name = `collision:${solid.surface}`;
      mesh.visible = false;
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      this.root.add(mesh);
      this.collisionMeshes.push(mesh);
      ctx.physics.addStatic(mesh, solid.surface);
    }
  }

  // =========================================================================
  // Lights, spawns, cover
  // =========================================================================

  private buildLights(ctx: GameContext): void {
    if (ctx.quality.preset === 'low') return;
    const lighting = ctx.system<LightingLike>('lighting');
    if (!lighting) return;

    // Warm practicals in the spaces the sun cannot reach. Intensities are set
    // against the sun's 5.0 rather than physical candela, so a bulb reads as a
    // bulb and not as a second sun.
    const bulbs: Array<[number, number, number, number, number, boolean]> = [
      [-13.0, 2.9, -9.0, 9.0, 9, true],
      [-21.5, 2.9, -12.0, 6.5, 7, false],
      [-23.0, 2.9 + FLOOR_H, -22.0, 6.0, 7, false],
      [-12.0, 2.9, 22.0, 7.0, 8, false],
      [22.0, 5.6, -22.0, 9.0, 10, false],
      [23.0, 3.4, -9.0, 8.0, 8, false],
      [12.0, 3.2, 8.0, 7.0, 7, false],
    ];
    for (const [x, y, z, range, intensity, shadow] of bulbs) {
      const l = lighting.addPointLight(_wv.set(x, y, z), 0xffc98a, intensity, range, shadow);
      l.name = 'practical';
      this.lights.push(l);
    }
  }

  private buildSpawns(): void {
    // The fourth column is the storey the spawn belongs to. Without it a spawn
    // inside a building resolves to the highest floor over that footprint —
    // which is the roof.
    const raw: Array<[number, number, number, number]> = [
      [-3.5, 30.0, 0, 0],
      [3.0, 31.5, 0, 0],
      [-11.8, 26.0, 0.28, 0.12],
      [12.5, 23.0, -0.22, 0.15],
      [-2.0, -30.0, Math.PI, 0],
      [3.8, -31.0, Math.PI, 0],
      [23.0, -26.5, Math.PI + 0.35, 0.03],
      [-13.5, -22.0, Math.PI - 0.5, 0],
      [32.0, -20.0, Math.PI, 0.01],
      [-28.0, -6.0, 0, 0.01],
      [-28.0, 20.0, Math.PI, 0.01],
      [24.0, 12.0, Math.PI * 0.5, 0.01],
    ];
    for (const [x, z, yaw, storey] of raw) {
      this.spawns.push({
        position: new THREE.Vector3(x, this.groundHeight(x, z, storey + 0.4) + 0.05, z),
        yaw,
      });
    }
  }

  /**
   * Cover points derived from the collision volumes: anything between waist and
   * head height gets a stand-off point off each of its four faces, rejected if
   * it lands inside something else. This is real, map-derived AI data rather
   * than a hand-typed list that rots the moment the layout changes.
   */
  private buildCover(): void {
    const boxes = this.collider.boxes;
    const tall: { center: THREE.Vector3; half: THREE.Vector3 }[] = [];
    for (const b of boxes) if (b.half.y > 0.85) tall.push(b);

    const inside = (x: number, y: number, z: number): boolean => {
      for (const t of tall) {
        if (
          Math.abs(x - t.center.x) < t.half.x + 0.34 &&
          Math.abs(z - t.center.z) < t.half.z + 0.34 &&
          y > t.center.y - t.half.y &&
          y < t.center.y + t.half.y
        ) {
          return true;
        }
      }
      return false;
    };

    const dirs: Array<[number, number]> = [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ];
    const seen = new Set<string>();
    for (const b of boxes) {
      const h = b.half.y * 2;
      if (h < 0.55 || h > 2.1) continue;
      const foot = b.half.x * b.half.z * 4;
      if (foot < 0.16 || foot > 16) continue;
      const top = b.center.y + b.half.y;
      // Cover height is measured from the floor the cover stands on, not from
      // world zero — otherwise nothing above the ground storey ever qualifies.
      const floorY = this.groundHeight(b.center.x, b.center.z, top + 0.4);
      if (top - floorY < 0.45 || top - floorY > 2.3) continue;

      for (const [dx, dz] of dirs) {
        const px = b.center.x + dx * (b.half.x + 0.62);
        const pz = b.center.z + dz * (b.half.z + 0.62);
        const py = this.groundHeight(px, pz, top + 0.4);
        if (Math.abs(py - (b.center.y - b.half.y)) > 0.6) continue;
        if (px < -40 || px > 42 || pz < -40 || pz > 40) continue;
        if (inside(px, py + 0.9, pz)) continue;
        const key = `${Math.round(px * 1.4)}:${Math.round(pz * 1.4)}:${Math.round(py)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        this.cover.push({
          position: new THREE.Vector3(px, py, pz),
          // Normal points from the cover toward the exposed side: the direction
          // the AI should be facing when it leans out.
          normal: new THREE.Vector3(dx, 0, dz),
        });
      }
    }
  }
}
