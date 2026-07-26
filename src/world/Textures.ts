/**
 * Procedural PBR texture generation.
 *
 * Every map in the game comes from here and nothing is loaded from disk. Maps
 * are generated on the GPU in two passes:
 *
 *   pass A  ->  RGBA16F "surface" target. .x is the height field, .yzw are three
 *               material-specific masks (rust, wetness, wear, ...). One noise
 *               evaluation per texel.
 *   pass B  ->  3 x RGBA8 MRT. 0 = albedo (sRGB attachment, hardware encoded),
 *               1 = tangent-space normal + height in alpha, 2 = ORM
 *               (r = ambient occlusion, g = roughness, b = metalness).
 *
 * Normals come from a Sobel of the pass-A height field, AO from a 8-direction
 * horizon sweep over the same field, so crevices genuinely darken instead of
 * being faked from albedo luminance.
 *
 * The ORM packing is the glTF convention, which means the *same* texture object
 * can be bound as aoMap (.r), roughnessMap (.g) and metalnessMap (.b) — three
 * samples exactly those channels. One texture, three slots, no extra memory.
 *
 * All noise is periodic on the tile so every map is seamless, including after
 * domain warping (warping a periodic field by a periodic offset stays periodic).
 */
import * as THREE from 'three';

export type SurfaceLook =
  | 'concrete_wall' | 'concrete_floor' | 'asphalt' | 'brick'
  | 'plaster_painted' | 'rusted_metal' | 'painted_metal' | 'corrugated_metal'
  | 'wood_plank' | 'wood_crate' | 'sand' | 'dirt_gravel' | 'rubble'
  | 'tile_floor' | 'glass_dirty' | 'fabric_canvas' | 'gun_metal' | 'gun_polymer';

export interface TextureSet {
  map: THREE.Texture;
  normalMap: THREE.Texture;
  roughnessMap: THREE.Texture;
  aoMap?: THREE.Texture;
  metalnessMap?: THREE.Texture;
  displacementMap?: THREE.Texture;
}

export interface TextureRequest {
  /** Square resolution per map. Rounded to a power of two, clamped 128..2048. */
  size?: number;
  /** Applied to wrapS/wrapT repeat of every returned map. */
  repeat?: number;
  /** Changes the noise field without changing the material's character. */
  seed?: number;
  /** Keep the half-float height field around and expose it as displacementMap. */
  displacement?: boolean;
}

/** Order must match the LOOK indices baked into the shader. */
const LOOKS: readonly SurfaceLook[] = [
  'concrete_wall', 'concrete_floor', 'asphalt', 'brick',
  'plaster_painted', 'rusted_metal', 'painted_metal', 'corrugated_metal',
  'wood_plank', 'wood_crate', 'sand', 'dirt_gravel', 'rubble',
  'tile_floor', 'glass_dirty', 'fabric_canvas', 'gun_metal', 'gun_polymer',
];

interface Tuning {
  /** Slope multiplier for the Sobel-derived normal. */
  normal: number;
  /** Height 1.0 expressed as a fraction of a tile — drives AO horizon angles. */
  relief: number;
  /** How much of the horizon term reaches the AO channel. */
  ao: number;
  /** Extra roughness added in occluded crevices (settled dust). */
  cavityRough: number;
}

const TUNING: Record<SurfaceLook, Tuning> = {
  concrete_wall: { normal: 1.35, relief: 0.013, ao: 1.05, cavityRough: 0.10 },
  concrete_floor: { normal: 1.3, relief: 0.012, ao: 0.95, cavityRough: 0.12 },
  // Asphalt relief is millimetre-scale aggregate and nothing else. The height
  // field is now a tenth as deep as it was, so the slope multiplier goes UP:
  // what the old value was amplifying was the low-frequency crack field, and
  // what this one amplifies is the grain.
  asphalt: { normal: 1.7, relief: 0.006, ao: 0.9, cavityRough: 0.07 },
  brick: { normal: 2.4, relief: 0.024, ao: 1.25, cavityRough: 0.10 },
  plaster_painted: { normal: 1.25, relief: 0.010, ao: 1.0, cavityRough: 0.10 },
  rusted_metal: { normal: 1.7, relief: 0.012, ao: 1.0, cavityRough: 0.06 },
  painted_metal: { normal: 1.1, relief: 0.008, ao: 0.8, cavityRough: 0.05 },
  corrugated_metal: { normal: 2.6, relief: 0.030, ao: 1.1, cavityRough: 0.06 },
  wood_plank: { normal: 1.6, relief: 0.016, ao: 1.05, cavityRough: 0.09 },
  wood_crate: { normal: 1.9, relief: 0.020, ao: 1.15, cavityRough: 0.10 },
  sand: { normal: 1.4, relief: 0.016, ao: 0.9, cavityRough: 0.04 },
  dirt_gravel: { normal: 2.3, relief: 0.022, ao: 1.2, cavityRough: 0.06 },
  rubble: { normal: 2.8, relief: 0.030, ao: 1.3, cavityRough: 0.08 },
  tile_floor: { normal: 1.9, relief: 0.014, ao: 1.15, cavityRough: 0.14 },
  glass_dirty: { normal: 0.7, relief: 0.004, ao: 0.4, cavityRough: 0.02 },
  fabric_canvas: { normal: 1.5, relief: 0.010, ao: 1.0, cavityRough: 0.05 },
  gun_metal: { normal: 0.9, relief: 0.005, ao: 0.7, cavityRough: 0.04 },
  gun_polymer: { normal: 1.2, relief: 0.006, ao: 0.75, cavityRough: 0.04 },
};

// ---------------------------------------------------------------------------
// GLSL
// ---------------------------------------------------------------------------

const VERT = /* glsl */ `
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4( position.xy, 0.0, 1.0 );
}
`;

/**
 * Periodic noise toolkit. Every generator takes an explicit vec2 period so a
 * field can be anisotropic (e.g. 120 cells across, 4 down for rain streaks) and
 * still tile in both axes.
 */
const NOISE = /* glsl */ `
#define PI 3.141592653589793

uniform vec2 uSeed;

float sat( float x ) { return clamp( x, 0.0, 1.0 ); }
vec3 sat3( vec3 x ) { return clamp( x, 0.0, 1.0 ); }

vec2 hash22( vec2 p, vec2 per ) {
  p = mod( p, max( per, vec2( 1.0 ) ) );
  vec2 q = vec2( dot( p, vec2( 127.1, 311.7 ) ), dot( p, vec2( 269.5, 183.3 ) ) );
  return fract( sin( q + uSeed ) * 43758.5453123 ) * 2.0 - 1.0;
}

float hash12( vec2 p, vec2 per ) {
  p = mod( p, max( per, vec2( 1.0 ) ) );
  return fract( sin( dot( p, vec2( 419.2, 371.9 ) ) + uSeed.y ) * 24634.6345 );
}

float gnoise( vec2 x, vec2 per ) {
  vec2 i = floor( x );
  vec2 f = x - i;
  vec2 u = f * f * f * ( f * ( f * 6.0 - 15.0 ) + 10.0 );
  float a = dot( hash22( i + vec2( 0.0, 0.0 ), per ), f - vec2( 0.0, 0.0 ) );
  float b = dot( hash22( i + vec2( 1.0, 0.0 ), per ), f - vec2( 1.0, 0.0 ) );
  float c = dot( hash22( i + vec2( 0.0, 1.0 ), per ), f - vec2( 0.0, 1.0 ) );
  float d = dot( hash22( i + vec2( 1.0, 1.0 ), per ), f - vec2( 1.0, 1.0 ) );
  return mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y ) * 1.35;
}

float fbm( vec2 x, vec2 per, int oct, float gain ) {
  float a = 0.5, s = 0.0, n = 0.0;
  for ( int i = 0; i < 8; i ++ ) {
    if ( i >= oct ) break;
    s += a * gnoise( x, per );
    n += a;
    x *= 2.0; per *= 2.0; a *= gain;
  }
  return s / max( n, 1e-5 );
}

/** Ridged multifractal in 0..1 — the crack / vein generator. */
float ridged( vec2 x, vec2 per, int oct ) {
  float a = 0.5, s = 0.0, n = 0.0;
  for ( int i = 0; i < 8; i ++ ) {
    if ( i >= oct ) break;
    float v = 1.0 - abs( gnoise( x, per ) );
    s += a * v * v;
    n += a;
    x *= 2.0; per *= 2.0; a *= 0.5;
  }
  return s / max( n, 1e-5 );
}

/** x = F1 distance, y = F2 distance, z = cell id in 0..1. */
vec3 worley( vec2 x, vec2 per ) {
  vec2 n = floor( x );
  vec2 f = x - n;
  float f1 = 8.0, f2 = 8.0, id = 0.0;
  for ( int j = -1; j <= 1; j ++ ) {
    for ( int i = -1; i <= 1; i ++ ) {
      vec2 g = vec2( float( i ), float( j ) );
      vec2 c = n + g;
      vec2 o = hash22( c, per ) * 0.5 + 0.5;
      vec2 r = g + o - f;
      float d = dot( r, r );
      if ( d < f1 ) { f2 = f1; f1 = d; id = hash12( c, per ); }
      else if ( d < f2 ) { f2 = d; }
    }
  }
  return vec3( sqrt( f1 ), sqrt( f2 ), id );
}

/** Domain warp — the single biggest reason procedural stops looking procedural. */
vec2 warp( vec2 p, vec2 per, float amt, int oct ) {
  return p + amt * vec2(
    fbm( p + vec2( 1.7, 9.2 ), per, oct, 0.5 ),
    fbm( p + vec2( 8.3, 2.8 ), per, oct, 0.5 )
  );
}

/** Downward-running streak field: high frequency across, stretched vertically. */
float streaks( vec2 uv, float across, float along ) {
  return fbm( vec2( uv.x * across, uv.y * along ), vec2( across, along ), 4, 0.55 ) * 0.5 + 0.5;
}
`;

/** Per-look height/mask + shading. Exactly one branch survives preprocessing. */
const LOOK_GLSL = /* glsl */ `
#if LOOK == 0
// ---------------------------------------------------------------- concrete wall
/** Bursts of rounds in tight groups — see the plaster look for the reasoning. */
float concPock( vec2 uv, out float rim ) {
  vec3 cl = worley( warp( uv * 3.0, vec2( 3.0 ), 0.35, 2 ), vec2( 3.0 ) );
  float cluster = step( 0.79, cl.z ) * ( 1.0 - smoothstep( 0.05, 0.24, cl.x ) );
  vec3 pk = worley( uv * 71.0, vec2( 71.0 ) );
  float hit = step( 0.50, pk.z ) * cluster;
  rim = ( 1.0 - smoothstep( 0.15, 0.31, pk.x ) ) * hit;
  return ( 1.0 - smoothstep( 0.03, 0.15, pk.x ) ) * hit;
}

vec4 surf( vec2 uv ) {
  // Large forms loud, fine detail quiet — one pour is a different age and a
  // different colour to the next, and that is what you should see first.
  float macro = fbm( warp( uv * 2.0, vec2( 2.0 ), 0.65, 3 ), vec2( 2.0 ), 5, 0.5 ) * 0.5 + 0.5;
  float mid = fbm( uv * 26.0, vec2( 26.0 ), 4, 0.5 ) * 0.5 + 0.5;
  float fine = fbm( uv * 200.0, vec2( 200.0 ), 3, 0.55 ) * 0.5 + 0.5;
  vec3 vor = worley( uv * 68.0, vec2( 68.0 ) );
  float pit = ( 1.0 - smoothstep( 0.0, 0.13, vor.x ) ) * step( 0.66, vor.z );
  float agg = smoothstep( 0.46, 0.82, fbm( uv * 44.0, vec2( 44.0 ), 3, 0.6 ) * 0.5 + 0.5 );
  // form-board seams: 4 pours per tile, edge wanders so it never reads as a ruler line
  float wob = fbm( vec2( uv.x * 11.0, 3.7 ), vec2( 11.0, 8.0 ), 3, 0.5 ) * 0.014;
  float sy = fract( uv.y * 4.0 + wob );
  float seam = 1.0 - smoothstep( 0.0, 0.028, min( sy, 1.0 - sy ) );
  vec3 tie = worley( uv * 4.0, vec2( 4.0 ) );
  float rod = ( 1.0 - smoothstep( 0.02, 0.10, tie.x ) ) * step( 0.82, tie.z );
  float cr = ridged( warp( uv * 9.0, vec2( 9.0 ), 0.35, 3 ), vec2( 9.0 ), 5 );
  float crack = smoothstep( 0.925, 0.995, cr ) * smoothstep( 0.32, 0.66, macro );
  // spalled face, exposing the coarse aggregate behind the fat layer
  vec3 sp = worley( warp( uv * 6.0, vec2( 6.0 ), 0.50, 3 ), vec2( 6.0 ) );
  float spall = ( 1.0 - smoothstep( 0.14, 0.27, sp.x ) ) * step( 0.70, sp.z ) * smoothstep( 0.28, 0.72, macro );
  float rim; float pock = concPock( uv, rim );

  float h = 0.64 + macro * 0.115 + mid * 0.050 + fine * 0.020 + agg * 0.028;
  h -= pit * 0.20 + seam * 0.075 + rod * 0.34 + crack * 0.150 + spall * 0.100 + pock * 0.220;
  h += rim * 0.016;

  // Water leaves the wall at the form-board seam and runs down from there, so
  // the staining is tied to the seam, not to uv.y. A uv.y gradient on a map
  // that tiles every 2.3 m is a horizontal dirt band on every storey.
  float st = streaks( uv, 130.0, 2.0 );
  float run = exp( -sy * 5.5 ) * smoothstep( 0.44, 0.92, st )
    * smoothstep( 0.40, 0.78, fbm( vec2( uv.x * 9.0, 5.3 ), vec2( 9.0, 1.0 ), 3, 0.5 ) * 0.5 + 0.5 );
  float wash = sat( streaks( uv, 80.0, 1.0 ) * 1.2 - 0.58 ) * 0.40;
  float stain = sat( run * 1.2 + wash );
  return vec4( sat( h ), stain, sat( crack + spall * 0.55 + pock * 0.8 ), agg );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float macro = fbm( warp( uv * 2.0, vec2( 2.0 ), 0.65, 3 ), vec2( 2.0 ), 5, 0.5 ) * 0.5 + 0.5;
  float fine = fbm( uv * 150.0, vec2( 150.0 ), 3, 0.5 ) * 0.5 + 0.5;
  // Bone-grey with the faintest warm cast. Wide in value, narrow in hue.
  vec3 base = vec3( 0.352, 0.340, 0.318 ) * ( 0.74 + 0.50 * macro );
  base *= 0.978 + 0.044 * fine;
  base = mix( base, vec3( 0.298, 0.288, 0.270 ), s.w * 0.45 );          // exposed aggregate
  base = mix( base, vec3( 0.150, 0.142, 0.128 ), sat( s.y ) * 0.68 );   // runoff staining
  base = mix( base, vec3( 0.470, 0.455, 0.424 ), sat( s.z ) * 0.45 );   // fresh fracture
  alb = base;
  rgh = 0.88 - 0.05 * s.w + 0.04 * sat( s.z );
  rgh = mix( rgh, 0.93, sat( s.y ) * 0.5 );
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 1
// --------------------------------------------------------------- concrete floor
vec4 surf( vec2 uv ) {
  float trowel = fbm( warp( uv * 5.0, vec2( 5.0 ), 0.70, 3 ), vec2( 5.0 ), 4, 0.55 ) * 0.5 + 0.5;
  float mid = fbm( uv * 40.0, vec2( 40.0 ), 5, 0.5 ) * 0.5 + 0.5;
  float fine = fbm( uv * 220.0, vec2( 220.0 ), 3, 0.55 ) * 0.5 + 0.5;
  vec3 vor = worley( uv * 80.0, vec2( 80.0 ) );
  float pit = ( 1.0 - smoothstep( 0.0, 0.13, vor.x ) ) * step( 0.66, vor.z );
  vec2 jw = uv * 2.0 + fbm( uv * 6.0, vec2( 6.0 ), 3, 0.5 ) * 0.012;
  vec2 j = abs( fract( jw ) - 0.5 );
  float joint = 1.0 - smoothstep( 0.004, 0.016, min( j.x, j.y ) );
  float crack = smoothstep( 0.82, 0.97, ridged( warp( uv * 10.0, vec2( 10.0 ), 0.5, 3 ), vec2( 10.0 ), 5 ) );
  float traffic = smoothstep( 0.42, 0.92, fbm( uv * 3.0 + vec2( 17.0 ), vec2( 3.0 ), 4, 0.5 ) * 0.5 + 0.5 );
  float spall = smoothstep( 0.72, 0.94, fbm( uv * 18.0 + vec2( 3.0 ), vec2( 18.0 ), 4, 0.55 ) * 0.5 + 0.5 ) * ( 1.0 - traffic * 0.8 );

  float h = 0.60 + trowel * 0.10 + mid * 0.09 + fine * 0.05;
  h -= pit * 0.26 + joint * 0.34 + crack * 0.28 + spall * 0.14;
  float dust = sat( ( 1.0 - traffic ) * ( fbm( uv * 12.0 + vec2( 31.0 ), vec2( 12.0 ), 4, 0.5 ) * 0.5 + 0.5 ) * 1.3 - 0.2 );
  return vec4( sat( h ), traffic, sat( crack + joint ), dust );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float blot = fbm( uv * 4.0 + vec2( 23.0 ), vec2( 4.0 ), 4, 0.5 ) * 0.5 + 0.5;
  vec3 base = vec3( 0.330, 0.325, 0.312 ) * ( 0.86 + 0.26 * blot );
  base = mix( base, vec3( 0.190, 0.184, 0.176 ), s.y * 0.55 );      // burnished traffic lane
  base = mix( base, vec3( 0.430, 0.424, 0.408 ), sat( s.z ) * 0.4 );
  base = mix( base, vec3( 0.455, 0.442, 0.408 ), s.w * 0.55 );      // pale settled dust
  alb = base;
  rgh = 0.84 - 0.34 * s.y + 0.10 * s.w;
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 2
// --------------------------------------------------------------------- asphalt
// The previous version drove a low-frequency ridged crack field through the
// height map at 0.14 amplitude, which produced metre-wide meandering ridges —
// dried mud, or brain coral, but never tarmac. Real asphalt relief is
// millimetre-scale aggregate: high frequency, very low amplitude. Everything
// below is authored to that rule, and the only things allowed to be large are
// the wheel lanes, the seams and the camber, none of which are relief.

/**
 * Wheel-polished lanes. A pure function of uv.x (the second fBm coordinate is a
 * constant), so the bands run unbroken the full length of the tile instead of
 * breaking into blobs, and they stay perfectly periodic across the seam.
 */
float roadLane( vec2 uv ) {
  float wander = fbm( vec2( uv.x * 2.0, uv.y * 1.0 ), vec2( 2.0, 1.0 ), 3, 0.5 ) * 0.09;
  float f = fbm( vec2( uv.x * 3.0 + wander, 0.37 ), vec2( 3.0, 1.0 ), 3, 0.5 ) * 0.5 + 0.5;
  return smoothstep( 0.36, 0.72, f );
}

/** Tar seam repairs: irregular ribbons poured over a trench, proud of the surface. */
float roadSeam( vec2 uv ) {
  float r = ridged( warp( uv * 4.0, vec2( 4.0 ), 0.28, 2 ), vec2( 4.0 ), 3 );
  float gate = smoothstep( 0.40, 0.78, fbm( uv * 2.0 + vec2( 12.0 ), vec2( 2.0 ), 3, 0.5 ) * 0.5 + 0.5 );
  return smoothstep( 0.780, 0.915, r ) * gate;
}

/** .x = utility patch (a different, coarser pour), .y = an iron cover. */
vec2 roadPatch( vec2 uv ) {
  vec3 pc = worley( warp( uv * 2.0, vec2( 2.0 ), 0.42, 2 ), vec2( 2.0 ) );
  float repair = ( 1.0 - smoothstep( 0.19, 0.34, pc.x ) ) * step( 0.55, pc.z );
  vec3 mh = worley( uv * 3.0, vec2( 3.0 ) );
  float cover = ( 1.0 - smoothstep( 0.052, 0.072, mh.x ) ) * step( 0.90, mh.z );
  return vec2( repair, cover );
}

vec4 surf( vec2 uv ) {
  // Dense fine aggregate. This is the only thing the eye should read as texture
  // at walking distance, and it is 3 mm deep.
  vec3 a1 = worley( uv * 148.0, vec2( 148.0 ) );
  vec3 a2 = worley( uv * 307.0, vec2( 307.0 ) );
  float stone = ( 1.0 - smoothstep( 0.10, 0.42, a1.x ) ) * ( 0.42 + 0.58 * step( 0.40, a1.z ) );
  float grit = 1.0 - smoothstep( 0.06, 0.30, a2.x );
  float bind = fbm( uv * 72.0, vec2( 72.0 ), 4, 0.5 ) * 0.5 + 0.5;

  // Long-wavelength camber. Far too subtle to see as a pattern; it exists so
  // the sun rakes unevenly across the street instead of lighting one flat plane.
  float camber = fbm( uv * 1.0, vec2( 1.0 ), 2, 0.5 ) * 0.5 + 0.5;

  float lane = roadLane( uv );
  float seam = roadSeam( uv );
  vec2 pat = roadPatch( uv );

  // Genuine cracks: thin, branching, and only where the binder has already
  // aged. The lanes are the last part of a road to crack — they get rolled.
  float cr = ridged( warp( uv * 9.0, vec2( 9.0 ), 0.32, 3 ), vec2( 9.0 ), 5 );
  float aged = smoothstep( 0.42, 0.78, fbm( uv * 2.0 + vec2( 34.0 ), vec2( 2.0 ), 3, 0.5 ) * 0.5 + 0.5 );
  float crack = smoothstep( 0.905, 0.985, cr ) * aged * ( 1.0 - lane * 0.6 );

  float h = 0.60 + camber * 0.030 + bind * 0.018 + stone * 0.042 + grit * 0.016;
  h += seam * 0.026 + pat.y * 0.026;
  h -= crack * 0.090 + pat.x * 0.012 + lane * 0.014;

  // Puddles pool in low spots only. The basin term reads the *low-frequency*
  // shape, not the aggregate, or the puddle mask comes out speckled.
  float basin = smoothstep( 0.46, 0.16, camber + lane * 0.30 );
  float rain = smoothstep( 0.66, 0.90, fbm( uv * 2.0 + vec2( 82.0 ), vec2( 2.0 ), 3, 0.5 ) * 0.5 + 0.5 );
  float wet = basin * rain * ( 1.0 - pat.y );

  return vec4( sat( h ), lane, sat( seam + pat.y * 0.85 ), sat( wet ) );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float stone = ( 1.0 - smoothstep( 0.10, 0.42, worley( uv * 148.0, vec2( 148.0 ) ).x ) );
  float g = fbm( uv * 44.0, vec2( 44.0 ), 3, 0.5 ) * 0.5 + 0.5;
  float macro = fbm( uv * 2.0 + vec2( 6.0 ), vec2( 2.0 ), 4, 0.5 ) * 0.5 + 0.5;
  vec2 pat = roadPatch( uv );
  float lane = s.y;

  // Near-neutral, very dark, the faintest warm cast. Anything with real chroma
  // here reads as painted tarmac; anything above ~0.10 reads as concrete.
  // The value spread across the whole road has to be wide, because at four
  // metres and a grazing angle the aggregate has already mipped away and the
  // large forms are all that is left. Flat and dark is as much a failure as
  // blue was.
  vec3 base = vec3( 0.0785, 0.0768, 0.0740 ) * ( 0.76 + 0.46 * macro );
  base *= 0.945 + 0.115 * g;
  // Exposed aggregate: low contrast per stone, but there is a lot of it.
  base = mix( base, vec3( 0.1210, 0.1170, 0.1105 ), stone * 0.62 );
  // Wheel paths are polished by tyres, so they are darker AND smoother.
  base = mix( base, vec3( 0.0512, 0.0505, 0.0500 ), lane * 0.85 );
  // The margin traffic never touches silts up with pale dust.
  float dust = ( 1.0 - lane ) * smoothstep( 0.28, 0.80, macro );
  base = mix( base, vec3( 0.1680, 0.1585, 0.1385 ), dust * 0.62 );
  base = mix( base, vec3( 0.0432, 0.0428, 0.0424 ), sat( s.z ) * 0.78 );   // fresh bitumen
  base = mix( base, vec3( 0.1010, 0.0980, 0.0920 ), pat.x * 0.75 );        // patch repair
  base = mix( base, vec3( 0.0690, 0.0680, 0.0672 ), pat.y * 0.85 );        // iron cover

  // The roughness floor is what actually keeps the road grey. A dark diffuse
  // with a low roughness hands the pixel to the sky's specular lobe, and a
  // clear-sky lobe is blue — which is exactly how a grey road renders navy.
  rgh = 0.90 - 0.05 * stone - 0.04 * macro;
  rgh = mix( rgh, 0.63, lane );                       // polished, still not glossy
  rgh = mix( rgh, 0.74, sat( s.z ) * 0.8 );           // tar skins over smooth
  rgh = mix( rgh, 0.95, dust );
  rgh = max( rgh, 0.56 );
  // Standing water: only inside the puddle mask, and it is the only place on
  // the whole road allowed below the floor.
  base *= 1.0 - 0.50 * s.w;
  rgh = mix( rgh, 0.13, s.w );

  alb = base; mtl = 0.0; opa = 1.0;
}

#elif LOOK == 3
// ----------------------------------------------------------------------- brick
vec4 surf( vec2 uv ) {
  vec2 w = uv + fbm( uv * 9.0, vec2( 9.0 ), 3, 0.5 ) * 0.006;
  // 18 courses / 6 stretchers per tile. Both counts are even so the running-bond
  // half-offset survives the tile wrap.
  float rows = 18.0, cols = 6.0;
  float row = floor( w.y * rows );
  float off = mod( row, 2.0 ) * 0.5;
  float colf = w.x * cols + off;
  vec2 id = vec2( floor( colf ), row );
  vec2 local = vec2( fract( colf ), fract( w.y * rows ) );
  float jitter = hash12( id, vec2( cols, rows ) );
  vec2 d = min( local, 1.0 - local );
  float jw = 0.052 + jitter * 0.016;
  float jh = 0.155 + jitter * 0.045;
  float edge = min( d.x / jw, d.y / jh );
  float brick = smoothstep( 0.35, 1.15, edge );

  float face = fbm( w * 70.0, vec2( 70.0 ), 4, 0.55 ) * 0.5 + 0.5;
  float mortar = fbm( w * 150.0, vec2( 150.0 ), 4, 0.55 ) * 0.5 + 0.5;
  vec3 pores = worley( w * 120.0, vec2( 120.0 ) );
  float pore = ( 1.0 - smoothstep( 0.0, 0.14, pores.x ) ) * step( 0.7, pores.z );
  // corner chipping — the noise only bites where we are already near an edge
  float chip = smoothstep( 0.55, 0.95, fbm( w * 55.0 + vec2( 12.0 ), vec2( 55.0 ), 4, 0.6 ) * 0.5 + 0.5 );
  chip *= 1.0 - smoothstep( 0.0, 1.5, edge );

  float h = mix( 0.34 + mortar * 0.10, 0.78 + face * 0.10 + jitter * 0.05, brick );
  h -= pore * 0.14 * brick + chip * 0.30;
  float eff = sat( ( 1.0 - brick ) * 0.6 + smoothstep( 0.55, 0.9, streaks( uv, 90.0, 5.0 ) ) * 0.5 );
  eff *= smoothstep( 0.25, 0.75, fbm( uv * 3.0 + vec2( 77.0 ), vec2( 3.0 ), 3, 0.5 ) * 0.5 + 0.5 );
  return vec4( sat( h ), brick, jitter, sat( eff * ( 1.0 - chip ) ) );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  vec2 w = uv + fbm( uv * 9.0, vec2( 9.0 ), 3, 0.5 ) * 0.006;
  float grain = fbm( w * 90.0, vec2( 90.0 ), 4, 0.55 ) * 0.5 + 0.5;
  float j = s.z;
  vec3 hot = vec3( 0.268, 0.104, 0.070 );
  vec3 cold = vec3( 0.148, 0.078, 0.062 );
  vec3 pale = vec3( 0.235, 0.170, 0.128 );
  vec3 clay = mix( cold, hot, sat( j * 1.4 ) );
  clay = mix( clay, pale, sat( j * 1.9 - 1.05 ) );
  clay *= 0.82 + 0.34 * grain;
  vec3 mortar = vec3( 0.395, 0.385, 0.360 ) * ( 0.85 + 0.28 * grain );
  vec3 base = mix( mortar, clay, s.y );
  base = mix( base, vec3( 0.62, 0.61, 0.58 ), s.w * 0.45 );          // efflorescence salt bloom
  float soot = sat( streaks( uv, 70.0, 3.0 ) * 1.5 - 0.7 ) * ( 1.0 - uv.y * 0.6 );
  base = mix( base, base * 0.42, soot * 0.5 );
  alb = base;
  rgh = mix( 0.93, 0.76 + 0.10 * grain, s.y ) + s.w * 0.06;
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 4
// ------------------------------------------------------------- painted plaster
// Weathering here is structural, not sprinkled. The old version scattered a
// paint-peel Worley field uniformly across the wall, which at this tiling gave
// evenly spaced dark dots of identical size — flyspecks, not a material. Every
// mark below has a cause and a direction: water runs down from a ledge, render
// spalls where it has debonded, rounds land in bursts.
//
// Rising damp and sun bleaching are deliberately NOT here. Both key off height
// above the ground, this map tiles vertically every ~2.3 m, and doing them in
// uv.y produced a horizontal dirt band on every storey. Materials.ts applies
// them in world space instead, where the wall actually knows where the floor is.

/** The wandering line a storey's sill / string course sits on. */
float plasterLedge( vec2 uv ) {
  return 0.74 + fbm( vec2( uv.x * 5.0, 2.3 ), vec2( 5.0, 1.0 ), 3, 0.5 ) * 0.030;
}

/** Block courses behind the render, seen wherever the render has come away. */
float plasterBlock( vec2 uv, out float mortar ) {
  const float ROWS = 10.0, COLS = 5.0;
  float row = floor( uv.y * ROWS );
  float off = mod( row, 2.0 ) * 0.5;
  vec2 local = vec2( fract( uv.x * COLS + off ), fract( uv.y * ROWS ) );
  vec2 d = min( local, 1.0 - local );
  mortar = 1.0 - smoothstep( 0.022, 0.060, min( d.x, d.y * 2.0 ) );
  return hash12( vec2( floor( uv.x * COLS + off ), row ), vec2( COLS, ROWS ) );
}

/**
 * Bullet pocking. Rounds arrive in bursts, so the craters come in tight groups
 * a few to a wall rather than evenly across it — the clustering is the entire
 * reason this reads as gunfire instead of as dirt.
 */
float plasterPock( vec2 uv, out float rim ) {
  vec3 cl = worley( warp( uv * 4.0, vec2( 4.0 ), 0.35, 2 ), vec2( 4.0 ) );
  float cluster = step( 0.70, cl.z ) * ( 1.0 - smoothstep( 0.05, 0.30, cl.x ) );
  vec3 pk = worley( uv * 83.0, vec2( 83.0 ) );
  float hit = step( 0.42, pk.z ) * cluster;
  rim = ( 1.0 - smoothstep( 0.16, 0.33, pk.x ) ) * hit;
  return ( 1.0 - smoothstep( 0.03, 0.16, pk.x ) ) * hit;
}

vec4 surf( vec2 uv ) {
  // The large forms carry the contrast; the fine detail is kept quiet. The old
  // balance was the other way round, which is why the wall read as a flat
  // pastel field with grit on it.
  float macro = fbm( warp( uv * 2.0, vec2( 2.0 ), 0.70, 3 ), vec2( 2.0 ), 5, 0.55 ) * 0.5 + 0.5;
  float trowel = fbm( warp( uv * 6.0, vec2( 6.0 ), 0.55, 3 ), vec2( 6.0 ), 4, 0.55 ) * 0.5 + 0.5;
  float orange = fbm( uv * 240.0, vec2( 240.0 ), 3, 0.5 ) * 0.5 + 0.5;      // roller stipple
  float craze = smoothstep( 0.895, 0.985, ridged( warp( uv * 16.0, vec2( 16.0 ), 0.30, 3 ), vec2( 16.0 ), 5 ) );

  // Render debonds in patches and falls away, taking the paint with it and
  // leaving a lipped edge and bare blockwork behind.
  vec3 sp = worley( warp( uv * 5.0, vec2( 5.0 ), 0.55, 3 ), vec2( 5.0 ) );
  float age = smoothstep( 0.34, 0.76, fbm( uv * 1.0 + vec2( 61.0 ), vec2( 1.0 ), 4, 0.5 ) * 0.5 + 0.5 );
  float spall = ( 1.0 - smoothstep( 0.15, 0.30, sp.x ) ) * step( 0.60, sp.z ) * age;
  float lip = ( 1.0 - smoothstep( 0.0, 0.035, abs( sp.x - 0.225 ) ) ) * step( 0.60, sp.z ) * age;

  float rim; float pock = plasterPock( uv, rim );

  // Runoff. Taking fract of the distance makes the fall-off wrap the tile seam,
  // so a trail starting under one ledge keeps running past the tile edge.
  float below = fract( plasterLedge( uv ) - uv.y );
  float runoff = exp( -below * 6.5 )
    * smoothstep( 0.40, 0.92, streaks( uv, 150.0, 2.0 ) )
    * smoothstep( 0.42, 0.80, fbm( vec2( uv.x * 8.0, 4.1 ), vec2( 8.0, 1.0 ), 3, 0.5 ) * 0.5 + 0.5 );
  float wash = sat( streaks( uv, 90.0, 1.0 ) * 1.25 - 0.55 ) * 0.42;
  float grime = sat( runoff * 1.15 + wash );

  float mortar; plasterBlock( uv, mortar );

  float h = 0.72 + macro * 0.085 + trowel * 0.040 + orange * 0.028;
  h -= spall * ( 0.085 + mortar * 0.045 ) + craze * 0.095 + pock * 0.250;
  h += lip * 0.030 + rim * 0.016;
  return vec4( sat( h ), sat( spall ), sat( pock * 0.9 + craze * 0.45 ), grime );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float macro = fbm( warp( uv * 2.0, vec2( 2.0 ), 0.70, 3 ), vec2( 2.0 ), 5, 0.55 ) * 0.5 + 0.5;
  float mid = fbm( warp( uv * 9.0, vec2( 9.0 ), 0.45, 3 ), vec2( 9.0 ), 4, 0.5 ) * 0.5 + 0.5;
  float repair = smoothstep( 0.54, 0.74, fbm( warp( uv * 3.0 + vec2( 9.0 ), vec2( 3.0 ), 0.50, 3 ), vec2( 3.0 ), 4, 0.5 ) * 0.5 + 0.5 );
  float fine = fbm( uv * 160.0, vec2( 160.0 ), 3, 0.5 ) * 0.5 + 0.5;

  // One warm-neutral band — bone through faded ochre. The light supplies the
  // colour of this town, not the paint.
  vec3 paint = vec3( 0.472, 0.444, 0.388 );
  paint *= 0.74 + 0.50 * macro;                                     // large forms, loud
  paint *= 0.90 + 0.19 * mid;                                       // 20-30 cm blotching
  paint *= 0.958 + 0.086 * fine;                                    // fine detail, quiet
  paint = mix( paint, vec3( 0.394, 0.374, 0.336 ), repair * 0.72 ); // patch repair, wrong tone

  float mortar; float blockId = plasterBlock( uv, mortar );
  vec3 block = mix( vec3( 0.236, 0.220, 0.196 ), vec3( 0.302, 0.284, 0.254 ), blockId );
  block = mix( block, vec3( 0.330, 0.318, 0.294 ), mortar * 0.75 );
  vec3 base = mix( paint, block, sat( s.y ) );

  base = mix( base, vec3( 0.176, 0.166, 0.148 ), sat( s.w ) * 0.60 );   // runoff, wash
  base = mix( base, vec3( 0.560, 0.540, 0.500 ), sat( s.z ) * 0.45 );   // impact scarring, clean render

  alb = base;
  rgh = 0.80 - 0.06 * macro;
  rgh = mix( rgh, 0.94, sat( s.y ) );          // bare block is dead matt
  rgh = mix( rgh, 0.90, sat( s.w ) * 0.7 );    // grime kills what sheen is left
  rgh = mix( rgh, 0.86, sat( s.z ) * 0.6 );
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 5
// ---------------------------------------------------------------- rusted metal
vec4 surf( vec2 uv ) {
  float plate = fbm( uv * 12.0, vec2( 12.0 ), 4, 0.5 ) * 0.5 + 0.5;
  // rust blooms: worley cores pushed around by fBm so they spread like real oxide
  vec2 rw = warp( uv * 5.0, vec2( 5.0 ), 0.95, 4 );
  vec3 bl = worley( rw, vec2( 5.0 ) );
  float bloom = 1.0 - smoothstep( 0.10, 0.62, bl.x );
  float field = fbm( uv * 8.0 + vec2( 19.0 ), vec2( 8.0 ), 5, 0.55 ) * 0.5 + 0.5;
  float rust = sat( bloom * 0.85 + smoothstep( 0.52, 0.86, field ) * 0.9 );
  // oxide runs downhill and stains clean steel below the bloom
  float run = sat( streaks( uv, 130.0, 3.0 ) * 1.5 - 0.4 );
  float below = smoothstep( 0.0, 0.35, rust ) * run;
  rust = sat( rust + below * 0.45 );
  float scale = fbm( uv * 90.0, vec2( 90.0 ), 4, 0.6 ) * 0.5 + 0.5;
  float pit = ( 1.0 - smoothstep( 0.0, 0.16, worley( uv * 130.0, vec2( 130.0 ) ).x ) ) * smoothstep( 0.4, 0.9, rust );
  // horizontal weld bead, one per tile
  float wy = abs( fract( uv.y * 1.0 + 0.27 ) - 0.5 ) * 2.0;
  float weld = ( 1.0 - smoothstep( 0.955, 1.0, wy ) ) * ( 0.6 + 0.4 * sin( uv.x * PI * 130.0 ) );
  float h = 0.68 + plate * 0.05 + rust * scale * 0.13 - pit * 0.26 + weld * 0.10;
  return vec4( sat( h ), rust, scale, weld );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float g = fbm( uv * 70.0, vec2( 70.0 ), 4, 0.55 ) * 0.5 + 0.5;
  vec3 steel = vec3( 0.165, 0.168, 0.175 ) * ( 0.85 + 0.3 * g );
  vec3 rustDeep = vec3( 0.140, 0.052, 0.026 );
  vec3 rustBright = vec3( 0.330, 0.140, 0.058 );
  vec3 oxide = mix( rustDeep, rustBright, sat( s.z * 1.3 ) );
  float r = sat( s.y );
  vec3 base = mix( steel, oxide, smoothstep( 0.06, 0.55, r ) );
  base = mix( base, oxide * 1.15, smoothstep( 0.55, 1.0, r ) );
  alb = base;
  mtl = 1.0 - smoothstep( 0.15, 0.6, r );                              // oxide is a dielectric
  rgh = mix( 0.34 + 0.12 * g, 0.90, smoothstep( 0.05, 0.65, r ) );
  rgh = mix( rgh, 0.55, s.w * 0.7 );                                   // weld bead
  opa = 1.0;
}

#elif LOOK == 6
// ---------------------------------------------------------------- painted metal
vec4 surf( vec2 uv ) {
  float panel = fbm( uv * 6.0, vec2( 6.0 ), 4, 0.5 ) * 0.5 + 0.5;
  float orange = fbm( uv * 300.0, vec2( 300.0 ), 3, 0.5 ) * 0.5 + 0.5;   // orange-peel spray
  // long directional scratches
  float sc = ridged( vec2( uv.x * 3.0 + fbm( uv * 8.0, vec2( 8.0 ), 3, 0.5 ) * 0.35, uv.y * 190.0 ), vec2( 3.0, 190.0 ), 4 );
  float scratch = smoothstep( 0.88, 0.99, sc );
  // chipping concentrated where fBm is already high — reads as impact damage
  vec2 cw = warp( uv * 11.0, vec2( 11.0 ), 0.6, 3 );
  float chip = smoothstep( 0.70, 0.86, fbm( cw, vec2( 11.0 ), 5, 0.55 ) * 0.5 + 0.5 );
  float core = smoothstep( 0.80, 0.93, fbm( cw, vec2( 11.0 ), 5, 0.55 ) * 0.5 + 0.5 );
  // rivets on a 1/8 grid along the panel seams
  vec2 rg = fract( uv * vec2( 8.0, 3.0 ) ) - 0.5;
  float rivet = 1.0 - smoothstep( 0.16, 0.26, length( rg * vec2( 1.0, 2.6 ) ) );
  rivet *= step( 0.86, fract( uv.y * 3.0 ) ) + step( fract( uv.y * 3.0 ), 0.14 );
  float h = 0.66 + panel * 0.06 + orange * 0.03 + rivet * 0.24;
  h -= chip * 0.06 + core * 0.06 + scratch * 0.05;
  float grime = sat( streaks( uv, 100.0, 3.0 ) * ( 1.0 - uv.y * 0.7 ) * 1.4 - 0.45 );
  return vec4( sat( h ), chip, core, sat( scratch * 0.6 + grime * 0.6 ) );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float g = fbm( uv * 120.0, vec2( 120.0 ), 3, 0.5 ) * 0.5 + 0.5;
  vec3 paint = vec3( 0.108, 0.118, 0.092 ) * ( 0.9 + 0.2 * g );        // olive drab
  vec3 primer = vec3( 0.245, 0.142, 0.088 );                            // red oxide primer
  vec3 steel = vec3( 0.190, 0.192, 0.198 );
  vec3 base = mix( paint, primer, s.y );
  base = mix( base, steel, s.z );
  float scratch = sat( s.w * 1.4 - 0.4 );
  base = mix( base, steel * 1.3, scratch * 0.55 );
  base = mix( base, base * 0.55, sat( s.w ) * 0.35 );
  alb = base;
  mtl = sat( s.z * 0.95 + scratch * 0.7 );
  rgh = mix( 0.44 + 0.08 * g, 0.66, s.y );
  rgh = mix( rgh, 0.28, s.z );
  rgh = mix( rgh, 0.22, scratch * 0.6 );
  opa = 1.0;
}

#elif LOOK == 7
// ------------------------------------------------------------- corrugated metal
vec4 surf( vec2 uv ) {
  const float RIBS = 7.0;
  float phase = uv.x * RIBS * 2.0 * PI;
  float rib = 0.5 + 0.5 * cos( phase );
  rib = pow( rib, 0.75 );                                              // flatter crowns, sharper valleys
  float valley = 1.0 - rib;
  float dent = fbm( uv * 5.0 + vec2( 13.0 ), vec2( 5.0 ), 4, 0.5 ) * 0.5 + 0.5;
  // galvanised spangle: large flat crystals with slightly different heights
  vec3 sp = worley( uv * 26.0, vec2( 26.0 ) );
  float spangle = sp.z;
  float grain = fbm( uv * 200.0, vec2( 200.0 ), 3, 0.5 ) * 0.5 + 0.5;
  float rustField = fbm( warp( uv * 7.0, vec2( 7.0 ), 0.8, 3 ), vec2( 7.0 ), 5, 0.55 ) * 0.5 + 0.5;
  float rust = sat( smoothstep( 0.50, 0.85, rustField ) * 0.8 + valley * valley * 0.55 * smoothstep( 0.35, 0.8, rustField ) );
  rust = sat( rust + sat( streaks( uv, 140.0, 3.0 ) * 1.4 - 0.55 ) * smoothstep( 0.2, 0.6, rust ) );
  // fastener washers, two rows
  vec2 fg = fract( uv * vec2( RIBS, 2.0 ) + vec2( 0.5, 0.25 ) ) - 0.5;
  float screw = 1.0 - smoothstep( 0.10, 0.20, length( fg * vec2( 1.0, RIBS / 2.0 ) ) );
  float h = 0.30 + rib * 0.52 + dent * 0.06 + grain * 0.02 + screw * 0.10 - rust * 0.05;
  return vec4( sat( h ), rust, spangle, screw );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float g = fbm( uv * 150.0, vec2( 150.0 ), 3, 0.5 ) * 0.5 + 0.5;
  vec3 zinc = vec3( 0.335, 0.345, 0.350 ) * ( 0.80 + 0.36 * s.z ) * ( 0.92 + 0.16 * g );
  vec3 oxide = mix( vec3( 0.128, 0.048, 0.024 ), vec3( 0.315, 0.135, 0.055 ), sat( g * 1.4 ) );
  float r = sat( s.y );
  vec3 base = mix( zinc, oxide, smoothstep( 0.05, 0.55, r ) );
  base = mix( base, vec3( 0.20, 0.205, 0.21 ), s.w * 0.6 );
  alb = base;
  mtl = ( 1.0 - smoothstep( 0.12, 0.6, r ) );
  rgh = mix( 0.30 + 0.22 * ( 1.0 - s.z ), 0.88, smoothstep( 0.05, 0.6, r ) );
  opa = 1.0;
}

#elif LOOK == 8
// ------------------------------------------------------------------ wood plank
vec4 surf( vec2 uv ) {
  const float PLANKS = 6.0;
  float pf = uv.y * PLANKS;
  float pi = floor( pf );
  float pl = fract( pf );
  float pid = hash12( vec2( pi, 3.0 ), vec2( PLANKS, 16.0 ) );
  // per-plank longitudinal offset so end joints never line up
  float shift = pid * 7.0;
  vec2 g = vec2( uv.x * 3.0 + shift, pl * 0.75 + pid * 4.0 );
  vec2 gw = warp( g * vec2( 4.0, 26.0 ), vec2( 12.0, 26.0 ), 0.55, 3 );
  // knots pull the growth rings into tight ellipses around them
  vec3 kn = worley( vec2( uv.x * 5.0 + shift, pl * 1.6 ), vec2( 5.0, 2.0 ) );
  float knot = 1.0 - smoothstep( 0.03, 0.22, kn.x );
  knot *= step( 0.80, kn.z );
  float rings = fract( gw.y * 0.55 + knot * 3.0 + fbm( gw * 0.5, vec2( 6.0, 13.0 ), 3, 0.5 ) * 1.5 );
  float ring = smoothstep( 0.30, 0.5, rings ) * ( 1.0 - smoothstep( 0.5, 0.72, rings ) );
  float fibre = fbm( vec2( uv.x * 400.0 + shift * 50.0, pl * 40.0 ), vec2( 400.0, 40.0 ), 3, 0.55 ) * 0.5 + 0.5;
  float gap = 1.0 - smoothstep( 0.0, 0.035, min( pl, 1.0 - pl ) );
  vec2 ng = fract( vec2( uv.x * 4.0 + 0.5, pl * 1.0 + 0.5 ) ) - 0.5;
  float nail = ( 1.0 - smoothstep( 0.02, 0.05, length( ng * vec2( 1.0, 4.0 ) ) ) ) * step( 0.35, pid );
  float wear = smoothstep( 0.35, 0.85, fbm( uv * 3.0 + vec2( 61.0 ), vec2( 3.0 ), 3, 0.5 ) * 0.5 + 0.5 );

  float h = 0.66 + ring * 0.10 + fibre * 0.07 + pid * 0.04;
  h -= gap * 0.55 + knot * 0.09 + nail * 0.30 + ( 1.0 - wear ) * 0.03;
  return vec4( sat( h ), ring, pid, sat( knot + gap * 0.6 ) );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float wear = smoothstep( 0.35, 0.85, fbm( uv * 3.0 + vec2( 61.0 ), vec2( 3.0 ), 3, 0.5 ) * 0.5 + 0.5 );
  vec3 light = vec3( 0.265, 0.180, 0.105 );
  vec3 dark = vec3( 0.115, 0.070, 0.038 );
  vec3 base = mix( light, dark, sat( s.y * 0.85 + 0.10 ) );
  base *= 0.82 + 0.36 * s.z;                                            // per-plank value shift
  base = mix( base, vec3( 0.055, 0.033, 0.020 ), sat( s.w ) * 0.8 );    // knots and shadowed gaps
  base = mix( base, base * 1.18 + vec3( 0.02, 0.017, 0.012 ), wear * 0.45 );
  alb = base;
  rgh = mix( 0.86, 0.42, wear ) + sat( s.w ) * 0.06;
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 9
// ------------------------------------------------------------------ wood crate
vec4 surf( vec2 uv ) {
  const float SLATS = 5.0;
  float pf = uv.y * SLATS;
  float pi = floor( pf ), pl = fract( pf );
  float pid = hash12( vec2( pi, 9.0 ), vec2( SLATS, 16.0 ) );
  float saw = fbm( vec2( uv.x * 30.0, pl * 260.0 + pid * 20.0 ), vec2( 30.0, 260.0 ), 3, 0.6 ) * 0.5 + 0.5;
  float grain = fbm( vec2( uv.x * 9.0 + pid * 5.0, pl * 30.0 ), vec2( 9.0, 30.0 ), 4, 0.55 ) * 0.5 + 0.5;
  float ring = smoothstep( 0.42, 0.58, fract( grain * 6.0 ) );
  float gap = 1.0 - smoothstep( 0.0, 0.030, min( pl, 1.0 - pl ) );
  // vertical batten across the slats
  float bx = abs( fract( uv.x * 2.0 + 0.25 ) - 0.5 ) * 2.0;
  float batten = smoothstep( 0.72, 0.80, bx );
  float splinter = smoothstep( 0.72, 0.95, fbm( uv * 40.0 + vec2( 7.0 ), vec2( 40.0 ), 4, 0.6 ) * 0.5 + 0.5 );
  float h = 0.60 + grain * 0.09 + saw * 0.06 + batten * 0.13 + pid * 0.04;
  h -= gap * 0.45 * ( 1.0 - batten ) + splinter * 0.10;
  float edgeWear = smoothstep( 0.55, 0.95, fbm( uv * 16.0 + vec2( 88.0 ), vec2( 16.0 ), 4, 0.55 ) * 0.5 + 0.5 );
  return vec4( sat( h ), ring, pid, sat( edgeWear * 0.7 + splinter * 0.5 ) );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  vec3 pale = vec3( 0.300, 0.222, 0.140 );
  vec3 deep = vec3( 0.155, 0.100, 0.055 );
  vec3 base = mix( pale, deep, sat( s.y * 0.7 + 0.15 ) );
  base *= 0.85 + 0.30 * s.z;
  base = mix( base, base * 0.60, sat( s.w ) * 0.5 );
  float dirt = sat( streaks( uv, 60.0, 4.0 ) * ( 1.0 - uv.y * 0.55 ) * 1.4 - 0.55 );
  base = mix( base, vec3( 0.075, 0.060, 0.045 ), dirt * 0.55 );
  alb = base;
  rgh = 0.90 - 0.10 * s.z + sat( s.w ) * 0.05;
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 10
// ------------------------------------------------------------------------ sand
vec4 surf( vec2 uv ) {
  // Integer cycle counts keep the ripple crests continuous across the tile seam;
  // the phase wobble comes from a periodic fBm so the wave never reads as a sine.
  float wobble = fbm( uv * 3.0, vec2( 3.0 ), 4, 0.5 ) * 2.4;
  float ripple = 0.5 + 0.5 * cos( ( uv.y * 13.0 + uv.x * 5.0 ) * 2.0 * PI + wobble );
  ripple = pow( ripple, 1.6 );
  float dune = fbm( uv * 3.0, vec2( 3.0 ), 4, 0.5 ) * 0.5 + 0.5;
  float med = fbm( uv * 40.0, vec2( 40.0 ), 4, 0.5 ) * 0.5 + 0.5;
  float grain = fbm( uv * 420.0, vec2( 420.0 ), 3, 0.55 ) * 0.5 + 0.5;
  vec3 pb = worley( uv * 55.0, vec2( 55.0 ) );
  float pebble = ( 1.0 - smoothstep( 0.02, 0.10, pb.x ) ) * step( 0.86, pb.z );
  float h = 0.5 + dune * 0.16 + ripple * 0.14 + med * 0.06 + grain * 0.035 + pebble * 0.12;
  float damp = smoothstep( 0.62, 0.95, fbm( uv * 3.0 + vec2( 91.0 ), vec2( 3.0 ), 3, 0.5 ) * 0.5 + 0.5 );
  return vec4( sat( h ), ripple, pebble, damp );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float g = fbm( uv * 300.0, vec2( 300.0 ), 3, 0.5 ) * 0.5 + 0.5;
  float tint = fbm( uv * 5.0 + vec2( 12.0 ), vec2( 5.0 ), 4, 0.5 ) * 0.5 + 0.5;
  vec3 base = mix( vec3( 0.430, 0.352, 0.238 ), vec3( 0.505, 0.432, 0.312 ), tint );
  base *= 0.88 + 0.24 * g;
  base = mix( base, base * 1.10, s.y * 0.25 );                          // lit crests are paler
  base = mix( base, vec3( 0.245, 0.215, 0.175 ), s.z * 0.7 );
  base *= 1.0 - 0.40 * s.w;                                             // damp sand darkens
  alb = base;
  rgh = 0.94 - 0.05 * s.y;
  rgh = mix( rgh, 0.55, s.w );
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 11
// ----------------------------------------------------------------- dirt gravel
vec4 surf( vec2 uv ) {
  vec3 big = worley( warp( uv * 22.0, vec2( 22.0 ), 0.25, 2 ), vec2( 22.0 ) );
  vec3 small = worley( uv * 60.0 + vec2( 4.0 ), vec2( 60.0 ) );
  float sBig = ( 1.0 - smoothstep( 0.06, 0.34, big.x ) ) * step( 0.42, big.z );
  float sSml = ( 1.0 - smoothstep( 0.04, 0.24, small.x ) ) * step( 0.30, small.z );
  float soil = fbm( uv * 30.0, vec2( 30.0 ), 5, 0.5 ) * 0.5 + 0.5;
  float fine = fbm( uv * 300.0, vec2( 300.0 ), 3, 0.55 ) * 0.5 + 0.5;
  float rut = 1.0 - smoothstep( 0.0, 0.11, abs( fract( uv.x * 2.0 + fbm( uv * 4.0, vec2( 4.0 ), 3, 0.5 ) * 0.05 ) - 0.5 ) );
  float h = 0.48 + soil * 0.12 + fine * 0.04 + sBig * 0.30 + sSml * 0.14 - rut * 0.16;
  float wet = smoothstep( 0.55, 0.9, fbm( uv * 3.0 + vec2( 71.0 ), vec2( 3.0 ), 3, 0.5 ) * 0.5 + 0.5 );
  wet = sat( wet + rut * 0.5 );
  return vec4( sat( h ), sat( sBig + sSml * 0.7 ), mix( big.z, small.z, 0.5 ), wet );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float g = fbm( uv * 200.0, vec2( 200.0 ), 3, 0.5 ) * 0.5 + 0.5;
  vec3 soil = vec3( 0.108, 0.082, 0.055 ) * ( 0.82 + 0.36 * g );
  vec3 stoneA = vec3( 0.245, 0.238, 0.222 );
  vec3 stoneB = vec3( 0.175, 0.148, 0.118 );
  vec3 stone = mix( stoneB, stoneA, sat( s.z * 1.3 ) ) * ( 0.85 + 0.3 * g );
  vec3 base = mix( soil, stone, sat( s.y ) );
  base *= 1.0 - 0.45 * s.w;
  alb = base;
  rgh = mix( 0.93, 0.74, sat( s.y ) );
  rgh = mix( rgh, 0.30, s.w * 0.8 );
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 12
// ---------------------------------------------------------------------- rubble
vec4 surf( vec2 uv ) {
  vec3 chunk = worley( warp( uv * 11.0, vec2( 11.0 ), 0.45, 3 ), vec2( 11.0 ) );
  vec3 mid = worley( warp( uv * 26.0 + vec2( 3.0 ), vec2( 26.0 ), 0.3, 2 ), vec2( 26.0 ) );
  vec3 grit = worley( uv * 85.0, vec2( 85.0 ) );
  // F2 - F1 gives the facet edges; the chunks read as broken, not as blobs
  float facet = smoothstep( 0.02, 0.16, chunk.y - chunk.x );
  float cA = ( 1.0 - smoothstep( 0.10, 0.42, chunk.x ) );
  float cB = ( 1.0 - smoothstep( 0.06, 0.28, mid.x ) );
  float cC = ( 1.0 - smoothstep( 0.03, 0.18, grit.x ) );
  float dustField = fbm( uv * 18.0, vec2( 18.0 ), 5, 0.5 ) * 0.5 + 0.5;
  float h = 0.28 + cA * 0.42 * facet + cB * 0.24 + cC * 0.10 + dustField * 0.06;
  float top = smoothstep( 0.55, 0.90, h );                              // dust settles on up-faces
  float dust = sat( top * ( 0.5 + 0.5 * dustField ) );
  float brickBit = step( 0.72, mid.z ) * cB;
  return vec4( sat( h ), chunk.z, brickBit, dust );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float g = fbm( uv * 160.0, vec2( 160.0 ), 3, 0.5 ) * 0.5 + 0.5;
  vec3 conc = mix( vec3( 0.235, 0.230, 0.220 ), vec3( 0.375, 0.368, 0.352 ), sat( s.y * 1.4 ) );
  conc *= 0.85 + 0.30 * g;
  vec3 brick = vec3( 0.225, 0.098, 0.068 ) * ( 0.85 + 0.3 * g );
  vec3 base = mix( conc, brick, s.z * 0.85 );
  base = mix( base, vec3( 0.400, 0.390, 0.368 ), s.w * 0.62 );
  alb = base;
  rgh = 0.90 + 0.06 * s.w - 0.04 * s.z;
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 13
// ------------------------------------------------------------------ tile floor
vec4 surf( vec2 uv ) {
  const float N = 6.0;
  vec2 w = uv + fbm( uv * 12.0, vec2( 12.0 ), 3, 0.5 ) * 0.0025;
  vec2 cell = floor( w * N );
  vec2 local = fract( w * N );
  float tid = hash12( cell, vec2( N ) );
  vec2 d = min( local, 1.0 - local );
  float e = min( d.x, d.y );
  float grout = 1.0 - smoothstep( 0.030, 0.052, e );
  float bevel = smoothstep( 0.030, 0.085, e );
  float marble = fbm( warp( w * 18.0 + tid * 30.0, vec2( 18.0 ), 0.9, 4 ), vec2( 18.0 ), 5, 0.55 ) * 0.5 + 0.5;
  float scuff = smoothstep( 0.62, 0.92, ridged( vec2( w.x * 60.0, w.y * 12.0 ), vec2( 60.0, 12.0 ), 3 ) );
  // one tile in ~10 is cracked through
  float crackedTile = step( 0.90, tid );
  float crack = smoothstep( 0.86, 0.97, ridged( local * 6.0 + tid * 12.0, vec2( 6.0 ), 4 ) ) * crackedTile;
  float chip = ( 1.0 - smoothstep( 0.0, 0.9, e * 14.0 ) ) * smoothstep( 0.6, 0.9, fbm( w * 46.0, vec2( 46.0 ), 4, 0.6 ) * 0.5 + 0.5 );
  float groutGrain = fbm( w * 180.0, vec2( 180.0 ), 3, 0.55 ) * 0.5 + 0.5;
  float h = mix( 0.42 + groutGrain * 0.08, 0.86 + marble * 0.025 + tid * 0.02, bevel );
  h -= crack * 0.28 + chip * 0.24;
  float dirt = sat( ( 1.0 - bevel ) * 1.1 + smoothstep( 0.6, 0.95, fbm( uv * 8.0 + vec2( 33.0 ), vec2( 8.0 ), 4, 0.5 ) * 0.5 + 0.5 ) * 0.4 );
  return vec4( sat( h ), bevel, tid, sat( dirt * 0.8 + scuff * 0.35 + crack * 0.6 ) );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  vec2 w = uv + fbm( uv * 12.0, vec2( 12.0 ), 3, 0.5 ) * 0.0025;
  float marble = fbm( warp( w * 18.0 + s.z * 30.0, vec2( 18.0 ), 0.9, 4 ), vec2( 18.0 ), 5, 0.55 ) * 0.5 + 0.5;
  vec3 tileA = vec3( 0.520, 0.498, 0.455 );
  vec3 tileB = vec3( 0.365, 0.348, 0.318 );
  vec3 tile = mix( tileB, tileA, sat( s.z * 0.5 + 0.4 ) );
  tile = mix( tile * 0.86, tile * 1.10, marble );
  vec3 grout = vec3( 0.225, 0.215, 0.198 );
  vec3 base = mix( grout, tile, s.y );
  base = mix( base, base * 0.62, sat( s.w ) * 0.55 );
  alb = base;
  rgh = mix( 0.90, 0.20 + 0.14 * marble, s.y );
  rgh = mix( rgh, 0.62, sat( s.w ) * 0.7 );
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 14
// ----------------------------------------------------------------- dirty glass
// The only look that is deliberately NOT tileable: the wipe arcs, the edge grime
// and the impact star are all placed relative to a single pane, so glass is
// authored for repeat = 1 (one pane per quad).
vec4 surf( vec2 uv ) {
  float dustField = fbm( uv * 26.0, vec2( 26.0 ), 5, 0.55 ) * 0.5 + 0.5;
  float dust = sat( smoothstep( 0.42, 0.9, dustField ) * 0.75 );
  // wiped arcs leave clean crescents
  float arc = 0.5 + 0.5 * sin( length( uv - vec2( 0.35, 0.55 ) ) * 46.0 + fbm( uv * 6.0, vec2( 6.0 ), 3, 0.5 ) * 3.0 );
  float wipe = smoothstep( 0.35, 0.85, arc ) * smoothstep( 0.7, 0.15, length( uv - vec2( 0.35, 0.55 ) ) );
  dust = sat( dust * ( 1.0 - wipe * 0.75 ) );
  vec3 dr = worley( uv * 40.0, vec2( 40.0 ) );
  float drop = ( 1.0 - smoothstep( 0.02, 0.09, dr.x ) ) * step( 0.66, dr.z );
  float run = sat( streaks( uv, 150.0, 4.0 ) * 1.5 - 0.62 );
  float edge = sat( 1.0 - smoothstep( 0.0, 0.16, min( min( uv.x, 1.0 - uv.x ), min( uv.y, 1.0 - uv.y ) ) ) );
  float grime = sat( dust + run * 0.6 + edge * 0.55 + drop * 0.8 );
  // impact star: radial ridged cracks from a single off-centre point
  vec2 c = uv - vec2( 0.68, 0.38 );
  float ang = atan( c.y, c.x );
  float rad = length( c );
  float star = smoothstep( 0.72, 0.95, ridged( vec2( ang * 3.4, rad * 8.0 ), vec2( 22.0, 8.0 ), 4 ) );
  star *= 1.0 - smoothstep( 0.05, 0.34, rad );
  float ring = ( 1.0 - smoothstep( 0.0, 0.012, abs( rad - 0.10 ) ) ) + ( 1.0 - smoothstep( 0.0, 0.010, abs( rad - 0.185 ) ) );
  float crack = sat( star + ring * 0.7 * ( 1.0 - smoothstep( 0.05, 0.30, rad ) ) );
  float h = 0.80 + dust * 0.05 + drop * 0.10 - crack * 0.35;
  return vec4( sat( h ), grime, crack, drop );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float g = fbm( uv * 120.0, vec2( 120.0 ), 3, 0.5 ) * 0.5 + 0.5;
  vec3 grimeCol = vec3( 0.185, 0.178, 0.160 ) * ( 0.8 + 0.4 * g );
  vec3 base = mix( vec3( 0.045, 0.050, 0.052 ), grimeCol, sat( s.y ) );
  base = mix( base, vec3( 0.62, 0.64, 0.66 ), sat( s.z ) * 0.75 );      // fractured glass scatters white
  alb = base;
  rgh = mix( 0.035, 0.52, sat( s.y ) );
  rgh = mix( rgh, 0.72, sat( s.z ) );
  mtl = 0.0;
  opa = sat( 0.10 + sat( s.y ) * 0.70 + sat( s.z ) * 0.85 );
}

#elif LOOK == 15
// --------------------------------------------------------------- canvas fabric
vec4 surf( vec2 uv ) {
  const float T = 110.0;
  vec2 w = uv + fbm( uv * 30.0, vec2( 30.0 ), 3, 0.5 ) * 0.004;         // thread wander
  vec2 t = w * T;
  vec2 c = floor( t ), f = fract( t );
  float over = mod( c.x + c.y, 2.0 );
  float thickX = 0.82 + 0.36 * hash12( vec2( c.x, 0.0 ), vec2( T, 1.0 ) );
  float thickY = 0.82 + 0.36 * hash12( vec2( 0.0, c.y ), vec2( 1.0, T ) );
  float cx = sin( f.x * PI ) * thickY;
  float cy = sin( f.y * PI ) * thickX;
  float weave = mix( cy, cx, over );
  float under = mix( cx, cy, over ) * 0.35;
  float fuzz = fbm( uv * 520.0, vec2( 520.0 ), 3, 0.6 ) * 0.5 + 0.5;
  float slack = fbm( uv * 4.0, vec2( 4.0 ), 4, 0.5 ) * 0.5 + 0.5;       // sag between tie points
  // stitched seam every half tile
  float sy = abs( fract( uv.y * 2.0 ) - 0.5 ) * 2.0;
  float seam = ( 1.0 - smoothstep( 0.93, 1.0, sy ) );
  float stitch = seam * step( 0.5, fract( uv.x * 60.0 ) );
  float h = 0.42 + weave * 0.30 + under * 0.10 + fuzz * 0.05 + slack * 0.10 + stitch * 0.10 - seam * 0.06;
  float stain = sat( smoothstep( 0.55, 0.95, fbm( uv * 5.0 + vec2( 27.0 ), vec2( 5.0 ), 4, 0.5 ) * 0.5 + 0.5 ) + streaks( uv, 40.0, 4.0 ) * 0.5 - 0.3 );
  return vec4( sat( h ), weave, over, stain );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float g = fbm( uv * 260.0, vec2( 260.0 ), 3, 0.55 ) * 0.5 + 0.5;
  vec3 base = vec3( 0.198, 0.176, 0.124 ) * ( 0.86 + 0.28 * g );
  base *= 0.82 + 0.30 * s.y;                                            // raised threads catch light
  base = mix( base, vec3( 0.078, 0.066, 0.048 ), sat( s.w ) * 0.7 );
  alb = base;
  rgh = 0.92 - 0.10 * s.y + sat( s.w ) * 0.03;
  mtl = 0.0; opa = 1.0;
}

#elif LOOK == 16
// -------------------------------------------------------------------- gun metal
vec4 surf( vec2 uv ) {
  // Bead-blasted parkerised finish over machined steel.
  vec3 blast = worley( uv * 300.0, vec2( 300.0 ) );
  float bead = 1.0 - smoothstep( 0.0, 0.30, blast.x );
  // Machining and brush passes run ALONG the part, not across it: low frequency
  // in x, very high in y. The old sine at 220 cycles beat against the mip chain
  // and moired into a plastic sheen, which is most of why this read as tubing.
  float mach = fbm( vec2( uv.x * 4.0, uv.y * 270.0 ), vec2( 4.0, 270.0 ), 3, 0.6 ) * 0.5 + 0.5;
  float brush = fbm( vec2( uv.x * 2.0, uv.y * 90.0 ), vec2( 2.0, 90.0 ), 3, 0.5 ) * 0.5 + 0.5;
  float broad = fbm( uv * 12.0, vec2( 12.0 ), 4, 0.5 ) * 0.5 + 0.5;
  float scratch = smoothstep( 0.92, 0.998, ridged( vec2( uv.x * 2.0 + fbm( uv * 9.0, vec2( 9.0 ), 3, 0.5 ) * 0.3, uv.y * 300.0 ), vec2( 2.0, 300.0 ), 3 ) );
  // Holster and handling wear takes the finish off the high points and edges,
  // never off a whole flat. Biasing it with the brush field keeps it directional.
  float wear = smoothstep( 0.66, 0.90, broad * 0.55 + brush * 0.45 );
  float h = 0.74 + bead * 0.050 + mach * 0.014 + broad * 0.022 - scratch * 0.035;
  return vec4( sat( h ), wear, mach, scratch );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float g = fbm( uv * 160.0, vec2( 160.0 ), 3, 0.5 ) * 0.5 + 0.5;
  float bead = 1.0 - smoothstep( 0.0, 0.30, worley( uv * 300.0, vec2( 300.0 ) ).x );
  // Phosphate black, neutral to a hair warm. At metalness 1 the albedo IS the
  // Fresnel colour, so a blue-biased value here tints every sky reflection on
  // the weapon blue — which is exactly what it was doing.
  vec3 park = vec3( 0.0500, 0.0488, 0.0470 ) * ( 0.85 + 0.36 * g );
  vec3 bare = vec3( 0.3450, 0.3370, 0.3230 );                 // steel through the finish
  float polish = sat( s.y * 0.85 + s.w * 0.45 );
  alb = mix( park, bare, polish );
  mtl = 1.0;
  // The roughness map is the thing that actually sells metal. Parkerising is
  // matt down in the bead-blast recesses; the machining passes cut fine
  // directional lines through it; wear polishes the high points bright, but
  // nothing on a service weapon is a mirror.
  rgh = 0.60 - 0.10 * bead;
  rgh -= ( s.z - 0.5 ) * 0.16;                                // machining lines
  rgh = mix( rgh, 0.285, polish );
  rgh = mix( rgh, 0.225, s.w * 0.7 );                         // bright scratches
  rgh += smoothstep( 0.55, 0.95, 1.0 - g ) * 0.05;            // oil film in the corners
  rgh = clamp( rgh, 0.215, 0.80 );
  opa = 1.0;
}

#else
// ------------------------------------------------------------------ gun polymer
vec4 surf( vec2 uv ) {
  vec3 stip = worley( uv * 120.0, vec2( 120.0 ) );
  float stipple = ( 1.0 - smoothstep( 0.05, 0.28, stip.x ) ) * step( 0.25, stip.z );
  float micro = fbm( uv * 480.0, vec2( 480.0 ), 3, 0.6 ) * 0.5 + 0.5;
  float mold = fbm( uv * 9.0, vec2( 9.0 ), 4, 0.5 ) * 0.5 + 0.5;
  // parting line from the injection mould, dead straight but faint
  float seam = 1.0 - smoothstep( 0.0, 0.006, abs( uv.y - 0.5 ) );
  float wear = smoothstep( 0.60, 0.90, fbm( uv * 6.0 + vec2( 44.0 ), vec2( 6.0 ), 5, 0.55 ) * 0.5 + 0.5 );
  float h = 0.70 + stipple * 0.16 + micro * 0.035 + mold * 0.02 + seam * 0.05 - wear * 0.03;
  return vec4( sat( h ), wear, stipple, micro );
}
void shade( vec2 uv, vec4 s, out vec3 alb, out float rgh, out float mtl, out float opa ) {
  float g = fbm( uv * 220.0, vec2( 220.0 ), 3, 0.5 ) * 0.5 + 0.5;
  vec3 base = vec3( 0.030, 0.031, 0.033 ) * ( 0.85 + 0.4 * g );
  base = mix( base, vec3( 0.062, 0.062, 0.064 ), s.z * 0.5 );
  base = mix( base, vec3( 0.085, 0.084, 0.082 ), s.y * 0.6 );           // rubbed-shiny handling wear
  alb = base;
  mtl = 0.0;
  // Never below ~0.4: a near-black dielectric with a low roughness under a
  // bright sky is a mirror, and a mirror of a clear sky is blue plastic.
  rgh = 0.72 - 0.06 * s.w;
  rgh = mix( rgh, 0.42, s.y * 0.85 );
  opa = 1.0;
}
#endif
`;

const FRAG_HEIGHT = /* glsl */ `
precision highp float;
in vec2 vUv;
layout( location = 0 ) out vec4 oSurf;
${NOISE}
${LOOK_GLSL}
void main() {
  oSurf = surf( vUv );
}
`;

const FRAG_MAPS = /* glsl */ `
precision highp float;
in vec2 vUv;
uniform sampler2D uHeight;
uniform float uSize;
uniform float uNormalStrength;
uniform float uRelief;
uniform float uAoStrength;
uniform float uCavityRough;
layout( location = 0 ) out vec4 oAlbedo;
layout( location = 1 ) out vec4 oNormal;
layout( location = 2 ) out vec4 oORM;
${NOISE}
${LOOK_GLSL}

float H( vec2 uv ) { return texture( uHeight, uv ).x; }

void main() {
  vec2 uv = vUv;
  vec4 s = texture( uHeight, uv );
  float h = s.x;
  float t = 1.0 / uSize;

  // Sobel gradient: wider support than a 2-tap difference, so high-frequency
  // grain turns into shading instead of into aliasing.
  float h00 = H( uv + vec2( -t, -t ) ), h10 = H( uv + vec2( 0.0, -t ) ), h20 = H( uv + vec2( t, -t ) );
  float h01 = H( uv + vec2( -t, 0.0 ) ),                                  h21 = H( uv + vec2( t, 0.0 ) );
  float h02 = H( uv + vec2( -t,  t ) ), h12 = H( uv + vec2( 0.0,  t ) ), h22 = H( uv + vec2( t,  t ) );
  float gx = ( h20 + 2.0 * h21 + h22 ) - ( h00 + 2.0 * h01 + h02 );
  float gy = ( h02 + 2.0 * h12 + h22 ) - ( h00 + 2.0 * h10 + h20 );
  // /4 undoes the Sobel kernel weight; uSize/1024 keeps slope resolution-independent.
  float k = uNormalStrength * uSize * ( 1.0 / 1024.0 ) * 0.25 * 12.0;
  vec3 n = normalize( vec3( -gx * k, -gy * k, 1.0 ) );
  oNormal = vec4( n * 0.5 + 0.5, h );

  // Horizon-sweep AO over the height field. Radii are fixed in UV so the result
  // is identical at every resolution.
  float occ = 0.0;
  for ( int d = 0; d < 8; d ++ ) {
    float a = ( float( d ) + 0.5 ) * ( 6.283185307 / 8.0 );
    vec2 dir = vec2( cos( a ), sin( a ) );
    float horizon = 0.0;
    for ( int r = 0; r < 5; r ++ ) {
      float dist = exp2( float( r ) ) * ( 1.0 / 1024.0 );
      float hs = H( uv + dir * dist );
      horizon = max( horizon, ( hs - h ) * uRelief / dist );
    }
    occ += sat( horizon );
  }
  occ /= 8.0;
  float ao = sat( 1.0 - occ * uAoStrength );
  // Sharpen: local curvature adds bite to the small crevices the sweep misses.
  float blur = ( h00 + h10 + h20 + h01 + h21 + h02 + h12 + h22 ) * 0.125;
  ao *= sat( 1.0 + ( h - blur ) * 3.0 );
  ao = clamp( ao, 0.04, 1.0 );

  vec3 alb; float rgh, mtl, opa;
  shade( uv, s, alb, rgh, mtl, opa );
  rgh = clamp( mix( rgh, min( rgh + uCavityRough, 1.0 ), 1.0 - ao ), 0.03, 1.0 );

  // Nothing real is darker than fresh soot or brighter than new gypsum. Holding
  // the whole library inside 0.03..0.85 keeps every surface responding to the
  // light instead of crushing to black or clipping to white.
  oAlbedo = vec4( clamp( alb, vec3( 0.030 ), vec3( 0.850 ) ), opa );
  oORM = vec4( ao, rgh, clamp( mtl, 0.0, 1.0 ), 1.0 );
}
`;

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

interface CacheEntry {
  set: TextureSet;
  maps: THREE.WebGLRenderTarget;
  height: THREE.WebGLRenderTarget | null;
  bytes: number;
}

const DEFAULT_SIZE = 1024;
/**
 * Soft ceiling on generated texture memory. Once crossed, later requests halve
 * their resolution rather than pushing the GPU into swapping.
 */
const MEMORY_BUDGET_BYTES = 384 * 1024 * 1024;

function pot(v: number): number {
  const clamped = Math.min(2048, Math.max(128, v));
  return 1 << Math.round(Math.log2(clamped));
}

export class TextureFactory {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly anisotropy: number;

  private readonly cache = new Map<string, CacheEntry>();
  private readonly programs = new Map<string, THREE.ShaderMaterial>();
  private readonly quad: THREE.Mesh;
  private readonly quadCamera = new THREE.Camera();
  private readonly quadGeometry: THREE.BufferGeometry;
  private readonly idleMaterial = new THREE.MeshBasicMaterial();
  private readonly heightType: THREE.TextureDataType;
  private bytesUsed = 0;
  private disposed = false;

  constructor(renderer: THREE.WebGLRenderer, anisotropy: number) {
    this.renderer = renderer;
    this.anisotropy = Math.max(1, Math.min(anisotropy, renderer.capabilities.getMaxAnisotropy()));
    // Half-float height keeps the Sobel gradients smooth. Without a renderable
    // float attachment we fall back to 8-bit — banded, but never broken.
    const ext = renderer.extensions;
    const floatRT = ext.has('EXT_color_buffer_float') || ext.has('EXT_color_buffer_half_float');
    this.heightType = floatRT ? THREE.HalfFloatType : THREE.UnsignedByteType;

    // Single oversized triangle: no diagonal seam, one fewer vertex than a quad.
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 2, 0, 0, 2]), 2));
    this.quadGeometry = g;
    this.quad = new THREE.Mesh(g, this.idleMaterial);
    this.quad.frustumCulled = false;
  }

  /** Suggested resolution for a quality preset. */
  static sizeForPreset(preset: 'low' | 'medium' | 'high' | 'ultra'): number {
    switch (preset) {
      case 'low': return 512;
      case 'medium': return 512;
      case 'high': return 1024;
      case 'ultra': return 2048;
    }
  }

  get(kind: SurfaceLook, opts: TextureRequest = {}): TextureSet {
    if (this.disposed) throw new Error('TextureFactory: used after dispose()');

    const repeat = opts.repeat ?? 1;
    const seed = opts.seed ?? 0;
    const wantDisplacement = opts.displacement === true;
    let size = pot(opts.size ?? DEFAULT_SIZE);
    if (this.bytesUsed > MEMORY_BUDGET_BYTES) size = Math.max(256, size >> 1);

    const key = `${kind}|${size}|${repeat}|${seed}|${wantDisplacement ? 'd' : ''}`;
    const hit = this.cache.get(key);
    if (hit) return hit.set;

    const entry = this.build(kind, size, repeat, seed, wantDisplacement);
    this.cache.set(key, entry);
    this.bytesUsed += entry.bytes;
    return entry.set;
  }

  private build(kind: SurfaceLook, size: number, repeat: number, seed: number, keepHeight: boolean): CacheEntry {
    const look = LOOKS.indexOf(kind);
    const tuning = TUNING[kind];
    const seedVec = new THREE.Vector2(
      // Golden-ratio decorrelated offsets; consecutive seeds give unrelated fields.
      (seed * 0.6180339887 % 1) * 512.0 + 13.0,
      (seed * 0.3819660113 % 1) * 512.0 + 71.0,
    );

    const height = new THREE.WebGLRenderTarget(size, size, {
      format: THREE.RGBAFormat,
      type: this.heightType,
      depthBuffer: false,
      stencilBuffer: false,
    });
    height.texture.name = `${kind}_height`;
    height.texture.wrapS = THREE.RepeatWrapping;
    height.texture.wrapT = THREE.RepeatWrapping;
    height.texture.minFilter = THREE.LinearFilter;
    height.texture.magFilter = THREE.LinearFilter;
    height.texture.generateMipmaps = false;
    height.texture.colorSpace = THREE.NoColorSpace;

    const maps = new THREE.WebGLRenderTarget(size, size, {
      count: 3,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: false,
      stencilBuffer: false,
    });
    const names = ['albedo', 'normal', 'orm'];
    for (let i = 0; i < 3; i++) {
      const tex = maps.textures[i];
      tex.name = `${kind}_${names[i]}`;
      tex.wrapS = THREE.RepeatWrapping;
      tex.wrapT = THREE.RepeatWrapping;
      tex.minFilter = THREE.LinearMipmapLinearFilter;
      tex.magFilter = THREE.LinearFilter;
      tex.generateMipmaps = true;
      tex.anisotropy = this.anisotropy;
      // Set before the target is allocated so the albedo attachment is created
      // as SRGB8_ALPHA8 and the hardware does the encode/decode for free.
      tex.colorSpace = i === 0 ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      tex.repeat.set(repeat, repeat);
    }
    if (keepHeight) height.texture.repeat.set(repeat, repeat);

    const heightMat = this.program(`h${look}`, FRAG_HEIGHT, look, 1);
    (heightMat.uniforms.uSeed.value as THREE.Vector2).copy(seedVec);
    this.renderPass(heightMat, height);

    const mapMat = this.program(`m${look}`, FRAG_MAPS, look, 3);
    (mapMat.uniforms.uSeed.value as THREE.Vector2).copy(seedVec);
    mapMat.uniforms.uHeight.value = height.texture;
    mapMat.uniforms.uSize.value = size;
    mapMat.uniforms.uNormalStrength.value = tuning.normal;
    mapMat.uniforms.uRelief.value = tuning.relief;
    mapMat.uniforms.uAoStrength.value = tuning.ao;
    mapMat.uniforms.uCavityRough.value = tuning.cavityRough;
    this.renderPass(mapMat, maps);
    mapMat.uniforms.uHeight.value = null;

    const orm = maps.textures[2];
    const set: TextureSet = {
      map: maps.textures[0],
      normalMap: maps.textures[1],
      roughnessMap: orm,
      aoMap: orm,
      metalnessMap: orm,
    };

    let bytes = size * size * 4 * 3 * 1.34; // three RGBA8 maps plus mip chain
    if (keepHeight) {
      set.displacementMap = height.texture;
      bytes += size * size * (this.heightType === THREE.HalfFloatType ? 8 : 4);
    } else {
      height.dispose();
    }

    return { set, maps, height: keepHeight ? height : null, bytes: Math.round(bytes) };
  }

  private program(key: string, frag: string, look: number, outputs: number): THREE.ShaderMaterial {
    const existing = this.programs.get(key);
    if (existing) return existing;
    const uniforms: Record<string, THREE.IUniform> = {
      uSeed: { value: new THREE.Vector2() },
    };
    if (outputs > 1) {
      uniforms.uHeight = { value: null };
      uniforms.uSize = { value: 1024 };
      uniforms.uNormalStrength = { value: 1 };
      uniforms.uRelief = { value: 0.01 };
      uniforms.uAoStrength = { value: 1 };
      uniforms.uCavityRough = { value: 0 };
    }
    const mat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      defines: { LOOK: look },
      uniforms,
      vertexShader: VERT,
      fragmentShader: frag,
      depthTest: false,
      depthWrite: false,
    });
    this.programs.set(key, mat);
    return mat;
  }

  private renderPass(material: THREE.ShaderMaterial, target: THREE.WebGLRenderTarget): void {
    const r = this.renderer;
    const prevTarget = r.getRenderTarget();
    const prevActiveCubeFace = r.getActiveCubeFace();
    const prevActiveMipLevel = r.getActiveMipmapLevel();
    const prevAutoClear = r.autoClear;

    this.quad.material = material;
    r.autoClear = true;
    r.setRenderTarget(target);
    r.render(this.quad, this.quadCamera);

    r.setRenderTarget(prevTarget, prevActiveCubeFace, prevActiveMipLevel);
    r.autoClear = prevAutoClear;
    this.quad.material = this.idleMaterial;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const entry of this.cache.values()) {
      entry.maps.dispose();
      entry.height?.dispose();
    }
    this.cache.clear();
    for (const mat of this.programs.values()) mat.dispose();
    this.programs.clear();
    this.idleMaterial.dispose();
    this.quadGeometry.dispose();
  }
}
