/**
 * The shared material library.
 *
 * Everything visible in the game asks for its surface here, so materials are
 * built once and handed out by reference. Two things make that safe:
 *
 *  - Textures come out of the TextureFactory at repeat = 1 and are shared by
 *    every material that uses the same look. Tiling density is therefore *not*
 *    a texture property (which would be global state) but a per-material
 *    `uUvScale` uniform injected into the vertex stage. Different repeats cost
 *    one uniform, not another 12 MB of VRAM.
 *  - The albedo/normal/ORM maps are authoritative: `color` is white and
 *    `roughness`/`metalness` are 1.0 so the maps pass through unmodified, and
 *    the option values act as multipliers on top.
 */
import * as THREE from 'three';
import type { QualitySettings, SurfaceKind } from '../core/Contracts';
import { TextureFactory, type SurfaceLook } from './Textures';

export type { SurfaceLook } from './Textures';

export interface MaterialOptions {
  /** Tiles of texture per world metre of UV. Applied in-shader, not on the texture. */
  repeat?: number;
  /** Override the resolution chosen from the quality preset. */
  size?: number;
  /** Varies the noise field; two walls with different seeds never twin. */
  seed?: number;
  /** Multiplied into albedo. Leave undefined for the authored colour. */
  color?: THREE.ColorRepresentation;
  /** Multiplier on the roughness map, 0..1. */
  roughness?: number;
  /** Multiplier on the metalness map, 0..1. */
  metalness?: number;
  /** Multiplier on the normal map strength. */
  normalScale?: number;
  aoIntensity?: number;
  envMapIntensity?: number;
  side?: THREE.Side;
  /**
   * When > 0, sample world-space triplanar at this many tiles per metre instead
   * of using the mesh UVs. Use on large or procedurally-cut geometry.
   */
  triplanar?: number;
  /** Emissive colour, for signage and screens built from these looks. */
  emissive?: THREE.ColorRepresentation;
  emissiveIntensity?: number;
}

/**
 * World-space grading applied on top of the tiling maps. See GRADE_BODY for
 * what each term does and why the texture cannot do it itself.
 */
interface GradeSettings {
  /** Ground-contact darkening / damp / ambient steal on near-vertical faces. */
  contact: number;
  /** Height above the ground over which `contact` falls off, in metres. */
  contactHeight: number;
  /** Sun-bleaching of the upper storeys: lighter, lower in chroma. */
  bleach: number;
  /** Large-scale value break-up, to hide that the map repeats. */
  macro: number;
  /**
   * Multiplier on the *specular* half of the image-based lighting only.
   *
   * `envMapIntensity` scales the diffuse irradiance and the specular lobe
   * together, which is the wrong knob: turning it down far enough to stop a
   * clear sky mirroring off the road also takes away the only fill light the
   * road has in shadow, and the street goes to crushed black. This scales the
   * lobe alone. It stands in for the specular-occlusion and F90 terms a ground
   * surface should have and does not, and it is the single number that decides
   * whether tarmac reads as grey asphalt or as blue water.
   */
  iblSpecular: number;
}

interface LookDefaults {
  /** Base tiles per UV unit when the caller does not say. */
  repeat: number;
  normalScale: number;
  aoIntensity: number;
  envMapIntensity: number;
  /** Minimum texture resolution regardless of preset (viewmodel surfaces). */
  minSize: number;
  /** Omitted for the viewmodel and for glass, which are not part of the world. */
  grade?: GradeSettings;
  physical?: {
    clearcoat?: number;
    clearcoatRoughness?: number;
    sheen?: number;
    sheenRoughness?: number;
    sheenColor?: number;
    ior?: number;
    specularIntensity?: number;
    transparent?: boolean;
    opacity?: number;
    depthWrite?: boolean;
  };
}

/** Walls: strong ground contact, real sun bleaching, strong macro break-up. */
const WALL_GRADE: GradeSettings = { contact: 0.85, contactHeight: 1.5, bleach: 0.50, macro: 0.24, iblSpecular: 0.45 };
/** Props and street furniture: they need grounding, they do not need bleaching. */
const PROP_GRADE: GradeSettings = { contact: 0.60, contactHeight: 0.9, bleach: 0.0, macro: 0.16, iblSpecular: 0.75 };
/**
 * Ground planes. `contact` is gated by how far a face is from horizontal, so a
 * road takes almost none of it and only kerbs, steps and the sides of things
 * pick it up — but the macro term still matters, because a floor is the largest
 * uninterrupted run of one texture in the frame.
 */
const FLOOR_GRADE: GradeSettings = { contact: 0.35, contactHeight: 0.7, bleach: 0.0, macro: 0.22, iblSpecular: 0.34 };

const DEFAULTS: Record<SurfaceLook, LookDefaults> = {
  concrete_wall: { repeat: 1, normalScale: 1.0, aoIntensity: 1.05, envMapIntensity: 0.9, minSize: 512, grade: WALL_GRADE },
  concrete_floor: { repeat: 1, normalScale: 0.9, aoIntensity: 1.05, envMapIntensity: 0.95, minSize: 512, grade: FLOOR_GRADE },
  // A 0.07-albedo horizontal surface under a clear sky is almost all specular
  // response, and a clear sky's lobe is blue — that, not the albedo, is what
  // made the street render navy. The fix is the specular-only knob below, not
  // envMapIntensity: cutting that far enough to stop the mirroring also took
  // away the road's only fill light and left it crushed black in shadow.
  asphalt: {
    repeat: 1, normalScale: 0.95, aoIntensity: 1.1, envMapIntensity: 0.95, minSize: 512,
    grade: { contact: 0.35, contactHeight: 0.7, bleach: 0.0, macro: 0.22, iblSpecular: 0.15 },
  },
  brick: { repeat: 1, normalScale: 1.1, aoIntensity: 1.15, envMapIntensity: 0.85, minSize: 512, grade: WALL_GRADE },
  plaster_painted: { repeat: 1, normalScale: 0.85, aoIntensity: 1.0, envMapIntensity: 0.85, minSize: 512, grade: WALL_GRADE },
  rusted_metal: { repeat: 1, normalScale: 1.0, aoIntensity: 1.0, envMapIntensity: 1.1, minSize: 512, grade: PROP_GRADE },
  painted_metal: {
    repeat: 1, normalScale: 0.85, aoIntensity: 0.9, envMapIntensity: 1.0, minSize: 512, grade: PROP_GRADE,
    physical: { clearcoat: 0.18, clearcoatRoughness: 0.55 },
  },
  corrugated_metal: { repeat: 1, normalScale: 1.0, aoIntensity: 1.0, envMapIntensity: 1.1, minSize: 512, grade: PROP_GRADE },
  wood_plank: { repeat: 1, normalScale: 1.0, aoIntensity: 1.05, envMapIntensity: 0.85, minSize: 512, grade: PROP_GRADE },
  wood_crate: { repeat: 1, normalScale: 1.05, aoIntensity: 1.1, envMapIntensity: 0.8, minSize: 512, grade: PROP_GRADE },
  sand: { repeat: 1, normalScale: 0.9, aoIntensity: 0.85, envMapIntensity: 0.95, minSize: 512, grade: FLOOR_GRADE },
  dirt_gravel: { repeat: 1, normalScale: 1.1, aoIntensity: 1.15, envMapIntensity: 0.9, minSize: 512, grade: FLOOR_GRADE },
  rubble: { repeat: 1, normalScale: 1.2, aoIntensity: 1.25, envMapIntensity: 0.9, minSize: 512, grade: FLOOR_GRADE },
  tile_floor: {
    // The plaza was a half-mirror of the sky. A dusty paved square is not.
    repeat: 1, normalScale: 0.95, aoIntensity: 1.1, envMapIntensity: 0.95, minSize: 512, grade: FLOOR_GRADE,
    physical: { clearcoat: 0.16, clearcoatRoughness: 0.45 },
  },
  glass_dirty: {
    repeat: 1, normalScale: 0.5, aoIntensity: 0.4, envMapIntensity: 1.6, minSize: 512,
    physical: { ior: 1.52, specularIntensity: 1.0, transparent: true, opacity: 1.0, depthWrite: false },
  },
  fabric_canvas: {
    repeat: 1, normalScale: 1.0, aoIntensity: 1.0, envMapIntensity: 0.8, minSize: 512, grade: PROP_GRADE,
    physical: { sheen: 0.55, sheenRoughness: 0.85, sheenColor: 0x9c8f6e },
  },
  // The viewmodel is lit by the same environment as the world, and at 1.35 that
  // environment was the only thing you could see on it: a clear-sky mirror on a
  // near-black metal reads as blue plastic tubing, not as a parkerised receiver.
  gun_metal: { repeat: 1, normalScale: 0.8, aoIntensity: 0.9, envMapIntensity: 0.85, minSize: 1024 },
  gun_polymer: {
    repeat: 1, normalScale: 1.0, aoIntensity: 0.9, envMapIntensity: 0.6, minSize: 1024,
    physical: { clearcoat: 0.06, clearcoatRoughness: 0.72 },
  },
};

/**
 * Looks whose tint is architectural and therefore has to stay inside the town's
 * palette. Signage and the painted-metal accents are deliberately absent: a
 * war-torn town is allowed one or two saturated things, and they should be the
 * ones a level designer chose rather than every facade on the street.
 */
const PALETTE_LOOKS: ReadonlySet<SurfaceLook> = new Set<SurfaceLook>([
  'concrete_wall', 'concrete_floor', 'asphalt', 'brick', 'plaster_painted',
  'tile_floor', 'rubble', 'sand', 'dirt_gravel', 'fabric_canvas',
]);

const _hsl = { h: 0, s: 0, l: 0 };

/**
 * Pulls a requested architectural tint into a narrow warm-neutral band — bone,
 * sand, dust-grey, faded ochre — keeping its value and discarding most of its
 * hue and chroma.
 *
 * The town was mint, salmon, cream and tan, which is three hue families too
 * many. Saturation is capped low enough that the *light* provides the colour of
 * a frame; the difference between one building and the next survives as a
 * difference in value, which is what actually reads at 40 metres anyway.
 */
function gradeTint(color: THREE.ColorRepresentation, out: THREE.Color): THREE.Color {
  out.set(color);
  out.getHSL(_hsl, THREE.SRGBColorSpace);
  const h = _hsl.h > 0.93 ? _hsl.h - 1 : _hsl.h;
  // The warm half of the wheel is clamped into 16deg..43deg. The cool half has
  // no near edge worth preserving, so it folds onto the middle of that band.
  const warm = h < 0.5 ? Math.min(Math.max(h, 0.044), 0.119) : 0.086;
  const hue = (h + (warm - h) * 0.88 + 1) % 1;
  out.setHSL(hue, Math.min(_hsl.s, 0.135) * 0.82, _hsl.l, THREE.SRGBColorSpace);
  return out;
}

const SURFACE_OF: Record<SurfaceLook, SurfaceKind> = {
  concrete_wall: 'concrete',
  concrete_floor: 'concrete',
  asphalt: 'concrete',
  brick: 'concrete',
  plaster_painted: 'plaster',
  rusted_metal: 'metal',
  painted_metal: 'metal',
  corrugated_metal: 'metal',
  wood_plank: 'wood',
  wood_crate: 'wood',
  sand: 'sand',
  dirt_gravel: 'dirt',
  rubble: 'concrete',
  tile_floor: 'concrete',
  glass_dirty: 'glass',
  fabric_canvas: 'fabric',
  gun_metal: 'metal',
  gun_polymer: 'rubber',
};

// ---------------------------------------------------------------------------
// Shader injection
// ---------------------------------------------------------------------------

/**
 * Scales every map's UV by a per-material uniform. This is what lets one shared
 * texture serve materials at different tiling densities — `texture.repeat` is
 * global to the texture object and would fight between materials.
 */
const UV_SCALE_BODY = /* glsl */ `
#ifdef USE_MAP
  vMapUv *= uUvScale;
#endif
#ifdef USE_NORMALMAP
  vNormalMapUv *= uUvScale;
#endif
#ifdef USE_ROUGHNESSMAP
  vRoughnessMapUv *= uUvScale;
#endif
#ifdef USE_METALNESSMAP
  vMetalnessMapUv *= uUvScale;
#endif
#ifdef USE_AOMAP
  vAoMapUv *= uUvScale;
#endif
#ifdef USE_ALPHAMAP
  vAlphaMapUv *= uUvScale;
#endif
#ifdef USE_EMISSIVEMAP
  vEmissiveMapUv *= uUvScale;
#endif
#ifdef USE_DISPLACEMENTMAP
  vDisplacementMapUv *= uUvScale;
#endif
`;

/** World position and world normal, needed by both the triplanar and grade paths. */
const WORLD_PARS = /* glsl */ `
varying vec3 vCodWPos;
varying vec3 vCodWNor;
`;

const WORLD_VERT = /* glsl */ `
  vCodWPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;
  vCodWNor = normalize( mat3( modelMatrix ) * objectNormal );
`;

const MACRO_NOISE = /* glsl */ `
float codMacroHash( vec2 p ) {
  return fract( sin( dot( floor( p ), vec2( 127.1, 311.7 ) ) ) * 43758.5453123 );
}

float codMacroNoise( vec2 p ) {
  vec2 i = floor( p ), f = fract( p );
  f = f * f * ( 3.0 - 2.0 * f );
  return mix(
    mix( codMacroHash( i ), codMacroHash( i + vec2( 1.0, 0.0 ) ), f.x ),
    mix( codMacroHash( i + vec2( 0.0, 1.0 ) ), codMacroHash( i + vec2( 1.0, 1.0 ) ), f.x ),
    f.y
  );
}
`;

const TRI_COMMON = /* glsl */ `
uniform float uTriScale;
uniform float uTriSharp;
uniform float uMacroScale;

vec3 codTriWeights() {
  vec3 w = pow( abs( normalize( vCodWNor ) ), vec3( uTriSharp ) );
  return w / max( w.x + w.y + w.z, 1e-4 );
}

vec4 codTriSample( sampler2D tex, vec3 w ) {
  return texture2D( tex, vCodWPos.zy * uTriScale ) * w.x
       + texture2D( tex, vCodWPos.xz * uTriScale ) * w.y
       + texture2D( tex, vCodWPos.xy * uTriScale ) * w.z;
}
`;

/**
 * World-space grading. Three things a tiling map physically cannot do, because
 * a texture has no idea where it has been placed:
 *
 *  - **Ground contact.** Splash-back dust, rising damp, and the ambient light
 *    the ground steals from the bottom of a wall all key off height above the
 *    floor. Without it every building, barrier and prop meets the ground on a
 *    hard line and reads as pasted onto the frame rather than sitting in it.
 *    The term is gated by how far off horizontal the face is, so roads and
 *    tabletops take almost none of it and only the vertical faces darken.
 *  - **Sun bleaching.** Upper storeys take more UV and more rain-washing than
 *    the street, so they sit lighter and lower in chroma. Done in the map this
 *    became a horizontal band on every tile; done here it is one gradient up
 *    the whole facade.
 *  - **Macro break-up.** A repeat is only invisible when something at a scale
 *    *larger* than the tile modulates it. Everything else in the frame can be
 *    perfect and visible tiling will still give the engine away.
 *
 * The whole thing is one vec4 uniform, two value-noise lookups and about twenty
 * ALU. It adds nothing to the draw call or triangle count.
 */
const GRADE_PARS = /* glsl */ `
uniform vec4 uGrade;   // x contact, y bleach, z macro, w contact height (metres)
uniform float uIblSpecular;
`;

/** Computed once, before <map_fragment>, and consumed by three later hooks. */
const GRADE_SETUP = /* glsl */ `
  float codMacro = codMacroNoise( vCodWPos.xz * 0.147 ) * 0.62
                 + codMacroNoise( vCodWPos.yz * 0.061 + 7.3 ) * 0.38;
  float codUpright = 1.0 - abs( normalize( vCodWNor ).y );
  float codNear = 1.0 - smoothstep( 0.0, uGrade.w, max( vCodWPos.y, 0.0 ) );
  float codContact = codNear * codNear * codUpright * uGrade.x * ( 0.62 + 0.38 * codMacro );
  float codBleach = smoothstep( 2.5, 9.0, vCodWPos.y ) * uGrade.y * ( 0.55 + 0.45 * codMacro );
`;

const GRADE_ALBEDO = /* glsl */ `
  diffuseColor.rgb *= 1.0 + ( codMacro - 0.5 ) * uGrade.z;
  // Damp and splash-back: darker, and pulled towards the colour of the dirt
  // that threw it there rather than towards neutral grey.
  diffuseColor.rgb = mix( diffuseColor.rgb, diffuseColor.rgb * vec3( 0.40, 0.385, 0.352 ), codContact );
  float codLum = dot( diffuseColor.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
  diffuseColor.rgb = mix( diffuseColor.rgb, mix( vec3( codLum ), diffuseColor.rgb, 0.70 ) * 1.12, codBleach );
`;

/** Appended after <aomap_fragment>: this is the part that actually grounds things. */
const GRADE_AO = /* glsl */ `
  {
    float codAmbient = 1.0 - codContact * 0.55;
    reflectedLight.indirectDiffuse *= codAmbient;
    reflectedLight.indirectSpecular *= codAmbient * uIblSpecular;
    // The clearcoat lobe is a second, separate reflection of the same sky, and
    // it is not covered by envMapIntensity's effect on the base layer. Left
    // alone it puts the whole sky back onto a paved floor after the base layer
    // has been told not to mirror it.
    #if defined( USE_CLEARCOAT )
      clearcoatSpecularIndirect *= codAmbient * uIblSpecular;
    #endif
    #if defined( USE_SHEEN )
      sheenSpecularIndirect *= codAmbient;
    #endif
  }
`;

interface Injection {
  /** UV repeat for the mesh-UV path. 1 means no UV injection is needed. */
  repeat: number;
  /** Triplanar tiles per metre, or 0 to use mesh UVs. */
  tri: number;
  /** World-space grading, or undefined for surfaces that are not in the world. */
  grade?: GradeSettings;
}

/**
 * One `onBeforeCompile` for all three injections. They have to share the hook —
 * a material only gets one — and they overlap anyway: triplanar and the world
 * grade both want the world position varying, and both want to be the last
 * thing that touches albedo and roughness.
 */
function applyInjection(mat: THREE.MeshStandardMaterial, inj: Injection): void {
  const { repeat, tri, grade } = inj;
  const needsWorld = tri > 0 || grade !== undefined;
  if (repeat === 1 && !needsWorld) return;

  const packed =
    tri > 0 &&
    mat.roughnessMap !== null &&
    mat.roughnessMap === mat.metalnessMap &&
    mat.roughnessMap === mat.aoMap;

  const uUvScale = { value: new THREE.Vector2(repeat, repeat) };
  const uTriScale = { value: tri };
  const uTriSharp = { value: 5.0 };
  const uMacroScale = { value: tri / 11 };
  const uGrade = {
    value: new THREE.Vector4(
      grade ? grade.contact : 0,
      grade ? grade.bleach : 0,
      grade ? grade.macro : 0,
      grade ? grade.contactHeight : 1,
    ),
  };
  const uIblSpecular = { value: grade ? grade.iblSpecular : 1 };

  const extraSamples = packed
    ? ''
    : /* glsl */ `
  #ifdef USE_AOMAP
    triAO = codTriSample( aoMap, triW );
  #endif
  #ifdef USE_METALNESSMAP
    triMet = codTriSample( metalnessMap, triW );
  #endif
`;

  mat.onBeforeCompile = (shader) => {
    if (repeat !== 1) shader.uniforms.uUvScale = uUvScale;
    if (grade) {
      shader.uniforms.uGrade = uGrade;
      shader.uniforms.uIblSpecular = uIblSpecular;
    }
    if (tri > 0) {
      shader.uniforms.uTriScale = uTriScale;
      shader.uniforms.uTriSharp = uTriSharp;
      shader.uniforms.uMacroScale = uMacroScale;
    }

    // ----------------------------------------------------------------- vertex
    let vert = shader.vertexShader;
    if (needsWorld) {
      vert = vert
        .replace('#include <common>', `#include <common>\n${WORLD_PARS}`)
        .replace('#include <project_vertex>', `#include <project_vertex>\n${WORLD_VERT}`);
    }
    if (repeat !== 1) {
      vert = vert
        .replace('#include <uv_pars_vertex>', '#include <uv_pars_vertex>\nuniform vec2 uUvScale;')
        .replace('#include <uv_vertex>', `#include <uv_vertex>\n${UV_SCALE_BODY}`);
    }
    shader.vertexShader = vert;

    // --------------------------------------------------------------- fragment
    let frag = shader.fragmentShader;
    const pars = [
      needsWorld ? WORLD_PARS : '',
      needsWorld ? MACRO_NOISE : '',
      tri > 0 ? TRI_COMMON : '',
      grade ? GRADE_PARS : '',
    ].join('\n');
    frag = frag.replace('#include <common>', `#include <common>\n${pars}`);

    // Setup goes at the logdepth hook, the last thing before <map_fragment>;
    // everything downstream reads what it leaves behind.
    const setup = [
      tri > 0
        ? /* glsl */ `
  vec3 triW = codTriWeights();
  vec4 triORM = vec4( 1.0 );
  #ifdef USE_ROUGHNESSMAP
    triORM = codTriSample( roughnessMap, triW );
  #endif
  vec4 triAO = triORM;
  vec4 triMet = triORM;${extraSamples}
  float codTriMacro = codMacroNoise( vCodWPos.xz * uMacroScale ) * 0.6
                    + codMacroNoise( vCodWPos.yz * uMacroScale * 1.73 + 11.3 ) * 0.4;`
        : '',
      grade ? GRADE_SETUP : '',
    ].join('\n');
    frag = frag.replace('#include <logdepthbuf_fragment>', `#include <logdepthbuf_fragment>\n${setup}`);

    const mapBody =
      tri > 0
        ? /* glsl */ `#ifdef USE_MAP
    vec4 triAlbedo = codTriSample( map, triW );
    triAlbedo.rgb *= 0.84 + 0.32 * codTriMacro;
    diffuseColor *= triAlbedo;
  #endif`
        : '#include <map_fragment>';
    frag = frag.replace('#include <map_fragment>', `${mapBody}\n${grade ? GRADE_ALBEDO : ''}`);

    const roughBody =
      tri > 0
        ? /* glsl */ `float roughnessFactor = roughness;
  #ifdef USE_ROUGHNESSMAP
    roughnessFactor *= triORM.g;
  #endif
  roughnessFactor = clamp( roughnessFactor * ( 1.06 - 0.12 * codTriMacro ), 0.025, 1.0 );`
        : '#include <roughnessmap_fragment>';
    // Damp masonry is rougher than dry masonry; bleached render is very
    // slightly less so, the loose surface having washed off it years ago.
    const roughGrade = grade
      ? '\n  roughnessFactor = clamp( roughnessFactor + codContact * 0.14 - codBleach * 0.03, 0.025, 1.0 );'
      : '';
    frag = frag.replace('#include <roughnessmap_fragment>', `${roughBody}${roughGrade}`);

    if (tri > 0) {
      frag = frag
        .replace(
          '#include <metalnessmap_fragment>',
          `float metalnessFactor = metalness;
  #ifdef USE_METALNESSMAP
    metalnessFactor *= triMet.b;
  #endif`,
        )
        .replace(
          '#include <normal_fragment_maps>',
          `#ifdef USE_NORMALMAP
  {
    vec3 wn = normalize( vCodWNor );
    vec3 nx = texture2D( normalMap, vCodWPos.zy * uTriScale ).xyz * 2.0 - 1.0;
    vec3 ny = texture2D( normalMap, vCodWPos.xz * uTriScale ).xyz * 2.0 - 1.0;
    vec3 nz = texture2D( normalMap, vCodWPos.xy * uTriScale ).xyz * 2.0 - 1.0;
    nx.xy *= normalScale; ny.xy *= normalScale; nz.xy *= normalScale;
    // Whiteout blend: keeps detail from every plane instead of letting the
    // dominant axis flatten the other two.
    vec3 tx = vec3( nx.xy + wn.zy, abs( nx.z ) * wn.x );
    vec3 ty = vec3( ny.xy + wn.xz, abs( ny.z ) * wn.y );
    vec3 tz = vec3( nz.xy + wn.xy, abs( nz.z ) * wn.z );
    vec3 worldN = normalize( tx.zyx * triW.x + ty.xzy * triW.y + tz.xyz * triW.z );
    normal = normalize( ( viewMatrix * vec4( worldN, 0.0 ) ).xyz );
    #ifdef DOUBLE_SIDED
      normal *= faceDirection;
    #endif
  }
  #endif`,
        );
    }

    const aoBody =
      tri > 0
        ? /* glsl */ `#ifdef USE_AOMAP
    float ambientOcclusion = ( triAO.r - 1.0 ) * aoMapIntensity + 1.0;
    reflectedLight.indirectDiffuse *= ambientOcclusion;
    #if defined( USE_CLEARCOAT )
      clearcoatSpecularIndirect *= ambientOcclusion;
    #endif
    #if defined( USE_SHEEN )
      sheenSpecularIndirect *= ambientOcclusion;
    #endif
    #if defined( USE_ENVMAP ) && defined( STANDARD )
      float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
      reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
    #endif
  #endif`
        : '#include <aomap_fragment>';
    frag = frag.replace('#include <aomap_fragment>', `${aoBody}\n${grade ? GRADE_AO : ''}`);

    shader.fragmentShader = frag;
  };

  const key = `cod|${repeat !== 1 ? 'u' : ''}${tri > 0 ? (packed ? 'T' : 't') : ''}${grade ? 'g' : ''}`;
  mat.customProgramCacheKey = () => key;
  if (repeat !== 1) mat.userData.uvScale = uUvScale;
  if (tri > 0) mat.userData.triplanarScale = tri;
  mat.needsUpdate = true;
}

/**
 * World-space triplanar sampling with a whiteout normal blend and a large-scale
 * brightness/roughness variation on top. Without the macro term a triplanar
 * wall still reads as one texture repeated; with it, it reads as a wall.
 */
export function triplanarMaterial(
  base: THREE.MeshStandardMaterial,
  scale: number,
  grade?: GradeSettings,
): THREE.MeshStandardMaterial {
  const mat = base.clone();
  applyInjection(mat, { repeat: 1, tri: scale, grade });
  return mat;
}

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

const _tmpColor = new THREE.Color();
const _gradedColor = new THREE.Color();

export class MaterialLibrary {
  readonly textures: TextureFactory;

  private readonly quality: QualitySettings;
  private readonly cache = new Map<string, THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial>();
  private disposed = false;

  constructor(renderer: THREE.WebGLRenderer, quality: QualitySettings) {
    this.quality = quality;
    this.textures = new TextureFactory(renderer, quality.anisotropy);
  }

  surfaceOf(look: SurfaceLook): SurfaceKind {
    return SURFACE_OF[look];
  }

  get(look: SurfaceLook, opts: MaterialOptions = {}): THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial {
    if (this.disposed) throw new Error('MaterialLibrary: used after dispose()');
    const d = DEFAULTS[look];
    const repeat = opts.repeat ?? d.repeat;
    const seed = opts.seed ?? 0;
    const size = Math.max(d.minSize, opts.size ?? TextureFactory.sizeForPreset(this.quality.preset));
    const tri = opts.triplanar ?? 0;

    const key = [
      look, repeat, seed, size, tri,
      opts.color !== undefined ? _tmpColor.set(opts.color).getHexString() : '-',
      opts.roughness ?? '-', opts.metalness ?? '-', opts.normalScale ?? '-',
      opts.aoIntensity ?? '-', opts.envMapIntensity ?? '-', opts.side ?? '-',
      opts.emissive !== undefined ? _tmpColor.set(opts.emissive).getHexString() : '-',
      opts.emissiveIntensity ?? '-',
    ].join('|');

    const hit = this.cache.get(key);
    if (hit) return hit;

    const mat = this.build(look, opts, repeat, seed, size, tri);
    mat.name = `${look}@${repeat}`;
    this.cache.set(key, mat);
    return mat;
  }

  private build(
    look: SurfaceLook,
    opts: MaterialOptions,
    repeat: number,
    seed: number,
    size: number,
    tri: number,
  ): THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial {
    const d = DEFAULTS[look];
    const set = this.textures.get(look, { size, seed, repeat: 1 });

    const params: THREE.MeshPhysicalMaterialParameters = {
      map: set.map,
      normalMap: set.normalMap,
      roughnessMap: set.roughnessMap,
      metalnessMap: set.metalnessMap ?? null,
      aoMap: set.aoMap ?? null,
      // The maps are authoritative; these scalars are pure multipliers.
      color: opts.color === undefined
        ? 0xffffff
        : PALETTE_LOOKS.has(look) ? gradeTint(opts.color, _gradedColor) : opts.color,
      roughness: opts.roughness ?? 1,
      metalness: opts.metalness ?? 1,
      aoMapIntensity: opts.aoIntensity ?? d.aoIntensity,
      envMapIntensity: opts.envMapIntensity ?? d.envMapIntensity,
      side: opts.side ?? THREE.FrontSide,
      dithering: true,
    };

    const ns = (opts.normalScale ?? 1) * d.normalScale;

    let mat: THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial;
    if (d.physical) {
      const p = d.physical;
      const phys = new THREE.MeshPhysicalMaterial(params);
      if (p.clearcoat !== undefined) phys.clearcoat = p.clearcoat;
      if (p.clearcoatRoughness !== undefined) phys.clearcoatRoughness = p.clearcoatRoughness;
      if (p.sheen !== undefined) phys.sheen = p.sheen;
      if (p.sheenRoughness !== undefined) phys.sheenRoughness = p.sheenRoughness;
      if (p.sheenColor !== undefined) phys.sheenColor.setHex(p.sheenColor);
      if (p.ior !== undefined) phys.ior = p.ior;
      if (p.specularIntensity !== undefined) phys.specularIntensity = p.specularIntensity;
      if (p.transparent !== undefined) phys.transparent = p.transparent;
      if (p.opacity !== undefined) phys.opacity = p.opacity;
      if (p.depthWrite !== undefined) phys.depthWrite = p.depthWrite;
      mat = phys;
    } else {
      mat = new THREE.MeshStandardMaterial(params);
    }

    mat.normalScale.set(ns, ns);
    if (opts.emissive !== undefined) {
      mat.emissive.set(opts.emissive);
      mat.emissiveIntensity = opts.emissiveIntensity ?? 1;
    }

    if (tri > 0) {
      const t = triplanarMaterial(mat, tri, d.grade);
      mat.dispose();
      return t;
    }

    applyInjection(mat, { repeat, tri: 0, grade: d.grade });
    return mat;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const mat of this.cache.values()) mat.dispose();
    this.cache.clear();
    this.textures.dispose();
  }
}
