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

interface LookDefaults {
  /** Base tiles per UV unit when the caller does not say. */
  repeat: number;
  normalScale: number;
  aoIntensity: number;
  envMapIntensity: number;
  /** Minimum texture resolution regardless of preset (viewmodel surfaces). */
  minSize: number;
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

const DEFAULTS: Record<SurfaceLook, LookDefaults> = {
  concrete_wall: { repeat: 1, normalScale: 1.0, aoIntensity: 1.0, envMapIntensity: 1.0, minSize: 512 },
  concrete_floor: { repeat: 1, normalScale: 0.9, aoIntensity: 1.0, envMapIntensity: 1.0, minSize: 512 },
  // Asphalt takes a reduced env weight: at full strength the sky's specular
  // lobe dominates a 0.1-albedo horizontal surface and the street renders navy.
  asphalt: { repeat: 1, normalScale: 0.85, aoIntensity: 1.0, envMapIntensity: 0.55, minSize: 512 },
  brick: { repeat: 1, normalScale: 1.1, aoIntensity: 1.15, envMapIntensity: 0.9, minSize: 512 },
  plaster_painted: { repeat: 1, normalScale: 0.8, aoIntensity: 0.9, envMapIntensity: 1.0, minSize: 512 },
  rusted_metal: { repeat: 1, normalScale: 1.0, aoIntensity: 1.0, envMapIntensity: 1.2, minSize: 512 },
  painted_metal: {
    repeat: 1, normalScale: 0.85, aoIntensity: 0.9, envMapIntensity: 1.1, minSize: 512,
    physical: { clearcoat: 0.18, clearcoatRoughness: 0.55 },
  },
  corrugated_metal: { repeat: 1, normalScale: 1.0, aoIntensity: 1.0, envMapIntensity: 1.25, minSize: 512 },
  wood_plank: { repeat: 1, normalScale: 1.0, aoIntensity: 1.05, envMapIntensity: 0.9, minSize: 512 },
  wood_crate: { repeat: 1, normalScale: 1.05, aoIntensity: 1.1, envMapIntensity: 0.85, minSize: 512 },
  sand: { repeat: 1, normalScale: 0.9, aoIntensity: 0.85, envMapIntensity: 1.0, minSize: 512 },
  dirt_gravel: { repeat: 1, normalScale: 1.1, aoIntensity: 1.15, envMapIntensity: 0.9, minSize: 512 },
  rubble: { repeat: 1, normalScale: 1.2, aoIntensity: 1.25, envMapIntensity: 0.9, minSize: 512 },
  tile_floor: {
    repeat: 1, normalScale: 0.95, aoIntensity: 1.1, envMapIntensity: 1.15, minSize: 512,
    physical: { clearcoat: 0.5, clearcoatRoughness: 0.14 },
  },
  glass_dirty: {
    repeat: 1, normalScale: 0.5, aoIntensity: 0.4, envMapIntensity: 1.6, minSize: 512,
    physical: { ior: 1.52, specularIntensity: 1.0, transparent: true, opacity: 1.0, depthWrite: false },
  },
  fabric_canvas: {
    repeat: 1, normalScale: 1.0, aoIntensity: 1.0, envMapIntensity: 0.8, minSize: 512,
    physical: { sheen: 0.55, sheenRoughness: 0.85, sheenColor: 0x9c8f6e },
  },
  gun_metal: { repeat: 1, normalScale: 0.9, aoIntensity: 0.8, envMapIntensity: 1.35, minSize: 1024 },
  gun_polymer: {
    repeat: 1, normalScale: 1.0, aoIntensity: 0.85, envMapIntensity: 1.0, minSize: 1024,
    physical: { clearcoat: 0.24, clearcoatRoughness: 0.52 },
  },
};

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

const TRI_COMMON = /* glsl */ `
varying vec3 vTriPos;
varying vec3 vTriNor;
uniform float uTriScale;
uniform float uTriSharp;
uniform float uMacroScale;

vec3 codTriWeights() {
  vec3 w = pow( abs( normalize( vTriNor ) ), vec3( uTriSharp ) );
  return w / max( w.x + w.y + w.z, 1e-4 );
}

vec4 codTriSample( sampler2D tex, vec3 w ) {
  return texture2D( tex, vTriPos.zy * uTriScale ) * w.x
       + texture2D( tex, vTriPos.xz * uTriScale ) * w.y
       + texture2D( tex, vTriPos.xy * uTriScale ) * w.z;
}

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

/**
 * World-space triplanar sampling with a whiteout normal blend and a large-scale
 * brightness/roughness variation on top. Without the macro term a triplanar
 * wall still reads as one texture repeated; with it, it reads as a wall.
 */
export function triplanarMaterial(
  base: THREE.MeshStandardMaterial,
  scale: number,
): THREE.MeshStandardMaterial {
  const mat = base.clone();
  const packed =
    mat.roughnessMap !== null &&
    mat.roughnessMap === mat.metalnessMap &&
    mat.roughnessMap === mat.aoMap;

  const uTriScale = { value: scale };
  const uTriSharp = { value: 5.0 };
  const uMacroScale = { value: scale / 11 };

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
    shader.uniforms.uTriScale = uTriScale;
    shader.uniforms.uTriSharp = uTriSharp;
    shader.uniforms.uMacroScale = uMacroScale;

    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\nvarying vec3 vTriPos;\nvarying vec3 vTriNor;`)
      .replace(
        '#include <project_vertex>',
        `#include <project_vertex>
  vTriPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;
  vTriNor = normalize( mat3( modelMatrix ) * objectNormal );`,
      );

    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${TRI_COMMON}`)
      .replace(
        '#include <logdepthbuf_fragment>',
        `#include <logdepthbuf_fragment>
  vec3 triW = codTriWeights();
  vec4 triORM = vec4( 1.0 );
  #ifdef USE_ROUGHNESSMAP
    triORM = codTriSample( roughnessMap, triW );
  #endif
  vec4 triAO = triORM;
  vec4 triMet = triORM;${extraSamples}
  float codMacro = codMacroNoise( vTriPos.xz * uMacroScale ) * 0.6
                 + codMacroNoise( vTriPos.yz * uMacroScale * 1.73 + 11.3 ) * 0.4;`,
      )
      .replace(
        '#include <map_fragment>',
        `#ifdef USE_MAP
    vec4 triAlbedo = codTriSample( map, triW );
    triAlbedo.rgb *= 0.84 + 0.32 * codMacro;
    diffuseColor *= triAlbedo;
  #endif`,
      )
      .replace(
        '#include <roughnessmap_fragment>',
        `float roughnessFactor = roughness;
  #ifdef USE_ROUGHNESSMAP
    roughnessFactor *= triORM.g;
  #endif
  roughnessFactor = clamp( roughnessFactor * ( 1.06 - 0.12 * codMacro ), 0.025, 1.0 );`,
      )
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
    vec3 wn = normalize( vTriNor );
    vec3 nx = texture2D( normalMap, vTriPos.zy * uTriScale ).xyz * 2.0 - 1.0;
    vec3 ny = texture2D( normalMap, vTriPos.xz * uTriScale ).xyz * 2.0 - 1.0;
    vec3 nz = texture2D( normalMap, vTriPos.xy * uTriScale ).xyz * 2.0 - 1.0;
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
      )
      .replace(
        '#include <aomap_fragment>',
        `#ifdef USE_AOMAP
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
  #endif`,
      );
  };

  const cacheKey = packed ? 'cod-triplanar-packed' : 'cod-triplanar';
  mat.customProgramCacheKey = () => cacheKey;
  mat.userData.triplanarScale = scale;
  mat.needsUpdate = true;
  return mat;
}

function injectUvScale(mat: THREE.MeshStandardMaterial, repeat: number): void {
  const uUvScale = { value: new THREE.Vector2(repeat, repeat) };
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uUvScale = uUvScale;
    shader.vertexShader = shader.vertexShader
      .replace('#include <uv_pars_vertex>', '#include <uv_pars_vertex>\nuniform vec2 uUvScale;')
      .replace('#include <uv_vertex>', `#include <uv_vertex>\n${UV_SCALE_BODY}`);
  };
  mat.customProgramCacheKey = () => 'cod-uvscale';
  mat.userData.uvScale = uUvScale;
}

// ---------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------

const _tmpColor = new THREE.Color();

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
      color: opts.color ?? 0xffffff,
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
      const t = triplanarMaterial(mat, tri);
      mat.dispose();
      return t;
    }

    // repeat === 1 needs no injection at all — keep those on the stock program.
    if (repeat !== 1) injectUvScale(mat, repeat);
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
