/**
 * Shared GLSL for the render pipeline, sky and lighting.
 *
 * Everything here is written against three's ShaderMaterial preamble: the code
 * is authored in GLSL1 style (varying, texture2D, gl_FragColor) which
 * three transparently upgrades to GLSL ES 3.00 on WebGL2, so ES3-only builtins
 * (textureLod, texelFetch, textureSize, texture(sampler2DShadow, vec3))
 * are also available and used where they matter.
 */

// ---------------------------------------------------------------------------
// Vertex shaders
// ---------------------------------------------------------------------------

export const FULLSCREEN_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`;

// ---------------------------------------------------------------------------
// Common helpers shared by every post fragment shader
// ---------------------------------------------------------------------------

export const POST_COMMON = /* glsl */ `
#define PST_PI 3.141592653589793

float sat( float x ) { return clamp( x, 0.0, 1.0 ); }
vec2 sat2( vec2 x ) { return clamp( x, vec2( 0.0 ), vec2( 1.0 ) ); }
vec3 sat3( vec3 x ) { return clamp( x, vec3( 0.0 ), vec3( 1.0 ) ); }
float lumaOf( vec3 c ) { return dot( c, vec3( 0.2126, 0.7152, 0.0722 ) ); }

float linearDepth( float d, float n, float f ) {
  float z = d * 2.0 - 1.0;
  return ( 2.0 * n * f ) / ( f + n - z * ( f - n ) );
}

vec3 viewFromDepth( vec2 uv, float d, mat4 invProj ) {
  vec4 c = vec4( uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
  vec4 v = invProj * c;
  return v.xyz / v.w;
}

// Interleaved gradient noise — the cheapest good per-pixel dither there is.
float ignNoise( vec2 p ) {
  return fract( 52.9829189 * fract( dot( p, vec2( 0.06711056, 0.00583715 ) ) ) );
}

float hash12( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx ) * 0.1031 );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.x + p3.y ) * p3.z );
}

vec2 hash22( vec2 p ) {
  vec3 p3 = fract( vec3( p.xyx ) * vec3( 0.1031, 0.1030, 0.0973 ) );
  p3 += dot( p3, p3.yzx + 33.33 );
  return fract( ( p3.xx + p3.yz ) * p3.zy );
}

vec3 rgb2ycocg( vec3 c ) {
  return vec3(
    0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
    0.5 * c.r - 0.5 * c.b,
    -0.25 * c.r + 0.5 * c.g - 0.25 * c.b
  );
}

vec3 ycocg2rgb( vec3 c ) {
  return vec3( c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z );
}

float henyeyGreenstein( float cosT, float g ) {
  float g2 = g * g;
  float d = 1.0 + g2 - 2.0 * g * cosT;
  return ( 1.0 - g2 ) / ( 4.0 * PST_PI * max( 1e-4, d * sqrt( max( 1e-4, d ) ) ) );
}
`;

// ---------------------------------------------------------------------------
// The analytic sky, shared by *everything* that needs to know what colour the
// air is in a given direction
// ---------------------------------------------------------------------------

/**
 * One Preetham evaluation, one set of constants, one call site shape.
 *
 * The dome, the aerial-perspective term in the composite and the SSR fallback
 * all call skyDomeRadiance with the same parameters, so it is structurally
 * impossible for a distant facade to haze toward a colour the sky above it is
 * not. Previously the composite used a three-colour lerp fitted on the CPU: it
 * carried the *horizon* colour (sampled across from the sun, and therefore warm
 * from the long slant path) into every near-horizontal view ray, including the
 * ones looking at a cool blue quarter of the sky. That is exactly the seam the
 * far plaza wall was sitting on — cream haze under a blue sky.
 *
 * Nothing here is scaled by intensity or rolled; callers do that via
 * skyDomeRadiance so the roll happens once, at the end, on the whole value.
 */
export const ATMOSPHERE_GLSL = /* glsl */ `
const vec3 ATM_UP = vec3( 0.0, 1.0, 0.0 );
const vec3 ATM_TOTAL_RAYLEIGH = vec3( 5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5 );
const vec3 ATM_MIE_CONST = vec3( 1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14 );

float atmRayleighPhase( float c ) { return ( 3.0 / ( 16.0 * PST_PI ) ) * ( 1.0 + c * c ); }

float atmMiePhase( float c, float g ) {
  float g2 = g * g;
  float inv = 1.0 / pow( max( 1e-4, 1.0 - 2.0 * g * c + g2 ), 1.5 );
  return ( 1.0 / ( 4.0 * PST_PI ) ) * ( ( 1.0 - g2 ) * inv );
}

float atmSunE( vec3 sunDir ) {
  return 1000.0 * max( 0.0, 1.0 - exp( -( ( 1.5707963 - acos( clamp( sunDir.y, -1.0, 1.0 ) ) ) / 1.5 ) ) );
}

/** Extinction along the sun ray from this direction's air mass. */
vec3 atmExtinction( float zenithAngle, float turbidity, float rayleigh, float mieCoefficient ) {
  vec3 betaR = ATM_TOTAL_RAYLEIGH * rayleigh;
  float c = ( 0.2 * turbidity ) * 10.0e-18;
  vec3 betaM = 0.434 * c * ATM_MIE_CONST * mieCoefficient;
  float denom = cos( zenithAngle ) + 0.15 * pow( max( 1e-3, 93.885 - ( zenithAngle * 180.0 / PST_PI ) ), -1.253 );
  float inverse = 1.0 / max( 1e-4, denom );
  return exp( -( betaR * 8.4e3 * inverse + betaM * 1.25e3 * inverse ) );
}

/** Diffuse sky radiance in dir. No sun disc, no intensity scale, no roll. */
vec3 atmosphereRadiance( vec3 dir, vec3 sunDir, float turbidity, float rayleigh, float mieCoefficient, float mieG ) {
  float sunE = atmSunE( sunDir );

  vec3 betaR = ATM_TOTAL_RAYLEIGH * rayleigh;
  float c = ( 0.2 * turbidity ) * 10.0e-18;
  vec3 betaM = 0.434 * c * ATM_MIE_CONST * mieCoefficient;

  float zenithAngle = acos( max( 0.0, dot( ATM_UP, dir ) ) );
  vec3 Fex = atmExtinction( zenithAngle, turbidity, rayleigh, mieCoefficient );

  float cosTheta = dot( dir, sunDir );
  vec3 betaRTheta = betaR * atmRayleighPhase( cosTheta * 0.5 + 0.5 );
  vec3 betaMTheta = betaM * atmMiePhase( cosTheta, mieG );

  vec3 base = ( betaRTheta + betaMTheta ) / ( betaR + betaM );
  vec3 Lin = pow( sunE * base * ( 1.0 - Fex ), vec3( 1.5 ) );
  Lin *= mix(
    vec3( 1.0 ),
    pow( max( vec3( 0.0 ), sunE * base * Fex ), vec3( 0.5 ) ),
    sat( pow( 1.0 - dot( ATM_UP, sunDir ), 5.0 ) )
  );

  return ( Lin + 0.1 * Fex ) * 0.04 + vec3( 0.0, 0.00035, 0.00085 );
}

/**
 * Hyperbolic roll for sky radiance.
 *
 * The old code did min( sky, 4.0 ). A hard min on the Mie aureole is what
 * produced a 300-pixel plateau of *identical* cream pixels around the sun: not
 * a sun, a hole cut in the sky. This leaves everything under knee untouched
 * and asymptotes to maxV above it, so the aureole always has a gradient in
 * it and the only thing that can reach the top of the range is the disc.
 */
vec3 skyRoll( vec3 x, float knee, float maxV ) {
  float range = max( 1e-3, maxV - knee );
  vec3 over = max( vec3( 0.0 ), x - knee );
  vec3 rolled = knee + range * ( over / ( over + range ) );
  return mix( x, rolled, step( vec3( knee ), x ) );
}

/**
 * The dome as the frame sees it, minus clouds: atmosphere, ground hemisphere,
 * horizon haze band, rolled. This is the function the aerial perspective fades
 * into, so by construction it cannot disagree with the background.
 */
vec3 skyDomeRadianceH(
  vec3 dir, vec3 sunDir, float turbidity, float rayleigh, float mieCoefficient, float mieG,
  float intensity, vec3 groundColor, float rollKnee, float rollMax, out vec3 horizonCol
) {
  vec3 sky = atmosphereRadiance( dir, sunDir, turbidity, rayleigh, mieCoefficient, mieG ) * intensity;

  vec3 horizonDir = normalize( vec3( dir.x, 0.02, dir.z ) );
  horizonCol = atmosphereRadiance( horizonDir, sunDir, turbidity, rayleigh, mieCoefficient, mieG ) * intensity;

  float below = smoothstep( 0.02, -0.10, dir.y );
  vec3 ground = groundColor * ( 0.35 + 0.65 * sat( sunDir.y ) ) + horizonCol * 0.35;
  sky = mix( sky, ground, below );

  // Real air never lets the horizon meet the ground clean. Rolled first so the
  // band cannot smear the aureole across the bottom third of a backlit sky.
  float hazeBand = exp( -abs( dir.y ) * 9.0 );
  vec3 haze = skyRoll( horizonCol, rollKnee, rollMax ) * 1.04 + vec3( 0.012, 0.013, 0.015 ) * intensity;
  sky = mix( sky, mix( sky, haze, 0.5 ) * vec3( 1.04, 1.01, 0.975 ), hazeBand * 0.5 );

  return skyRoll( max( vec3( 0.0 ), sky ), rollKnee, rollMax );
}

vec3 skyDomeRadiance(
  vec3 dir, vec3 sunDir, float turbidity, float rayleigh, float mieCoefficient, float mieG,
  float intensity, vec3 groundColor, float rollKnee, float rollMax
) {
  vec3 ignored;
  return skyDomeRadianceH(
    dir, sunDir, turbidity, rayleigh, mieCoefficient, mieG, intensity, groundColor,
    rollKnee, rollMax, ignored
  );
}
`;

/** Uniform block every caller of skyDomeRadiance declares, verbatim. */
export const SKY_UNIFORMS_GLSL = /* glsl */ `
uniform vec3 uSunDir;
uniform float uTurbidity;
uniform float uRayleigh;
uniform float uMieCoefficient;
uniform float uMieG;
uniform float uSkyIntensity;
uniform vec3 uGroundColor;
uniform float uSkyRollKnee;
uniform float uSkyRollMax;
`;

/** The call, with the uniforms above already bound. */
export const SKY_CALL_GLSL = /* glsl */ `
vec3 skyInDirection( vec3 d ) {
  return skyDomeRadiance(
    d, uSunDir, uTurbidity, uRayleigh, uMieCoefficient, uMieG,
    uSkyIntensity, uGroundColor, uSkyRollKnee, uSkyRollMax
  );
}
`;

// ---------------------------------------------------------------------------
// Value-noise fBm, used by the sky's cloud layers
// ---------------------------------------------------------------------------

export const NOISE_GLSL = /* glsl */ `
float vhash( vec2 p ) {
  p = fract( p * vec2( 233.34, 851.73 ) );
  p += dot( p, p + 23.45 );
  return fract( p.x * p.y );
}

float vnoise( vec2 p ) {
  vec2 i = floor( p );
  vec2 f = fract( p );
  f = f * f * ( 3.0 - 2.0 * f );
  float a = vhash( i );
  float b = vhash( i + vec2( 1.0, 0.0 ) );
  float c = vhash( i + vec2( 0.0, 1.0 ) );
  float d = vhash( i + vec2( 1.0, 1.0 ) );
  return mix( mix( a, b, f.x ), mix( c, d, f.x ), f.y );
}

float fbm5( vec2 p ) {
  float s = 0.0;
  float a = 0.5;
  float n = 0.0;
  for ( int i = 0; i < 5; i ++ ) {
    s += a * vnoise( p );
    n += a;
    // Irrational rotation + scale per octave so octaves never line up.
    p = mat2( 1.62, 1.18, -1.18, 1.62 ) * p + vec2( 3.71, 7.13 );
    a *= 0.5;
  }
  return s / n;
}

float fbm3( vec2 p ) {
  float s = 0.0;
  float a = 0.5;
  float n = 0.0;
  for ( int i = 0; i < 3; i ++ ) {
    s += a * vnoise( p );
    n += a;
    p = mat2( 1.62, 1.18, -1.18, 1.62 ) * p + vec2( 3.71, 7.13 );
    a *= 0.5;
  }
  return s / n;
}
`;

// ---------------------------------------------------------------------------
// Velocity
// ---------------------------------------------------------------------------

/** Reprojects the depth buffer through the previous frame's camera. */
export const CAMERA_VELOCITY_FRAG = /* glsl */ `
varying vec2 vUv;
uniform highp sampler2D tDepth;
uniform mat4 uInvViewProj;
uniform mat4 uPrevViewProj;

void main() {
  float d = texture2D( tDepth, vUv ).x;
  vec4 clip = vec4( vUv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0 );
  vec4 world = uInvViewProj * clip;
  world /= world.w;
  vec4 prev = uPrevViewProj * vec4( world.xyz, 1.0 );
  vec2 prevUv = ( prev.xy / prev.w ) * 0.5 + 0.5;
  gl_FragColor = vec4( vUv - prevUv, 0.0, 1.0 );
}
`;

/**
 * Object velocity. Rasterised with the jittered matrices so fragments land on
 * the same texels as the main pass, but the reported motion uses unjittered
 * matrices so TAA does not chase its own jitter.
 */
export const OBJECT_VELOCITY_VERT = /* glsl */ `
#include <common>
#include <skinning_pars_vertex>

uniform mat4 uPrevModelMatrix;
uniform mat4 uPrevViewProj;
uniform mat4 uCurrViewProj;

varying vec4 vCurrClip;
varying vec4 vPrevClip;

void main() {
  #include <skinbase_vertex>
  vec3 transformed = vec3( position );
  #include <skinning_vertex>

  vec4 wp = modelMatrix * vec4( transformed, 1.0 );
  vCurrClip = uCurrViewProj * wp;
  vPrevClip = uPrevViewProj * ( uPrevModelMatrix * vec4( transformed, 1.0 ) );

  gl_Position = projectionMatrix * modelViewMatrix * vec4( transformed, 1.0 );
}
`;

export const OBJECT_VELOCITY_FRAG = /* glsl */ `
varying vec4 vCurrClip;
varying vec4 vPrevClip;

#ifndef ZERO_VELOCITY
uniform highp sampler2D tSceneDepth;
uniform vec2 uInvRes;
#endif

void main() {
  #ifdef ZERO_VELOCITY
    gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
  #else
    // Manual depth test against the main pass: the velocity target owns no
    // depth buffer, so occluded fragments are rejected here instead.
    vec2 uv = gl_FragCoord.xy * uInvRes;
    float sceneD = texture2D( tSceneDepth, uv ).x;
    if ( gl_FragCoord.z > sceneD + 0.00035 ) discard;
    vec2 a = ( vCurrClip.xy / vCurrClip.w ) * 0.5 + 0.5;
    vec2 b = ( vPrevClip.xy / vPrevClip.w ) * 0.5 + 0.5;
    gl_FragColor = vec4( a - b, 0.0, 1.0 );
  #endif
}
`;

// ---------------------------------------------------------------------------
// GTAO
// ---------------------------------------------------------------------------

export const GTAO_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}

uniform highp sampler2D tDepth;
uniform mat4 uInvProj;
uniform vec2 uInvFullRes;
uniform float uProjScaleY;
uniform float uRadius;
uniform float uPower;
uniform float uFrame;
uniform vec2 uFadeRange;

float depthAt( vec2 uv ) { return textureLod( tDepth, uv, 0.0 ).x; }

void main() {
  float d = depthAt( vUv );
  if ( d >= 0.9999995 ) { gl_FragColor = vec4( 1.0 ); return; }

  vec3 P = viewFromDepth( vUv, d, uInvProj );
  // Past the fade range the result is forced to 1.0 anyway, so there is no
  // reason to have marched for it. In a street shot most of the frame is
  // beyond this, which makes the test worth more than it looks.
  if ( -P.z > uFadeRange.y ) { gl_FragColor = vec4( 1.0 ); return; }
  vec3 V = normalize( -P );

  // Depth-derived normal. Picking the closer of the two one-sided differences
  // keeps the normal correct across depth discontinuities instead of shearing
  // across them, which is what makes naive SSAO halo along silhouettes.
  vec2 e = uInvFullRes;
  vec3 Pr = viewFromDepth( vUv + vec2( e.x, 0.0 ), depthAt( vUv + vec2( e.x, 0.0 ) ), uInvProj );
  vec3 Pl = viewFromDepth( vUv - vec2( e.x, 0.0 ), depthAt( vUv - vec2( e.x, 0.0 ) ), uInvProj );
  vec3 Pt = viewFromDepth( vUv + vec2( 0.0, e.y ), depthAt( vUv + vec2( 0.0, e.y ) ), uInvProj );
  vec3 Pb = viewFromDepth( vUv - vec2( 0.0, e.y ), depthAt( vUv - vec2( 0.0, e.y ) ), uInvProj );
  vec3 ddx = ( abs( Pr.z - P.z ) < abs( P.z - Pl.z ) ) ? ( Pr - P ) : ( P - Pl );
  vec3 ddy = ( abs( Pt.z - P.z ) < abs( P.z - Pb.z ) ) ? ( Pt - P ) : ( P - Pb );
  vec3 N = normalize( cross( ddx, ddy ) );
  if ( dot( N, V ) < 0.0 ) N = -N;

  float noise = ignNoise( gl_FragCoord.xy + uFrame * 5.588238 );
  float radiusPix = clamp( uRadius * uProjScaleY / max( 0.05, -P.z ), 4.0, 110.0 );

  float visibility = 0.0;

  for ( int s = 0; s < GTAO_SLICES; s ++ ) {
    float phi = ( float( s ) + noise ) * ( PST_PI / float( GTAO_SLICES ) );
    vec2 dir = vec2( cos( phi ), sin( phi ) );

    // In-slice orthonormal frame: V forward, T along +dir.
    vec3 sliceV = vec3( dir, 0.0 );
    vec3 T = sliceV - V * dot( sliceV, V );
    float tl = length( T );
    if ( tl < 1e-5 ) continue;
    T /= tl;
    vec3 axis = cross( V, T );

    vec3 projN = N - axis * dot( N, axis );
    float projNLen = length( projN );
    if ( projNLen < 1e-4 ) continue;
    vec3 pn = projN / projNLen;

    float nAng = atan( dot( pn, T ), dot( pn, V ) );

    float cosH1 = -1.0;
    float cosH2 = -1.0;

    for ( int k = 0; k < GTAO_STEPS; k ++ ) {
      float t = ( float( k ) + 0.5 + noise * 0.75 ) / float( GTAO_STEPS );
      // Bias samples toward the centre, where contact detail lives. Squaring
      // is the usual choice but it clusters too hard to survive a low step
      // count; 1.6 keeps the outer sample far enough out that a two-step slice
      // still spans the whole contact band.
      t = pow( t, 1.6 );
      vec2 off = dir * ( t * radiusPix ) * uInvFullRes;

      vec2 uvA = vUv + off;
      vec2 uvB = vUv - off;

      float dA = depthAt( uvA );
      float dB = depthAt( uvB );

      if ( dA < 0.9999995 && uvA.x > 0.0 && uvA.x < 1.0 && uvA.y > 0.0 && uvA.y < 1.0 ) {
        vec3 D = viewFromDepth( uvA, dA, uInvProj ) - P;
        float l = length( D );
        if ( l > 1e-4 ) {
          float c = dot( D / l, V );
          // Attenuate the horizon with distance instead of range-checking the
          // sample: this is what stops GTAO from ringing around silhouettes.
          c = mix( c, -1.0, sat( ( l - uRadius * 0.65 ) / ( uRadius * 0.35 ) ) );
          cosH1 = max( cosH1, c );
        }
      }

      if ( dB < 0.9999995 && uvB.x > 0.0 && uvB.x < 1.0 && uvB.y > 0.0 && uvB.y < 1.0 ) {
        vec3 D = viewFromDepth( uvB, dB, uInvProj ) - P;
        float l = length( D );
        if ( l > 1e-4 ) {
          float c = dot( D / l, V );
          c = mix( c, -1.0, sat( ( l - uRadius * 0.65 ) / ( uRadius * 0.35 ) ) );
          cosH2 = max( cosH2, c );
        }
      }
    }

    float h1 = acos( clamp( cosH1, -1.0, 1.0 ) );
    float h2 = -acos( clamp( cosH2, -1.0, 1.0 ) );
    h1 = nAng + min( h1 - nAng, 0.5 * PST_PI );
    h2 = nAng + max( h2 - nAng, -0.5 * PST_PI );

    float sinN = sin( nAng );
    float cosN = cos( nAng );
    float arc =
      0.25 * ( -cos( 2.0 * h1 - nAng ) + cosN + 2.0 * h1 * sinN ) +
      0.25 * ( -cos( 2.0 * h2 - nAng ) + cosN + 2.0 * h2 * sinN );

    visibility += projNLen * arc;
  }

  visibility = sat( visibility / float( GTAO_SLICES ) );
  float ao = pow( visibility, uPower );

  // Screen-space occlusion has no business darkening the far field.
  ao = mix( ao, 1.0, smoothstep( uFadeRange.x, uFadeRange.y, -P.z ) );

  gl_FragColor = vec4( ao, ao, ao, 1.0 );
}
`;

// ---------------------------------------------------------------------------
// Depth-aware separable blur (used for AO and for SSR)
// ---------------------------------------------------------------------------

export const BILATERAL_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}

uniform sampler2D tDiffuse;
uniform highp sampler2D tDepth;
uniform vec2 uDir;
uniform vec2 uNearFar;
uniform float uDepthSigma;

void main() {
  float cd = linearDepth( texture2D( tDepth, vUv ).x, uNearFar.x, uNearFar.y );
  vec4 sum = texture2D( tDiffuse, vUv ) * 0.2270270270;
  float wsum = 0.2270270270;

  // 9-tap gaussian collapsed to 4 linear-sampled offsets per side.
  const float offs[ 4 ] = float[ 4 ]( 1.3846153846, 3.2307692308, 5.1794871795, 7.1282051282 );
  const float wts[ 4 ] = float[ 4 ]( 0.3162162162, 0.0702702703, 0.0155844156, 0.0031168831 );

  for ( int i = 0; i < 4; i ++ ) {
    vec2 o = uDir * offs[ i ];

    vec2 ua = vUv + o;
    float da = linearDepth( texture2D( tDepth, ua ).x, uNearFar.x, uNearFar.y );
    float wa = wts[ i ] * exp( -abs( da - cd ) * uDepthSigma );
    sum += texture2D( tDiffuse, ua ) * wa;
    wsum += wa;

    vec2 ub = vUv - o;
    float db = linearDepth( texture2D( tDepth, ub ).x, uNearFar.x, uNearFar.y );
    float wb = wts[ i ] * exp( -abs( db - cd ) * uDepthSigma );
    sum += texture2D( tDiffuse, ub ) * wb;
    wsum += wb;
  }

  gl_FragColor = sum / max( wsum, 1e-4 );
}
`;

// ---------------------------------------------------------------------------
// Screen-space reflections
// ---------------------------------------------------------------------------

export const SSR_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}
${ATMOSPHERE_GLSL}
${SKY_UNIFORMS_GLSL}
${SKY_CALL_GLSL}

uniform sampler2D tScene;
uniform highp sampler2D tDepth;
uniform mat4 uProj;
uniform mat4 uInvProj;
uniform mat4 uCamWorld;
uniform vec2 uInvFullRes;
uniform vec2 uNearFar;
uniform float uFrame;
uniform float uThickness;
uniform float uMaxDistance;
uniform float uReflectivity;

float depthAt( vec2 uv ) { return textureLod( tDepth, uv, 0.0 ).x; }

void main() {
  float d = depthAt( vUv );
  if ( d >= 0.9999995 ) { gl_FragColor = vec4( 0.0 ); return; }

  vec3 P = viewFromDepth( vUv, d, uInvProj );
  vec2 e = uInvFullRes;
  vec3 Pr = viewFromDepth( vUv + vec2( e.x, 0.0 ), depthAt( vUv + vec2( e.x, 0.0 ) ), uInvProj );
  vec3 Pl = viewFromDepth( vUv - vec2( e.x, 0.0 ), depthAt( vUv - vec2( e.x, 0.0 ) ), uInvProj );
  vec3 Pt = viewFromDepth( vUv + vec2( 0.0, e.y ), depthAt( vUv + vec2( 0.0, e.y ) ), uInvProj );
  vec3 Pb = viewFromDepth( vUv - vec2( 0.0, e.y ), depthAt( vUv - vec2( 0.0, e.y ) ), uInvProj );
  vec3 ddx = ( abs( Pr.z - P.z ) < abs( P.z - Pl.z ) ) ? ( Pr - P ) : ( P - Pl );
  vec3 ddy = ( abs( Pt.z - P.z ) < abs( P.z - Pb.z ) ) ? ( Pt - P ) : ( P - Pb );
  vec3 N = normalize( cross( ddx, ddy ) );
  vec3 V = normalize( -P );
  if ( dot( N, V ) < 0.0 ) N = -N;

  vec3 worldN = normalize( mat3( uCamWorld ) * N );
  vec3 worldP = ( uCamWorld * vec4( P, 1.0 ) ).xyz;

  // Without a G-buffer there is no authored roughness, so reflections are
  // restricted to near-horizontal surfaces — wet tarmac, sills, puddles —
  // where a grazing-angle reflection is both plausible and flattering.
  float upMask = smoothstep( 0.55, 0.86, worldN.y );
  if ( upMask <= 0.001 ) { gl_FragColor = vec4( 0.0 ); return; }

  // Break the reflectivity up so the street is not uniformly wet.
  float wet;
  {
    vec2 wp = worldP.xz * 0.09;
    vec2 wi = floor( wp );
    vec2 wf = fract( wp );
    wf = wf * wf * ( 3.0 - 2.0 * wf );
    float a0 = hash12( wi );
    float a1 = hash12( wi + vec2( 1.0, 0.0 ) );
    float a2 = hash12( wi + vec2( 0.0, 1.0 ) );
    float a3 = hash12( wi + vec2( 1.0, 1.0 ) );
    wet = mix( mix( a0, a1, wf.x ), mix( a2, a3, wf.x ), wf.y );
    wet = sat( ( wet - 0.34 ) * 2.0 );
  }
  float rough = mix( 0.32, 0.06, wet );
  float refl = uReflectivity * mix( 0.25, 1.0, wet ) * upMask;

  float rnd = hash12( gl_FragCoord.xy + uFrame * 17.31 );
  vec2 rnd2 = hash22( gl_FragCoord.xy * 1.37 + uFrame * 11.7 );

  vec3 jitteredN = normalize( N + ( vec3( rnd2, rnd ) - 0.5 ) * rough * 0.55 );
  vec3 R = reflect( -V, jitteredN );

  float fres = 0.04 + 0.96 * pow( 1.0 - sat( dot( jitteredN, V ) ), 5.0 );
  float strength = refl * fres;
  if ( strength <= 0.003 ) { gl_FragColor = vec4( 0.0 ); return; }

  vec3 worldR = normalize( mat3( uCamWorld ) * R );
  vec3 fallback = skyInDirection( worldR );

  vec3 hitColor = fallback;
  float hitConfidence = 0.0;

  if ( R.z < 0.999 ) {
    float stepLen = uMaxDistance / float( SSR_STEPS );
    float t = stepLen * ( 0.35 + rnd * 0.65 );
    float prevT = 0.0;

    for ( int i = 0; i < SSR_STEPS; i ++ ) {
      vec3 S = P + R * t;
      if ( S.z > -uNearFar.x ) break;

      vec4 clip = uProj * vec4( S, 1.0 );
      vec2 suv = ( clip.xy / clip.w ) * 0.5 + 0.5;
      if ( suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0 ) break;

      float sd = depthAt( suv );
      if ( sd < 0.9999995 ) {
        float sceneZ = viewFromDepth( suv, sd, uInvProj ).z;
        float delta = S.z - sceneZ; // negative until the ray goes behind geometry

        if ( delta < 0.0 && delta > -uThickness ) {
          // Binary refine between the last two samples for a crisp contact.
          float lo = prevT;
          float hi = t;
          for ( int j = 0; j < 4; j ++ ) {
            float mid = ( lo + hi ) * 0.5;
            vec3 M = P + R * mid;
            vec4 mc = uProj * vec4( M, 1.0 );
            vec2 muv = ( mc.xy / mc.w ) * 0.5 + 0.5;
            float md = depthAt( muv );
            float mz = viewFromDepth( muv, md, uInvProj ).z;
            if ( M.z - mz < 0.0 ) hi = mid; else lo = mid;
          }
          vec3 H = P + R * hi;
          vec4 hc = uProj * vec4( H, 1.0 );
          vec2 huv = ( hc.xy / hc.w ) * 0.5 + 0.5;

          hitColor = textureLod( tScene, huv, 0.0 ).rgb;

          // Fade at the screen border and as the ray runs long, then fade to
          // the sky so the transition to the fallback is never a hard seam.
          vec2 edge = smoothstep( vec2( 0.0 ), vec2( 0.14 ), huv ) *
                      smoothstep( vec2( 0.0 ), vec2( 0.14 ), vec2( 1.0 ) - huv );
          hitConfidence = edge.x * edge.y;
          hitConfidence *= 1.0 - sat( hi / uMaxDistance );
          hitConfidence *= sat( 1.0 + delta / uThickness );
          break;
        }
      }

      prevT = t;
      t += stepLen * ( 1.0 + float( i ) * 0.14 ); // geometric growth: dense near, cheap far
    }
  }

  vec3 color = mix( fallback, hitColor, hitConfidence );
  gl_FragColor = vec4( color * strength, strength );
}
`;

// ---------------------------------------------------------------------------
// Volumetric light shafts
// ---------------------------------------------------------------------------

export const VOLUMETRIC_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}

uniform highp sampler2D tDepth;
uniform mat4 uInvProj;
uniform mat4 uCamWorld;
uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform float uDensity;
uniform float uHeightFalloff;
uniform float uBaseHeight;
uniform float uG;
uniform float uMaxDistance;
uniform float uFrame;

#if CSM_COUNT > 0
uniform highp sampler2DShadow uShadow0;
uniform mat4 uShadowMat0;
#endif
#if CSM_COUNT > 1
uniform highp sampler2DShadow uShadow1;
uniform mat4 uShadowMat1;
#endif
#if CSM_COUNT > 2
uniform highp sampler2DShadow uShadow2;
uniform mat4 uShadowMat2;
#endif
#if CSM_COUNT > 3
uniform highp sampler2DShadow uShadow3;
uniform mat4 uShadowMat3;
#endif
uniform vec4 uCascadeFar;

float sampleShadowAt( mat4 m, highp sampler2DShadow smap, vec3 w ) {
  vec4 c = m * vec4( w, 1.0 );
  vec3 p = c.xyz / c.w;
  if ( p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z < 0.0 || p.z > 1.0 ) return -1.0;
  return texture( smap, vec3( p.xy, p.z - 0.0012 ) );
}

float sunVisibility( vec3 w, float viewDist ) {
  float v = -1.0;
  #if CSM_COUNT > 0
  if ( viewDist < uCascadeFar.x ) v = sampleShadowAt( uShadowMat0, uShadow0, w );
  #endif
  #if CSM_COUNT > 1
  if ( v < 0.0 && viewDist < uCascadeFar.y ) v = sampleShadowAt( uShadowMat1, uShadow1, w );
  #endif
  #if CSM_COUNT > 2
  if ( v < 0.0 && viewDist < uCascadeFar.z ) v = sampleShadowAt( uShadowMat2, uShadow2, w );
  #endif
  #if CSM_COUNT > 3
  if ( v < 0.0 ) v = sampleShadowAt( uShadowMat3, uShadow3, w );
  #endif
  return v < 0.0 ? 1.0 : v;
}

void main() {
  float d = texture2D( tDepth, vUv ).x;
  vec3 P = viewFromDepth( vUv, d, uInvProj );
  vec3 worldEnd = ( uCamWorld * vec4( P, 1.0 ) ).xyz;

  vec3 ray = worldEnd - uCamPos;
  float dist = length( ray );
  if ( dist < 1e-3 ) { gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 ); return; }
  vec3 rd = ray / dist;
  float marchLen = min( dist, uMaxDistance );

  float cosT = dot( rd, uSunDir );
  float phase = henyeyGreenstein( cosT, uG );

  float stepLen = marchLen / float( VOL_STEPS );
  // Blue-noise-ish dither so the 12 steps read as haze, not as bands.
  float dither = ignNoise( gl_FragCoord.xy + uFrame * 3.7 );

  vec3 scatter = vec3( 0.0 );
  float transmittance = 1.0;

  for ( int i = 0; i < VOL_STEPS; i ++ ) {
    float t = ( float( i ) + dither ) * stepLen;
    if ( t > marchLen ) break;
    vec3 w = uCamPos + rd * t;

    float h = exp( -max( 0.0, w.y - uBaseHeight ) * uHeightFalloff );
    float sigma = uDensity * h;
    if ( sigma < 1e-6 ) continue;

    float vis = sunVisibility( w, t );
    scatter += transmittance * vis * sigma * stepLen * phase;
    transmittance *= exp( -sigma * stepLen * 1.35 );
    if ( transmittance < 0.01 ) break;
  }

  gl_FragColor = vec4( scatter * uSunColor, 1.0 );
}
`;

// ---------------------------------------------------------------------------
// Composite: AO + SSR + volumetrics + aerial perspective
// ---------------------------------------------------------------------------

export const COMPOSITE_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}
${ATMOSPHERE_GLSL}
${SKY_UNIFORMS_GLSL}
${SKY_CALL_GLSL}

uniform sampler2D tScene;
uniform highp sampler2D tDepth;
uniform sampler2D tAo;
uniform sampler2D tSsr;
uniform sampler2D tVolume;
uniform mat4 uInvProj;
uniform mat4 uCamWorld;
uniform vec3 uCamPos;
uniform float uAoStrength;
uniform vec2 uAoLitRange;
uniform float uAoDirectKeep;
uniform float uSsrEnabled;
uniform float uVolumeEnabled;
uniform float uFogDensity;
uniform float uFogHeightFalloff;
uniform float uFogBaseHeight;
uniform float uFogStart;
uniform float uFogDesaturate;

// GTAO multi-bounce: occlusion on a coloured albedo should not go neutral grey.
vec3 multiBounce( float ao, vec3 albedo ) {
  vec3 a = 2.0404 * albedo - 0.3324;
  vec3 b = -4.7951 * albedo + 0.6417;
  vec3 c = 2.7552 * albedo + 0.6903;
  return sat3( vec3( ao ) * ( vec3( ao ) * ( vec3( ao ) * a + b ) + c ) );
}

void main() {
  vec3 color = texture2D( tScene, vUv ).rgb;
  float d = texture2D( tDepth, vUv ).x;
  bool isSky = d >= 0.9999995;

  if ( !isSky ) {
    float ao = texture2D( tAo, vUv ).r;

    // Occlusion belongs to the ambient term. Nothing here has a split
    // direct/indirect buffer, but with the sun running roughly eight times the
    // fill, scene luminance separates the two well enough: a pixel above the
    // lit band is being hit by the key, and darkening it would be painting dirt
    // into a sunlit corner. It keeps a fraction, because contact does block
    // some of the bounce even in full sun.
    float lit = smoothstep( uAoLitRange.x, uAoLitRange.y, lumaOf( color ) );
    float aoW = uAoStrength * mix( 1.0, uAoDirectKeep, lit );
    ao = mix( 1.0, ao, aoW );
    color *= multiBounce( ao, sat3( color * 1.6 ) );

    if ( uSsrEnabled > 0.5 ) {
      vec4 ssr = texture2D( tSsr, vUv );
      color = mix( color, ssr.rgb / max( ssr.a, 1e-3 ), sat( ssr.a ) );
    }
  }

  // Height-based aerial perspective. Distant geometry is not tinted with a
  // single fog colour but with the sky radiance along the view ray, which is
  // what makes a horizon read as air rather than as a grey wash.
  if ( !isSky ) {
    vec3 P = viewFromDepth( vUv, d, uInvProj );
    vec3 worldP = ( uCamWorld * vec4( P, 1.0 ) ).xyz;
    vec3 ray = worldP - uCamPos;
    float dist = length( ray );
    vec3 rd = dist > 1e-4 ? ray / dist : vec3( 0.0, 0.0, -1.0 );

    float hCam = max( 0.0, uCamPos.y - uFogBaseHeight );
    float hEnd = max( 0.0, worldP.y - uFogBaseHeight );
    float dy = hEnd - hCam;
    // Analytic integral of exp(-k*h) along the ray.
    float optical;
    if ( abs( dy ) < 1e-3 ) {
      optical = exp( -uFogHeightFalloff * hCam ) * dist;
    } else {
      optical = dist * ( exp( -uFogHeightFalloff * hCam ) - exp( -uFogHeightFalloff * hEnd ) ) /
                ( uFogHeightFalloff * dy );
    }
    optical = max( 0.0, optical - uFogStart * exp( -uFogHeightFalloff * hCam ) );
    float fog = 1.0 - exp( -uFogDensity * optical );

    float aer = min( fog, 0.94 );

    // Below a percent of haze the analytic sky evaluation cannot change the
    // pixel by a code value, and most of a street frame is inside that. Worth
    // the branch: two Preetham evaluations at full resolution are not free.
    if ( aer > 0.01 ) {
      // Contrast and saturation go first, and they go faster than the colour
      // replacement does. Air scatters a distant facade's own light out of the
      // ray before it fills the ray back up with sky, so what you see at range
      // is a *flatter, greyer* version of the surface with the sky laid over it
      // — not the surface at full chroma under a wash. Doing only the wash is
      // what makes engine fog look like a coloured sheet of glass.
      float ls = lumaOf( color );
      color = mix( color, vec3( ls ), aer * uFogDesaturate );
      // Contrast toward the local mean as well, so distant value structure
      // compresses instead of staying razor sharp under a lifted sky.
      color = mix( color, vec3( mix( ls, 0.5, 0.35 ) ), aer * 0.22 );

      // The haze itself is the sky radiance *along this pixel's view ray*, from
      // the exact function that draws the dome — same Preetham evaluation, same
      // roll, same horizon band. Not an approximation of it, and not a
      // CPU-fitted three-colour lerp: the far wall now fades into the pixel
      // directly above it, whatever colour that pixel happens to be.
      vec3 air = skyInDirection( rd );
      color = mix( color, air, aer );
    }
  }

  if ( uVolumeEnabled > 0.5 ) {
    color += texture2D( tVolume, vUv ).rgb;
  }

  // Alpha 0 marks "world"; the viewmodel pass overwrites it with 1.
  gl_FragColor = vec4( color, 0.0 );
}
`;

// ---------------------------------------------------------------------------
// Auto-exposure metering
// ---------------------------------------------------------------------------

/**
 * Log-average luminance, folded 1344x756 -> 64x64 -> 8x8 -> 1x1 in three draws.
 *
 * Metering runs on the *scene* target, before the viewmodel is composited, so
 * the gun — which is dark, close and covers a fifth of the frame — cannot drag
 * the whole world a stop brighter every time the player looks down.
 *
 * The result is stored as an 8-bit encoding of log2(luminance) over a 28-stop
 * window, which quantises to 0.11 stops: below the point at which a change in
 * exposure is visible, and cheap enough to read back a single texel per frame.
 */
export const LUMA_INIT_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}
uniform sampler2D tDiffuse;
uniform highp sampler2D tDepth;
uniform vec2 uSrcTexel;
uniform vec2 uBlock;
uniform float uSkyWeight;

void main() {
  float s = 0.0;
  float wsum = 0.0;
  for ( int y = 0; y < 4; y ++ ) {
    for ( int x = 0; x < 4; x ++ ) {
      vec2 o = ( ( vec2( float( x ), float( y ) ) + 0.5 ) * 0.25 - 0.5 ) * uBlock;
      vec2 uv = vUv + o * uSrcTexel;
      vec3 c = max( vec3( 0.0 ), textureLod( tDiffuse, uv, 0.0 ).rgb );
      // The sky is not what the camera is exposing for. A frame that happens to
      // point up the street into the sun contains three times the sky of one
      // pointed down it, and metering them equally swings the exposure a stop
      // and a half between two shots of the same town at the same hour. It
      // keeps a small weight so a frame that is *only* sky still meters.
      float w = textureLod( tDepth, uv, 0.0 ).x >= 0.9999995 ? uSkyWeight : 1.0;
      // Clamped at the top so the sun's disc — four orders of magnitude above
      // anything else in the frame — cannot swing the whole meter on its own.
      s += w * log2( clamp( lumaOf( c ), 1e-4, 8.0 ) );
      wsum += w;
    }
  }
  s /= max( wsum, 1e-4 );
  gl_FragColor = vec4( ( s + 14.0 ) / 28.0, 0.0, 0.0, 1.0 );
}
`;

export const LUMA_DOWN_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D tDiffuse;
uniform vec2 uSrcTexel;
uniform vec2 uBlock;

void main() {
  // 4x4 bilinear taps on texel boundaries: each tap is the mean of four texels,
  // so a block of 8x8 source texels is covered exactly once with no gaps and no
  // double counting.
  float s = 0.0;
  for ( int y = 0; y < 4; y ++ ) {
    for ( int x = 0; x < 4; x ++ ) {
      vec2 o = ( ( vec2( float( x ), float( y ) ) + 0.5 ) * 0.25 - 0.5 ) * uBlock;
      s += textureLod( tDiffuse, vUv + o * uSrcTexel, 0.0 ).r;
    }
  }
  gl_FragColor = vec4( s / 16.0, 0.0, 0.0, 1.0 );
}
`;

// ---------------------------------------------------------------------------
// Bloom: threshold + progressive down/up chain
// ---------------------------------------------------------------------------

export const BLOOM_PREFILTER_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}

uniform sampler2D tDiffuse;
uniform vec2 uTexel;
uniform float uThreshold;
uniform float uKnee;
uniform float uClamp;
uniform float uExposure;

// Thresholding happens in exposed space. Doing it on raw scene radiance means
// the threshold drifts with exposure and a bright sky blooms over everything.
vec3 fetch( vec2 uv ) { return min( textureLod( tDiffuse, uv, 0.0 ).rgb * uExposure, vec3( uClamp ) ); }
float karis( vec3 c ) { return 1.0 / ( 1.0 + lumaOf( c ) ); }

void main() {
  // Karis-weighted 4-tap box: a single blown pixel cannot dominate the mip.
  vec3 a = fetch( vUv + uTexel * vec2( -1.0, -1.0 ) );
  vec3 b = fetch( vUv + uTexel * vec2( 1.0, -1.0 ) );
  vec3 c = fetch( vUv + uTexel * vec2( -1.0, 1.0 ) );
  vec3 e = fetch( vUv + uTexel * vec2( 1.0, 1.0 ) );
  float wa = karis( a ), wb = karis( b ), wc = karis( c ), we = karis( e );
  vec3 col = ( a * wa + b * wb + c * wc + e * we ) / max( 1e-4, wa + wb + wc + we );

  float br = max( col.r, max( col.g, col.b ) );
  float knee = max( 1e-4, uThreshold * uKnee );
  float soft = clamp( br - uThreshold + knee, 0.0, 2.0 * knee );
  soft = soft * soft / ( 4.0 * knee );
  float contrib = max( soft, br - uThreshold ) / max( br, 1e-4 );

  gl_FragColor = vec4( col * contrib, 1.0 );
}
`;

export const BLOOM_DOWN_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D tDiffuse;
uniform vec2 uTexel;

vec3 f( vec2 o ) { return textureLod( tDiffuse, vUv + o * uTexel, 0.0 ).rgb; }

void main() {
  // Jimenez 13-tap downsample — no aliasing crawl on the bloom mips.
  vec3 a = f( vec2( -2.0, 2.0 ) ), b = f( vec2( 0.0, 2.0 ) ), c = f( vec2( 2.0, 2.0 ) );
  vec3 d = f( vec2( -2.0, 0.0 ) ), e = f( vec2( 0.0, 0.0 ) ), g = f( vec2( 2.0, 0.0 ) );
  vec3 h = f( vec2( -2.0, -2.0 ) ), i = f( vec2( 0.0, -2.0 ) ), j = f( vec2( 2.0, -2.0 ) );
  vec3 k = f( vec2( -1.0, 1.0 ) ), l = f( vec2( 1.0, 1.0 ) );
  vec3 m = f( vec2( -1.0, -1.0 ) ), n = f( vec2( 1.0, -1.0 ) );

  vec3 col = e * 0.125;
  col += ( a + c + h + j ) * 0.03125;
  col += ( b + d + g + i ) * 0.0625;
  col += ( k + l + m + n ) * 0.125;

  gl_FragColor = vec4( col, 1.0 );
}
`;

export const BLOOM_UP_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D tDiffuse;
uniform vec2 uTexel;
uniform float uRadius;
uniform float uLevelWeight;

vec3 f( vec2 o ) { return textureLod( tDiffuse, vUv + o * uTexel * uRadius, 0.0 ).rgb; }

void main() {
  // 9-tap tent upsample, blended additively into the finer level. Each coarser
  // level lands at a fraction of the one below it, which is what makes the
  // chain a tight bright core with a wide faint skirt instead of an even glow.
  vec3 col = f( vec2( 0.0, 0.0 ) ) * 4.0;
  col += ( f( vec2( -1.0, 0.0 ) ) + f( vec2( 1.0, 0.0 ) ) + f( vec2( 0.0, -1.0 ) ) + f( vec2( 0.0, 1.0 ) ) ) * 2.0;
  col += f( vec2( -1.0, -1.0 ) ) + f( vec2( 1.0, -1.0 ) ) + f( vec2( -1.0, 1.0 ) ) + f( vec2( 1.0, 1.0 ) );
  col *= 0.0625 * uLevelWeight;

  gl_FragColor = vec4( col, 1.0 );
}
`;

// ---------------------------------------------------------------------------
// Motion blur
// ---------------------------------------------------------------------------

export const MOTION_BLUR_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}

uniform sampler2D tDiffuse;
uniform sampler2D tVelocity;
uniform highp sampler2D tDepth;
uniform vec2 uResolution;
uniform float uShutter;
uniform float uMaxRadius;
uniform float uFrame;

void main() {
  vec4 center = texture2D( tDiffuse, vUv );
  vec2 vel = texture2D( tVelocity, vUv ).xy * uShutter;

  float lenPx = length( vel * uResolution );
  if ( lenPx < 0.75 ) { gl_FragColor = center; return; }

  float maxPx = uMaxRadius * uResolution.y;
  if ( lenPx > maxPx ) vel *= maxPx / lenPx;

  float centerDepth = texture2D( tDepth, vUv ).x;
  float jitter = ignNoise( gl_FragCoord.xy + uFrame * 2.137 ) - 0.5;

  vec4 sum = center;
  float wsum = 1.0;

  for ( int i = 1; i <= MB_TAPS; i ++ ) {
    float t = ( float( i ) + jitter ) / float( MB_TAPS );
    vec2 o = vel * ( t * 0.5 );

    vec2 ua = vUv + o;
    vec2 ub = vUv - o;

    vec4 ca = texture2D( tDiffuse, ua );
    vec4 cb = texture2D( tDiffuse, ub );

    // Do not smear the viewmodel into the world or the world onto the gun.
    float ma = 1.0 - abs( ca.a - center.a );
    float mb = 1.0 - abs( cb.a - center.a );

    float w = 1.0 - t * 0.35;
    sum += ca * ( w * ma ) + cb * ( w * mb );
    wsum += w * ma + w * mb;
  }

  gl_FragColor = vec4( ( sum / max( wsum, 1e-4 ) ).rgb, center.a );
}
`;

// ---------------------------------------------------------------------------
// Depth of field — hexagonal aperture
// ---------------------------------------------------------------------------

export const DOF_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}

uniform sampler2D tDiffuse;
uniform highp sampler2D tDepth;
uniform vec2 uResolution;
uniform vec2 uNearFar;
uniform float uFocusDistance;
uniform float uAperture;
uniform float uFocalLength;
uniform float uMaxCoc;
uniform float uFrame;

float cocFor( float viewZ ) {
  float f = uFocalLength;
  float s = max( uFocusDistance, f + 1e-3 );
  float c = uAperture * f * abs( viewZ - s ) / max( viewZ * ( s - f ), 1e-4 );
  return min( c * uResolution.y, uMaxCoc );
}

// Circle -> hexagon: the aperture blades are what make bokeh read as a lens.
float hexRadius( float ang ) {
  float a = mod( ang, PST_PI / 3.0 ) - PST_PI / 6.0;
  return 0.8660254 / max( 0.35, cos( a ) );
}

void main() {
  vec4 center = texture2D( tDiffuse, vUv );
  float d = texture2D( tDepth, vUv ).x;

  // The viewmodel is composited after the world; it is always in focus and it
  // never receives near-field bleed from geometry behind it.
  if ( center.a > 0.5 ) { gl_FragColor = center; return; }

  float viewZ = linearDepth( d, uNearFar.x, uNearFar.y );
  float coc = cocFor( viewZ );
  if ( coc < 1.0 ) { gl_FragColor = center; return; }

  float rot = ignNoise( gl_FragCoord.xy + uFrame * 1.618 ) * 6.2831853;
  vec2 texel = 1.0 / uResolution;

  vec4 sum = center;
  float wsum = 1.0;

  for ( int ring = 1; ring <= DOF_RINGS; ring ++ ) {
    float rr = float( ring ) / float( DOF_RINGS );
    int count = 6 * ring;
    for ( int i = 0; i < 6 * DOF_RINGS; i ++ ) {
      if ( i >= count ) break;
      float ang = ( float( i ) / float( count ) ) * 6.2831853 + rot;
      vec2 dir = vec2( cos( ang ), sin( ang ) ) * hexRadius( ang );
      vec2 uv = vUv + dir * rr * coc * texel;

      vec4 s = texture2D( tDiffuse, uv );
      if ( s.a > 0.5 ) continue; // never pull the gun into the world's bokeh

      float sd = texture2D( tDepth, uv ).x;
      float sCoc = cocFor( linearDepth( sd, uNearFar.x, uNearFar.y ) );
      // A sample only spreads this far if its own circle of confusion reaches.
      float w = sat( ( sCoc - rr * coc ) * 0.5 + 0.6 );
      sum += s * w;
      wsum += w;
    }
  }

  gl_FragColor = vec4( ( sum / max( wsum, 1e-4 ) ).rgb, center.a );
}
`;

// ---------------------------------------------------------------------------
// TAA
// ---------------------------------------------------------------------------

export const TAA_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}

uniform sampler2D tCurrent;
uniform sampler2D tHistory;
uniform sampler2D tVelocity;
uniform highp sampler2D tDepth;
uniform vec2 uResolution;
uniform float uHistoryValid;
uniform float uMinBlend;
uniform float uMaxBlend;

void main() {
  vec2 texel = 1.0 / uResolution;
  vec4 cur = texture2D( tCurrent, vUv );

  // Dilate the velocity toward the closest fragment in the neighbourhood so
  // silhouettes reproject with the object in front, not the background.
  vec2 bestUv = vUv;
  float bestD = texture2D( tDepth, vUv ).x;
  vec2 offs[ 4 ];
  offs[ 0 ] = vec2( -1.0, -1.0 );
  offs[ 1 ] = vec2( 1.0, -1.0 );
  offs[ 2 ] = vec2( -1.0, 1.0 );
  offs[ 3 ] = vec2( 1.0, 1.0 );
  for ( int i = 0; i < 4; i ++ ) {
    vec2 u = vUv + offs[ i ] * texel;
    float dd = texture2D( tDepth, u ).x;
    if ( dd < bestD ) { bestD = dd; bestUv = u; }
  }
  vec2 vel = texture2D( tVelocity, bestUv ).xy;

  vec2 prevUv = vUv - vel;
  if ( prevUv.x < 0.0 || prevUv.x > 1.0 || prevUv.y < 0.0 || prevUv.y > 1.0 || uHistoryValid < 0.5 ) {
    gl_FragColor = cur;
    return;
  }

  // Neighbourhood statistics in YCoCg: chroma clipping there kills the purple
  // fringing that RGB clamping leaves behind on moving edges.
  vec3 m1 = vec3( 0.0 );
  vec3 m2 = vec3( 0.0 );
  vec3 mn = vec3( 1e9 );
  vec3 mx = vec3( -1e9 );
  for ( int y = -1; y <= 1; y ++ ) {
    for ( int x = -1; x <= 1; x ++ ) {
      vec3 c = rgb2ycocg( texture2D( tCurrent, vUv + vec2( float( x ), float( y ) ) * texel ).rgb );
      m1 += c;
      m2 += c * c;
      mn = min( mn, c );
      mx = max( mx, c );
    }
  }
  vec3 mu = m1 / 9.0;
  vec3 sigma = sqrt( max( vec3( 0.0 ), m2 / 9.0 - mu * mu ) );
  vec3 lo = max( mn, mu - sigma * 1.45 );
  vec3 hi = min( mx, mu + sigma * 1.45 );

  vec3 hist = rgb2ycocg( texture2D( tHistory, prevUv ).rgb );
  vec3 curY = rgb2ycocg( cur.rgb );

  // Clip (not clamp) toward the current colour along the history ray.
  vec3 centre = 0.5 * ( hi + lo );
  vec3 extent = max( 0.5 * ( hi - lo ), vec3( 1e-4 ) );
  vec3 rayDir = hist - centre;
  vec3 unit = rayDir / extent;
  float maxUnit = max( abs( unit.x ), max( abs( unit.y ), abs( unit.z ) ) );
  if ( maxUnit > 1.0 ) hist = centre + rayDir / maxUnit;

  float speed = length( vel * uResolution );
  float blend = mix( uMinBlend, uMaxBlend, sat( speed / 12.0 ) );

  vec3 outC = ycocg2rgb( mix( hist, curY, blend ) );
  gl_FragColor = vec4( max( vec3( 0.0 ), outC ), cur.a );
}
`;

// ---------------------------------------------------------------------------
// Tonemap + grade
// ---------------------------------------------------------------------------

export const TONEMAP_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}

uniform sampler2D tDiffuse;
uniform sampler2D tBloom;
uniform float uExposure;
uniform float uBloomStrength;
uniform vec3 uLift;
uniform vec3 uGamma;
uniform vec3 uGain;
uniform vec3 uShadowTint;
uniform vec3 uHighlightTint;
uniform float uSaturation;
uniform float uContrast;
uniform float uToe;
uniform float uToeKnee;
uniform float uShoulder;
uniform float uWhitePoint;

// Full ACES RRT+ODT fit (Stephen Hill), not the Narkowicz approximation:
// the input/output matrices are what give ACES its highlight hue rotation.
const mat3 ACES_IN = mat3(
  0.59719, 0.07600, 0.02840,
  0.35458, 0.90834, 0.13383,
  0.04823, 0.01566, 0.83777
);
const mat3 ACES_OUT = mat3(
  1.60475, -0.10208, -0.00327,
  -0.53108, 1.10813, -0.07276,
  -0.07367, -0.00605, 1.07602
);

vec3 rrtOdtFit( vec3 v ) {
  vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
  vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
  return a / b;
}

vec3 acesFitted( vec3 c ) {
  c = ACES_IN * c;
  c = rrtOdtFit( c );
  c = ACES_OUT * c;
  return max( vec3( 0.0 ), c );
}

/**
 * Filmic S-curve applied on top of ACES.
 *
 * Three separate jobs, in order, and none of them clip:
 *
 *  - contrast as a power around the 18% pivot. A power has no discontinuity at
 *    either end, unlike ( x - p ) * k + p, which is a straight line that runs
 *    off both ends of the range and then gets clamped. That clamp is a hard
 *    clip and it is why the old grade had neither black detail nor a highlight
 *    roll — it traded both for a slightly steeper middle.
 *  - a toe: extra density in the bottom couple of stops, on a smoothstep so it
 *    fades out rather than banding. This is what puts a real black in the frame.
 *    Its knee is an explicit parameter and it belongs *low*: at the old hard
 *    coded 0.2 the toe was engaging on everything under sRGB 0.48, which in a
 *    frame whose content sat between sRGB 0.13 and 0.40 meant every pixel. A
 *    curve that shapes the whole image is not a toe, it is a gamma, and the
 *    result was a midtone crush that flattened exactly the range the eye reads
 *    for form. At 0.07 it touches the bottom two stops and nothing else.
 *  - a shoulder: a hyperbolic roll above uShoulder. It leaves the curve with
 *    unit slope exactly at the knee — so there is no visible break where it
 *    engages — and asymptotes to 1.0, so an arbitrarily bright highlight
 *    compresses toward white and never reaches a flat clipped plateau. Only
 *    the sun's own disc gets close.
 *
 * A shoulder that is normalised to put 1.0 at 1.0 is not a shoulder, it is a
 * knee: forcing that endpoint makes the slope above the pivot *greater* than
 * one, which expands the very highlights it is supposed to be compressing.
 */
vec3 toneShape( vec3 c, float contrast, float toe, float toeKnee, float shoulder, float white ) {
  vec3 x = max( c, vec3( 0.0 ) );

  x = pow( max( x / 0.18, vec3( 1e-5 ) ), vec3( contrast ) ) * 0.18;

  vec3 t = sat3( x / max( 1e-4, toeKnee ) );
  x *= mix( vec3( 1.0 ), t * t * ( 3.0 - 2.0 * t ), toe );

  float s = clamp( shoulder, 0.05, 0.95 );
  // The shoulder is normalised so that a chosen white point maps to exactly
  // 1.0. Without this the curve asymptotes *below* one — ACES tops out near
  // 1.0165, the contrast power lifts that to about 1.54, and a shoulder whose
  // asymptote is 1.0 then compresses it to 0.94 no matter how bright the input.
  // Nothing in the game could reach white: not a window onto full sun, not the
  // sun itself. A filmic curve has to be told where white is.
  float w = max( white, 1.0 + ( 1.0 - s ) * 0.02 );
  float range = ( ( 1.0 - s ) * ( w - s ) ) / max( 1e-3, ( w - s ) - ( 1.0 - s ) );
  vec3 over = max( vec3( 0.0 ), x - s );
  vec3 rolled = s + range * ( over / ( over + range ) );
  x = mix( x, rolled, step( vec3( s ), x ) );

  return sat3( x );
}

void main() {
  vec3 color = texture2D( tDiffuse, vUv ).rgb;
  vec3 bloom = texture2D( tBloom, vUv ).rgb;

  color = max( vec3( 0.0 ), color * uExposure );
  // Bloom arrives already exposed from the prefilter, and is *added* rather
  // than cross-faded: mix( color, bloom, 0.055 ) quietly took 5.5% off every
  // pixel in the frame whether or not there was any bloom there to take it for.
  color += bloom * max( 0.0, uBloomStrength );

  color = acesFitted( color );

  color = toneShape( color, uContrast, uToe, uToeKnee, uShoulder, uWhitePoint );

  // Lift / gamma / gain.
  //
  // uLift is fed 0 and must stay there. The buffer is display-LINEAR at this
  // point and FINAL_FRAG still has to run linearToSrgb over it, so a lift
  // authored as an sRGB pedestal lands two and a half stops too high: 0.017
  // linear encodes to sRGB 0.138, which is why five different frames all
  // measured a 1st-percentile luminance of 0.133 to three decimals. There was
  // no black anywhere in the game. The film black now lives in FINAL_FRAG,
  // after the encode, where a print black is actually specified.
  color = sat3( color * ( uGain - uLift ) + uLift );
  color = pow( max( color, vec3( 1e-5 ) ), uGamma );

  // Split tone: cool shadows, warm highlights — the golden-hour signature.
  //
  // The luma is taken in an approximately perceptual space. The breakpoints
  // below were authored against sRGB code values, but color here is
  // display-linear, and feeding one to the other put content at sRGB 0.34 —
  // linear 0.095 — at shadowW 0.92 and highW 0.0. That painted 92% of the COOL
  // tint onto every sunlit surface in the game and never once fired the warm
  // one, which is why the sunlit half of a facade measured *bluer* than its own
  // shadow. Two ways to fix it; this is the one that keeps the numbers legible.
  float l = lumaOf( pow( max( color, vec3( 0.0 ) ), vec3( 1.0 / 2.2 ) ) );
  vec3 shadowW = vec3( 1.0 - smoothstep( 0.0, 0.84, l ) );
  vec3 highW = vec3( smoothstep( 0.44, 1.0, l ) );
  color *= mix( vec3( 1.0 ), uShadowTint, shadowW );
  color *= mix( vec3( 1.0 ), uHighlightTint, highW * 0.8 );

  // Saturation shaping: lift the mids, let the extremes fall off.
  float sl = lumaOf( color );
  float satScale = uSaturation * ( 1.0 - 0.35 * smoothstep( 0.65, 1.0, sl ) );
  color = mix( vec3( sl ), color, satScale );

  gl_FragColor = vec4( sat3( color ), 1.0 );
}
`;

// ---------------------------------------------------------------------------
// Final: distortion, aberration, sharpen, vignette, grain, sRGB
// ---------------------------------------------------------------------------

export const FINAL_FRAG = /* glsl */ `
varying vec2 vUv;
${POST_COMMON}

uniform sampler2D tDiffuse;
uniform vec2 uResolution;
uniform float uTime;
uniform float uAberration;
uniform float uDistortion;
uniform float uVignette;
uniform float uGrain;
uniform float uSharpen;
uniform float uFxaa;
uniform float uFilmBlack;
uniform vec3 uFilmBlackTint;

/**
 * Luma FXAA 3.11, trimmed to the console-quality preset. This is the *only*
 * anti-aliasing when TAA is off — the renderer is created with antialias:false
 * because the frame is composited through render targets, where MSAA does not
 * apply. Without it a low preset ships raw, crawling geometry edges.
 *
 * Two things were wrong with the way it was being used. It ran on a *linear*
 * buffer while its thresholds were authored for gamma-encoded luma, so a
 * silhouette running sRGB 0.32 -> 0.61 in one pixel presented as a linear step
 * of 0.09 and slipped under the absolute gate. And the pipeline latched it off
 * whenever a frame took longer than 45ms — which is every frame on the software
 * rasteriser the whole review is captured on, so the shipped screenshots had no
 * anti-aliasing at all. The luma is now taken in perceptual space, on FXAA's
 * own published thresholds.
 */
float fxaaLuma( vec3 c ) { return lumaOf( sqrt( max( c, vec3( 0.0 ) ) ) ); }

vec3 fxaaFilter( sampler2D tex, vec2 uv, vec2 texel ) {
  vec3 rgbM = texture2D( tex, uv ).rgb;
  vec3 rgbNW = texture2D( tex, uv + vec2( -1.0, -1.0 ) * texel ).rgb;
  vec3 rgbNE = texture2D( tex, uv + vec2(  1.0, -1.0 ) * texel ).rgb;
  vec3 rgbSW = texture2D( tex, uv + vec2( -1.0,  1.0 ) * texel ).rgb;
  vec3 rgbSE = texture2D( tex, uv + vec2(  1.0,  1.0 ) * texel ).rgb;

  float lM = fxaaLuma( rgbM );
  float lNW = fxaaLuma( rgbNW );
  float lNE = fxaaLuma( rgbNE );
  float lSW = fxaaLuma( rgbSW );
  float lSE = fxaaLuma( rgbSE );
  float lMin = min( lM, min( min( lNW, lNE ), min( lSW, lSE ) ) );
  float lMax = max( lM, max( max( lNW, lNE ), max( lSW, lSE ) ) );
  // Contrast gate: flat regions are left alone so texture detail survives.
  if ( lMax - lMin < max( 0.0312, lMax * 0.125 ) ) return rgbM;

  vec2 dir = vec2( -( ( lNW + lNE ) - ( lSW + lSE ) ), ( lNW + lSW ) - ( lNE + lSE ) );
  float reduce = max( ( lNW + lNE + lSW + lSE ) * 0.03125, 0.0078125 );
  float rcpMin = 1.0 / ( min( abs( dir.x ), abs( dir.y ) ) + reduce );
  dir = clamp( dir * rcpMin, -8.0, 8.0 ) * texel;

  vec3 rgbA = 0.5 * (
    texture2D( tex, uv + dir * ( 1.0 / 3.0 - 0.5 ) ).rgb +
    texture2D( tex, uv + dir * ( 2.0 / 3.0 - 0.5 ) ).rgb
  );
  vec3 rgbB = rgbA * 0.5 + 0.25 * (
    texture2D( tex, uv - dir * 0.5 ).rgb +
    texture2D( tex, uv + dir * 0.5 ).rgb
  );
  float lB = fxaaLuma( rgbB );
  return ( lB < lMin || lB > lMax ) ? rgbA : rgbB;
}

vec3 linearToSrgb( vec3 c ) {
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow( max( c, vec3( 1e-5 ) ), vec3( 1.0 / 2.4 ) ) - 0.055;
  return mix( lo, hi, step( vec3( 0.0031308 ), c ) );
}

void main() {
  vec2 texel = 1.0 / uResolution;
  vec2 c = vUv - 0.5;
  float r2 = dot( c, c );

  // Barrel distortion, deliberately at the edge of perceptible.
  vec2 uvD = 0.5 + c * ( 1.0 + uDistortion * r2 );

  vec3 color = uFxaa > 0.5 ? fxaaFilter( tDiffuse, uvD, texel ) : texture2D( tDiffuse, uvD ).rgb;

  // Radial chromatic aberration: zero at centre, grows with r^2. Applied as a
  // per-channel *delta* so it composes with the resolved (FXAA'd) colour rather
  // than throwing that resolve away on two of the three channels.
  float ca = uAberration * r2;
  if ( ca > 0.0 ) {
    vec3 centre = texture2D( tDiffuse, uvD ).rgb;
    color.r += texture2D( tDiffuse, 0.5 + ( uvD - 0.5 ) * ( 1.0 + ca ) ).r - centre.r;
    color.b += texture2D( tDiffuse, 0.5 + ( uvD - 0.5 ) * ( 1.0 - ca ) ).b - centre.b;
  }

  // Unsharp mask. Every console title ships one; without it a 1080p frame
  // upscaled from an internal buffer reads soft.
  if ( uSharpen > 0.0 ) {
    vec3 blur =
      texture2D( tDiffuse, uvD + vec2( texel.x, 0.0 ) ).rgb +
      texture2D( tDiffuse, uvD - vec2( texel.x, 0.0 ) ).rgb +
      texture2D( tDiffuse, uvD + vec2( 0.0, texel.y ) ).rgb +
      texture2D( tDiffuse, uvD - vec2( 0.0, texel.y ) ).rgb;
    blur *= 0.25;
    color = clamp( color + ( color - blur ) * uSharpen, 0.0, 1.0 );
  }

  float vig = 1.0 - uVignette * smoothstep( 0.12, 0.78, r2 );
  color *= vig;

  color = linearToSrgb( color );

  // Film black, and it goes *here* — after the display encode, because a print
  // black is specified as a code value, not as a radiance. sRGB code 3 on a
  // 0..255 scale, tinted very slightly cool, which is what the base density of
  // a projection print actually measures. Applied as a range compression rather
  // than an add, so the top of the scale is untouched and the frame keeps its
  // white. The whole visible effect is that the darkest pixels stop being
  // absolute zero; it is 2% of what the old linear lift was doing.
  color = color * ( 1.0 - uFilmBlack ) + uFilmBlack * uFilmBlackTint;

  // Grain scales with darkness: film has more visible grain in the toe.
  float lum = lumaOf( color );
  float g = hash12( gl_FragCoord.xy + fract( uTime ) * 431.7 ) - 0.5;
  color += g * uGrain * ( 0.35 + 0.9 * ( 1.0 - lum ) );

  // Ordered dither to kill 8-bit banding in the sky gradient.
  float dither = ( ignNoise( gl_FragCoord.xy ) - 0.5 ) / 255.0;
  gl_FragColor = vec4( clamp( color + dither, 0.0, 1.0 ), 1.0 );
}
`;

export const COPY_FRAG = /* glsl */ `
varying vec2 vUv;
uniform sampler2D tDiffuse;
void main() { gl_FragColor = texture2D( tDiffuse, vUv ); }
`;

// ---------------------------------------------------------------------------
// Sky dome
// ---------------------------------------------------------------------------

export const SKY_VERT = /* glsl */ `
varying vec3 vWorldDir;
void main() {
  vec4 wp = modelMatrix * vec4( position, 1.0 );
  vWorldDir = wp.xyz - cameraPosition;
  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`;

/**
 * Preetham analytic sky with two parallax cloud layers.
 *
 * Ray/Mie constants are the standard Preetham fits for sea-level air; the
 * cloud layers are projected onto flat planes at two altitudes so they take on
 * real perspective foreshortening toward the horizon instead of sliding across
 * a dome like a texture.
 */
export const SKY_FRAG = /* glsl */ `
varying vec3 vWorldDir;
${POST_COMMON}
${ATMOSPHERE_GLSL}
${NOISE_GLSL}
${SKY_UNIFORMS_GLSL}
${SKY_CALL_GLSL}

uniform float uTime;
uniform float uCloudCover;
uniform float uSunDiscIntensity;

/**
 * The solar disc, as a separate term added *after* the sky has been rolled.
 *
 * This is the whole of finding 5. The disc was already the right angular size —
 * 0.53 degrees, with limb darkening — and already carried a radiance four
 * orders of magnitude above the sky around it. Then min( sky, 4.0 ) flattened
 * the disc and three hundred pixels of Mie aureole to the *same constant*, and
 * what reached the frame was a cream plateau with no core, no edge and a radial
 * profile that read 0.979 at the centre and 0.955 a hundred and fifty pixels
 * out. A hole, not a sun. Rolling the sky and adding the disc on top leaves the
 * core three orders of magnitude clear of its surroundings, so it survives the
 * ACES shoulder as a small hard white core and the bloom chain gives it the
 * steep skirt instead of glowing the whole sky uniformly.
 *
 * It is excluded from the IBL — the sun is already a directional light, and
 * putting a 10,000-unit disc into the PMREM as well would double-count the key
 * and blow every specular highlight in the level.
 */
vec3 sunDiscRadiance( vec3 dir, vec3 sunDir ) {
  if ( uSunDiscIntensity <= 0.0 ) return vec3( 0.0 );
  float cosTheta = dot( dir, sunDir );
  const float cosDisc = 0.999956;   // ~0.53 degrees
  const float softEdge = 0.0000075; // roughly one pixel at this field of view
  float disc = smoothstep( cosDisc, cosDisc + softEdge, cosTheta );
  if ( disc <= 0.0 ) return vec3( 0.0 );
  // Limb darkening across the disc itself, not across its antialiased edge:
  // the rim of the photosphere is genuinely ~35% down on the centre.
  float r = sat( ( 1.0 - cosTheta ) / ( 1.0 - cosDisc ) );
  float limb = 0.35 + 0.65 * sqrt( max( 0.0, 1.0 - r * r ) );
  float zenithAngle = acos( clamp( sunDir.y, -1.0, 1.0 ) );
  vec3 Fex = atmExtinction( zenithAngle, uTurbidity, uRayleigh, uMieCoefficient );
  return atmSunE( sunDir ) * uSunDiscIntensity * Fex * disc * limb;
}

/**
 * One cloud layer projected onto a flat plane at the given altitude.
 *
 * Returns ( sunlight reaching this point, coverage alpha, thin-edge factor ).
 *
 * A cloud is not an alpha mask with a gradient on it — it is a volume, and what
 * makes it read as one is that light entering the sun-facing side has to travel
 * through the volume to reach the side you are looking at. So instead of one
 * gradient tap, this walks the density field toward the sun and accumulates
 * optical depth, then applies Beer's law. The result is a dark base and a hot
 * sunward shoulder on the *same* cloud, which is the whole difference between a
 * cloud and a wisp of grey paint.
 */
vec4 cloudLayer( vec3 dir, float height, float scale, vec2 drift, float cover, float sharpness ) {
  if ( dir.y <= 0.012 ) return vec4( 0.0 );
  float t = height / dir.y;
  if ( t > 90000.0 ) return vec4( 0.0 );

  vec2 p = ( dir.xz * t ) * scale + drift * uTime;

  float d = fbm5( p );
  // A second, slower-moving warp keeps the layer from reading as a flat mask.
  d = mix( d, fbm3( p * 0.47 + vec2( 11.3, -4.7 ) ), 0.35 );

  float density = sat( ( d - cover ) * sharpness );
  if ( density <= 0.001 ) return vec4( 0.0 );

  // Three taps along the sun's projection into the layer's plane, weighted so
  // the near ones dominate. fbm3 rather than fbm5 — self-shadowing needs the
  // low frequencies of the field, not its detail, and this keeps the layer at
  // roughly the cost of the single fbm5 tap it replaces.
  vec2 sunStep = normalize( uSunDir.xz + vec2( 1e-4 ) ) * 0.19;
  float od =
    sat( ( fbm3( p + sunStep ) - cover ) * sharpness ) * 1.00 +
    sat( ( fbm3( p + sunStep * 2.3 ) - cover ) * sharpness ) * 0.62 +
    sat( ( fbm3( p + sunStep * 4.4 ) - cover ) * sharpness ) * 0.34;

  // Beer's law through the accumulated depth, plus a self-occlusion term from
  // the point's own density so the middle of a thick cell is darker than its
  // shoulder even when nothing upwind of it is.
  float lightThrough = exp( -( od * 1.5 + density * 0.9 ) );

  // Thin edges: where the cell has only just crossed the coverage threshold,
  // light passes almost straight through. This is the term that becomes the
  // silver lining when the sun is behind that edge.
  float thin = ( 1.0 - density ) * smoothstep( 0.0, 0.28, density );

  float horizonFade = smoothstep( 0.012, 0.14, dir.y );
  float distFade = 1.0 - sat( ( t * scale - 26.0 ) / 60.0 );

  return vec4( lightThrough, density * horizonFade * mix( 0.35, 1.0, distFade ), thin, 0.0 );
}

void main() {
  vec3 dir = normalize( vWorldDir );

  // Atmosphere, ground hemisphere, horizon band and roll — the exact call the
  // composite's aerial perspective and the SSR fallback make.
  vec3 horizonCol;
  vec3 sky = skyDomeRadianceH(
    dir, uSunDir, uTurbidity, uRayleigh, uMieCoefficient, uMieG,
    uSkyIntensity, uGroundColor, uSkyRollKnee, uSkyRollMax, horizonCol
  );

  if ( dir.y > 0.012 ) {
    float cosSun = dot( dir, uSunDir );
    float sunAmt = sat( cosSun * 0.5 + 0.5 );

    vec4 low = cloudLayer( dir, 1.0, 1.35, vec2( 0.0042, 0.0017 ), uCloudCover, 3.4 );
    vec4 high = cloudLayer( dir, 2.6, 0.62, vec2( 0.0115, -0.0038 ), uCloudCover + 0.12, 2.2 );

    // Shadowed cloud is not grey: its underside is lit by the sky above it and
    // by the ground below, so it goes blue-grey with a warm floor. The sunward
    // shoulder takes the sun's own colour, warm and well over unity so it can
    // clip into the bloom the way a real cloud edge does at this hour.
    vec3 shadeLow = vec3( 0.20, 0.23, 0.31 ) * uSkyIntensity + horizonCol * 0.25;
    vec3 shadeHigh = vec3( 0.34, 0.37, 0.45 ) * uSkyIntensity + horizonCol * 0.18;
    vec3 litLow = mix( shadeLow, vec3( 1.55, 1.36, 1.10 ) * uSkyIntensity, low.x );
    vec3 litHigh = mix( shadeHigh, vec3( 1.32, 1.22, 1.08 ) * uSkyIntensity, high.x );

    // Silver lining. Forward scatter through a thin edge, so it only appears
    // where the sun is actually behind that edge, and it falls off sharply with
    // angle rather than glowing everywhere on the sunward half of the sky.
    float forward = pow( sat( cosSun ), 7.0 );
    vec3 silver = vec3( 1.9, 1.62, 1.24 ) * uSkyIntensity * forward;
    litLow += silver * low.z * 2.6;
    litHigh += silver * high.z * 1.8;

    litLow *= mix( 0.88, 1.22, sunAmt );
    litHigh *= mix( 0.92, 1.16, sunAmt );

    float aHigh = high.y * 0.55;
    float aLow = low.y * 0.94;

    sky = mix( sky, litHigh * 0.85, sat( aHigh ) );
    sky = mix( sky, litLow * 0.9, sat( aLow ) );

    // Cloud tops are the one part of the sky allowed above the roll, so a lit
    // shoulder can still clip into the bloom the way a real one does. The disc
    // is occluded by whatever cloud is in front of it.
    sky = skyRoll( sky, uSkyRollKnee, uSkyRollMax * 1.35 );
    sky += sunDiscRadiance( dir, uSunDir ) * ( 1.0 - sat( aLow ) ) * ( 1.0 - sat( aHigh ) * 0.75 );
  } else {
    sky += sunDiscRadiance( dir, uSunDir );
  }

  gl_FragColor = vec4( max( vec3( 0.0 ), sky ), 1.0 );
}
`;

/**
 * Cascade coverage helper appended to three's common chunk so every lit
 * material can weight its directional lights by cascade.
 */
export const CSM_COVERAGE_GLSL = /* glsl */ `
float csmBoxCoverage( vec4 sc ) {
  vec3 p = sc.xyz / sc.w;
  if ( p.z < 0.0 || p.z > 1.0 ) return 0.0;
  vec2 e = abs( p.xy - 0.5 ) * 2.0;
  float m = max( e.x, e.y );
  return 1.0 - smoothstep( 0.84, 1.0, m );
}
`;

/** Builds the snippet injected into the directional-light loop. */
export function csmLoopSnippet(cascades: number): string {
  const lines: string[] = [];
  lines.push(
    `#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS ) && ( NUM_DIR_LIGHT_SHADOWS == ${cascades} )`,
  );
  lines.push('{');
  lines.push('float csmW = 1.0;');
  for (let k = 1; k < cascades; k++) {
    lines.push(`#if UNROLLED_LOOP_INDEX > ${k - 1}`);
    lines.push(`csmW *= 1.0 - csmBoxCoverage( vDirectionalShadowCoord[ UNROLLED_LOOP_INDEX - ${k} ] );`);
    lines.push('#endif');
  }
  // The last cascade keeps whatever the tighter cascades did not claim, so
  // geometry beyond the shadow range stays lit instead of falling to black.
  lines.push(`#if UNROLLED_LOOP_INDEX < ${cascades - 1}`);
  lines.push('csmW *= csmBoxCoverage( vDirectionalShadowCoord[ UNROLLED_LOOP_INDEX ] );');
  lines.push('#endif');
  lines.push('directLight.color *= csmW;');
  lines.push('}');
  lines.push('#endif');
  return lines.join('\n');
}
