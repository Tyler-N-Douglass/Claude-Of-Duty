/**
 * Travelling tracers.
 *
 * A tracer is not a line drawn from muzzle to impact — it is a lit projectile
 * that leaves the barrel and covers the distance at the round's real muzzle
 * velocity. The quad is stretched along the flight axis and billboarded around
 * it, so it reads as a streak from any angle without ever facing the camera
 * flat. Simulation is entirely in the vertex shader from the spawn time.
 */
import * as THREE from 'three';

const TRACER_VERT = /* glsl */ `
precision highp float;

attribute vec3 aStart;
attribute vec3 aDir;
attribute vec4 aParams; // spawnTime, speed, distance, width
attribute vec4 aStyle;  // tailLength, seed, intensity, flightFadeExponent
attribute vec3 aColor;

uniform float uTime;

varying vec2 vLine;   // x: 0 at tail, 1 at head; y: -1..1 across the streak
varying vec4 vColor;  // rgb + intensity

void main() {
  float t = uTime - aParams.x;
  float speed = aParams.y;
  float dist = aParams.z;
  float tail = aStyle.x;
  float travel = t * speed;

  if (t < 0.0 || travel - tail > dist) {
    vLine = vec2(0.0);
    vColor = vec4(0.0);
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  float head = min(travel, dist);
  float back = max(travel - tail, 0.0);
  float s = mix(back, head, position.x);

  vec3 dir = normalize(aDir);
  vec3 wp = aStart + dir * s;

  vec3 toCam = cameraPosition - wp;
  float camDist = length(toCam);
  vec3 side = cross(dir, toCam / max(camDist, 1e-4));
  float sl = length(side);
  // Looking straight down the flight path there is no stable side vector; fall
  // back to any perpendicular so the quad never degenerates to a point.
  if (sl < 1e-4) {
    side = abs(dir.y) > 0.9 ? vec3(1.0, 0.0, 0.0) : normalize(cross(dir, vec3(0.0, 1.0, 0.0)));
  } else {
    side /= sl;
  }

  // Keep a minimum apparent thickness at range, otherwise distant tracers
  // alias into a dotted line.
  float width = aParams.w * max(1.0, camDist * 0.012);
  wp += side * (position.y * width);

  float u = clamp(travel / max(dist, 0.001), 0.0, 1.0);
  float flight = 1.0 - pow(u, max(aStyle.w, 0.1));
  // The last stretch before impact dims fast rather than popping out.
  flight *= 1.0 - smoothstep(0.92, 1.0, u);

  vLine = vec2(position.x, position.y * 2.0);
  vColor = vec4(aColor, aStyle.z * flight);
  gl_Position = projectionMatrix * viewMatrix * vec4(wp, 1.0);
}
`;

const TRACER_FRAG = /* glsl */ `
precision highp float;

varying vec2 vLine;
varying vec4 vColor;

void main() {
  float cross_ = abs(vLine.y);
  if (cross_ > 1.0 || vColor.a <= 0.0) discard;
  float falloff = 1.0 - cross_;
  float core = pow(falloff, 14.0);
  float glow = pow(falloff, 2.6);
  float along = vLine.x;
  // Bright at the head, trailing off toward the tail.
  float tail = smoothstep(0.0, 0.45, along);
  float tip = pow(along, 8.0);
  float a = (core * 1.9 + glow * 0.36) * tail;
  vec3 rgb = mix(vColor.rgb, vec3(1.0, 0.96, 0.88), min(1.0, core * 0.85 + tip * 0.6));
  rgb *= vColor.a * (1.0 + tip * 1.6);
  float alpha = a * vColor.a;
  if (alpha < 0.002) discard;
  gl_FragColor = vec4(rgb * alpha, alpha);
}
`;

interface TracerRecord {
  active: boolean;
  /** Absolute fx-clock time the round reaches its closest point to the camera. */
  crackAt: number;
  crackDone: boolean;
  crackDistance: number;
  crackX: number;
  crackY: number;
  crackZ: number;
  endAt: number;
}

const _dir = new THREE.Vector3();
const _rel = new THREE.Vector3();
const _closest = new THREE.Vector3();

export interface TracerSpawn {
  from: THREE.Vector3;
  to: THREE.Vector3;
  speed: number;
  /** 0..1 — bright ammunition tracer vs. faint vapour streak. */
  intensity: number;
  color: THREE.Color;
  width: number;
  tail: number;
}

export type CrackCallback = (x: number, y: number, z: number, distance: number, speed: number) => void;

export class Tracers {
  private readonly geometry: THREE.InstancedBufferGeometry;
  private readonly material: THREE.ShaderMaterial;
  private readonly mesh: THREE.Mesh;
  private readonly aStart: THREE.InstancedBufferAttribute;
  private readonly aDir: THREE.InstancedBufferAttribute;
  private readonly aParams: THREE.InstancedBufferAttribute;
  private readonly aStyle: THREE.InstancedBufferAttribute;
  private readonly aColor: THREE.InstancedBufferAttribute;
  private readonly records: TracerRecord[] = [];
  private readonly capacity: number;
  private cursor = 0;
  private time = 0;
  private liveUntil = -1;

  /** Called when a round passes close enough to the camera to crack the air. */
  onCrack: CrackCallback | null = null;

  constructor(private readonly scene: THREE.Scene, capacity: number) {
    this.capacity = Math.max(16, capacity);

    this.geometry = new THREE.InstancedBufferGeometry();
    this.geometry.setAttribute('position', new THREE.Float32BufferAttribute(
      [0, -0.5, 0, 1, -0.5, 0, 1, 0.5, 0, 0, 0.5, 0], 3,
    ));
    this.geometry.setIndex([0, 1, 2, 0, 2, 3]);
    this.geometry.instanceCount = 0;
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);

    const mk = (size: number): THREE.InstancedBufferAttribute => {
      const a = new THREE.InstancedBufferAttribute(new Float32Array(this.capacity * size), size);
      a.setUsage(THREE.DynamicDrawUsage);
      return a;
    };
    this.aStart = mk(3);
    this.aDir = mk(3);
    this.aParams = mk(4);
    this.aStyle = mk(4);
    this.aColor = mk(3);
    this.geometry.setAttribute('aStart', this.aStart);
    this.geometry.setAttribute('aDir', this.aDir);
    this.geometry.setAttribute('aParams', this.aParams);
    this.geometry.setAttribute('aStyle', this.aStyle);
    this.geometry.setAttribute('aColor', this.aColor);

    this.material = new THREE.ShaderMaterial({
      vertexShader: TRACER_VERT,
      fragmentShader: TRACER_FRAG,
      uniforms: { uTime: { value: 0 } },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      toneMapped: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      side: THREE.DoubleSide,
    });

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'fx-tracers';
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.renderOrder = 13;
    this.mesh.visible = false;
    scene.add(this.mesh);

    for (let i = 0; i < this.capacity; i++) {
      this.records.push({
        active: false, crackAt: 0, crackDone: true, crackDistance: 1e9,
        crackX: 0, crackY: 0, crackZ: 0, endAt: 0,
      });
    }
  }

  spawn(s: TracerSpawn, cameraPosition: THREE.Vector3): void {
    _dir.subVectors(s.to, s.from);
    const distance = _dir.length();
    if (distance < 0.05) return;
    _dir.multiplyScalar(1 / distance);

    const speed = Math.max(60, s.speed);
    const i = this.cursor;
    this.cursor = (i + 1) % this.capacity;

    const p3 = i * 3;
    const p4 = i * 4;
    const st = this.aStart.array as Float32Array;
    st[p3] = s.from.x; st[p3 + 1] = s.from.y; st[p3 + 2] = s.from.z;
    const dr = this.aDir.array as Float32Array;
    dr[p3] = _dir.x; dr[p3 + 1] = _dir.y; dr[p3 + 2] = _dir.z;

    // A real tracer streak is roughly the distance the round covers in one
    // frame of persistence — long enough to read, short enough to stay a round.
    const tail = s.tail > 0 ? s.tail : THREE.MathUtils.clamp(speed * 0.022, 3, 22);
    const pr = this.aParams.array as Float32Array;
    pr[p4] = this.time; pr[p4 + 1] = speed; pr[p4 + 2] = distance; pr[p4 + 3] = s.width;
    const sy = this.aStyle.array as Float32Array;
    sy[p4] = tail; sy[p4 + 1] = Math.random(); sy[p4 + 2] = s.intensity; sy[p4 + 3] = 2.2;
    const cl = this.aColor.array as Float32Array;
    cl[p3] = s.color.r; cl[p3 + 1] = s.color.g; cl[p3 + 2] = s.color.b;

    for (const a of [this.aStart, this.aDir, this.aParams, this.aStyle, this.aColor]) {
      a.addUpdateRange(i * a.itemSize, a.itemSize);
      a.needsUpdate = true;
    }

    // Closest approach of the segment to the camera, for the supersonic crack.
    _rel.subVectors(cameraPosition, s.from);
    const proj = THREE.MathUtils.clamp(_rel.dot(_dir), 0, distance);
    _closest.copy(s.from).addScaledVector(_dir, proj);
    const miss = _closest.distanceTo(cameraPosition);

    const rec = this.records[i]!;
    rec.active = true;
    rec.crackDistance = miss;
    rec.crackAt = this.time + proj / speed;
    rec.crackX = _closest.x;
    rec.crackY = _closest.y;
    rec.crackZ = _closest.z;
    // Only rounds that were not fired from the player's own muzzle can crack;
    // 2.5m of travel is well past the barrel.
    rec.crackDone = !(miss < 7 && proj > 2.5 && speed > 220);
    rec.endAt = this.time + (distance + tail) / speed;

    if (rec.endAt > this.liveUntil) this.liveUntil = rec.endAt;
  }

  update(dt: number): void {
    this.time += dt;
    this.material.uniforms.uTime.value = this.time;

    const alive = this.time < this.liveUntil;
    this.mesh.visible = alive;
    this.geometry.instanceCount = alive ? this.capacity : 0;

    if (!alive) return;
    for (let i = 0; i < this.capacity; i++) {
      const rec = this.records[i]!;
      if (!rec.active) continue;
      if (!rec.crackDone && this.time >= rec.crackAt) {
        rec.crackDone = true;
        this.onCrack?.(rec.crackX, rec.crackY, rec.crackZ, rec.crackDistance, 0);
      }
      if (this.time > rec.endAt) rec.active = false;
    }
  }

  dispose(): void {
    this.scene.remove(this.mesh);
    this.geometry.dispose();
    this.material.dispose();
    this.records.length = 0;
    this.onCrack = null;
  }
}
