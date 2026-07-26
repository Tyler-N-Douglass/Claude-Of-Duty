import type { QualityPreset, QualitySettings } from './Contracts';

const PRESETS: Record<QualityPreset, Omit<QualitySettings, 'preset'>> = {
  low: {
    renderScale: 0.7, shadowMapSize: 1024, shadowCascades: 2, ssao: false, ssr: false,
    bloom: true, motionBlur: false, taa: false, volumetrics: false, anisotropy: 4,
    particleBudget: 1500, decalBudget: 64, targetFps: 60,
  },
  medium: {
    renderScale: 0.85, shadowMapSize: 2048, shadowCascades: 3, ssao: true, ssr: false,
    bloom: true, motionBlur: true, taa: true, volumetrics: false, anisotropy: 8,
    particleBudget: 4000, decalBudget: 128, targetFps: 60,
  },
  high: {
    renderScale: 1.0, shadowMapSize: 2048, shadowCascades: 4, ssao: true, ssr: true,
    bloom: true, motionBlur: true, taa: true, volumetrics: true, anisotropy: 16,
    particleBudget: 8000, decalBudget: 256, targetFps: 60,
  },
  ultra: {
    renderScale: 1.0, shadowMapSize: 4096, shadowCascades: 4, ssao: true, ssr: true,
    bloom: true, motionBlur: true, taa: true, volumetrics: true, anisotropy: 16,
    particleBudget: 16000, decalBudget: 512, targetFps: 60,
  },
};

export function createQuality(preset: QualityPreset = 'high'): QualitySettings {
  return { preset, ...PRESETS[preset] };
}

export function applyPreset(q: QualitySettings, preset: QualityPreset): void {
  Object.assign(q, PRESETS[preset], { preset });
}

/**
 * Picks a starting preset from the GPU string and screen size. Deliberately
 * conservative: the adaptive resolution controller will claw quality back up.
 */
export function detectPreset(renderer: { getContext(): WebGLRenderingContext | WebGL2RenderingContext }): QualityPreset {
  try {
    const gl = renderer.getContext();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const gpu = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : '';
    const s = gpu.toLowerCase();
    // Software rasterisers (SwiftShader in headless CI, llvmpipe on a bare
    // Linux box) are one to two orders of magnitude slower than any real GPU.
    // Treating them as low-end is the difference between a capture that
    // completes and one that times out mid-frame.
    if (/swiftshader|llvmpipe|softpipe|software|basic render/.test(s)) return 'low';
    const mobile = /adreno|mali|apple gpu|powervr/.test(s);
    if (mobile) return 'low';
    if (/rtx\s*(40|50)|rx\s*7[89]|m[123]\s*(max|ultra)/.test(s)) return 'ultra';
    if (/rtx|radeon rx|apple m\d/.test(s)) return 'high';
    if (/intel|uhd|iris/.test(s)) return 'medium';
  } catch {
    /* fall through */
  }
  return 'high';
}
