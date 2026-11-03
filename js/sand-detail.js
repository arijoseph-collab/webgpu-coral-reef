// -----------------------------------------------------------------------------
// Procedural sand detail (art-director item 4 / undersea P5): one tileable 512x512 RGBA8 texture generated
// at load (no download), sampled at two scales by the seabed shader:
//   R  low-frequency fbm albedo patches (detritus / turf / biofilm)
//   G  ripple amplitude (patchy ripple height, so the crests come and go)
//   B  fine grain hash
//   A  Voronoi flecks (coralline crust / shell grit / rubble mottling)
// -----------------------------------------------------------------------------
import * as THREE from "three/webgpu";

export function makeSandDetailTexture(size = 512, seed = 0x5a4d) {
  let s = seed >>> 0;
  const rnd = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  // tileable value-noise lattice of period P (cells across the texture)
  const lattice = (P) => {
    const g = new Float32Array(P * P);
    for (let i = 0; i < g.length; i++) g[i] = rnd();
    return (u, v) => {
      const x = u * P, y = v * P;
      const xi = Math.floor(x), yi = Math.floor(y);
      const fx = x - xi, fy = y - yi;
      const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
      const x0 = ((xi % P) + P) % P, y0 = ((yi % P) + P) % P, x1 = (x0 + 1) % P, y1 = (y0 + 1) % P;
      const a = g[y0 * P + x0], b = g[y0 * P + x1], c = g[y1 * P + x0], d = g[y1 * P + x1];
      return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy;
    };
  };
  const albedoOct = [lattice(3), lattice(6), lattice(12), lattice(24), lattice(48)];
  const rippleOct = [lattice(2), lattice(5), lattice(10)];
  const fbm = (oct, u, v) => {
    let sum = 0, amp = 0.5, norm = 0;
    for (const n of oct) {
      sum += n(u, v) * amp;
      norm += amp;
      amp *= 0.55;
    }
    return sum / norm;
  };
  // tileable Voronoi feature points, VP cells across
  const VP = 28;
  const feat = new Float32Array(VP * VP * 3);
  for (let i = 0; i < VP * VP; i++) {
    feat[i * 3] = rnd();
    feat[i * 3 + 1] = rnd();
    feat[i * 3 + 2] = rnd();
  }
  const data = new Uint8Array(size * size * 4);
  for (let j = 0; j < size; j++) {
    const v = j / size;
    for (let i = 0; i < size; i++) {
      const u = i / size;
      const o = (j * size + i) * 4;
      // R: albedo patches, contrast-stretched
      const a = fbm(albedoOct, u, v);
      data[o] = Math.max(0, Math.min(255, (0.5 + (a - 0.5) * 2.2) * 255));
      // G: ripple amplitude (broad, mostly present)
      const r = fbm(rippleOct, u + 0.37, v + 0.11);
      data[o + 1] = Math.max(0, Math.min(255, (0.35 + (r - 0.5) * 2.4) * 255));
      // B: grain
      data[o + 2] = Math.floor(rnd() * 255);
      // A: Voronoi flecks (only ~35% of the cells carry a fleck)
      const x = u * VP, y = v * VP, xi = Math.floor(x), yi = Math.floor(y);
      let best = 9, bestRnd = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const cx = xi + dx, cy = yi + dy;
          const k = ((((cy % VP) + VP) % VP) * VP + (((cx % VP) + VP) % VP)) * 3;
          const px = cx + 0.15 + feat[k] * 0.7, py = cy + 0.15 + feat[k + 1] * 0.7;
          const d = Math.hypot(x - px, y - py);
          if (d < best) (best = d), (bestRnd = feat[k + 2]);
        }
      const fleck = bestRnd > 0.65 ? Math.max(0, 1 - best / (0.18 + bestRnd * 0.22)) : 0;
      data[o + 3] = Math.min(255, Math.floor(Math.pow(fleck, 0.7) * 255));
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  tex.name = "sand_detail";
  return tex;
}

/**
 * Coral footprint texture (from env.terrainInfo.footprint): R = horizontal distance to coral (m),
 * G = highest coral point above (m, 0 if none). Half-float so it filters on every WebGPU device.
 * Created empty at startup (materials compile before the GLB arrives) and filled by fillFootprintTexture().
 */
export function makeFootprintTexture(G = 128) {
  const tex = new THREE.DataTexture(new Uint16Array(G * G * 4), G, G, THREE.RGBAFormat, THREE.HalfFloatType);
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.NoColorSpace;
  tex.name = "coral_footprint";
  // far from any coral until filled
  const d = tex.image.data;
  for (let i = 0; i < G * G; i++) d[i * 4] = THREE.DataUtils.toHalfFloat(9);
  tex.needsUpdate = true;
  return tex;
}

export function fillFootprintTexture(tex, fp) {
  const d = tex.image.data;
  const n = fp.G * fp.G;
  for (let i = 0; i < n; i++) {
    d[i * 4] = THREE.DataUtils.toHalfFloat(Math.min(fp.dist[i], 60));
    d[i * 4 + 1] = THREE.DataUtils.toHalfFloat(Math.max(0, fp.top[i]));
    d[i * 4 + 2] = THREE.DataUtils.toHalfFloat(fp.occ[i]);
  }
  tex.needsUpdate = true;
}
