// -----------------------------------------------------------------------------
// OceanSurface — simulated open-water surface + physically based caustics
//
// Adapted from jeantimex/webgpu-water (MIT, https://github.com/jeantimex/webgpu-water), the WebGPU
// port of Evan Wallace's WebGL Water (MIT). The same three GPU passes, re-built on three.js r172
// render targets + TSL and changed from a closed pool to a periodic open-water tile:
//   1. update   - heightfield wave equation (velocity += (neighbour average - height) * 2, damping) on a
//                 periodic grid, excited by cosine drops (drop.frag.wgsl): user clicks and optional rain.
//                 The simulation only runs while ripples are alive, so it costs nothing when calm.
//   2. normals  - surface = spectral wind waves (a deterministic set of periodic deep-water waves, slope
//                 computed analytically) + simulated ripples (central differences).
//   3. caustics - a refracted projected grid (caustics.vert/frag.wgsl): every surface vertex refracts the
//                 sun (Snell, eta = 1/1.333) and is projected to a plane `depth` metres below the surface;
//                 intensity = flat area / refracted area (screen-space derivatives, additive blending).
// Consumers sample the periodic caustic tile by tracing a world point back to the surface along the
// flat refracted sun direction (see causticUV()), so the pattern lands consistently on corals and sand.
// -----------------------------------------------------------------------------
import * as THREE from "three/webgpu";
import {
  Fn,
  uniform,
  texture,
  uv,
  vec2,
  vec3,
  vec4,
  float,
  positionGeometry,
  varying,
  dFdx,
  dFdy,
  abs,
  max,
  min,
  sqrt,
  sin,
  cos,
  dot,
  normalize,
  refract,
  uniformArray,
  fract,
  length,
  clamp,
  log2,
  mix,
  smoothstep,
  instanceIndex,
} from "three/tsl";

// Seeded PRNG (mulberry32) so waves and ripples are identical on every load (deterministic screenshots)
function rng32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const MAX_DROPS = 4;
const _c = new THREE.Color();
const SIM_HZ = 120;

/**
 * Deterministic directional wind-sea: `count` deep-water waves with integer wave numbers per tile (so the
 * surface tiles seamlessly), log-uniform wavelengths in [minLambda, maxLambda], spread around `windDir`.
 * Amplitudes follow a k^-2 law (every scale contributes equally to surface curvature, i.e. to focusing,
 * which gives the characteristic multi-scale caustic network) and are normalised so the RMS surface
 * curvature focuses refracted sunlight at roughly `depth` metres times `focus`.
 */
function makeWindSea(o) {
  const rand = rng32(o.seed);
  const L = o.tileSize;
  const used = new Set();
  const waves = [];
  for (let guard = 0; waves.length < o.count && guard < 20000; guard++) {
    const lam = Math.exp(Math.log(o.minLambda) + rand() * (Math.log(o.maxLambda) - Math.log(o.minLambda)));
    const dev = (rand() + rand() + rand() - 1.5) * o.spread; // ~gaussian directional spreading
    const th = o.windDir + dev + (rand() < 0.18 ? Math.PI * 0.5 * (rand() < 0.5 ? -1 : 1) : 0); // some cross-sea
    const kc = L / lam;
    const m = Math.round(Math.cos(th) * kc);
    const n = Math.round(Math.sin(th) * kc);
    if (m === 0 && n === 0) continue;
    if (used.has(`${m},${n}`) || used.has(`${-m},${-n}`)) continue;
    used.add(`${m},${n}`);
    const kLen = (2 * Math.PI * Math.hypot(m, n)) / L; // rad / m
    waves.push({ m, n, kLen, amp: (0.7 + 0.6 * rand()) / (kLen * kLen), phase: rand() * Math.PI * 2, omega: Math.sqrt(9.81 * kLen) });
  }
  // RMS Laplacian of the height field = RMS curvature; deflection ~ (1 - 1/1.333) * slope
  let curv2 = 0;
  for (const w of waves) curv2 += (w.amp * w.kLen * w.kLen) ** 2 / 2;
  const norm = o.focus / (0.25 * o.depth * Math.sqrt(curv2));
  for (const w of waves) w.amp *= norm;
  return waves;
}

export class OceanSurface {
  /**
   * @param {THREE.WebGPURenderer} renderer
   * @param {object} o
   * @param {number} o.tileSize     world size (m) of one periodic surface / caustic tile
   * @param {number} o.simSize      heightfield resolution (texels per tile side)
   * @param {number} o.causticSize  caustic texture resolution
   * @param {number} o.gridSize     caustic projection grid resolution (quads per tile side)
   * @param {number} o.depth        depth (m) below the surface where the caustic pattern is focused
   * @param {THREE.Vector3} o.sunAir  unit vector towards the sun above the water
   * @param {object} o.surfaceY     TSL uniform: world height of the water surface
   * @param {object} o.sunDirWater  TSL uniform: refracted (underwater) unit vector towards the sun
   */
  constructor(renderer, o) {
    this.renderer = renderer;
    this.tileSize = o.tileSize ?? 7.5;
    this.simSize = o.simSize ?? 256;
    this.causticSize = o.causticSize ?? 1024;
    this.gridSize = o.gridSize ?? 230;
    // The tile frame is rotated against the world so the 7.5 m repeat never lines up with the scene axes
    this.tileRot = ((o.tileRotationDeg ?? 23) * Math.PI) / 180;
    this.surfaceY = o.surfaceY;
    this.sunDirWater = o.sunDirWater;
    this.timeScale = o.timeScale ?? 0.6; // slowed down a little: calm, readable caustics
    this.waves = makeWindSea({
      seed: o.seed ?? 0x0cea7,
      tileSize: this.tileSize,
      count: o.waveCount ?? 40,
      minLambda: o.minLambda ?? 0.11,
      maxLambda: o.maxLambda ?? 0.55,
      windDir: o.windDir ?? 0.6,
      spread: o.windSpread ?? 1.2,
      depth: o.focusDepth ?? 4.0, // refracted sunlight comes into focus ~4 m down: crisp lines on the sand,
      focus: o.focus ?? 1.0, //       softer, broader bands on the coral tops (real depth behaviour)
    });

    this.uTexel = uniform(1 / this.simSize);
    this.uDamping = uniform(o.damping ?? 0.988);
    this.uSimAmp = uniform(o.rippleAmplitude ?? 0.024); // metres of surface height per unit sim height
    this._baseWaveAmp = o.waveAmplitude ?? 1.0;
    this.uWaveAmp = uniform(this._baseWaveAmp); // scales the whole wind sea (1 = calibrated focus)
    this.uWaveTime = uniform(0);
    // Caustic layers (one instanced draw): R = deep layer at sand depth, G = shallow layer at coral-top depth.
    // (Per-wavelength IOR layers were tried: real water dispersion gives ~5 cm rainbow fringes at 4 m without
    // the sun-disk blur, and two extra layers doubled the raster cost - both experts vetoed rainbow fringes.)
    this.depthDeep = o.depthDeep ?? 4.3;
    this.depthShallow = o.depthShallow ?? 2.4;
    this.uLayers = uniformArray([new THREE.Vector4(this.depthDeep, 1 / 1.336, 0, 0), new THREE.Vector4(this.depthShallow, 1 / 1.336, 0, 0)], "vec4");
    this.uMasks = uniformArray([new THREE.Vector4(1, 0, 0, 0), new THREE.Vector4(0, 1, 0, 0)], "vec4");
    this.uSunAir = uniform(new THREE.Vector3()); // sun direction (air) expressed in the rotated tile frame
    this.setSun(o.sunAir);
    this.uDrops = uniformArray(new Array(MAX_DROPS).fill(0).map(() => new THREE.Vector4(0, 0, 0.05, 0)), "vec4");

    const rtOpts = () => ({
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      wrapS: THREE.RepeatWrapping,
      wrapT: THREE.RepeatWrapping,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      generateMipmaps: false,
    });
    this.simA = new THREE.RenderTarget(this.simSize, this.simSize, rtOpts());
    this.simB = new THREE.RenderTarget(this.simSize, this.simSize, rtOpts());
    // Surface (height, nx, nz) at exactly the caustic grid resolution: every grid vertex lands on a texel
    // centre and reads the exact analytic wind-sea normal (no interpolation -> no beaded focal lines).
    // (linear filtering is still exact for the grid: every vertex samples a texel centre)
    this.surfaceRT = new THREE.RenderTarget(this.gridSize, this.gridSize, rtOpts());
    // Mipmapped (regenerated after every caustic pass): distant sand doesn't shimmer, and consumers blur
    // the pattern with a mip bias where it is out of focus.
    this.causticRT = new THREE.RenderTarget(this.causticSize, this.causticSize, {
      ...rtOpts(),
      format: THREE.RGFormat,
      minFilter: THREE.LinearMipmapLinearFilter,
      generateMipmaps: true,
    });
    // 4x MSAA raster target: at the focal lines the refracted grid collapses into sub-pixel slivers;
    // single-sample rasterisation only lights the pixels whose centres they happen to cover (beaded, dotted
    // lines), while multisampled additive blending integrates their coverage (smooth filaments). r172 cannot
    // create a multisampled texture with mips, so the resolved image is copied into the mipmapped causticRT.
    this.causticSamples = o.causticSamples ?? 4;
    if (this.causticSamples > 1) {
      this.causticMSRT = new THREE.RenderTarget(this.causticSize, this.causticSize, {
        ...rtOpts(),
        format: THREE.RGFormat,
        samples: this.causticSamples,
      });
      this.causticMSRT.texture.name = "ocean";
      const copyMat = new THREE.MeshBasicNodeMaterial({ name: "ocean_caustic_resolve", depthTest: false, depthWrite: false });
      const src = texture(this.causticMSRT.texture);
      copyMat.fragmentNode = vec4(src.sample(uv()).xy, 0.0, 1.0);
      this.copyQuad = new THREE.QuadMesh(copyMat);
    }
    for (const rt of [this.simA, this.simB, this.surfaceRT, this.causticRT]) rt.texture.name = "ocean";

    this.updateAB = new THREE.QuadMesh(this._makeUpdateMaterial(this.simA));
    this.updateBA = new THREE.QuadMesh(this._makeUpdateMaterial(this.simB));
    this.normalQuad = new THREE.QuadMesh(this._makeNormalMaterial(this.simA));
    this._makeCausticPass();

    this.rainRate = 0; // drops per second (0 = calm; ripples only from addDrop / dropAt)
    this.clearColor = new THREE.Color(0, 0, 0);
    this.reset();
  }

  // ---- 1. Wave equation step (+ up to MAX_DROPS periodic cosine drops, drop.frag.wgsl) ----
  _makeUpdateMaterial(srcRT) {
    const src = texture(srcRT.texture);
    const mat = new THREE.MeshBasicNodeMaterial({ name: "ocean_update", depthTest: false, depthWrite: false });
    mat.fragmentNode = Fn(() => {
      const p = uv();
      const info = src.sample(p);
      const d = this.uTexel;
      const avg = src
        .sample(p.add(vec2(d, 0)))
        .x.add(src.sample(p.sub(vec2(d, 0))).x)
        .add(src.sample(p.add(vec2(0, d))).x)
        .add(src.sample(p.sub(vec2(0, d))).x)
        .mul(0.25);
      const vel = info.y.add(avg.sub(info.x).mul(2.0)).mul(this.uDamping).toVar();
      let height = info.x.add(vel);
      for (let i = 0; i < MAX_DROPS; i++) {
        const drop = this.uDrops.element(i);
        const offs = fract(p.sub(drop.xy).add(0.5)).sub(0.5); // periodic distance on the tile
        const f = max(float(0.0), float(1.0).sub(length(offs).div(drop.z)));
        height = height.add(float(0.5).sub(f.mul(Math.PI).cos().mul(0.5)).mul(drop.w));
      }
      return vec4(height.mul(this.uDamping), vel, 0.0, 1.0);
    })();
    return mat;
  }

  /**
   * TSL: surface at tile uv `p` -> { h (m), sx, sz (slope dh/dx, dh/dz) }: the spectral wind sea evaluated
   * analytically (exact slopes, no texel interpolation) + the simulated ripples (central differences).
   * Used per vertex by the caustic grid and per texel by the surface pass.
   */
  _surfaceAt(p, src, level = null) {
    const L = this.tileSize;
    let h = float(0.0);
    let sx = float(0.0);
    let sz = float(0.0);
    const t = this.uWaveTime;
    for (const w of this.waves) {
      const ph = p.x.mul(2 * Math.PI * w.m).add(p.y.mul(2 * Math.PI * w.n)).sub(t.mul(w.omega)).add(w.phase).toVar();
      const ak = (w.amp * 2 * Math.PI) / L;
      h = h.add(sin(ph).mul(w.amp));
      const c = cos(ph);
      sx = sx.add(c.mul(ak * w.m));
      sz = sz.add(c.mul(ak * w.n));
    }
    const d = this.uTexel;
    const simH = (q) => (level === null ? src.sample(q) : src.sample(q).level(level)).x;
    const step2 = d.mul(2.0 * L); // metres between the central-difference samples
    const rx = simH(p.add(vec2(d, 0))).sub(simH(p.sub(vec2(d, 0)))).div(step2);
    const rz = simH(p.add(vec2(0, d))).sub(simH(p.sub(vec2(0, d)))).div(step2);
    const a = this.uWaveAmp;
    return {
      h: h.mul(a).add(simH(p).mul(this.uSimAmp)),
      sx: sx.mul(a).add(rx.mul(this.uSimAmp)),
      sz: sz.mul(a).add(rz.mul(this.uSimAmp)),
    };
  }

  // ---- 2. Surface height + normal texture (stored as height, nx, nz), one texel per caustic-grid vertex ----
  _makeNormalMaterial(srcRT) {
    this.normalSrc = texture(srcRT.texture);
    const mat = new THREE.MeshBasicNodeMaterial({ name: "ocean_normals", depthTest: false, depthWrite: false });
    mat.fragmentNode = Fn(() => {
      const f = this._surfaceAt(uv(), this.normalSrc);
      const n = normalize(vec3(f.sx.negate(), 1.0, f.sz.negate()));
      return vec4(f.h, n.x, n.z, 1.0);
    })();
    return mat;
  }

  // ---- 3. Caustics: refracted projected grid with additive blending (caustics.vert/frag.wgsl) ----
  // One instanced draw renders both layers: instance i refracts to depth_i and adds its intensity into
  // channel i only (R: sand-depth layer, G: shallow coral-top layer).
  _makeCausticPass() {
    // Grid covers the tile plus a ~7.5% border (> the largest refracted lateral shift, ~0.3 m at 4.3 m), so
    // light refracted across the tile edge wraps around. The border is a whole number of texels plus one
    // half, so every vertex sits exactly on a texel centre of the surface texture.
    const N = this.gridSize;
    const margin = Math.round(0.075 * N) + 0.5;
    const ext = 1 + (2 * margin) / N;
    const segs = N + 2 * margin;
    const geo = new THREE.PlaneGeometry(ext, ext, segs, segs);
    const L = this.tileSize;
    const mat = new THREE.MeshBasicNodeMaterial({
      name: "ocean_caustics",
      side: THREE.DoubleSide, // focusing folds the grid: back-facing triangles still carry light
      transparent: true,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendEquation: THREE.AddEquation,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneFactor,
      depthTest: false,
      depthWrite: false,
    });
    const layer = this.uLayers.element(instanceIndex);
    const depth = layer.x;
    const eta = layer.y;
    const s = positionGeometry.xy.add(0.5); // tile uv in [-0.075, 1.075]
    // exact analytic surface at every grid vertex (bilinearly interpolated normals made the curvature jump
    // at texel borders, which beaded the focal lines)
    const info = texture(this.surfaceRT.texture).sample(s).level(0);
    const nxz = info.yz;
    const n = vec3(nxz.x, sqrt(max(float(0.0), float(1.0).sub(dot(nxz, nxz)))), nxz.y);
    const incident = this.uSunAir.negate();
    const ray = refract(incident, n, eta);
    const ray0 = refract(incident, vec3(0, 1, 0), eta);
    // flat surface: hits the layer plane at S + ray0 * t0 ; displaced surface: S + h + ray * t
    const t0 = depth.div(ray0.y.negate());
    const t1 = depth.add(info.x).div(ray.y.negate());
    // Tile-uv displacement relative to the flat-water hit of the green (1.336) ray, so all four layers
    // stay registered with the trace-back lookup in causticUV()
    const g0 = refract(incident, vec3(0, 1, 0), 1.0 / 1.336);
    const shift = ray.xz.mul(t1).sub(g0.xz.mul(depth.div(g0.y.negate()))).div(L);
    const uvNew = s.add(shift);
    const qOld = varying(s.mul(L));
    const qNew = varying(uvNew.mul(L));
    const mask = varying(this.uMasks.element(instanceIndex));
    mat.vertexNode = vec4(uvNew.x.mul(2.0).sub(1.0), float(1.0).sub(uvNew.y.mul(2.0)), 0.0, 1.0);
    mat.fragmentNode = Fn(() => {
      // Light lands where refracted cells shrink: intensity = |det J_flat| / |det J_refracted|
      const ox = dFdx(qOld);
      const oy = dFdy(qOld);
      const nx = dFdx(qNew);
      const ny = dFdy(qNew);
      const oldArea = abs(ox.x.mul(oy.y).sub(ox.y.mul(oy.x)));
      const newArea = abs(nx.x.mul(ny.y).sub(nx.y.mul(ny.x)));
      const intensity = min(oldArea.div(max(newArea, float(1e-12))), 24.0);
      return mask.mul(intensity);
    })();
    void t0;
    this.causticMesh = new THREE.Mesh(geo, mat);
    this.causticMesh.count = 2;
    this.causticMesh.frustumCulled = false;
    this.causticScene = new THREE.Scene();
    this.causticScene.add(this.causticMesh);
    this.causticCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1, 1);
  }

  /** Set the sun direction above the water (unit vector towards the sun, world frame). */
  setSun(sunAir) {
    const d = sunAir.clone().normalize();
    const c = Math.cos(this.tileRot), sn = Math.sin(this.tileRot);
    this.uSunAir.value.set(c * d.x - sn * d.z, d.y, sn * d.x + c * d.z);
  }

  _clearSim() {
    const r = this.renderer;
    const prev = r.getRenderTarget();
    const prevColor = r.getClearColor(new THREE.Color());
    const prevAlpha = r.getClearAlpha();
    r.setClearColor(0x000000, 0);
    for (const rt of [this.simA, this.simB]) {
      r.setRenderTarget(rt);
      r.clear(true, false, false);
    }
    r.setRenderTarget(prev);
    r.setClearColor(prevColor, prevAlpha);
    this.steps = 0;
    this.normalSrc.value = this.simA.texture;
  }

  /** Flat water: clear the ripple simulation and the drop/rain schedule. */
  reset() {
    this._clearSim();
    this.rand = rng32(0x0cea7);
    this.simTime = 0;
    this.nextRainTime = 0;
    this.activeUntil = -1; // the ripple simulation steps only while simTime < activeUntil
    this.accum = 0;
    this.lastTime = undefined;
    this._pendingDrops = [];
  }

  /** Queue a ripple at tile uv (u, v) in [0,1); radius in tile units; strength in sim units. */
  addDrop(u, v, radius = 0.03, strength = 0.6) {
    if (this._pendingDrops.length < MAX_DROPS * 8) this._pendingDrops.push([u - Math.floor(u), v - Math.floor(v), radius, strength]);
    this.activeUntil = Math.max(this.activeUntil, this.simTime + 6.0);
  }

  /**
   * Drop a ripple on the surface so that its caustic ring is centred on world point `p` (THREE.Vector3):
   * the surface point is found by tracing back along the refracted sun direction.
   */
  dropAt(p, radiusMeters = 0.12, strength = 1.0) {
    const s = this.sunDirWater.value;
    const toSurf = Math.max(0, this.surfaceY.value - p.y) / Math.max(0.25, s.y);
    this.addDrop((p.x + s.x * toSurf) / this.tileSize, (p.z + s.z * toSurf) / this.tileSize, radiusMeters / this.tileSize, strength);
  }

  _rain(time) {
    if (this.rainRate <= 0) {
      this.nextRainTime = time;
      return;
    }
    while (this.nextRainTime <= time) {
      this.nextRainTime += 1 / this.rainRate;
      const r = this.rand;
      this.addDrop(r(), r(), 0.011 + r() * 0.013, (1.35 + r() * 0.85) * (r() < 0.5 ? 1 : -1));
    }
  }

  _step() {
    const r = this.renderer;
    const pend = this._pendingDrops;
    for (let i = 0; i < MAX_DROPS; i++) {
      const d = pend[i];
      this.uDrops.array[i].set(d ? d[0] : 0, d ? d[1] : 0, d ? d[2] : 0.05, d ? d[3] : 0);
    }
    this._pendingDrops = pend.slice(MAX_DROPS);
    const quad = this.steps % 2 === 0 ? this.updateAB : this.updateBA; // A -> B, then B -> A
    r.setRenderTarget(this.steps % 2 === 0 ? this.simB : this.simA);
    quad.render(r);
    this.steps++;
  }

  /**
   * Advance to `time` (seconds): ripple simulation at a fixed 120 Hz (max 4 steps per call, only while
   * ripples are alive), then rebuild the surface normals and the caustic tile.
   */
  update(time, opts = {}) {
    const r = this.renderer;
    const prev = r.getRenderTarget();
    const prevAutoClear = r.autoClear;
    const STEP = 1 / SIM_HZ;
    if (this.lastTime === undefined || time < this.lastTime || time - this.lastTime > 1) this.lastTime = time - STEP;
    this.accum += time - this.lastTime;
    this.lastTime = time;
    let n = 0;
    while (this.accum >= STEP && n < 4) {
      this._rain(this.simTime);
      this.simTime += STEP;
      if (this.simTime < this.activeUntil) this._step();
      else if (this.steps !== 0) this._clearSim(); // ripples have decayed: back to exact flat water
      this.accum -= STEP;
      n++;
    }
    if (n === 4) this.accum = 0;

    this.uWaveTime.value = time * this.timeScale;
    this.uWaveAmp.value = this.rainRate > 0 ? this._baseWaveAmp * 0.82 : this._baseWaveAmp;
    // the latest heights live in A after an even number of steps, otherwise in B
    this.normalSrc.value = this.steps % 2 === 0 ? this.simA.texture : this.simB.texture;
    r.setRenderTarget(this.surfaceRT);
    this.normalQuad.render(r);

    if (opts.caustics !== false) {
      this.causticRT.texture.generateMipmaps = opts.mips !== false;
      // clear inside the raster pass itself (a separate clear pass would be discarded for MSAA targets by the
      // tile-memory storeOp patch in index.html)
      const prevClear = r.getClearColor(_c);
      const prevClearAlpha = r.getClearAlpha();
      r.autoClear = true;
      r.setClearColor(this.clearColor, 0);
      r.setRenderTarget(this.causticMSRT || this.causticRT);
      r.render(this.causticScene, this.causticCamera);
      if (this.causticMSRT) {
        r.setRenderTarget(this.causticRT);
        this.copyQuad.render(r);
      }
      r.setClearColor(prevClear, prevClearAlpha);
    }

    r.autoClear = prevAutoClear;
    r.setRenderTarget(prev);
  }

  /** Deterministic state for screenshots: replay the ripple simulation from flat water to `time`. */
  warmTo(time, seconds = 4) {
    this.reset();
    this.simTime = Math.max(0, time - seconds);
    this.nextRainTime = this.simTime;
    if (this.rainRate > 0) this.activeUntil = time + 6.0;
    this.lastTime = this.simTime;
    if (this.rainRate > 0) for (let t = this.simTime; t < time; t += 1 / 30) this.update(t);
    this.update(time);
  }

  /** TSL: tile uv of the caustic pattern reaching world position `posW` (trace back along the sun). */
  causticUV(posW) {
    const toSurf = max(float(0.0), this.surfaceY.sub(posW.y)).div(max(float(0.25), this.sunDirWater.y));
    const q = vec2(posW.x, posW.z).add(vec2(this.sunDirWater.x, this.sunDirWater.z).mul(toSurf));
    const c = Math.cos(this.tileRot) / this.tileSize, sn = Math.sin(this.tileRot) / this.tileSize;
    return vec2(q.x.mul(c).sub(q.y.mul(sn)), q.x.mul(sn).add(q.y.mul(c)));
  }

  /** TSL: raw caustic texel (r = sand-depth layer, g = shallow coral-top layer), mean 1 (vertex or fragment). */
  causticSample(posW) {
    return texture(this.causticRT.texture, this.causticUV(posW));
  }

  /**
   * TSL (fragment stage): caustic irradiance at `posW` (float, 1 = flat water), blended between the two
   * layers by depth and blurred by the sun-disk penumbra (the 0.53 deg sun widens every filament by
   * ~7 mm per metre of water; `sharpness` < 1 keeps the art-directed crispness). The blur also removes the
   * dotted look of sub-pixel focal triangles in the projected grid.
   */
  causticIrradiance(posW, sharpness = 0.3) {
    const depth = max(float(0.0), this.surfaceY.sub(posW.y)).toVar();
    const texelM = this.tileSize / this.causticSize;
    const uvc = this.causticUV(posW).toVar();
    // LOD floor rather than a bias (a bias does nothing while the texture is magnified next to the camera):
    // max(footprint LOD, sun-disk penumbra LOD) -> ~LOD 1 on the sand, ~0 on coral tops
    const fp = max(length(dFdx(uvc)), length(dFdy(uvc))).mul(this.causticSize);
    const lod = max(log2(max(fp, float(1e-6))), log2(max(float(1.0), depth.mul((0.0071 * sharpness) / texelM))));
    const smp = texture(this.causticRT.texture, uvc).level(lod);
    return mix(smp.y, smp.x, smoothstep(this.depthShallow, this.depthDeep, depth));
  }
}
