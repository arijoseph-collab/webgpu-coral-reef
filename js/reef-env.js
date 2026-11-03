// -----------------------------------------------------------------------------
// reef-env.js — shared "water + light" contract for every material in the reef viewer
//
// Built from the two expert reviews (undersea realism + art direction). Every material — the
// Houdini corals, fish, flora, props, marine snow — must go through these helpers so the scene reads
// as one body of water:
//   waterColor(dir)        radiance of the open water column along `dir` (the backdrop at infinity)
//   fogTerms(posW)         per-channel transmittance + fog colour; fog converges EXACTLY to waterColor
//   depthLight(posW)       direct-sun / ambient attenuation with depth (normalised at the hero height)
//   causticLight(posW)     projected-grid caustic irradiance (js/ocean-surface.js), mean ~1, multiplies
//                          the direct sun so caustics take the surface colour (no white veins)
//   sunVis(posW, nrmW)     baked refracted-sun shadow map (static reef, rendered once)
//   surge(xz, t)           shared oscillatory surge displacement: fans, whips, grass, fish and snow all
//                          move with the same water (coherence is the strongest "underwater" cue)
//   litProp({...})         the reef lighting model for props (wrap key x shadow x caustics, sky, fill, rim)
//   output(col)            soft highlight shoulder, vignette toward deep water, +-0.5 LSB dither and the
//                          fused display encode used by the coral shader (direct-to-canvas sRGB)
// CPU helpers (after the GLB loads, `await env.terrain`): heightAt / normalAt / coralDist / coralTop.
// -----------------------------------------------------------------------------
import * as THREE from "three/webgpu";
import {
  Fn,
  uniform,
  vec2,
  vec3,
  vec4,
  float,
  mix,
  smoothstep,
  clamp,
  max,
  min,
  exp,
  exp2,
  log2,
  dot,
  normalize,
  length,
  sin,
  cos,
  step,
  pow,
  sqrt,
  fract,
  texture,
  cameraPosition,
  screenUV,
  screenCoordinate,
  atan,
} from "three/tsl";

/** Unified underwater sun direction (towards the sun): key light, caustics, shadows and shafts agree. */
export const SUN_DIR_WATER = new THREE.Vector3(0.36, 0.8, 0.48).normalize();

/** Inverse Snell: direction towards the sun above the surface for an underwater direction `w`. */
export function airDirFromWater(w, ior = 1.333) {
  const h = Math.hypot(w.x, w.z);
  const ha = Math.min(0.9999, (h / Math.hypot(w.x, w.y, w.z)) * ior);
  const s = h > 0 ? ha / h : 0;
  return new THREE.Vector3(w.x * s, Math.sqrt(1 - ha * ha), w.z * s).normalize();
}

// Seeded PRNG (mulberry32) — every module must use seeded randomness (deterministic stills)
export function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Shared surge (oscillatory wave-driven water motion near the bottom)
const SURGE_DIR1 = [Math.sin((35 * Math.PI) / 180), -Math.cos((35 * Math.PI) / 180)]; // ~35 deg off the hero view axis
const SURGE_DIR2 = [
  SURGE_DIR1[0] * Math.cos(0.436) - SURGE_DIR1[1] * Math.sin(0.436),
  SURGE_DIR1[0] * Math.sin(0.436) + SURGE_DIR1[1] * Math.cos(0.436),
]; // rotated 25 deg
const SURGE = { p1: 7.5, a1: 0.12, p2: 4.6, a2: 0.042, ph2: 1.3, speed: 6.0 };

export function createReefEnv(o) {
  const { renderer, scene, camera, ocean, AB } = o;
  const u = o.uniforms; // existing uniforms from index.html (time, sunDir, keyDir, keyColor, ...)

  // ---- new water-column / grade uniforms (set per mood by applyMood) ----
  u.fogSigma = uniform(new THREE.Vector3(0.085, 0.1, 0.075)); // per-channel extinction (1/m) of the view ray
  u.fogStart = uniform(1.6); // metres of clear water in front of the camera
  u.fogWall = uniform(new THREE.Vector2(5.6, 12.2)); // radial distance over which the water closes completely (inside +-13 m seabed)
  u.depthSigma = uniform(new THREE.Vector3(0.11, 0.13, 0.095)); // extinction applied to light with depth
  u.lightRefY = uniform(1.6); // height where depth light is normalised (hero corals keep today's exposure)
  u.seaTint = uniform(new THREE.Vector3(0.58, 0.5, 0.7));
  u.sheenTint = uniform(new THREE.Vector3(0.94, 0.9, 1.0));
  u.causticMood = uniform(1.0);
  u.causticNorm = uniform(1.24); // 1 / mean(sqrt(irradiance)) so compressed caustics keep mean ~1
  u.vignette = uniform(0.2);
  u.rimColor = uniform(new THREE.Vector3(0.42, 0.4, 0.62));
  u.keyGain = uniform(0.78);
  u.exposure = uniform(1.08); // per mood (art director round 1: Pastel read ~15% too dark)
  u.fogDesat = uniform(0.15); // per mood: desaturate + lift the far water colour (Pastel horizon read grape)
  u.shadowFill = uniform(0.3); // forward-scattered sun reaching shadowed areas (underwater shadows are soft, ratio ~0.55)
  u.sandGain = uniform(0.9);
  u.shaftMood = uniform(1.0);
  u.skyGain = uniform(0.3);
  u.fillGain = uniform(0.22);
  if (!u.bumpStrength) u.bumpStrength = uniform(1.25);
  if (!u.directOut) u.directOut = uniform(1.0);

  const env = {
    renderer,
    scene,
    camera,
    ocean,
    AB: AB || new Set(),
    u,
    groups: {}, // name -> Object3D (variant tokens can hide them: see index.html applyRenderVariant)
    sites: {}, // shared placement info published by modules (e.g. sites.anemones)
    makeRng,
    SUN_DIR_WATER,
  };

  // ---------------------------------------------------------------------------
  // Water column
  // ---------------------------------------------------------------------------
  /** Radiance of the open water column seen along unit direction `d` (no sun disk / shafts). */
  /**
   * Radial sun shafts seen along `d` (beams are parallel to the refracted sun, so in perspective they are
   * radial lines through the sun / anti-sun points, indexed by the azimuth around the sun axis). Part of the
   * water colour, so the backdrop and every fogged surface carry the same shafts (no beams stopping at the
   * silhouettes of hazed geometry). Fades out right at the horizon.
   */
  env.radialShafts = (d) => {
    const sd = u.sunDir;
    const e1 = normalize(vec3(sd.z.negate(), 0.0, sd.x));
    const e2 = normalize(vec3(sd.y.mul(sd.x), sd.z.mul(sd.z).add(sd.x.mul(sd.x)).negate(), sd.y.mul(sd.z)));
    const phi = atan(dot(d, e2), dot(d, e1)).toVar();
    const t = u.time;
    const b1 = sin(phi.mul(12.0).add(sin(t.mul(0.28)).mul(0.65))).mul(0.5).add(0.5);
    const b2 = sin(phi.mul(21.0).sub(cos(t.mul(0.21).add(1.7)).mul(0.55)).add(1.4)).mul(0.5).add(0.5).toVar();
    const b3 = sin(phi.mul(32.0).sub(t.mul(0.32)).add(2.8)).mul(0.5).add(0.5).toVar();
    const b2s = b2.mul(b2).toVar();
    const b2_8 = b2s.mul(b2s).mul(b2s.mul(b2s));
    const b3s = b3.mul(b3).toVar();
    const b3_11 = b3s.mul(b3s).mul(b3s.mul(b3s)).mul(b3s).mul(b3);
    const haze = sin(d.x.mul(3.8).add(d.z.mul(2.7)).add(t.mul(0.15))).mul(cos(d.y.mul(4.2).sub(t.mul(0.12)))).mul(0.5).add(0.5);
    const radial = pow(b1, 5.5).mul(0.52).add(b2_8.mul(0.38)).add(b3_11.mul(0.26))
      .mul(smoothstep(-0.12, 0.4, d.y)).mul(float(0.7).add(haze.mul(0.6)));
    return radial.mul(0.42).mul(u.atmosphere).mul(float(0.35).add(u.causticsScale.mul(0.65))).mul(u.shaftMood).mul(smoothstep(-0.01, 0.08, d.y));
  };

  env.waterColor = (d) => {
    // wider gradient around d.y = 0 than the original backdrop (the horizon read as two stripes), a lifted
    // horizon band with its darkest point below the sand line, and a per-mood desaturation of the far water
    const upMix = smoothstep(-0.6, 0.35, d.y);
    const surfaceMix = smoothstep(0.04, 0.72, d.y);
    const grad = mix(mix(u.bgDeep, u.bgMid, upMix), u.bgShallow, surfaceMix.mul(0.72)).toVar();
    const band = exp(d.y.mul(d.y).mul(-22.0)).mul(0.17);
    const toSun = dot(d, u.sunDir).mul(0.5).add(0.5);
    const lit = grad.mul(mix(float(0.99), float(1.24), toSun.mul(toSun))).mul(float(1.0).add(band));
    const grey = dot(lit, vec3(0.3333));
    return mix(lit, vec3(grey), u.fogDesat).mul(float(1.0).add(u.fogDesat.mul(0.5))).add(u.shaftTint.mul(env.radialShafts(d)));
  };

  /**
   * Fog for a point: per-channel transmittance T of the view ray (Beer-Lambert past `fogStart`), closing
   * fully at the radial water wall so the 26 m seabed never shows an edge; colour = waterColor(view dir).
   * Returns { T: vec3, color: vec3, ray: vec3, dist: float }.
   */
  env.fogTerms = (posW) => {
    const toP = posW.sub(cameraPosition);
    const dist = length(toP).toVar();
    const ray = toP.div(max(dist, float(1e-4))).toVar();
    const atmo = clamp(u.atmosphere, 0.0, 1.6);
    const wr = clamp(length(vec2(posW.x, posW.z)).sub(u.fogWall.x).div(u.fogWall.y.sub(u.fogWall.x)), 0.0, 1.0);
    const wall = float(1.0).sub(wr.mul(wr).mul(wr).mul(wr.mul(wr.mul(6.0).sub(15.0)).add(10.0))); // smootherstep
    const T = exp(u.fogSigma.mul(max(dist.sub(u.fogStart), 0.0)).mul(atmo).negate()).mul(wall);
    return { T, color: env.waterColor(ray), ray, dist };
  };

  /** shaded -> fogged (call per pixel for props; the coral shader packs the terms into varyings). */
  env.applyFog = (col, posW) => {
    const f = env.fogTerms(posW);
    return col.mul(f.T).add(f.color.mul(vec3(1.0).sub(f.T)));
  };

  /** Light attenuation with depth, normalised at u.lightRefY: { sun: vec3, amb: vec3 }. */
  env.depthLight = (posW) => {
    const dy = u.lightRefY.sub(posW.y); // > 0 below the reference height
    const sun = exp(u.depthSigma.mul(dy.div(max(u.sunDir.y, 0.3))).mul(-0.6));
    const amb = exp(u.depthSigma.mul(dy).mul(-0.45));
    return { sun: min(sun, vec3(1.25)), amb: min(amb, vec3(1.2)) };
  };

  // ---------------------------------------------------------------------------
  // Caustics (projected-grid irradiance from js/ocean-surface.js)
  // ---------------------------------------------------------------------------
  /** vec3 multiplier for the direct sun at posW (mean ~1). Fragment stage. `gain` scales the contrast. */
  env.causticLight = (posW, gain = 1.0, power = 0.5) => {
    const c = ocean.causticIrradiance(posW); // float, mean 1, depth-layer blended, sun-disk blurred
    // compress the filaments (sqrt on sand; gentler on coral so soft 1.5-2.5x patches read), keep the mean
    const cc = power === 0.5 ? sqrt(max(c, 0.0)).mul(u.causticNorm) : pow(max(c, 0.0), power).mul(power === 0.75 ? 1.12 : 1.0);
    const k = u.causticsScale.mul(1.0 / 1.15).mul(u.causticMood).mul(gain);
    return vec3(max(float(1.0).add(cc.sub(1.0).mul(k)), 0.0));
  };

  // ---------------------------------------------------------------------------
  // Baked refracted-sun shadow map (static reef + static props)
  // ---------------------------------------------------------------------------
  const SHADOW_SIZE = 2048;
  const SHADOW_HALF = 4.2; // metres: ortho half-extent around the reef centre
  const shadowRT = new THREE.RenderTarget(SHADOW_SIZE, SHADOW_SIZE, {
    type: THREE.UnsignedByteType,
    format: THREE.RedFormat,
    depthBuffer: true,
    generateMipmaps: false,
  });
  shadowRT.depthTexture = new THREE.DepthTexture(SHADOW_SIZE, SHADOW_SIZE);
  shadowRT.depthTexture.type = THREE.FloatType;
  shadowRT.depthTexture.compareFunction = THREE.LessEqualCompare;
  shadowRT.depthTexture.minFilter = THREE.LinearFilter;
  shadowRT.depthTexture.magFilter = THREE.LinearFilter;
  const sunCam = new THREE.OrthographicCamera(-SHADOW_HALF, SHADOW_HALF, SHADOW_HALF, -SHADOW_HALF, 0.1, 18);
  sunCam.layers.set(1); // shadow casters are tagged with layer 1 (env.addShadowCaster)
  const uShadowMat = uniform(new THREE.Matrix4());
  const uShadowOn = uniform(0.0);
  const uShadowSoft = uniform(3.4 / SHADOW_SIZE); // ~1.4 cm penumbra: underwater shadows are soft
  const depthOnly = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide, colorWrite: false, name: "shadow_depth" });
  const casters = new Set();
  env.shadow = { rt: shadowRT, camera: sunCam, uMatrix: uShadowMat, uOn: uShadowOn, size: SHADOW_SIZE, half: SHADOW_HALF };

  /** Tag a static object (and its children) as a shadow caster; the map is re-baked on the next frame. */
  env.addShadowCaster = (obj) => {
    obj.traverse((c) => c.layers.enable(1));
    casters.add(obj);
    env.requestShadowBake();
  };
  let bakePending = false;
  env.requestShadowBake = () => {
    bakePending = true;
  };
  /** Render the shadow map now (one-time cost ~5 ms for the 1M-triangle reef). */
  env.bakeShadows = () => {
    const center = new THREE.Vector3(0, 0.9, 0);
    sunCam.position.copy(u.sunDir.value).normalize().multiplyScalar(9).add(center);
    sunCam.up.set(0, 1, 0);
    sunCam.lookAt(center);
    sunCam.updateMatrixWorld();
    sunCam.coordinateSystem = renderer.coordinateSystem;
    sunCam.updateProjectionMatrix();
    uShadowMat.value.multiplyMatrices(sunCam.projectionMatrix, sunCam.matrixWorldInverse);
    const prevTarget = renderer.getRenderTarget();
    const prevOverride = scene.overrideMaterial;
    const prevBg = scene.background;
    const prevBgNode = scene.backgroundNode;
    scene.background = null;
    scene.backgroundNode = null;
    scene.overrideMaterial = depthOnly;
    renderer.setRenderTarget(shadowRT);
    renderer.clear();
    renderer.render(scene, sunCam);
    scene.overrideMaterial = prevOverride;
    scene.background = prevBg;
    scene.backgroundNode = prevBgNode;
    renderer.setRenderTarget(prevTarget);
    uShadowOn.value = 1.0;
    bakePending = false;
  };
  /** Call once per frame before rendering: performs a requested (re)bake. */
  env.preRender = () => {
    if (bakePending && casters.size) env.bakeShadows();
  };

  /** TSL (fragment): sun visibility 0..1 from the baked map (4 bilinear PCF compare taps). */
  env.sunVis = (posW, nrmW) => {
    const p = nrmW ? posW.add(nrmW.mul(0.012)) : posW; // normal-offset bias
    const sc = uShadowMat.mul(vec4(p, 1.0)).toVar();
    const suv = vec2(sc.x.mul(0.5).add(0.5), sc.y.mul(-0.5).add(0.5)).toVar();
    const ref = sc.z.sub(0.0012);
    const dt = env.shadow.depthTexture || shadowRT.depthTexture;
    const tap = (x, y) => texture(dt, suv.add(vec2(x, y).mul(uShadowSoft))).compare(ref);
    const vis = tap(-0.6, -0.8).add(tap(0.8, -0.6)).add(tap(-0.8, 0.6)).add(tap(0.6, 0.8)).mul(0.25);
    const inside = step(0.002, suv.x).mul(step(suv.x, 0.998)).mul(step(0.002, suv.y)).mul(step(suv.y, 0.998));
    return mix(float(1.0), vis, inside.mul(uShadowOn));
  };
  /** TSL (fragment): cheap single bilinear-PCF tap (volumetric shadow beams in the water column). */
  env.sunVis1 = (posW) => {
    const sc = uShadowMat.mul(vec4(posW, 1.0)).toVar();
    const suv = vec2(sc.x.mul(0.5).add(0.5), sc.y.mul(-0.5).add(0.5)).toVar();
    const vis = texture(shadowRT.depthTexture, suv).compare(sc.z.sub(0.002));
    const inside = step(0.002, suv.x).mul(step(suv.x, 0.998)).mul(step(0.002, suv.y)).mul(step(suv.y, 0.998));
    return mix(float(1.0), vis, inside.mul(uShadowOn));
  };

  // ---------------------------------------------------------------------------
  // Volumetric sun beams (in-scatter of the caustic focal lines extended along the refracted sun)
  // ---------------------------------------------------------------------------
  /**
   * TSL: march `steps` fixed samples along a view ray through the blurred caustic texture of the simulated
   * surface; the blur grows with distance like a diverging beam cone so the integrand stays smooth (no
   * jitter, no banding). Returns the beam in-scatter (multiply by u.shaftTint).
   */
  env.beams = (ro, rd, maxDist, steps = 12) => {
    const len = min(maxDist, float(9.0)).toVar();
    let acc = float(0.0);
    for (let i = 0; i < steps; i++) {
      const tt = float((i + 0.5) / steps).mul(len).toVar();
      const p = ro.add(rd.mul(tt)).toVar();
      const c = texture(ocean.causticRT.texture, ocean.causticUV(p)).level(tt.mul(0.32).add(4.6)).x;
      acc = acc.add(max(c.sub(0.78), 0.0).mul(smoothstep(-0.05, 0.3, p.y)).mul(exp(tt.mul(-0.12))));
    }
    const cosT = dot(rd, u.sunDir);
    const denom = max(float(1.3364).sub(cosT.mul(1.16)), 1e-3).toVar(); // Henyey-Greenstein g = 0.58
    const phase = float(0.052808).div(denom.mul(sqrt(denom))).mul(1.8).add(0.28);
    return acc.mul(len.div(steps)).mul(phase).mul(u.shaftMood);
  };

  // ---------------------------------------------------------------------------
  // Shared surge field
  // ---------------------------------------------------------------------------
  const w1 = (2 * Math.PI) / SURGE.p1;
  const w2 = (2 * Math.PI) / SURGE.p2;
  const k1 = w1 / SURGE.speed;
  const k2 = w2 / SURGE.speed;
  /** TSL: horizontal water displacement (m) at world xz (vec2) and time t; `lag` delays (s). */
  env.surge = (xz, t, lag = 0) => {
    const tt = lag ? t.sub(lag) : t;
    const d1 = vec2(SURGE_DIR1[0], SURGE_DIR1[1]);
    const d2 = vec2(SURGE_DIR2[0], SURGE_DIR2[1]);
    const ph1 = tt.mul(w1).sub(dot(xz, d1).mul(k1));
    const ph2 = tt.mul(w2).sub(dot(xz, d2).mul(k2)).add(SURGE.ph2);
    return d1.mul(sin(ph1).mul(SURGE.a1)).add(d2.mul(sin(ph2).mul(SURGE.a2)));
  };
  /** TSL: water velocity (m/s) of the surge. */
  env.surgeVel = (xz, t) => {
    const d1 = vec2(SURGE_DIR1[0], SURGE_DIR1[1]);
    const d2 = vec2(SURGE_DIR2[0], SURGE_DIR2[1]);
    const ph1 = t.mul(w1).sub(dot(xz, d1).mul(k1));
    const ph2 = t.mul(w2).sub(dot(xz, d2).mul(k2)).add(SURGE.ph2);
    return d1.mul(cos(ph1).mul(SURGE.a1 * w1)).add(d2.mul(cos(ph2).mul(SURGE.a2 * w2)));
  };
  /** CPU twin of env.surge: returns [dx, dz]. */
  env.surgeCPU = (x, z, t) => {
    const ph1 = t * w1 - (x * SURGE_DIR1[0] + z * SURGE_DIR1[1]) * k1;
    const ph2 = t * w2 - (x * SURGE_DIR2[0] + z * SURGE_DIR2[1]) * k2 + SURGE.ph2;
    const s1 = Math.sin(ph1) * SURGE.a1;
    const s2 = Math.sin(ph2) * SURGE.a2;
    return [SURGE_DIR1[0] * s1 + SURGE_DIR2[0] * s2, SURGE_DIR1[1] * s1 + SURGE_DIR2[1] * s2];
  };
  env.surgeDir = new THREE.Vector2(SURGE_DIR1[0], SURGE_DIR1[1]);

  // ---------------------------------------------------------------------------
  // Lighting model for props (matches the coral shader's look: wrap key, sky gradient, fill)
  // ---------------------------------------------------------------------------
  /**
   * @param {object} p
   *  albedo (vec3, display-referred like the coral attributes), N (unit world normal, facing the viewer),
   *  posW (vec3), ao (float, default 1), wrap (key wrap, default 0.44), trans (float back-light
   *  transmission for thin leaves/fins, default 0), causticGain (default 1), shadow (bool, default true),
   *  rim (float, default 1), sheen (float velvet sheen, default 0)
   * @returns vec3 shaded colour before fog (feed into env.finish / env.output)
   */
  env.litProp = (p) => {
    const N = p.N;
    const Nmacro = p.Nmacro || N;
    const posW = p.posW;
    const ao = p.ao ?? float(1.0);
    const wrap = p.wrap ?? 0.44;
    const V = normalize(cameraPosition.sub(posW));
    const dl = env.depthLight(posW);
    const vis = p.shadow === false ? float(1.0) : mix(u.shadowFill, float(1.0), env.sunVis(posW, Nmacro));
    const caus = p.causticGain === 0 ? vec3(1.0) : env.causticLight(posW, p.causticGain ?? 1.0);
    const ndl = dot(Nmacro, u.sunDir);
    const key = u.keyColor.mul(clamp(ndl.add(wrap).div(1.0 + wrap), 0.0, 1.0)).mul(vis).mul(caus).mul(dl.sun).mul(u.keyGain);
    const sky = mix(u.skyLow, u.skyHigh, N.y.mul(0.5).add(0.5)).mul(u.skyGain).mul(dl.amb);
    const fill = u.fillColor.mul(clamp(dot(N, u.fillDir).add(0.52).div(1.52), 0.0, 1.0)).mul(u.fillGain).mul(dl.amb);
    let col = p.albedo.mul(sky.add(fill).mul(ao).add(key));
    if (p.Nmacro) {
      const raking = normalize(vec3(u.sunDir.x.mul(1.85), float(0.60), u.sunDir.z.mul(1.85)));
      const micro = clamp(float(1.0).add(dot(N, raking).sub(dot(Nmacro, raking)).mul(0.95)), 0.46, 1.40);
      col = col.mul(micro);
    }
    const nv = clamp(dot(N, V), 0.0, 1.0);
    const fr = float(1.0).sub(nv);
    const rimAmt = p.rim ?? 1.0;
    if (rimAmt) {
      const rim = fr.mul(fr).mul(fr).mul(clamp(dot(N, u.rimDir).mul(0.5).add(0.5), 0.0, 1.0)).mul(rimAmt);
      col = col.add(u.rimColor.mul(rim).mul(ao).mul(0.5));
    }
    if (p.trans) {
      // thin-tissue transmission: light arriving from behind the surface (sun behind the leaf/fin)
      const back = clamp(dot(N.negate(), u.sunDir).add(0.25).div(1.25), 0.0, 1.0);
      col = col.add(p.albedo.mul(u.keyColor).mul(back.mul(back)).mul(vis).mul(dl.sun).mul(p.trans));
    }
    if (p.sheen) {
      const sh = fr.mul(fr).mul(fr.mul(fr)).mul(float(0.35).add(clamp(ndl, 0.0, 1.0).mul(0.65)));
      col = col.add(mix(p.albedo, vec3(1.0), 0.5).mul(sh).mul(p.sheen));
    }
    return col;
  };

  // ---------------------------------------------------------------------------
  // Output: highlight shoulder, vignette toward deep water, dither, display encode
  // ---------------------------------------------------------------------------
  /** Soft C1 rational shoulder instead of a hard clamp (preserves 3D form on sunlit macro tips). */
  env.shoulder = (x) => {
    const k = 0.74;
    const r = 0.26;
    const u0 = max(x.sub(k), vec3(0.0));
    return min(x, vec3(k)).add(u0.mul(r).div(u0.add(r)));
  };
  /** Vignette that shifts the frame corners toward the deep-water colour (not black). */
  env.vignette = (col) => {
    const q = screenUV.sub(0.5).mul(vec2(1.0, 0.82));
    const v = smoothstep(0.32, 0.78, length(q)).mul(u.vignette);
    return mix(col, u.bgDeep.mul(0.9), v);
  };
  /**
   * Display encode shared with the coral shader: display-referred `mapped` -> x^2.05 -> sRGB OETF when
   * rendering straight to the canvas (u.directOut = 1), or linear for the renderer's output pass.
   */
  env.encode = (mapped) => {
    const m = max(mapped, vec3(1e-8));
    const l = log2(m).toVar();
    const curve = exp2(l.mul(2.05 * 0.41666)).mul(1.055).sub(0.055);
    const toe = exp2(l.mul(2.05)).mul(12.92);
    const direct = mix(curve, toe, step(m, vec3(0.060024)));
    return mix(exp2(l.mul(2.05)), direct, u.directOut);
  };
  /** +-0.5 LSB dither (interleaved gradient noise): removes 8-bit banding in the smooth water gradients. */
  env.dither = (encoded) => {
    const h = fract(fract(dot(screenCoordinate.xy, vec2(0.06711056, 0.00583715))).mul(52.9829189));
    return encoded.add(h.sub(0.5).mul(1.0 / 255.0).mul(u.directOut));
  };
  /** Fog + grade + encode for a prop: returns the vec4 for material.fragmentNode. */
  env.finish = (shaded, posW, alpha = 1.0) => {
    const fogged = env.applyFog(shaded, posW);
    return vec4(env.dither(env.encode(env.shoulder(env.vignette(fogged).mul(u.exposure.mul(0.92))))), alpha);
  };
  /** Grade + encode only (for colours that are already fogged). */
  env.output = (fogged, alpha = 1.0) => vec4(env.dither(env.encode(env.shoulder(env.vignette(fogged).mul(u.exposure.mul(0.92))))), alpha);

  // ---------------------------------------------------------------------------
  // CPU terrain / occupancy (resolved once the Houdini GLB is loaded: env.setReef(meshes))
  // ---------------------------------------------------------------------------
  let resolveTerrain;
  env.terrain = new Promise((r) => (resolveTerrain = r));
  env.setReef = (reefMeshes) => {
    const t = buildTerrain(reefMeshes);
    env.terrainInfo = t;
    resolveTerrain(t);
    return t;
  };

  // ---------------------------------------------------------------------------
  // Moods
  // ---------------------------------------------------------------------------
  const MOODS = {
    day: {
      fogSigma: [0.085, 0.1, 0.075], fogStart: 1.6, depthSigma: [0.11, 0.13, 0.095], seaTint: [0.58, 0.5, 0.7],
      sheenTint: [0.94, 0.9, 1.0], causticMood: 1.0, rimColor: [0.42, 0.4, 0.62], keyGain: 0.78, skyGain: 0.3, fillGain: 0.22, vignette: 0.18,
      exposure: 1.08, fogDesat: 0.18, shadowFill: 0.3, sandGain: 0.9, shaftMood: 1.0,
    },
    sunset: {
      fogSigma: [0.13, 0.075, 0.06], fogStart: 1.6, depthSigma: [0.2, 0.09, 0.07], seaTint: [0.64, 0.78, 0.8],
      sheenTint: [0.82, 0.96, 1.0], causticMood: 1.05, rimColor: [0.2, 0.5, 0.62], keyGain: 0.8, skyGain: 0.3, fillGain: 0.22, vignette: 0.2,
      exposure: 1.02, fogDesat: 0.06, shadowFill: 0.3, sandGain: 1.0, shaftMood: 1.0,
    },
    night: {
      fogSigma: [0.24, 0.19, 0.14], fogStart: 0.8, depthSigma: [0.24, 0.19, 0.14], seaTint: [0.22, 0.27, 0.45],
      sheenTint: [0.45, 0.65, 1.0], causticMood: 0.2, rimColor: [0.08, 0.2, 0.42], keyGain: 0.2, skyGain: 0.17, fillGain: 0.15, vignette: 0.32,
      exposure: 1.0, fogDesat: 0.0, shadowFill: 0.35, sandGain: 0.8, shaftMood: 0.35,
    },
  };
  env.applyMood = (mode) => {
    const m = MOODS[mode];
    if (!m) return;
    u.fogSigma.value.set(...m.fogSigma);
    u.fogStart.value = m.fogStart;
    u.depthSigma.value.set(...m.depthSigma);
    u.seaTint.value.set(...m.seaTint);
    u.sheenTint.value.set(...m.sheenTint);
    u.causticMood.value = m.causticMood;
    u.rimColor.value.set(...m.rimColor);
    u.keyGain.value = m.keyGain;
    u.skyGain.value = m.skyGain;
    u.fillGain.value = m.fillGain;
    u.vignette.value = m.vignette;
    u.exposure.value = m.exposure;
    u.fogDesat.value = m.fogDesat;
    u.shadowFill.value = m.shadowFill;
    u.sandGain.value = m.sandGain;
    u.shaftMood.value = m.shaftMood;
    env.moodFogStart = m.fogStart;
    env.mood = mode;
  };
  env.mood = "day";

  return env;
}

// -----------------------------------------------------------------------------
// Terrain helpers built from the Houdini meshes (CPU, once)
// -----------------------------------------------------------------------------
function buildTerrain(reefMeshes) {
  const seabed = reefMeshes.seabed;
  // --- seabed heightfield: the Houdini seabed is a regular 180x180 grid over [-13, 13]^2 ---
  let N = 180;
  let x0 = -13;
  let x1 = 13;
  const grid = new Float32Array(N * N).fill(0);
  if (seabed) {
    const pos = seabed.geometry.attributes.position;
    seabed.updateMatrixWorld(true);
    const m = seabed.matrixWorld;
    const v = new THREE.Vector3();
    const box = new THREE.Box3().setFromBufferAttribute(pos);
    x0 = box.min.x;
    x1 = box.max.x;
    N = Math.round(Math.sqrt(pos.count));
    const filled = new Uint8Array(N * N);
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(m);
      const gi = Math.round(((v.x - x0) / (x1 - x0)) * (N - 1));
      const gj = Math.round(((v.z - x0) / (x1 - x0)) * (N - 1));
      if (gi < 0 || gj < 0 || gi >= N || gj >= N) continue;
      grid[gj * N + gi] = v.y;
      filled[gj * N + gi] = 1;
    }
    // fill any holes (irregular grids) from neighbours
    for (let pass = 0; pass < 4; pass++)
      for (let j = 0; j < N; j++)
        for (let i = 0; i < N; i++) {
          if (filled[j * N + i]) continue;
          let s = 0, c = 0;
          for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const a = i + di, b = j + dj;
            if (a >= 0 && b >= 0 && a < N && b < N && filled[b * N + a]) { s += grid[b * N + a]; c++; }
          }
          if (c) { grid[j * N + i] = s / c; filled[j * N + i] = 1; }
        }
  }
  const cell = (x1 - x0) / (N - 1);
  const heightAt = (x, z) => {
    const fx = Math.min(N - 1.001, Math.max(0, (x - x0) / cell));
    const fz = Math.min(N - 1.001, Math.max(0, (z - x0) / cell));
    const i = Math.floor(fx), j = Math.floor(fz);
    const tx = fx - i, tz = fz - j;
    const h00 = grid[j * N + i], h10 = grid[j * N + i + 1], h01 = grid[(j + 1) * N + i], h11 = grid[(j + 1) * N + i + 1];
    return (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
  };
  const normalAt = (x, z) => {
    const e = cell * 0.5;
    const dx = heightAt(x + e, z) - heightAt(x - e, z);
    const dz = heightAt(x, z + e) - heightAt(x, z - e);
    return new THREE.Vector3(-dx, 2 * e, -dz).normalize();
  };

  // --- coral footprint: occupancy, top height and distance-to-coral on a 7 cm grid over [-4.5, 4.5]^2 ---
  const G = 128, gx0 = -4.5, gsize = 9.0, gc = gsize / G;
  const occ = new Uint8Array(G * G);
  const top = new Float32Array(G * G).fill(-1);
  const coralGeometry = [];
  const v = new THREE.Vector3();
  for (const [name, mesh] of Object.entries(reefMeshes)) {
    if (!mesh || name === "seabed") continue;
    mesh.updateMatrixWorld(true);
    const pos = mesh.geometry.attributes.position;
    const world = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
      world[i * 3] = v.x; world[i * 3 + 1] = v.y; world[i * 3 + 2] = v.z;
      const gi = Math.floor((v.x - gx0) / gc), gj = Math.floor((v.z - gx0) / gc);
      if (gi < 0 || gj < 0 || gi >= G || gj >= G) continue;
      occ[gj * G + gi] = 1;
      if (v.y > top[gj * G + gi]) top[gj * G + gi] = v.y;
    }
    coralGeometry.push({ name, positions: world, count: pos.count });
  }
  // two-pass chamfer distance transform (metres) from occupied cells
  const INF = 1e9;
  const dist = new Float32Array(G * G);
  for (let i = 0; i < G * G; i++) dist[i] = occ[i] ? 0 : INF;
  const D1 = gc, D2 = gc * Math.SQRT2;
  for (let j = 0; j < G; j++)
    for (let i = 0; i < G; i++) {
      let d = dist[j * G + i];
      if (i > 0) d = Math.min(d, dist[j * G + i - 1] + D1);
      if (j > 0) {
        d = Math.min(d, dist[(j - 1) * G + i] + D1);
        if (i > 0) d = Math.min(d, dist[(j - 1) * G + i - 1] + D2);
        if (i < G - 1) d = Math.min(d, dist[(j - 1) * G + i + 1] + D2);
      }
      dist[j * G + i] = d;
    }
  for (let j = G - 1; j >= 0; j--)
    for (let i = G - 1; i >= 0; i--) {
      let d = dist[j * G + i];
      if (i < G - 1) d = Math.min(d, dist[j * G + i + 1] + D1);
      if (j < G - 1) {
        d = Math.min(d, dist[(j + 1) * G + i] + D1);
        if (i < G - 1) d = Math.min(d, dist[(j + 1) * G + i + 1] + D2);
        if (i > 0) d = Math.min(d, dist[(j + 1) * G + i - 1] + D2);
      }
      dist[j * G + i] = d;
    }
  const sampleGrid = (arr, x, z, outside) => {
    const fx = (x - gx0) / gc - 0.5, fz = (z - gx0) / gc - 0.5;
    if (fx < 0 || fz < 0 || fx > G - 1 || fz > G - 1) return outside(fx, fz);
    const i = Math.min(G - 2, Math.floor(fx)), j = Math.min(G - 2, Math.floor(fz));
    const tx = fx - i, tz = fz - j;
    return (arr[j * G + i] * (1 - tx) + arr[j * G + i + 1] * tx) * (1 - tz) + (arr[(j + 1) * G + i] * (1 - tx) + arr[(j + 1) * G + i + 1] * tx) * tz;
  };
  /** Horizontal distance (m) from (x, z) to the nearest coral footprint (vertical projection of all coral). */
  const coralDist = (x, z) =>
    sampleGrid(dist, x, z, (fx, fz) => {
      const cx = Math.min(G - 1, Math.max(0, fx)), cz = Math.min(G - 1, Math.max(0, fz));
      const edge = dist[Math.round(cz) * G + Math.round(cx)];
      return edge + Math.hypot(fx - cx, fz - cz) * gc;
    });
  /** Highest coral point above (x, z), or -1 when there is no coral there. */
  const coralTop = (x, z) => {
    const gi = Math.floor((x - gx0) / gc), gj = Math.floor((z - gx0) / gc);
    if (gi < 0 || gj < 0 || gi >= G || gj >= G) return -1;
    return top[gj * G + gi];
  };
  return { heightAt, normalAt, coralDist, coralTop, coralGeometry, seabedExtent: [x0, x1], footprint: { G, x0: gx0, size: gsize, occ, top, dist } };
}
