// -----------------------------------------------------------------------------
// Marine snow v2 — GPU-only drifting particulate (expert review P7 / item 7)
//
// The old version was 650 `THREE.Points` (always 1 px in WebGPU, i.e. invisible at DPR 2) animated on the
// CPU every frame. Here a single instanced draw of camera-facing quads does everything in the vertex
// shader: per-particle seeds come from hash(instanceIndex), positions drift and slosh with the shared
// surge field (env.surge, so snow moves with the same water as fans, grass and fish) and wrap inside a box
// that follows the camera (constant density, no CPU work). Size has a pixel floor so specks stay visible,
// motes close to the lens become soft, faint bokeh discs (energy-conserving alpha), brightness follows
// forward scattering toward the sun, the baked shadow map and the caustic beam (motes sparkle as they cross
// light), and everything fades into the water colour with distance.
// -----------------------------------------------------------------------------
import * as THREE from "three/webgpu";
import {
  Fn,
  vec2,
  vec3,
  vec4,
  float,
  uniform,
  hash,
  instanceIndex,
  positionGeometry,
  cameraPosition,
  cameraViewMatrix,
  cameraProjectionMatrix,
  varying,
  fract,
  sin,
  cos,
  mix,
  max,
  min,
  abs,
  dot,
  length,
  normalize,
  smoothstep,
  clamp,
  exp,
  uv,
  step,
} from "three/tsl";

export function createMarineSnow(env, opts = {}) {
  const u = env.u;
  const COUNT = opts.count ?? 10000;
  const BOX = opts.box ?? 9.0; // metres: wrap box centred on the camera
  const group = new THREE.Group();
  group.name = "snow";
  env.groups.snow = group;
  env.scene.add(group);

  const uFocus = uniform(2.8); // metres: distance in focus (bokeh grows away from it, mostly near the lens)
  const uAperture = uniform(0.010); // world-size circle of confusion scale
  const uPixel = uniform(1.0 / 1660.0); // tan(fov/2)*2 / drawing-buffer height: world size of one pixel per metre
  const uNight = uniform(0.0);

  const mat = new THREE.MeshBasicNodeMaterial({
    name: "marine_snow",
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
  });

  const i = instanceIndex;
  const h0 = hash(i);
  const h1 = hash(i.add(7919));
  const h2 = hash(i.add(15013));
  const h3 = hash(i.add(23011));
  const h4 = hash(i.add(31387));
  const t = u.time;

  // Base position in a BOX-periodic world lattice + slow sinking/drift + surge slosh + small eddies
  const base = vec3(h0, h1, h2).mul(BOX);
  const drift = vec3(0.012, -0.018, 0.006).mul(t).mul(float(0.6).add(h3.mul(0.8)));
  const eddy = vec3(sin(t.mul(0.37).add(h3.mul(40.0))), cos(t.mul(0.29).add(h4.mul(31.0))), sin(t.mul(0.33).add(h0.mul(27.0)))).mul(0.05);
  const p0 = base.add(drift).add(eddy);
  const sl = env.surge(vec2(p0.x, p0.z), t);
  const pDrift = p0.add(vec3(sl.x, 0.0, sl.y).mul(0.85));
  // wrap into the box around the camera (constant density wherever the camera goes)
  const rel = fract(pDrift.sub(cameraPosition).div(BOX).add(0.5)).sub(0.5).mul(BOX);
  const center = cameraPosition.add(rel).toVar();
  // stay inside the water: above the sand, below the surface
  const centerY = max(center.y, float(0.05));
  const c = vec3(center.x, centerY, center.z).toVar();

  const viewC = cameraViewMatrix.mul(vec4(c, 1.0)).toVar();
  const dist = max(viewC.z.negate(), float(0.05)).toVar();
  // size: mostly 0.6-1.8 mm specks, ~6% larger flecks (3-6 mm); pixel floor so they always read
  const big = step(0.94, h4);
  const worldSize = mix(float(0.0008).add(h3.mul(0.0016)), float(0.0035).add(h0.mul(0.003)), big);
  const pxFloor = dist.mul(uPixel).mul(3.2);
  const sharp = max(worldSize, pxFloor);
  // circle of confusion: motes near the lens grow into soft discs
  const coc = abs(float(1.0).div(dist).sub(float(1.0).div(uFocus))).mul(uAperture).mul(dist);
  const size = max(sharp, coc).toVar();
  const energy = clamp(sharp.div(size), 0.0, 1.0); // energy-conserving fade of the bokeh discs
  const offs = positionGeometry.xy.mul(size);
  mat.vertexNode = cameraProjectionMatrix.mul(vec4(viewC.xy.add(offs), viewC.z, 1.0));

  // brightness terms per vertex (constant across the tiny quad)
  const ray = normalize(c.sub(cameraPosition));
  const cosSun = dot(ray, u.sunDir);
  const g = 0.62;
  const hg = float((1 - g * g) / (4 * Math.PI)).div(max(float(1 + g * g).sub(cosSun.mul(2 * g)), 1e-3).pow(1.5));
  const fogT = exp(u.fogSigma.mul(max(dist.sub(u.fogStart), 0.0)).mul(-1.0)).dot(vec3(0.333)).toVar();
  const nearFade = smoothstep(0.06, 0.25, dist);
  const vBright = varying(
    vec4(
      hg.mul(1.6).add(0.62).mul(fogT).mul(nearFade).mul(max(energy, float(0.05))).mul(float(0.45).add(h4.mul(0.35)).add(big.mul(0.6))),
      c.x,
      c.y,
      c.z
    )
  );
  const vSpark = varying(step(0.92, h1).mul(uNight)); // night: bioluminescent dinoflagellate sparks

  mat.fragmentNode = Fn(() => {
    const q = uv().sub(0.5);
    const r2 = dot(q, q).mul(4.0);
    const disc = exp(r2.mul(-3.2)).mul(step(r2, 1.0));
    const posW = vec3(vBright.y, vBright.z, vBright.w);
    // light reaching the mote: sun (shadowed, caustic beam) + ambient water glow
    const vis = env.sunVis(posW, null);
    const caus = env.causticLight(posW, 1.2).x;
    const sunPart = u.keyColor.mul(vis).mul(caus).mul(0.75);
    const amb = mix(u.bgMid, u.bgShallow, 0.5).mul(0.35);
    const col = sunPart.add(amb).mul(mix(vec3(1.0), u.shaftTint, 0.4));
    const spark = vec3(0.35, 0.95, 1.0).mul(vSpark).mul(2.5);
    const lum = col.mul(vBright.x).add(spark.mul(vBright.x.add(0.55))).mul(disc).mul(3.2);
    return vec4(env.encode(min(lum, vec3(1.0))), 1.0);
  })();

  const geo = new THREE.PlaneGeometry(1, 1);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.count = COUNT;
  mesh.frustumCulled = false;
  mesh.renderOrder = 900; // after the opaque reef, before the backdrop
  group.add(mesh);

  return {
    group,
    mesh,
    uniforms: { uFocus, uAperture, uNight },
    update() {
      const cam = env.camera;
      const h = env.renderer.getDrawingBufferSize(_v2).y || 1660;
      uPixel.value = (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov) * 0.5)) / h;
      uNight.value = env.mood === "night" ? 1.0 : 0.0;
    },
    warmTo() {},
  };
}
const _v2 = new THREE.Vector2();
