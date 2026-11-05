import * as THREE from "three/webgpu";
import {
  Fn,
  If,
  attribute,
  positionLocal,
  positionWorld,
  normalWorld,
  cameraPosition,
  uniform,
  vec2,
  vec3,
  vec4,
  float,
  floor,
  fract,
  sin,
  cos,
  abs,
  sqrt,
  exp,
  dot,
  length,
  normalize,
  mix,
  clamp,
  smoothstep,
  max,
  min,
  pow,
  step,
  varying,
  texture3D,
  positionGeometry,
  cameraProjectionMatrixInverse,
  cameraWorldMatrix,
  sRGBTransferOETF,
  output,
  dFdx,
  dFdy,
  fwidth,
  cross,
  atan,
  screenCoordinate,
  sign,
  log2,
  exp2,
  modelWorldMatrix,
  normalGeometry,
  cameraViewMatrix,
  cameraProjectionMatrix,
  texture,
  uv,
} from "three/tsl";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OceanSurface } from "./js/ocean-surface.js";
import { makeSandDetailTexture, makeFootprintTexture, fillFootprintTexture } from "./js/sand-detail.js";
import { createReefEnv, SUN_DIR_WATER, airDirFromWater } from "./js/reef-env.js";
import { createReefFish } from "./js/reef-fish.js";
import { createReefFlora } from "./js/reef-flora.js";
import { createReefProps } from "./js/reef-props.js";
import { createMarineSnow } from "./js/marine-snow.js";
import { createWaterSurface } from "./js/water-surface.js";

const ASSET_CDN_BASE = "https://raw.githubusercontent.com/arijoseph-collab/webgpu-coral-reef/main/public";

function toCdnUrl(relPath) {
  const clean = relPath.replace(/^\.\//, "").replace(/^\//, "").split("?")[0];
  return `${ASSET_CDN_BASE}/${clean}`;
}

async function fetchBinaryWithFallback(relPath, expectedBytes = 0) {
  const localUrl = relPath.startsWith(".") ? relPath : `./${relPath.replace(/^\//, "")}`;
  try {
    const r = await fetch(localUrl);
    const ct = (r.headers.get("content-type") || "").toLowerCase();
    if (!r.ok || ct.includes("text/html")) throw new Error(`Local HTTP ${r.status} (${ct})`);
    const buf = await r.arrayBuffer();
    if (expectedBytes > 0 && buf.byteLength !== expectedBytes) {
      throw new Error(`Unexpected byteLength ${buf.byteLength} (expected ${expectedBytes})`);
    }
    return buf;
  } catch (localErr) {
    const cdnUrl = toCdnUrl(relPath);
    const r2 = await fetch(cdnUrl, { mode: "cors" });
    if (!r2.ok) throw localErr;
    const buf2 = await r2.arrayBuffer();
    if (expectedBytes > 0 && buf2.byteLength !== expectedBytes) {
      throw new Error(`CDN unexpected byteLength ${buf2.byteLength}`);
    }
    return buf2;
  }
}

function loadGlbWithFallback(loader, relPath, onLoad, onProgress, onError) {
  loader.setCrossOrigin("anonymous");
  const localUrl = relPath.startsWith(".") ? relPath : `./${relPath.replace(/^\//, "")}`;
  loader.load(
    localUrl,
    onLoad,
    onProgress,
    (localErr) => {
      const cdnUrl = toCdnUrl(relPath);
      console.warn("Local GLB unavailable in sandbox, falling back to GitHub Raw CDN:", cdnUrl, localErr);
      loader.load(cdnUrl, onLoad, onProgress, onError);
    }
  );
}

(async () => {

const container = document.getElementById("canvas-container");
const loadingOverlay = document.getElementById("loading-overlay");
const rendererBadge = document.getElementById("renderer-badge");

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(33, window.innerWidth / window.innerHeight, 0.08, 120);
camera.position.set(-0.2, 1.12, 2.95);

// URL options: ?mood=day|sunset|night&cam=wide|brain|sponge
// Dev benchmarking: ?bench=<label>&frames=240&warmup=60&dpr=<ratio>&ab=<comma flags>
const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get("hud") === "0") document.body.classList.add("no-hud");
const BENCH = urlParams.get("bench");
const AB = new Set((urlParams.get("ab") || "").split(",").filter(Boolean));
const T_PAGE_START = performance.now();

// Initialize Three.js WebGPURenderer
const renderer = new THREE.WebGPURenderer({
  alpha: false, // opaque canvas: alpha-to-coverage foliage must not let the page show through
  antialias: !AB.has("nomsaa"),
  powerPreference: "high-performance",
});
const dprOverride = parseFloat(urlParams.get("dpr"));
renderer.setPixelRatio(Number.isFinite(dprOverride) ? dprOverride : Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.NoToneMapping;
container.appendChild(renderer.domElement);

await renderer.init();
const isWebGPU = renderer.backend && renderer.backend.isWebGPUBackend;
rendererBadge.textContent = isWebGPU ? "WebGPU Active (GPU)" : "WebGL2 Fallback";

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.05, 1.16, 0.02);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.32;
controls.maxPolarAngle = Math.PI * 0.485;
controls.minDistance = 0.75;
controls.maxDistance = 8.5; // beyond this the reef dissolves in the water column (expert review)

// (no three.js lights: every material shades itself through js/reef-env.js)

// Snell's Law Refracted Sun Direction in Water (ported from willwaltz/globes src/render/shaders.ts `sunInWater`)
function computeSunInWater(ax, ay, az) {
  const len = Math.hypot(ax, ay, az);
  const sx = ax / len, sy = ay / len, sz = az / len;
  const eta = 1.0 / 1.3333;
  const k = Math.max(0.0, 1.0 - eta * eta * (1.0 - sy * sy));
  return new THREE.Vector3(eta * sx, Math.sqrt(k), eta * sz).normalize();
}
const sunDirWaterVec = SUN_DIR_WATER.clone(); // one sun for key light, caustics, shadows and shafts

// Shared Interactive Uniforms for TSL Sub-Pixel 3D Honeycomb Pore, Bump/Displacement, AO, Caustics & Volumetric Water Shader
const uPoreStrength = uniform(1.15);
const uBumpStrength = uniform(urlParams.has("bump") ? parseFloat(urlParams.get("bump")) : 1.25);
const uAoIntensity = uniform(1.20);
const uWaterAtmosphere = uniform(1.05);
const uSssIntensity = uniform(0.28);
const uRoughScale = uniform(0.84);
const uCausticsScale = uniform(1.15);
const uTime = uniform(0.0);

// Underwater Sun & Column Uniforms (willwaltz/globes + willwaltz/webgpu-ray-tracing)
const uSunDirWater = uniform(sunDirWaterVec.clone());
const uWaterSurfaceY = uniform(4.4);
const uWaterExt = uniform(new THREE.Vector3(0.32, 0.18, 0.10)); // Beer-Lambert RGB extinction
const uWaterGlow = uniform(new THREE.Vector3(0.46, 0.34, 0.68)); // Deep water column extinction color
const uShaftTint = uniform(new THREE.Vector3(0.82, 0.98, 1.00)); // Volumetric god-ray & caustic tint
const uBgDeep = uniform(new THREE.Vector3(0.24, 0.15, 0.44));
const uBgMid = uniform(new THREE.Vector3(0.48, 0.36, 0.72));
const uBgShallow = uniform(new THREE.Vector3(0.68, 0.64, 0.90));

// Mood Lighting Uniforms (matching Houdini VEX rt_shader_wrangle in "Pastel Reference" mode!)
const uKeyDir = uniform(SUN_DIR_WATER.clone()); // == uSunDirWater (expert review: shading terminators agree with shadows/caustics)
const uFillDir = uniform(new THREE.Vector3(-0.65, 0.35, 0.65).normalize());
const uRimDir = uniform(new THREE.Vector3(0.0, 0.55, -0.82).normalize());
const uKeyColor = uniform(new THREE.Vector3(1.05, 0.99, 0.96));
const uFillColor = uniform(new THREE.Vector3(0.78, 0.90, 0.99));
const uSkyLow = uniform(new THREE.Vector3(0.60, 0.42, 0.78));
const uSkyHigh = uniform(new THREE.Vector3(0.98, 0.95, 1.00));

// Simulated open-water surface + projected-grid caustics (adapted from jeantimex/webgpu-water)
const SUN_AIR_DIR = airDirFromWater(SUN_DIR_WATER, 1.336);
// dev tuning knobs: ?wtile=&wgrid=&wfocus=&wfdepth=&wmin=&wmax=&wcount=&wdisp=&wtime=
const wp = (k, d) => (urlParams.has(k) ? parseFloat(urlParams.get(k)) : d);
const ocean = new OceanSurface(renderer, {
  tileSize: wp("wtile", 5.0),
  simSize: 256,
  causticSize: 1024,
  causticSamples: wp("wmsaa", 4),
  gridSize: wp("wgrid", 260),
  focus: wp("wfocus", 0.55), // undersea expert round 1: single narrower filaments instead of fold pairs
  focusDepth: wp("wfdepth", 4.0),
  minLambda: wp("wmin", 0.11),
  maxLambda: wp("wmax", 0.55), // art director round 1: caustic cells x0.65
  waveCount: wp("wcount", 40),
  timeScale: wp("wtime", 0.6),
  sunAir: SUN_AIR_DIR,
  surfaceY: uWaterSurfaceY,
  sunDirWater: uSunDirWater,
});
const uDirectOut = uniform(0.0); // 1 when rendering straight to the sRGB canvas (no output pass)
const uNightGlow = uniform(0.0); // Bio-Night bioluminescent polyp tips (0 by day)
// DOF-lite (art director): detail behind/in front of the focus distance loses pore relief, caustic
// sharpness and specular and shifts slightly toward the water colour; silhouettes stay sharp
const uFocusDist = uniform(2.8);
const uDofAmt = uniform(0.0); // per preset: macro 1, hero 0.5

// Shared water + light contract for every material (js/reef-env.js, from the two expert reviews)
const env = createReefEnv({
  renderer,
  scene,
  camera,
  ocean,
  AB,
  uniforms: {
    time: uTime, sunDir: uSunDirWater, keyDir: uKeyDir, keyColor: uKeyColor, fillDir: uFillDir, fillColor: uFillColor,
    rimDir: uRimDir, skyLow: uSkyLow, skyHigh: uSkyHigh, shaftTint: uShaftTint, waterGlow: uWaterGlow,
    bgDeep: uBgDeep, bgMid: uBgMid, bgShallow: uBgShallow, causticsScale: uCausticsScale, atmosphere: uWaterAtmosphere,
    surfaceY: uWaterSurfaceY, sssIntensity: uSssIntensity, bumpStrength: uBumpStrength, directOut: uDirectOut,
  },
});
env.sunAir = SUN_AIR_DIR;
// camera presets shared with the life modules (placement keeps these frames clean)
env.cameraPresets = {
  hero: { eye: [-0.2, 1.12, 2.95], target: [0.05, 1.16, 0.02], fov: 33 },
  wide: { eye: [0.35, 1.45, 5.2], target: [0.0, 0.95, 0.0], fov: 35 },
  macro: { eye: [-0.08, 1.24, 1.48], target: [-0.04, 1.08, 0.16], fov: 35 },
  up: { eye: [-1.6, 0.35, -2.4], target: [0.2, 3.4, 0.1], fov: 40 },
};

// Seabed detail (procedural, generated at load) and the coral footprint (distance / height, filled after the GLB loads)
const sandTex = makeSandDetailTexture(512);
const footprintTex = makeFootprintTexture(128);
const FOOT_X0 = -4.5, FOOT_SIZE = 9.0; // must match buildTerrain() in js/reef-env.js
const SURGE_D = env.surgeDir.clone(); // sand ripple crests form perpendicular to the oscillatory surge
// Houdini sand lab bake (houdini/build_sand_lab.py -> tools/pack_sand_textures.py): a 9 x 9 m patch around the
// reef with Swift-Hohenberg vortex ripples that bend and terminate around the corals, scour moats, burrows,
// grazing trails and projected shell grit, plus a tileable 60 cm grain tile. Gated by uHoudiniSand until loaded.
const uHoudiniSand = uniform(0.0);
const uScanBump = uniform(0.0);
const SAND_PATCH_X0 = -4.5, SAND_PATCH_SIZE = 9.0, SAND_GRAIN_TILE = 0.6;
const makePlaceholderTex = (repeat) => {
  const t = new THREE.Texture();
  t.flipY = false;
  t.colorSpace = THREE.NoColorSpace;
  t.wrapS = t.wrapT = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
  return t;
};
const sandPatchNH = makePlaceholderTex(false);
const sandPatchMat = makePlaceholderTex(false);
const sandGrainTex = makePlaceholderTex(true);
const coralBumpNH = makePlaceholderTex(true);
const reefSubstrateNH = makePlaceholderTex(true);
let houdiniSandReady = Promise.resolve();
{
  const texLoader = new THREE.TextureLoader();
  texLoader.setCrossOrigin("anonymous");
  const applyTexParams = (t, repeat) => {
    t.flipY = false; // row 0 = z = -4.5 (matches footprintTex DataTexture)
    t.colorSpace = THREE.NoColorSpace;
    t.wrapS = t.wrapT = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.anisotropy = 8;
    t.generateMipmaps = true;
    return t;
  };
  const loadTex = (url, repeat) => {
    const localUrl = url.startsWith(".") ? url : `./${url.replace(/^\//, "")}`;
    return new Promise((resolve, reject) =>
      texLoader.load(
        localUrl,
        (t) => resolve(applyTexParams(t, repeat)),
        undefined,
        () => texLoader.load(toCdnUrl(url), (t2) => resolve(applyTexParams(t2, repeat)), undefined, reject)
      )
    );
  };
  const copyTex = (dst, src) => {
    dst.image = src.image;
    for (const k of ["flipY", "colorSpace", "wrapS", "wrapT", "minFilter", "magFilter", "anisotropy", "generateMipmaps"]) dst[k] = src[k];
    dst.needsUpdate = true;
  };
  const bumpTask = Promise.all([
    loadTex("/textures/coral_bump_nh.png?v=2", true),
    loadTex("/textures/reef_substrate_nh.png?v=2", true),
  ])
    .then(([cb, rs]) => {
      copyTex(coralBumpNH, cb);
      copyTex(reefSubstrateNH, rs);
      uScanBump.value = 1.0;
    })
    .catch((e) => console.warn("Scan bump maps unavailable:", e));
  const sandTask =
    !AB.has("proceduralSand") && !urlParams.get("features")?.includes("proceduralSand")
      ? Promise.all([
          loadTex("/textures/sand/sand_patch_nh.png?v=2", false),
          loadTex("/textures/sand/sand_patch_mat.png?v=2", false),
          loadTex("/textures/sand/sand_grain.png?v=2", true),
        ])
          .then(([nh, mat, grain]) => {
            for (const [dst, src] of [[sandPatchNH, nh], [sandPatchMat, mat], [sandGrainTex, grain]]) copyTex(dst, src);
            uHoudiniSand.value = 1.0;
          })
          .catch((e) => console.warn("Houdini sand textures unavailable, using the procedural sand:", e))
      : Promise.resolve();
  houdiniSandReady = Promise.all([bumpTask, sandTask]);
}

// Exact 27-Neighbor (3x3x3) 3D Cellular Voronoi Offsets so f1 and f2 have zero cell-boundary seams!
const OFFSETS_27 = [];
for (let dz = -1; dz <= 1; dz++) {
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      OFFSETS_27.push([dx, dy, dz]);
    }
  }
}
const OFFSETS_8 = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];

// Sin-free 32-bit float 3D Hash (Dave Hoskins) — zero WGSL f32 sin() precision speckle!
function hash33TSL(cell) {
  let p3 = fract(cell.mul(vec3(0.1031, 0.1030, 0.0973)));
  p3 = p3.add(dot(p3, vec3(p3.y, p3.x, p3.z).add(33.33)));
  return fract(vec3(p3.x, p3.x, p3.y).add(vec3(p3.y, p3.x, p3.x)).mul(vec3(p3.z, p3.y, p3.x)));
}

// Direct-to-canvas output (F.srgbOut) skips the renderer's full-screen sRGB output pass; the
// custom shaders then encode sRGB themselves (see encodeOutputTSL / uDirectOut, declared above).

// Tile-memory friendly MSAA on Apple/mobile GPUs: three.js stores every multisampled color
// sample and the depth buffer back to memory after each pass. This app renders a single scene
// pass per frame and never reads them back, so the GPU can resolve in tile memory and drop the
// samples (storeOp "discard"), saving hundreds of MB of bandwidth per frame at Retina sizes.
const gpuPassOptions = { discardMultisample: false };
if (typeof GPUCommandEncoder !== "undefined") {
  const beginRenderPass = GPUCommandEncoder.prototype.beginRenderPass;
  GPUCommandEncoder.prototype.beginRenderPass = function (desc) {
    if (gpuPassOptions.discardMultisample && desc && desc.colorAttachments) {
      let multisampled = false;
      for (const ca of desc.colorAttachments) {
        if (ca && ca.resolveTarget) {
          ca.storeOp = "discard";
          multisampled = true;
        }
      }
      const ds = desc.depthStencilAttachment;
      if (multisampled && ds) {
        if (ds.depthStoreOp) ds.depthStoreOp = "discard";
        if (ds.stencilStoreOp) ds.stencilStoreOp = "discard";
      }
    }
    return beginRenderPass.call(this, desc);
  };
}

// ============================================================================
// Water, Refracted Ridge Caustics, Sun Shafts & Thin-Film Interference
// Ported from willwaltz/globes (src/render/shaders.ts, caustic_resolve.wgsl,
// src/reef/reefWGSL.ts) and willwaltz/webgpu-ray-tracing (src/rt/rtWGSL.ts)
//
// Performance architecture (measured with tools/bench.mjs at 2940x1660, 4x MSAA):
//  - Honeycomb pores: the per-pixel 27-cell (+8-cell) 3D Voronoi search is replaced by ONE
//    trilinear fetch of a precomputed, tileable Voronoi volume (tools/gen_honeycomb_volume.mjs);
//    the crater direction comes from screen-space derivatives of the stored f1 distance.
//  - Smooth, low-frequency water terms (fog, volumetric shaft in-scatter, thin-film optical path,
//    pore domain warp) are evaluated per-vertex and interpolated; only high-frequency detail
//    (caustics, pores, lighting) runs per-pixel.
//  - Closed meshes (staghorn, brain) and the seabed heightfield cull back faces (Houdini winds them clockwise);
//    world position/normal come straight from the geometry so no unused varyings are emitted.
//  - Integer powers use repeated squaring instead of pow() (no log2/exp2 per call).
//  - The seabed skips the pore field and the thin film (its pore strength is 0, sea_blend is 1).
//  - The ocean backdrop is drawn LAST with depth testing, so only uncovered pixels shade it.
//  - Rendering goes straight to the sRGB canvas (no full-screen output pass) and multisampled
//    attachments are resolved in tile memory without being stored.
// FEATURES_BASELINE reproduces the original per-pixel shader for A/B benchmarking
// (?bench=<label>&abv=baseline|optimized) and as a fallback if the volume fails to load.
// ============================================================================
const FEATURES_OPTIMIZED = {
  pores: "volume", perVertex: true, seabedPores: false, fastMath: true, bgLast: true, fragmentNode: true,
  srgbOut: true, discardMultisample: true, cullClosed: true, compactVaryings: true, seabedSpecialize: true,
  legacyLook: false,
};
const FEATURES_BASELINE = {
  pores: "procedural", perVertex: false, seabedPores: true, fastMath: false, bgLast: false, fragmentNode: false,
  srgbOut: false, discardMultisample: false, cullClosed: false, compactVaryings: false, seabedSpecialize: false,
  legacyLook: true,
};

// Integer powers by repeated squaring. `toVar()` keeps every intermediate a single WGSL variable:
// TSL does not create temporaries for multiply-referenced nodes inside vertex-stage varyings, so
// without it the per-vertex water terms expand into a ~60 KB expression tree.
const sq = (x) => {
  const v = x.toVar();
  return v.mul(v);
};
const pow6 = (x) => {
  const x2 = sq(x).toVar();
  return sq(x2).mul(x2);
};
const pow8 = (x) => sq(sq(sq(x)));
const pow10 = (x) => {
  const x2 = sq(x).toVar();
  return sq(sq(x2)).mul(x2);
};
const pow12 = (x) => {
  const x4 = sq(sq(x)).toVar();
  return sq(x4).mul(x4);
};

// Fast analytic wave-focus caustic for 3D volumetric sun shafts (`globes_waterLight`)
function evalShaftWaveFocusTSL(uvIn, t, fast) {
  const uv = uvIn.toVar();
  const w1 = float(1.0).sub(abs(sin(uv.x.mul(2.1).add(uv.y.mul(1.5)).add(t.mul(0.85)))));
  const w2 = float(1.0).sub(abs(sin(uv.x.mul(1.6).sub(uv.y.mul(2.3)).sub(t.mul(0.72)).add(1.7))));
  const focus = fast ? pow6(w1).add(pow6(w2)) : pow(clamp(w1, 0.0, 1.0), 6.0).add(pow(clamp(w2, 0.0, 1.0), 6.0));
  return focus.mul(0.65);
}

// Domain-Warped 4-Wave Refracted Ridge & Star-Cusp Caustic Network (`ocean_shimmer` &
// `caustic_resolve.wgsl` in willwaltz/globes). All 8 wavevectors k_i satisfy
// dot(k_i, uSunDirWater) = 0 at reef-scale frequency (~0.26m cells), so interlocking polygonal
// catenary cells & star-cusps wrap across every coral branch, fan lobe, and the seabed.
// Returns the scalar focus; callers apply the chromatic R/G/B dispersion tint.
function evalCausticFocusTSL(posW, t, fast) {
  const a1 = dot(posW, vec3(9.80, -4.22, 3.69)).add(t.mul(0.95)).add(sin(dot(posW, vec3(-3.18, -3.31, 11.20)).sub(t.mul(0.72))).mul(1.28));
  const a2 = dot(posW, vec3(-4.20, -2.39, 9.55)).sub(t.mul(0.82)).add(cos(dot(posW, vec3(10.56, -1.96, -2.93)).add(t.mul(0.78)).add(1.9)).mul(1.28));
  const a3 = dot(posW, vec3(7.51, -5.14, 7.89)).add(t.mul(0.88)).add(sin(dot(posW, vec3(8.65, 0.23, -7.25)).sub(t.mul(0.66)).add(3.4)).mul(1.24));
  const a4 = dot(posW, vec3(14.80, 0.31, -12.18)).add(t.mul(1.18)).add(cos(dot(posW, vec3(-11.45, -2.46, 15.28)).add(t.mul(0.90))).mul(0.95));
  const w1 = float(1.0).sub(abs(sin(a1)));
  const w2 = float(1.0).sub(abs(sin(a2)));
  const w3 = float(1.0).sub(abs(cos(a3)));
  const w4 = float(1.0).sub(abs(sin(a4)));
  const ridgeSum = fast
    ? pow10(w1).add(pow10(w2)).add(pow10(w3)).add(pow12(w4).mul(0.60))
    : pow(clamp(w1, 0.0, 1.0), 10.5)
        .add(pow(clamp(w2, 0.0, 1.0), 10.5))
        .add(pow(clamp(w3, 0.0, 1.0), 10.5))
        .add(pow(clamp(w4, 0.0, 1.0), 12.0).mul(0.60));
  // Non-linear outer power makes ridge intersections & triple-junction cusps spike brighter
  return clamp(pow(ridgeSum.mul(0.66), 1.38), 0.0, 1.55);
}
const CAUSTIC_DISPERSION = vec3(1.02, 1.06, 1.15);

// Henyey-Greenstein anisotropic forward scattering phase (`globes_hg`), g = 0.58
function globesHGTSL(cosTheta, fast) {
  if (fast) {
    // (1 - g^2) / (4 pi) / denom^1.5 with constants folded, denom^1.5 = denom * sqrt(denom)
    const denom = max(float(1e-3), float(1.3364).sub(cosTheta.mul(1.16))).toVar();
    return float(0.052808).div(denom.mul(sqrt(denom)));
  }
  const g = float(0.58);
  const g2 = g.mul(g);
  const denom = max(float(1e-3), float(1.0).add(g2).sub(g.mul(2.0).mul(cosTheta)));
  return float(1.0).sub(g2).div(float(12.56637).mul(pow(denom, 1.5)));
}

// Thin-film iridescent coral mucus interference (`reef_film` in willwaltz/globes/src/reef/reefWGSL.ts)
// split into the (smooth) optical path difference and the per-pixel RGB interference fringe
function reefFilmOpdTSL(cosTheta, thickness) {
  return thickness.mul(2.84).mul(sqrt(max(float(0.05), float(1.0).sub(float(1.0).sub(cosTheta).mul(0.49)))));
}
function reefFilmFringeTSL(opd) {
  return cos(vec3(9.666, 11.855, 13.963).mul(opd)).mul(0.5).add(0.5);
}

// Project any 3D world point `p` along `uSunDirWater` to the overhead water surface `y = uWaterSurfaceY`
function projectToWaterSurfaceTSL(pIn, scale) {
  const p = pIn.toVar();
  const toSurf = max(float(0.0), uWaterSurfaceY.sub(p.y)).div(max(float(0.25), uSunDirWater.y));
  return vec2(p.x, p.z).add(vec2(uSunDirWater.x, uSunDirWater.z).mul(toSurf)).mul(scale);
}

// Raymarched 3D Volumetric Sun-Shaft In-Scattering (`globes_waterLight` & `light_resolve.wgsl` in willwaltz/globes)
function evalVolumetricSunShaftsTSL(rayOrigin, rayDir, maxDist, t, fast) {
  const effDist = min(maxDist, float(7.5)).toVar();
  const p1 = rayOrigin.add(rayDir.mul(effDist.mul(0.32))).toVar();
  const p2 = rayOrigin.add(rayDir.mul(effDist.mul(0.68))).toVar();

  const c1 = evalShaftWaveFocusTSL(projectToWaterSurfaceTSL(p1, 1.25), t, fast).mul(smoothstep(-0.4, 3.8, p1.y));
  const c2 = evalShaftWaveFocusTSL(projectToWaterSurfaceTSL(p2, 1.25), t, fast).mul(smoothstep(-0.4, 3.8, p2.y));

  const beamAvg = c1.add(c2).mul(0.5);
  const sunAlign = dot(rayDir, uSunDirWater);
  const hg = globesHGTSL(sunAlign, fast).mul(1.8).add(0.28);
  const distWeight = smoothstep(0.45, 5.5, effDist);
  return beamAvg.mul(hg).mul(distWeight);
}

// Volumetric sun beams v2 (expert review P10 / item 11): march the view ray through the water column and
// sample the blurred caustic pattern of the simulated surface at every step - the beams ARE the caustic
// focal lines extended along the refracted sun, so shafts in the water line up with the light on the sand
// and converge correctly in perspective. In the fragment stage the baked shadow map adds volumetric
// shadow beams behind the corals. `jitter` (0..1, per pixel) hides the step banding.
function evalShaftBeamsTSL(ro, rd, maxDist, steps, jitter, shadowed) {
  const len = min(maxDist, float(9.0)).toVar();
  let acc = float(0.0);
  for (let i = 0; i < steps; i++) {
    // fixed (unjittered) samples; the blur grows with distance like a diverging beam cone, so the
    // integrand stays smooth enough that `steps` samples neither band nor sparkle
    const tt = float(i + 0.5).div(steps).mul(len).toVar();
    const p = ro.add(rd.mul(tt)).toVar();
    const c = texture(ocean.causticRT.texture, ocean.causticUV(p)).level(tt.mul(0.32).add(4.6)).x;
    let beam = max(c.sub(0.78), 0.0);
    if (shadowed) beam = beam.mul(env.sunVis1(p));
    acc = acc.add(beam.mul(smoothstep(-0.05, 0.3, p.y)).mul(exp(tt.mul(-0.12))));
  }
  void jitter;
  const phase = globesHGTSL(dot(rd, uSunDirWater), true).mul(1.8).add(0.28);
  return acc.mul(len.div(steps)).mul(phase);
}
const ignTSL = () => fract(fract(dot(screenCoordinate.xy, vec2(0.06711056, 0.00583715))).mul(52.9829189));

// Output transform shared by all custom shaders: display-referred `mapped` -> linear for the
// renderer's sRGB output pass, or encoded directly when rendering straight to the canvas.
// The direct path fuses pow(x, 2.05) with the sRGB OETF (exactly sRGBTransferOETF(x^2.05)),
// sharing one log2 per channel instead of two pow() calls.
function encodeOutputTSL(mapped, srgbOut) {
  if (!srgbOut) return pow(mapped, vec3(2.05, 2.05, 2.05));
  const l = log2(max(mapped, vec3(1e-8, 1e-8, 1e-8))).toVar();
  const curve = exp2(l.mul(2.05 * 0.41666)).mul(1.055).sub(0.055);
  const toe = exp2(l.mul(2.05)).mul(12.92);
  // x^2.05 <= 0.0031308  <=>  x <= 0.0031308^(1/2.05) = 0.060024
  return mix(curve, toe, step(mapped, vec3(0.060024, 0.060024, 0.060024)));
}

// Animated Ocean Backdrop with Volumetric Sun Shafts, FBM Haze & Overhead Ripple Shimmer
// Ported from `OceanBackdrop.ts` and `oceanWGSL` in willwaltz/globes/src/render/shaders.ts
function buildOceanBackdropNode(dirNode, fast, srgbOut, legacy = false, F_BG = {}) {
  return Fn(() => {
    const d = normalize(dirNode);
    const t = uTime;

    // 1. Elevation depth gradient (`ocean_depthLight`); v2: the shared water colour (js/reef-env.js) that
    //    fogged geometry converges to, so the seabed dissolves into the backdrop without a horizon line
    const upMix = smoothstep(-0.42, 0.20, d.y);
    const surfaceMix = smoothstep(0.04, 0.72, d.y);
    let col = legacy ? mix(mix(uBgDeep, uBgMid, upMix), uBgShallow, surfaceMix.mul(0.72)) : env.waterColor(d);
    // structured light (haze, shafts, shimmer) fades out right at the horizon, where fogged sand meets it
    const horizonFade = legacy ? float(1.0) : smoothstep(-0.01, 0.08, d.y);

    // 2. Drifting underwater particulate haze (`ocean_fbm`)
    const fbmHaze = sin(d.x.mul(3.8).add(d.z.mul(2.7)).add(t.mul(0.15))).mul(cos(d.y.mul(4.2).sub(t.mul(0.12)))).mul(0.5).add(0.5);
    col = col.add(uShaftTint.mul(fbmHaze.sub(0.45)).mul(0.07).mul(uWaterAtmosphere).mul(horizonFade));

    // 3. Diagonal Volumetric Sun Shafts fanning down from upper-right (`ocean_shafts` in willwaltz/globes)
    //    v2: beams are parallel to the refracted sun, so in perspective they are radial lines through the
    //    sun / anti-sun points: index them by the azimuth around the sun axis instead of a fixed diagonal
    const sunCos = clamp(dot(d, uSunDirWater), -1.0, 1.0);
    let shaftCoord;
    let sunCone;
    if (legacy) {
      shaftCoord = d.x.mul(0.86).sub(d.y.mul(0.48)).add(d.z.mul(0.18));
      sunCone = smoothstep(-0.18, 0.34, d.y).mul(smoothstep(-0.55, 0.65, d.x));
    } else {
      const e1 = normalize(vec3(uSunDirWater.z.negate(), 0.0, uSunDirWater.x));
      const e2 = normalize(vec3(uSunDirWater.y.mul(uSunDirWater.x), uSunDirWater.z.mul(uSunDirWater.z).add(uSunDirWater.x.mul(uSunDirWater.x)).negate(), uSunDirWater.y.mul(uSunDirWater.z)));
      shaftCoord = atan(dot(d, e2), dot(d, e1)); // azimuth around the sun axis (radians)
      sunCone = smoothstep(-0.12, 0.4, d.y);
    }
    // integer azimuthal frequencies keep the pattern continuous across the atan2 branch cut
    const [fq1, fq2, fq3] = legacy ? [19.0, 33.0, 51.0] : [12.0, 21.0, 32.0];

    const sway1 = sin(t.mul(0.28)).mul(0.65);
    const sway2 = cos(t.mul(0.21).add(1.7)).mul(0.55);
    const b1 = sin(shaftCoord.mul(fq1).add(sway1)).mul(0.5).add(0.5);
    const b2 = sin(shaftCoord.mul(fq2).sub(sway2).add(1.4)).mul(0.5).add(0.5);
    const b3 = sin(shaftCoord.mul(fq3).sub(t.mul(0.32)).add(2.8)).mul(0.5).add(0.5);
    const s1 = pow(b1, 5.5);
    const s2 = fast ? pow8(b2) : pow(b2, 8.0);
    const s3 = fast ? pow10(b3).mul(b3) : pow(b3, 11.0);
    const radialShafts = s1.mul(0.52).add(s2.mul(0.38)).add(s3.mul(0.26)).mul(sunCone).mul(float(0.70).add(fbmHaze.mul(0.60)));

    // 4. 3D Raymarched Caustic God-Rays through the water column (v2: caustic-texture beams + shadow shafts)
    const marchShafts = legacy || F_BG.noBeams
      ? evalVolumetricSunShaftsTSL(cameraPosition, d, float(7.0), t, fast)
      : evalShaftBeamsTSL(cameraPosition, d, float(8.0), 12, 0.0, !!F_BG.beamShadow).mul(0.6); // shadow taps measured +1.1 ms: opt-in
    // (v2: the radial shafts are part of env.waterColor, shared with the fog of every surface)
    const totalShafts = (legacy ? radialShafts.mul(0.68) : float(0.0)).add(marchShafts.mul(0.42)).mul(uWaterAtmosphere).mul(float(0.35).add(uCausticsScale.mul(0.65)))
      .mul(legacy ? float(1.0) : env.u.shaftMood);
    col = col.add(uShaftTint.mul(totalShafts).mul(horizonFade));

    // 5. Overhead Water Surface Caustic Shimmer & Snell's Window (`ocean_shimmer` in willwaltz/globes)
    const overheadMask = smoothstep(0.04, 0.48, d.y);
    const surfProj = vec2(d.x, d.z).div(max(float(0.18), d.y.add(0.20))).mul(1.85);
    const overheadCaustic = evalShaftWaveFocusTSL(surfProj, t, fast);
    const snellWindow = smoothstep(0.85, 0.18, length(vec2(d.x, d.z).div(max(float(0.22), d.y.add(0.22)))));
    col = col.add(uShaftTint.mul(overheadCaustic).mul(overheadMask).mul(float(0.30).add(snellWindow.mul(0.55))).mul(0.36).mul(uCausticsScale).mul(horizonFade));

    // Overhead refracted sun bloom
    const sunPos = max(float(0.0), sunCos);
    const sunBloom = fast
      ? sq(pow8(sunPos)).mul(sq(sunPos)).mul(0.34).add(sq(sq(sunPos)).mul(sqrt(sunPos)).mul(0.12))
      : pow(sunPos, 18.0).mul(0.34).add(pow(sunPos, 4.5).mul(0.12));
    col = col.add(uKeyColor.mul(sunBloom).mul(uWaterAtmosphere));

    // 6. Soft reef back-glow halo (`globesGlow` in willwaltz/globes)
    const reefCenterDir = normalize(vec3(0.0, 1.05, 0.0).sub(cameraPosition));
    const reefHalo = pow(max(float(0.0), dot(d, reefCenterDir)), 3.2).mul(0.12);
    col = col.add(uWaterGlow.mul(reefHalo).mul(uWaterAtmosphere).mul(horizonFade));

    if (!legacy) return env.dither(encodeOutputTSL(env.shoulder(env.vignette(col).mul(env.u.exposure.mul(0.92))), srgbOut));
    return encodeOutputTSL(clamp(col.mul(0.94), 0.0, 1.0), srgbOut);
  })();
}

// Full-screen backdrop triangle drawn LAST at the far plane with depth testing, so the ocean
// shader only runs on pixels not covered by reef geometry (three's scene.backgroundNode sphere
// is drawn first with depth testing disabled, i.e. it shades every pixel).
function makeBackdropMesh(fast, srgbOut, legacy, F_BG) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  const mat = new THREE.MeshBasicNodeMaterial({ name: "ocean_backdrop", depthWrite: false, depthTest: true });
  mat.fog = false;
  const viewPos = cameraProjectionMatrixInverse.mul(vec4(positionGeometry.xy, 1.0, 1.0));
  const rayWorld = varying(cameraWorldMatrix.mul(vec4(viewPos.xyz.div(viewPos.w), 0.0)).xyz);
  mat.vertexNode = vec4(positionGeometry.xy, 1.0, 1.0);
  // fragmentNode bypasses the (unused) lighting model entirely
  mat.fragmentNode = vec4(buildOceanBackdropNode(rayWorld, fast, srgbOut, legacy, F_BG), 1.0);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = "ocean_backdrop";
  mesh.frustumCulled = false;
  mesh.renderOrder = 1000;
  return mesh;
}

// ---------------------------------------------------------------------------
// Precomputed tileable 3D Voronoi honeycomb volume (tools/gen_honeycomb_volume.mjs)
//   R = dRoundA / D_SCALE, G = per-cell random, B = dRoundB / D_SCALE, A = f1A / F_SCALE
// ---------------------------------------------------------------------------
const HONEY_N = 128;
const HONEY_TILE_A = 8; // primary cells per texture period
const HONEY_D_SCALE = 1.25;
const HONEY_F_SCALE = 1.5;
const honeyFieldTex = new THREE.Data3DTexture(new Uint8Array(HONEY_N * HONEY_N * HONEY_N * 4), HONEY_N, HONEY_N, HONEY_N);
honeyFieldTex.format = THREE.RGBAFormat;
honeyFieldTex.type = THREE.UnsignedByteType;
honeyFieldTex.minFilter = THREE.LinearFilter;
honeyFieldTex.magFilter = THREE.LinearFilter;
honeyFieldTex.wrapS = THREE.RepeatWrapping;
honeyFieldTex.wrapT = THREE.RepeatWrapping;
honeyFieldTex.wrapR = THREE.RepeatWrapping;
honeyFieldTex.generateMipmaps = false;
honeyFieldTex.unpackAlignment = 1;
const honeycombReady = fetchBinaryWithFallback(`./textures/honeycomb_field_${HONEY_N}.bin`, honeyFieldTex.image.data.byteLength)
  .then((buf) => {
    honeyFieldTex.image.data.set(new Uint8Array(buf));
    honeyFieldTex.needsUpdate = true;
    return true;
  })
  .catch((e) => {
    console.warn("Honeycomb volume unavailable; falling back to the procedural 3D Voronoi:", e);
    return false;
  });

// Exact procedural 27-cell (+8-cell) 3D Voronoi honeycomb (original per-pixel path, FEATURES_BASELINE)
function proceduralHoneycombTSL(pQuery, freqU) {
  const pScaledA = pQuery.mul(freqU);
  const bA = floor(pScaledA);
  const fA = fract(pScaledA);
  let f1A = float(10.0);
  let f2A = float(10.0);
  let bestVecA = vec3(0.0, 0.0, 0.0);
  let bestRandA = float(0.5);
  for (let i = 0; i < 27; i++) {
    const [ox, oy, oz] = OFFSETS_27[i];
    const off = vec3(ox, oy, oz);
    const h = hash33TSL(bA.add(off));
    const seedPos = off.add(float(0.5)).add(h.sub(0.5).mul(0.80));
    const dVec = fA.sub(seedPos);
    const d = length(dVec);
    const m1 = step(d, f1A);
    f2A = mix(min(f2A, d), f1A, m1);
    f1A = min(f1A, d);
    bestVecA = mix(bestVecA, dVec, m1);
    bestRandA = mix(bestRandA, h.x, m1);
  }
  // Secondary fine calice pores (freq_b = freq_a * 1.85)
  const pScaledB = pQuery.mul(freqU.mul(1.85)).add(vec3(11.3, 7.1, 19.7)).sub(0.5);
  const bB = floor(pScaledB);
  const fB = fract(pScaledB);
  let f1B = float(10.0);
  let f2B = float(10.0);
  for (let i = 0; i < 8; i++) {
    const [ox, oy, oz] = OFFSETS_8[i];
    const off = vec3(ox, oy, oz);
    const h = hash33TSL(bB.add(off));
    const seedPos = off.add(h.sub(0.5).mul(0.74));
    const d = length(fB.sub(seedPos));
    const m1 = step(d, f1B);
    f2B = mix(min(f2B, d), f1B, m1);
    f1B = min(f1B, d);
  }
  return {
    dRoundA: f2A.sub(f1A).sub(max(float(0.0), f1A.sub(0.22)).mul(0.20)),
    dRoundB: f2B.sub(f1B).sub(max(float(0.0), f1B.sub(0.24)).mul(0.18)),
    bestRandA,
    bestVecA,
  };
}

// Photogrammetry coral-limestone bump + 3D parallax (PolyHaven coral_stone_wall + coral_fort_wall_01).
// Uses 3-plane blended projection with view-ray parallax driven by pore/mesh relief (3 mipmapped 2D taps, zero pre-taps).
function sampleCoralScanBumpTSL(pW, normW, viewDir, scale, parallaxOffset) {
  const an = abs(normW);
  const w = sq(sq(an)).toVar();
  const ws = w.div(w.x.add(w.y).add(w.z).add(1e-5)).toVar();
  const ndv = max(abs(dot(normW, viewDir)), 0.36);
  const vTan = viewDir.sub(normW.mul(dot(viewDir, normW))).div(ndv);
  const pPar = pW.sub(vTan.mul(parallaxOffset)).mul(scale).toVar();
  const sXY = texture(coralBumpNH, vec2(pPar.x, pPar.y)).toVar();
  const sXZ = texture(coralBumpNH, vec2(pPar.x, pPar.z)).toVar();
  const sYZ = texture(coralBumpNH, vec2(pPar.z, pPar.y)).toVar();
  const nXY = sXY.xy.mul(2.0).sub(1.0);
  const nXZ = sXZ.xy.mul(2.0).sub(1.0);
  const nYZ = sYZ.xy.mul(2.0).sub(1.0);
  const tVec = vec3(nXY.x, nXY.y, 0.0).mul(ws.z)
    .add(vec3(nXZ.x, 0.0, nXZ.y).mul(ws.y))
    .add(vec3(0.0, nYZ.y, nYZ.x).mul(ws.x));
  const surfPerturb = tVec.sub(normW.mul(dot(tVec, normW)));
  const scanAo = sXY.z.mul(ws.z).add(sXZ.z.mul(ws.y)).add(sYZ.z.mul(ws.x));
  const scanH = sXY.w.mul(ws.z).add(sXZ.w.mul(ws.y)).add(sYZ.w.mul(ws.x));
  return { perturb: surfPerturb, ao: scanAo, h: scanH };
}

// Precomputed-volume honeycomb: ONE full-screen 3D trilinear fetch in Wide/Hero, plus true 3D volumetric
// crater parallax & finite-difference smooth gradient inside the close-up (< 1.85 m) macro branch (.level(0)).
function volumeHoneycombTSL(pQuery, freqU, normW, camDist = null, viewDir = null, parallaxAmt = null) {
  const uvw0 = pQuery.mul(freqU.div(HONEY_TILE_A)).toVar();
  const field0 = texture3D(honeyFieldTex, uvw0);
  const dRoundA = field0.x.mul(HONEY_D_SCALE).toVar();
  const dRoundB = field0.z.mul(HONEY_D_SCALE).toVar();
  const bestRandA = field0.y.toVar();
  const f1 = field0.w.mul(HONEY_F_SCALE);
  const dpx = dFdx(pQuery);
  const dpy = dFdy(pQuery);
  const r1 = cross(dpy, normW);
  const r2 = cross(normW, dpx);
  const det = dot(dpx, r1);
  const bestVec = r1.mul(dFdx(f1)).add(r2.mul(dFdy(f1))).mul(sign(det).div(max(abs(det), float(1e-24)))).toVar();
  if (camDist) {
    // Close-up (< 1.85 m): apply true 3D volumetric view-ray crater parallax + smooth FD gradient (.level(0))
    If(camDist.lessThan(1.85), () => {
      let uvw = uvw0;
      let f0 = field0.w;
      if (viewDir && parallaxAmt) {
        const d0 = smoothstep(0.15, 0.44, dRoundA);
        const vTan = viewDir.sub(normW.mul(dot(viewDir, normW))).div(max(abs(dot(normW, viewDir)), 0.34));
        uvw = pQuery.sub(vTan.mul(d0.mul(0.0055).mul(parallaxAmt))).mul(freqU.div(HONEY_TILE_A)).toVar();
        const fPar = texture3D(honeyFieldTex, uvw).level(0);
        dRoundA.assign(fPar.x.mul(HONEY_D_SCALE));
        dRoundB.assign(fPar.z.mul(HONEY_D_SCALE));
        bestRandA.assign(fPar.y);
        f0 = fPar.w;
      }
      const t1 = normalize(cross(normW, vec3(0.31, 0.95, 0.07))).toVar();
      const t2 = cross(normW, t1);
      const e = float(0.16 * HONEY_TILE_A).div(freqU).div(HONEY_TILE_A);
      const fa = texture3D(honeyFieldTex, uvw.add(t1.mul(e))).level(0).w;
      const fb = texture3D(honeyFieldTex, uvw.add(t2.mul(e))).level(0).w;
      bestVec.assign(t1.mul(fa.sub(f0)).add(t2.mul(fb.sub(f0))));
    });
  }
  return {
    dRoundA,
    dRoundB,
    bestRandA,
    bestVecA: bestVec,
  };
}

// Dev-only stand-in with no texture fetch or cell search (isolates the pore shading math cost)
function constHoneycombTSL(pQuery, freqU) {
  const s = fract(pQuery.mul(freqU));
  return { dRoundA: s.x.mul(0.6), dRoundB: s.y.mul(0.6), bestRandA: s.z, bestVecA: s.sub(0.5) };
}

function buildCoralMaterial(sp, cfg, F) {
  const freqU = float(cfg.freq);
  const rawPoreU = float(cfg.poreStr);
  const hasPores = (cfg.poreStr > 0 || F.seabedPores) && !F.noPores;
  // The seabed's sea_blend is exactly 1, so its thin-film tint is the identity: skip it there
  const hasFilm = !(cfg.isSeabed && F.seabedSpecialize);
  // Smooth terms are computed in the vertex stage and interpolated when F.perVertex is set
  const smooth = F.perVertex ? (node) => varying(node) : (node) => node;

  const hasBump = !F.legacyLook && !F.noBump;
  const bumpGain = hasBump ? uBumpStrength : float(0.0);
  const coralDataS = attribute("_coral_data", "vec4");

  // ---- True 3D Vertex Geometric Displacement (seabed 360x360 ripple/gravel heightfield + coral skeleton relief) ----
  const rawWorldPos = modelWorldMatrix.mul(vec4(positionGeometry, 1.0)).xyz.toVar();
  const rawWorldNrm = normalize(modelWorldMatrix.mul(vec4(normalGeometry, 0.0)).xyz).toVar();
  let dispWorldPos = rawWorldPos;
  if (hasBump) {
    if (cfg.isSeabed) {
      const pV = vec2(rawWorldPos.x, rawWorldPos.z).toVar();
      const hsV = F.proceduralSand ? float(0.0) : uHoudiniSand;
      const puvV = clamp(pV.sub(SAND_PATCH_X0).div(SAND_PATCH_SIZE), 0.001, 0.999).toVar();
      const edgeV = min(min(puvV.x, puvV.y), min(float(1.0).sub(puvV.x), float(1.0).sub(puvV.y)));
      const wPV = smoothstep(0.0, 0.07, edgeV).mul(hsV);
      const fpV = texture(footprintTex, clamp(pV.sub(FOOT_X0).div(FOOT_SIZE), 0.001, 0.999)).level(0).x;
      const coralClear = smoothstep(0.03, 0.18, fpV);
      const hPatch = texture(sandPatchNH, puvV).level(0).w.sub(0.5).mul(0.076).mul(wPV);
      const apronMask = coralClear.mul(float(1.0).sub(smoothstep(0.35, 1.45, fpV))).mul(uScanBump);
      const hRock = texture(reefSubstrateNH, pV.mul(0.85)).level(0).w.sub(0.46).mul(0.034).mul(apronMask);
      const dispY = hPatch.add(hRock).mul(coralClear).mul(bumpGain);
      dispWorldPos = vec3(rawWorldPos.x, rawWorldPos.y.add(dispY), rawWorldPos.z).toVar();
    } else {
      const holdfastFade = float(1.0).sub(smoothstep(0.22, 0.88, coralDataS.w));
      const cUvV = vec2(rawWorldPos.x.add(rawWorldPos.y.mul(0.63)), rawWorldPos.z.sub(rawWorldPos.y.mul(0.63))).mul(2.4);
      const cDisp = texture(coralBumpNH, cUvV).level(0).w.sub(0.5).mul(0.0055).mul(holdfastFade).mul(uScanBump).mul(bumpGain);
      dispWorldPos = rawWorldPos.add(rawWorldNrm.mul(cDisp)).toVar();
    }
  }

  // World-space position & normal. F.compactVaryings builds them straight from the geometry
  // attributes, so three's positionLocal / normalView varyings (unused by this shader) are never
  // emitted: with ~1M triangles the per-vertex interpolant count is a real cost on tiled GPUs.
  const posWorld = F.compactVaryings ? varying(dispWorldPos) : positionWorld;
  const nrmWorld = F.compactVaryings ? varying(rawWorldNrm) : normalWorld;

  // ---- Smooth, low-frequency water & surface terms ----
  const camToP = posWorld.sub(cameraPosition);
  const camDist = length(camToP).toVar();
  const camRay = camToP.div(camDist).toVar();
  const shaftsTerm = F.noShafts
    ? float(0.0)
    : evalVolumetricSunShaftsTSL(cameraPosition, camRay, camDist, uTime, F.fastMath)
        .mul(uWaterAtmosphere)
        .mul(float(0.22).add(uCausticsScale.mul(0.32)));
  const filmTerm = hasFilm
    ? reefFilmOpdTSL(max(float(0.02), abs(dot(normalize(nrmWorld), camRay))), float(0.34).add(posWorld.y.mul(0.14)).add(coralDataS.y.mul(0.20)))
    : float(0.0);
  // x: depth fog factor, y: volumetric sun-shaft in-scatter along the view ray (`globes_waterLight`),
  // z: thin-film optical path difference (`reef_film`), driven by the macro normal & height
  const waterTerms = F.legacyLook
    ? smooth(vec3(smoothstep(2.8, 12.5, camDist).mul(clamp(uWaterAtmosphere, 0.0, 1.5)).mul(0.56), shaftsTerm, filmTerm))
    : null;
  // Water column v2 (expert review): per-channel Beer-Lambert transmittance of the view ray that closes
  // completely at the radial water wall, and the fog colour = the backdrop's water colour along the ray,
  // so fully fogged sand matches the backdrop exactly (no horizon line). Packed into two vec4 varyings.
  const fogA = F.legacyLook ? null : (() => {
    const f = env.fogTerms(posWorld);
    return { tShafts: smooth(vec4(f.T, shaftsTerm)), colFilm: smooth(vec4(f.color, filmTerm)) };
  })();
  // Organic 3D domain warp matching Houdini's rt_shader_wrangle (+ radial slit-pore stretch on fan petals)
  let pQueryNode = posWorld.add(sin(posWorld.mul(11.5).add(vec3(1.3, 4.7, 8.2))).mul(0.011));
  if (cfg.isFan) {
    const fanRad = normalize(vec3(posWorld.x, posWorld.y.mul(0.5), posWorld.z).add(vec3(1e-4, 1e-4, 1e-4))).toVar();
    pQueryNode = pQueryNode.toVar();
    pQueryNode = pQueryNode.sub(fanRad.mul(dot(pQueryNode, fanRad).mul(0.34)));
  }
  const pQuery = hasPores ? smooth(pQueryNode) : null;

  const evalHoudiniVEXShader = Fn(() => {
    const wallCd = attribute("_wall_cd", "vec3");
    const cavityCd = attribute("_cavity_cd", "vec3");
    const coralData = attribute("_coral_data", "vec4"); // x: 14-ray baked AO, y: tip_mask, z: pore_depth, w: sea_blend

    const bakedAo = coralData.x;
    const tipMask = coralData.y;
    const meshPore = coralData.z;
    const seaBlend = coralData.w;

    const posW = posWorld;
    const viewDir = normalize(cameraPosition.sub(posW));
    let normW = normalize(nrmWorld);
    // Flip normal if viewing back-face on double-sided foliose fan petals
    const facing = step(0.0, dot(normW, viewDir)).mul(2.0).sub(1.0);
    normW = normW.mul(facing);

    // ---- 3D honeycomb calice pores + volumetric parallax + PolyHaven scan bump ----
    let holeDepth = float(0.0);
    let cellRand = float(0.5);
    let N = normW;
    let rimOverhangShadow = float(1.0);
    let scanBumpAo = float(1.0);
    let Cd = wallCd;
    const bumpDistFade = float(1.0).sub(smoothstep(2.2, 6.2, camDist)).toVar();
    if (hasPores) {
      const tipSuppress = float(1.0).sub(smoothstep(0.42, 0.88, tipMask).mul(0.94));
      const defocusP = F.legacyLook ? float(0.0) : smoothstep(0.3, 1.4, abs(camDist.sub(uFocusDist))).mul(uDofAmt);
      const effPoreStr = mix(rawPoreU, float(0.0), smoothstep(0.0, 0.95, seaBlend)).mul(uPoreStrength).mul(float(1.0).sub(defocusP.mul(0.55))).toVar();
      const parAmt = hasBump ? effPoreStr.mul(bumpGain).mul(float(1.0).sub(smoothstep(1.8, 4.6, camDist))) : null;
      const hc =
        F.pores === "volume"
          ? volumeHoneycombTSL(pQuery, freqU, normW, F.legacyLook || F.noFdGrad ? null : camDist, hasBump ? viewDir : null, parAmt)
          : F.pores === "const"
            ? constHoneycombTSL(pQuery, freqU)
            : proceduralHoneycombTSL(pQuery, freqU);

      cellRand = hc.bestRandA;
      // Trabecular wall thickness (~65% plump calcified coral skeleton wall, ~35% deep rounded oval calice cavities)
      const wallW = float(0.17).add(hc.bestRandA.mul(0.05)).add(tipMask.mul(0.22)).add(seaBlend.mul(0.16));
      const holeA = smoothstep(wallW, wallW.add(0.20), hc.dRoundA);
      const holeB = smoothstep(wallW.add(0.04), wallW.add(0.22), hc.dRoundB);

      const procHole = clamp(holeA.mul(0.78).add(holeB.mul(0.22)).mul(tipSuppress).mul(effPoreStr), 0.0, 1.0);
      holeDepth = cfg.isVerrucose
        ? clamp(max(meshPore.mul(0.80).mul(uPoreStrength), procHole.mul(0.35)), 0.0, 1.0)
        : clamp(max(procHole, meshPore.mul(0.45).mul(effPoreStr)), 0.0, 1.0);

      // Perturb surface normal N strongly inward on the steep rounded crater wall of each honeycomb cell!
      const cellTangent = normalize(hc.bestVecA.sub(normW.mul(dot(hc.bestVecA, normW))).add(normW.mul(1e-4)));
      const funnelSlope = sin(clamp(hc.dRoundA.sub(wallW.mul(0.5)).div(0.24), 0.0, 1.0).mul(3.14159265))
        .mul(tipSuppress)
        .mul(effPoreStr);
      let nAccum = normW.sub(cellTangent.mul(funnelSlope.mul(hasBump ? 0.92 : 0.68))).toVar();

      // Stony calcified limestone micro-bump + parallax relief (PolyHaven coral_stone_wall + species sculpt)
      if (hasBump) {
        const scanW = bumpDistFade.mul(uScanBump).mul(bumpGain).toVar();
        const sb = sampleCoralScanBumpTSL(posW, normW, viewDir, 2.8, holeDepth.mul(0.0038).mul(scanW));
        // Stronger stony micro-relief on trabecular walls, Acropora trunks, and brain gyri; gentler on glossy tips
        const wallBumpW = mix(float(0.38), float(0.92), float(1.0).sub(holeDepth)).mul(mix(float(1.0), float(0.42), tipMask));
        nAccum = nAccum.add(sb.perturb.mul(scanW.mul(wallBumpW).mul(1.06)));
        scanBumpAo = mix(float(1.0), sb.ao, clamp(scanW.mul(0.46), 0.0, 0.68));

        if (sp === "brain") {
          // Comb-like meandroid septa teeth (~3.2 mm spacing) cutting across the brain-coral gyri slopes
          const horizN = vec3(normW.x, 0.0, normW.z);
          const slopeLen = length(horizN);
          const ridgeDir = horizN.div(max(slopeLen, 1e-3));
          const alongDir = vec3(ridgeDir.z.negate(), 0.0, ridgeDir.x);
          const septaPh = dot(posW, alongDir).mul(195.0).add(sin(dot(posW, ridgeDir).mul(42.0)).mul(1.6));
          const septaSlope = cos(septaPh).mul(smoothstep(0.12, 0.65, slopeLen)).mul(float(1.0).sub(seaBlend));
          nAccum = nAccum.add(alongDir.mul(septaSlope.mul(scanW.mul(0.58))));
          scanBumpAo = scanBumpAo.mul(float(1.0).sub(float(0.5).sub(sin(septaPh).mul(0.5)).mul(smoothstep(0.15, 0.65, slopeLen)).mul(0.20)));
        } else if (cfg.isFan) {
          // Radial vascular costae ribs radiating out across the cupped foliose fan petals
          const radDir = normalize(vec3(posW.x, 0.0, posW.z).add(vec3(1e-4, 0.0, 1e-4)));
          const tangRib = normalize(cross(normW, radDir).add(vec3(1e-4, 0.0, 0.0)));
          const ribAng = atan(posW.z, posW.x);
          const ribPh = ribAng.mul(46.0).add(sin(posW.y.mul(16.0)).mul(1.3));
          nAccum = nAccum.add(tangRib.mul(cos(ribPh).mul(scanW.mul(0.46)).mul(float(1.0).sub(tipMask.mul(0.6)))));
        }
      }
      N = normalize(nAccum);

      // Directional crater lip self-shadowing: overhanging trabecular wall shadows the side of the hole facing away from the light!
      rimOverhangShadow = clamp(
        float(1.0).sub(holeA.mul(effPoreStr).mul(hasBump ? 0.48 : 0.38).mul(clamp(dot(cellTangent, uKeyDir).negate().add(0.24), 0.0, 1.0))),
        hasBump ? 0.52 : 0.60,
        1.0
      );

      // Brighten plump raised trabecular walls and fill rounded calice cavities with Karma-matched Random-Walk SSS coral shadow!
      const wallRidge = float(1.0).sub(holeA).mul(tipSuppress).mul(effPoreStr);
      const raisedWall = wallCd.mul(float(1.0).add(wallRidge.mul(0.15)));
      const karmaCavityCd = max(cavityCd, wallCd.mul(vec3(0.42, 0.34, 0.44)));
      Cd = mix(raisedWall, karmaCavityCd, pow(holeDepth, 0.95).mul(0.76));
    }

    // ---- Seabed v2 + Steep Parallax Occlusion Mapping (POM) + Houdini Horizon Relief ----
    let sandAo = float(1.0);
    if (cfg.isSeabed && !F.legacyLook && !F.noSand) {
      const p = vec2(posW.x, posW.z).toVar();
      const d1 = texture(sandTex, p.mul(1.0 / 2.3)); // 2.3 m repeat: patches + ripple amplitude
      const d2 = texture(sandTex, p.mul(1.0 / 0.37).add(vec2(0.37, 0.71))); // 0.37 m repeat: grain + flecks
      // Houdini bake: patch (9 x 9 m around the reef) + 60 cm grain tile; wP fades the patch out at its border
      const hs = F.proceduralSand ? float(0.0) : uHoudiniSand;
      const puv = p.sub(SAND_PATCH_X0).div(SAND_PATCH_SIZE).toVar();
      const edge = min(min(puv.x, puv.y), min(float(1.0).sub(puv.x), float(1.0).sub(puv.y)));
      const wP = smoothstep(0.0, 0.07, edge).mul(hs).toVar();
      const fp = texture(footprintTex, p.sub(FOOT_X0).div(FOOT_SIZE)); // r: distance to coral (m), g: coral top (m)

      // 2-tap Parallax Relief on PolyHaven reef gravel/rock substrate (seabed vertices already carry 3D sandPatchNH displacement)
      const pomFade = float(1.0).sub(smoothstep(1.6, 5.2, camDist)).mul(bumpGain).toVar();
      const vStepXZ = vec2(viewDir.x, viewDir.z).div(max(viewDir.y, float(0.28))).toVar();
      const apronW = float(0.26).add(float(0.74).mul(float(1.0).sub(smoothstep(0.25, 1.65, fp.x)))).mul(uScanBump).toVar();
      let subUV = p.mul(0.92);
      if (hasBump) {
        const pomDepth = float(0.015).mul(apronW).mul(pomFade);
        const hS0 = float(1.0).sub(texture(reefSubstrateNH, subUV).w);
        subUV = p.sub(vStepXZ.mul(hS0.mul(pomDepth))).mul(0.92);
      }
      const subSmp = texture(reefSubstrateNH, subUV).toVar();
      const pnh = texture(sandPatchNH, puv); // xy: normal x,z  z: cavity+horizon AO  w: height
      const pmat = texture(sandPatchMat, puv); // x: luminance  y: detritus  z: shell grit  w: coralline
      const grainFade = float(1.0).sub(smoothstep(0.6, 2.2, camDist));
      const gtUV = p.sub(vStepXZ.mul(grainFade.mul(0.003).mul(bumpGain))).div(SAND_GRAIN_TILE);
      const gt = texture(sandGrainTex, gtUV); // xy: grain normal x,z  z: albedo  w: shell bits

      const along = dot(p, vec2(SURGE_D.x, SURGE_D.y));
      const across = dot(p, vec2(-SURGE_D.y, SURGE_D.x));
      const warp = sin(across.mul(1.7).add(sin(along.mul(0.9).add(across.mul(0.43))).mul(1.3))).mul(2.2).add(d1.y.mul(3.0));
      const ph = along.mul((2 * Math.PI) / 0.105).add(warp).toVar(); // ~10 cm crests
      const slope = cos(ph).add(cos(ph.mul(2.0)).mul(0.35)); // asymmetric ripple profile
      const aaFade = clamp(float(2.0).sub(fwidth(ph).mul(1.6)), 0.0, 1.0);
      const nearCoral = smoothstep(0.08, 0.6, fp.x);
      const ripAmp = d1.y.mul(0.55).add(0.2).mul(0.42).mul(aaFade).mul(nearCoral).mul(float(1.0).sub(wP)).mul(bumpGain).toVar();
      const hn = pnh.xy.mul(2.0).sub(1.0).mul(wP.mul(1.82).mul(bumpGain)); // baked vortex ripples, scour, burrows, grit
      const gn = gt.xy.mul(2.0).sub(1.0).mul(grainFade.mul(hs).mul(1.12).mul(bumpGain));
      const subN = subSmp.xy.mul(2.0).sub(1.0).mul(apronW.mul(bumpDistFade).mul(0.86).mul(bumpGain));
      N = normalize(
        N.sub(vec3(SURGE_D.x, 0.0, SURGE_D.y).mul(slope.mul(ripAmp)))
          .add(vec3(hn.x.add(gn.x).add(subN.x), 0.0, hn.y.add(gn.y).add(subN.y)))
      );
      // albedo: +-12% patches, ripple troughs darker (detritus), grain fading with distance,
      // sparse cream/pink flecks (shell grit, coralline crust), denser close to the reef
      const crest = sin(ph).mul(0.5).add(0.5);
      const fleck = d1.w.mul(float(0.15).add(float(0.85).mul(float(1.0).sub(smoothstep(0.25, 1.5, fp.x)))));
      Cd = mix(Cd, vec3(0.925, 0.9, 0.885), 0.35).mul(env.u.sandGain);
      Cd = Cd.mul(float(0.88).add(d1.x.mul(0.24)))
        .mul(float(1.0).sub(float(1.0).sub(crest).mul(0.14).mul(ripAmp.mul(2.5))))
        .mul(float(1.0).add(d2.z.sub(0.5).mul(0.05).mul(grainFade)));
      Cd = mix(Cd, mix(vec3(0.97, 0.9, 0.92), vec3(0.95, 0.66, 0.74), d2.x), fleck.mul(0.45).mul(float(1.0).sub(wP)));
      // Houdini + PolyHaven substrate material masks
      const hLum = mix(float(1.0), pmat.x.mul(2.0), wP);
      Cd = Cd.mul(hLum).mul(float(1.0).add(gt.z.sub(0.5).mul(0.26).mul(grainFade).mul(hs)));
      Cd = Cd.mul(mix(float(1.0), float(0.86).add(subSmp.w.mul(0.24)), apronW.mul(bumpDistFade).mul(0.55)));
      Cd = mix(Cd, Cd.mul(vec3(0.80, 0.76, 0.64)), pmat.y.mul(wP).mul(0.62));
      Cd = mix(Cd, vec3(0.97, 0.94, 0.9), max(pmat.z.mul(wP).mul(0.58), gt.w.mul(grainFade).mul(hs).mul(0.5)));
      Cd = mix(Cd, vec3(0.93, 0.64, 0.72), pmat.w.mul(wP).mul(0.55));
      // contact occlusion + Houdini cavity & sun-horizon self-shadow + PolyHaven gravel crevice AO
      const tall = smoothstep(0.15, 0.9, fp.y.sub(posW.y));
      const patchAo = pow(pnh.z, clamp(bumpGain, 0.0, 2.0));
      const rockAo = mix(float(1.0), subSmp.z, apronW.mul(bumpDistFade).mul(0.55).mul(clamp(bumpGain, 0.0, 1.6)));
      sandAo = float(1.0).sub(float(1.0).sub(smoothstep(0.0, 0.38, fp.x)).mul(float(0.18).add(tall.mul(0.2))))
        .mul(mix(float(1.0), patchAo, wP))
        .mul(rockAo);
    }

    // Velvety calcium-carbonate macro wrap diffuse (on normW so pastel shadow sides never darken) +
    // full-contrast un-wrapped raking micro-bump relief (on N - normW so 3D bump & ripples pop crisply!)
    const ndlKey = clamp(dot(normW, uKeyDir).add(0.44).div(1.44), 0.0, 1.0);
    const ndlFill = clamp(dot(N, uFillDir).add(0.52).div(1.52), 0.0, 1.0);
    const ndlRim = clamp(dot(N, uRimDir), 0.0, 1.0);
    const fresnel = pow(clamp(float(1.0).sub(max(0.0, dot(N, viewDir))), 0.0, 1.0), 2.3);
    const rakingKey = normalize(vec3(uKeyDir.x.mul(cfg.isSeabed ? 2.3 : 1.9), float(cfg.isSeabed ? 0.44 : 0.62), uKeyDir.z.mul(cfg.isSeabed ? 2.3 : 1.9)));
    const microDelta = dot(N, rakingKey).sub(dot(normW, rakingKey));
    const microRelief = hasBump
      ? clamp(float(1.0).add(microDelta.mul(cfg.isSeabed ? 1.72 : 1.32)), 0.34, 1.54)
      : float(1.0);

    // Subsurface forward/back scatter glow on raised trabecular walls, polyp domes, & creamy tips (with Karma Random-Walk bleed into thin pore walls)
    const wallMask = clamp(float(1.0).sub(holeDepth.mul(0.72)), 0.0, 1.0);
    const sssFactor = clamp(tipMask.mul(0.76).add(wallMask.mul(0.46)).mul(wallMask), 0.0, 1.0);
    const sssTint = mix(wallCd, vec3(1.0, 0.96, 0.98), 0.32);
    const sssGlow = sssTint
      .mul(sssFactor)
      .mul(float(0.30).add(fresnel.mul(0.30)).add(ndlRim.mul(0.18)))
      .mul(uSssIntensity.div(0.28))
      .mul(mix(float(1.0), float(0.18), uNightGlow));

    // Dual-lobe specular matched to Karma mtlxstandard_surface:
    // 1) Rough calcium-carbonate base lobe + 2) Tight wet-mucus clearcoat (coat_roughness 0.068, IOR 1.37) on coral tips & walls
    const filmOpd = F.legacyLook ? waterTerms.z : fogA.colFilm.w;
    const filmTint = hasFilm
      ? mix(vec3(1.0, 1.0, 1.0), reefFilmFringeTSL(filmOpd), float(1.0).sub(seaBlend).mul(0.45))
      : vec3(1.0, 1.0, 1.0);
    const H = normalize(uKeyDir.add(viewDir));
    const ndh = max(0.0, dot(N, H));
    const specPower = clamp(float(24.0).sub(uRoughScale.mul(14.0)), 6.0, 28.0);
    const specGain = cfg.isSeabed ? 0.13 : 0.14;
    const baseSpec = filmTint.mul(pow(ndh, specPower).mul(specGain));
    let specRaw = baseSpec.mul(wallMask).mul(rimOverhangShadow);
    if (!cfg.isSeabed) {
      const coatPower = clamp(float(128.0).sub(uRoughScale.mul(72.0)), 24.0, 140.0);
      const coatMask = mix(float(0.34), float(1.0), tipMask).mul(wallMask).mul(float(1.0).sub(seaBlend));
      const coatSpec = mix(vec3(1.0, 1.0, 1.0), filmTint, 0.35).mul(pow(ndh, coatPower).mul(0.24).mul(coatMask).mul(rimOverhangShadow));
      specRaw = specRaw.add(coatSpec);
    }

    // Combine Houdini's 14-Ray Baked Vertex AO + Per-Pixel Pore Cavity AO + Scan Micro-AO + Rim Overhang Shadow
    const poreAo = mix(float(1.0), float(0.64), pow(holeDepth, 1.10));
    const rawAo = bakedAo.mul(poreAo).mul(scanBumpAo);
    const effAo = clamp(mix(float(1.0), rawAo, uAoIntensity), 0.16, 1.0);
    const shadow = rimOverhangShadow.mul(mix(float(1.0), bakedAo, float(0.35).mul(uAoIntensity)));

    if (!F.legacyLook) {
      // =======================================================================
      // Light transport v2 (expert review): one refracted sun drives the key light, the baked shadow
      // map (sunVis), and the projected-grid caustics of the simulated surface (js/ocean-surface.js,
      // adapted from jeantimex/webgpu-water). Caustics MULTIPLY the direct sun, so they take the
      // surface's own colour, vanish in shadow and conserve energy (no white veins on the sand);
      // light dims and shifts colour with depth; fog converges to the backdrop's water colour.
      // =======================================================================
      const nLit = normW.mul(sign(dot(normW, uKeyDir)).add(0.001)); // offset toward the lit side (thin petals)
      const vis = F.noShadow ? float(1.0) : env.sunVis(posW, nLit);
      const dl = env.depthLight(posW);
      // Caustic contrast is lower on the sand (value hierarchy: the floor must not be the brightest,
      // busiest area) and fades inside pores / occluded crevices
      const causMask = mix(float(0.25), float(1.0), wallMask).mul(smoothstep(0.18, 0.74, bakedAo));
      // (art director + undersea expert round 1: sand 0.45 so the floor is not the brightest, busiest area;
      //  coral 1.4 with gentler compression so soft 1.5-2.5x patches read on sun-facing coral)
      const caus = F.noCaustics
        ? vec3(1.0, 1.0, 1.0)
        : mix(vec3(1.0, 1.0, 1.0), cfg.isSeabed ? env.causticLight(posW, 0.45) : env.causticLight(posW, 1.4, 0.75), causMask);
      const seaTint = mix(vec3(1.0, 1.0, 1.0), env.u.seaTint, seaBlend.mul(cfg.isSeabed ? 0.45 : 0.75));
      // underwater shadows are soft and partly filled by forward-scattered sun (shadow/lit ~0.55 on sand);
      // caustics only exist in the directly lit part
      const sunBeam = uKeyColor.mul(ndlKey).mul(microRelief).mul(shadow).mul(dl.sun).mul(env.u.keyGain);
      const sunLit = caus.mul(vis).add(env.u.shadowFill.mul(float(1.0).sub(vis)).mul(mix(float(1.0), microRelief, 0.65)));
      const skyLight = mix(uSkyLow, uSkyHigh, N.y.mul(0.5).add(0.5)).mul(env.u.skyGain).mul(dl.amb).mul(mix(float(1.0), microRelief, 0.62));
      const fillLight = uFillColor.mul(ndlFill).mul(env.u.fillGain).mul(dl.amb).mul(mix(float(1.0), microRelief, 0.35));
      // brightest filaments get a faint luminous lift (15% toward the shaft tint, art-director note)
      const sparkle = uShaftTint.mul(max(caus.y.sub(1.35), 0.0)).mul(sunBeam).mul(vis).mul(cfg.isSeabed ? 0.04 : 0.1);
      const diffuse = Cd.mul(seaTint).mul(skyLight.add(fillLight).mul(sandAo).add(sunBeam.mul(sunLit).mul(mix(float(1.0), sandAo, 0.4)))).mul(effAo);
      // Coral material response (art-director item 8 + Karma Random-Walk SSS benchmark calibration):
      // velvet polyp sheen, warm subsurface band along the light terminator, back-lit transmission through thin petals,
      // and (Bio-Night) slowly pulsing bioluminescent polyp tips
      const sheen = mix(wallCd, vec3(1.0), 0.5).mul(env.u.sheenTint).mul(fresnel).mul(cfg.isFan ? 0.10 : 0.16)
        .mul(effAo).mul(wallMask).mul(clamp(env.u.keyGain.mul(1.28), 0.3, 1.0));
      const nDotL = dot(N, uKeyDir);
      const termBand = clamp(nDotL.add(0.5).div(1.5), 0.0, 1.0).sub(clamp(nDotL, 0.0, 1.0));
      const sssTerm = wallCd.mul(wallCd).mul(termBand).mul(vis).mul(wallMask).mul(uSssIntensity.div(0.28)).mul(0.44);
      let extra = sssTerm;
      if (cfg.isFan) {
        const back = clamp(dot(normW.negate(), uKeyDir), 0.0, 1.0);
        const towardSun = sq(sq(clamp(dot(viewDir.negate(), uKeyDir), 0.0, 1.0))); // only when looking into the light
        extra = extra.add(wallCd.mul(vec3(1.0, 0.85, 0.75)).mul(back.mul(sqrt(back))).mul(towardSun).mul(vis).mul(wallMask).mul(dl.sun).mul(0.68));
      }
      if (cfg.glow && hasPores) {
        // bioluminescent polyps + axial tips: saturated GFP/RFP glow inside calice cups and smooth
        // growing tips, varying smoothly across branches without grid blocks or Voronoi mosaic facets
        const branch = sin(posW.x.mul(2.7).add(posW.z.mul(2.1))).mul(cos(posW.y.mul(2.3).sub(posW.x.mul(1.4)))).mul(0.5).add(0.5);
        const region = smoothstep(0.18, 0.78, branch.add(tipMask.mul(0.55)));
        const pulse = sin(uTime.mul(0.9).add(cellRand.mul(6.283)).add(posW.y.mul(3.1))).mul(0.32).add(0.68);
        const cupGlow = holeDepth.mul(step(0.50, cellRand)).mul(region).mul(pulse).mul(1.95);
        const tipGlow = sq(tipMask).mul(tipMask).mul(sin(uTime.mul(0.85).add(posW.y.mul(3.4))).mul(0.25).add(0.75)).mul(0.38);
        extra = extra.add(vec3(cfg.glow[0], cfg.glow[1], cfg.glow[2]).mul(cupGlow.add(tipGlow)).mul(uNightGlow));
      }
      const sssLit = sssGlow.mul(float(1.0).sub(clamp(ndlKey.mul(vis), 0.0, 1.0).mul(tipMask).mul(0.50)));
      const shadedRaw = diffuse
        .add(sparkle)
        .add(sssLit)
        .add(sheen)
        .add(extra)
        .add(specRaw.mul(mix(float(0.35), float(1.0), vis)).mul(float(1.0).sub(smoothstep(0.3, 1.4, abs(camDist.sub(uFocusDist))).mul(uDofAmt))));
      const defocus = smoothstep(0.3, 1.4, abs(camDist.sub(uFocusDist))).mul(uDofAmt);
      const T = fogA.tShafts.xyz.mul(float(1.0).sub(defocus.mul(0.15)));
      const shaded = shadedRaw.mul(T).add(fogA.colFilm.xyz.mul(vec3(1.0).sub(T))).add(uShaftTint.mul(fogA.tShafts.w).mul(0.14));
      return env.dither(encodeOutputTSL(env.shoulder(env.vignette(shaded).mul(env.u.exposure.mul(0.92))), F.srgbOut));
    }
    const spec = specRaw;

    // =========================================================================
    // Refracted Water Surface Caustic Network + Beam Cross-Section Projection
    // Ported from willwaltz/globes (`globes_sandShade`, `caustic_resolve.wgsl`, `ocean_shimmer`)
    // (legacy look, kept for A/B: ?features=optimized+legacyLook)
    // =========================================================================
    const causticRaw = F.noCaustics ? vec3(0.0, 0.0, 0.0) : evalCausticFocusTSL(posW, uTime, F.fastMath).mul(CAUSTIC_DISPERSION);

    // Sun-facing weight along the key beam using macro `normW` so caustics glide smoothly across coral & seabed!
    const keyFacing = clamp(dot(normW, uKeyDir).add(0.22).div(1.22), 0.0, 1.0);
    const poreCausticMask = mix(float(0.16), float(1.0), wallMask).mul(smoothstep(0.18, 0.74, bakedAo));

    // Beer-Lambert water path absorption from the water surface to the coral point (`globes_sunIn`)
    const sunWaterTrans = exp(uWaterExt.negate().mul(max(float(0.0), uWaterSurfaceY.sub(posW.y)).mul(0.09).mul(uWaterAtmosphere)));
    const causticNet = causticRaw.mul(keyFacing).mul(poreCausticMask).mul(sunWaterTrans).mul(uCausticsScale);

    // Physical underwater trough divergence vs ridge convergence (`willwaltz/globes` `globes_sandShade` & `reef_shade`)
    const troughShade = mix(float(0.82), float(0.42), clamp(uCausticsScale.mul(0.80), 0.0, 1.0));
    const seaTint = mix(vec3(1.0, 1.0, 1.0), vec3(0.58, 0.50, 0.70), seaBlend.mul(0.75));

    const skyLight = mix(uSkyLow, uSkyHigh, N.y.mul(0.5).add(0.5)).mul(0.36);
    const keyLight = uKeyColor.mul(ndlKey).mul(shadow).mul(troughShade);
    const fillLight = uFillColor.mul(ndlFill).mul(0.26);

    // Refracted caustic sunlight filaments
    const causticContribution = mix(wallCd.mul(seaTint).mul(uKeyColor), uShaftTint, 0.56)
      .mul(causticNet)
      .mul(shadow)
      .mul(1.95);

    const shadedRaw = Cd.mul(seaTint).mul(skyLight.add(keyLight).add(fillLight)).mul(effAo)
      .add(causticContribution)
      .add(sssGlow)
      .add(vec3(0.94, 0.90, 1.0).mul(fresnel).mul(0.14).mul(effAo).mul(wallMask))
      .add(spec);

    const shaded = mix(shadedRaw, uWaterGlow, waterTerms.x).add(uShaftTint.mul(waterTerms.y).mul(0.14));
    return encodeOutputTSL(clamp(shaded.mul(0.92), 0.0, 1.0), F.srgbOut);
  });

  const mat = new THREE.MeshBasicNodeMaterial({
    name: `houdini_${sp}`,
    // Houdini exports clockwise (outward) winding, so for the closed, consistently wound meshes and
    // the seabed heightfield BackSide culls the hidden far-side triangles; open meshes (fan petals,
    // polyp tubes) stay double-sided.
    side: F.cullClosed && cfg.cullBack ? THREE.BackSide : THREE.DoubleSide,
  });
  if (F.compactVaryings) {
    mat.vertexNode = cameraProjectionMatrix.mul(cameraViewMatrix).mul(vec4(dispWorldPos, 1.0));
  }
  if (F.flat) {
    mat.colorNode = attribute("_wall_cd", "vec3");
  } else if (F.fragmentNode) {
    // Output the shaded color directly (skips MeshBasic's no-op lighting model)
    mat.fragmentNode = vec4(evalHoudiniVEXShader(), 1.0);
  } else {
    mat.colorNode = evalHoudiniVEXShader();
  }
  return mat;
}

const speciesList = ["staghorn", "brain", "table", "sponge", "seabed"];
const speciesConfig = {
  staghorn: { freq: 76.0, poreStr: 0.95, isFan: false, isVerrucose: false, cullBack: true, glow: [0.30, 0.95, 1.0] },
  brain:    { freq: 62.0, poreStr: 0.36, isFan: false, isVerrucose: false, cullBack: true, glow: [1.0, 0.42, 0.88] },
  table:    { freq: 84.0, poreStr: 0.84, isFan: true,  isVerrucose: false, cullBack: false, glow: [0.62, 0.48, 1.0] },
  sponge:   { freq: 68.0, poreStr: 0.52, isFan: false, isVerrucose: true,  cullBack: false, glow: [0.35, 1.0, 0.78] },
  seabed:   { freq: 76.0, poreStr: 0.00, isFan: false, isVerrucose: false, cullBack: true, isSeabed: true },
};

// Resolve a feature-set spec such as "optimized", "baseline+volume" or "optimized+pixel+bgFirst"
// ("+" may arrive URL-decoded as a space, so spaces and "." also separate tokens)
function resolveFeatures(spec) {
  const tokens = spec.split(/[+\s.]+/).filter(Boolean);
  const F = { ...(tokens[0] === "baseline" ? FEATURES_BASELINE : FEATURES_OPTIMIZED) };
  for (const tok of tokens.slice(1)) {
    if (tok === "volume" || tok === "procedural") F.pores = tok;
    else if (tok === "vertex") F.perVertex = true;
    else if (tok === "pixel") F.perVertex = false;
    else if (tok === "seabedPores") F.seabedPores = true;
    else if (tok === "fastMath") F.fastMath = true;
    else if (tok === "slowMath") F.fastMath = false;
    else if (tok === "bgLast") F.bgLast = true;
    else if (tok === "bgFirst") F.bgLast = false;
    else if (tok === "flat") F.flat = true;
    else if (tok === "noCaustics") F.noCaustics = true;
    else if (tok === "legacyLook") F.legacyLook = true;
    else if (tok === "newLook") F.legacyLook = false;
    else if (tok === "noShadow") F.noShadow = true;
    else if (tok === "noOcean") F.noOcean = true;
    else if (tok === "noSand") F.noSand = true;
    else if (tok === "noBump") F.noBump = true;
    else if (tok === "proceduralSand") F.proceduralSand = true;
    else if (tok === "noBeams") F.noBeams = true;
    else if (tok === "noFdGrad") F.noFdGrad = true;
    else if (tok === "beamShadow") F.beamShadow = true;
    else if (tok === "noOceanCaustics") F.noOceanCaustics = true;
    else if (tok === "noOceanMips") F.noOceanMips = true;
    else if (/^no(Fish|Flora|Props|Snow|Life)$/.test(tok)) (F.hideGroups || (F.hideGroups = [])).push(tok.slice(2).toLowerCase());
    else if (tok === "noShafts") F.noShafts = true;
    else if (tok === "fragNode") F.fragmentNode = true;
    else if (tok === "colorNode") F.fragmentNode = false;
    else if (tok === "direct") F.srgbOut = true;
    else if (tok === "outPass") F.srgbOut = false;
    else if (tok === "discard") F.discardMultisample = true;
    else if (tok === "store") F.discardMultisample = false;
    else if (tok === "noPores") F.noPores = true;
    else if (tok === "constPores") F.pores = "const";
    else if (tok === "cull") F.cullClosed = true;
    else if (tok === "noCull") F.cullClosed = false;
    else if (tok === "compact") F.compactVaryings = true;
    else if (tok === "noCompact") F.compactVaryings = false;
    else if (tok === "noBg") F.noBg = true;
    else if (tok === "seabedFilm") F.seabedSpecialize = false;
    else if (tok === "hideSeabed") F.hide = ["seabed"];
    else if (tok === "hideCoral") F.hide = speciesList.filter((s) => s !== "seabed");
  }
  return F;
}

// A render variant = coral material set + ocean backdrop strategy for a feature set.
// Backdrop nodes/meshes are shared per (fastMath, srgbOut) so switching variants never rebuilds them.
const backdropCache = new Map();
function getBackdrop(F) {
  const key = `${F.bgLast}|${F.fastMath}|${F.srgbOut}|${!!F.legacyLook}|${!!F.noBeams}|${!!F.beamShadow}`;
  if (!backdropCache.has(key)) {
    if (F.bgLast) {
      const mesh = makeBackdropMesh(F.fastMath, F.srgbOut, !!F.legacyLook, F);
      mesh.visible = false;
      scene.add(mesh);
      backdropCache.set(key, { mesh, node: null });
    } else {
      backdropCache.set(key, { mesh: null, node: buildOceanBackdropNode(positionLocal, F.fastMath, F.srgbOut, !!F.legacyLook, F) });
    }
  }
  return backdropCache.get(key);
}
function buildRenderVariant(spec, F) {
  const materials = {};
  for (const sp of speciesList) materials[sp] = buildCoralMaterial(sp, speciesConfig[sp], F);
  const backdrop = getBackdrop(F);
  return { spec, F, materials, backdropMesh: backdrop.mesh, backdropNode: backdrop.node };
}

const reefMeshes = {}; // species -> mesh (filled when the GLB loads)
let sceneReady = false; // live loop renders once the reef and all pipelines are compiled
let activeVariant = null;
let wireframeOn = false;
function applyRenderVariant(variant) {
  if (activeVariant && activeVariant.backdropMesh) activeVariant.backdropMesh.visible = false;
  activeVariant = variant;
  renderer.outputColorSpace = variant.F.srgbOut ? THREE.LinearSRGBColorSpace : THREE.SRGBColorSpace;
  uDirectOut.value = variant.F.srgbOut ? 1.0 : 0.0;
  gpuPassOptions.discardMultisample = !!variant.F.discardMultisample;
  for (const sp of speciesList) {
    if (variant.materials[sp].wireframe !== wireframeOn) variant.materials[sp].wireframe = wireframeOn;
    if (!reefMeshes[sp]) continue;
    reefMeshes[sp].material = variant.materials[sp];
    reefMeshes[sp].visible = !(variant.F.hide && variant.F.hide.includes(sp));
  }
  const hidden = new Set(variant.F.hideGroups || []);
  for (const [name, group] of Object.entries(env.groups)) group.visible = !(hidden.has(name) || hidden.has("life") || AB.has("no" + name));
  if (AB.has("nobg") || variant.F.noBg) {
    scene.backgroundNode = null;
    scene.background = new THREE.Color(0x6a58a8);
  } else if (variant.backdropMesh) {
    scene.backgroundNode = null;
    scene.background = null;
    variant.backdropMesh.visible = true;
  } else {
    scene.background = null;
    scene.backgroundNode = variant.backdropNode;
  }
}

const useHoneycombVolume = await honeycombReady;
const defaultSpec = urlParams.get("features") || (useHoneycombVolume ? "optimized" : "optimized+procedural");
const defaultVariant = buildRenderVariant(defaultSpec, resolveFeatures(defaultSpec));
applyRenderVariant(defaultVariant);

// Upgrade the coarse 180x180 Houdini seabed (14.5 cm spacing) into a graded 360x360 heightfield mesh
// (~2.7 cm triangles across the reef & foreground sand) so the GPU vertex shader displaces real 3D
// vortex sand ripples, scour moats, burrows, and rocky reef-apron clasts with clockwise BackSide winding.
function subdivideSeabedForDisplacement(seabedMesh, S = 360) {
  const old = seabedMesh.geometry;
  const pos0 = old.attributes.position;
  const nrm0 = old.attributes.normal;
  const wcd0 = old.attributes._wall_cd;
  const ccd0 = old.attributes._cavity_cd;
  const dat0 = old.attributes._coral_data;
  const N0 = Math.round(Math.sqrt(pos0.count));
  if (N0 * N0 !== pos0.count || N0 < 16) return { verts: 0, tris: 0 };
  const box = new THREE.Box3().setFromBufferAttribute(pos0);
  const x0 = box.min.x, x1 = box.max.x;
  const span = x1 - x0, mid = 0.5 * (x0 + x1), half = 0.5 * span;
  const nCells = N0 * N0;
  const gY = new Float32Array(nCells);
  const gN = new Float32Array(nCells * 3);
  const gW = new Float32Array(nCells * 3);
  const gC = new Float32Array(nCells * 3);
  const gD = new Float32Array(nCells * 4);
  for (let k = 0; k < pos0.count; k++) {
    const gi = Math.min(N0 - 1, Math.max(0, Math.round(((pos0.getX(k) - x0) / span) * (N0 - 1))));
    const gj = Math.min(N0 - 1, Math.max(0, Math.round(((pos0.getZ(k) - x0) / span) * (N0 - 1))));
    const o = gj * N0 + gi;
    gY[o] = pos0.getY(k);
    gN[o * 3] = nrm0.getX(k); gN[o * 3 + 1] = nrm0.getY(k); gN[o * 3 + 2] = nrm0.getZ(k);
    gW[o * 3] = wcd0.getX(k); gW[o * 3 + 1] = wcd0.getY(k); gW[o * 3 + 2] = wcd0.getZ(k);
    gC[o * 3] = ccd0.getX(k); gC[o * 3 + 1] = ccd0.getY(k); gC[o * 3 + 2] = ccd0.getZ(k);
    gD[o * 4] = dat0.getX(k); gD[o * 4 + 1] = dat0.getY(k); gD[o * 4 + 2] = dat0.getZ(k); gD[o * 4 + 3] = dat0.getW(k);
  }
  const V = S * S;
  const pos = new Float32Array(V * 3);
  const nrm = new Float32Array(V * 3);
  const wcd = new Float32Array(V * 3);
  const ccd = new Float32Array(V * 3);
  const dat = new Float32Array(V * 4);
  const coords = new Float32Array(S);
  for (let i = 0; i < S; i++) {
    const u = (i / (S - 1)) * 2 - 1;
    coords[i] = mid + half * u * (0.38 + 0.62 * u * u);
  }
  const bilerp = (arr, stride, i0, j0, tx, tz, dst, doff) => {
    const a00 = (j0 * N0 + i0) * stride, a10 = a00 + stride, a01 = a00 + N0 * stride, a11 = a01 + stride;
    for (let c = 0; c < stride; c++) {
      const r0 = arr[a00 + c] * (1 - tx) + arr[a10 + c] * tx;
      const r1 = arr[a01 + c] * (1 - tx) + arr[a11 + c] * tx;
      dst[doff + c] = r0 * (1 - tz) + r1 * tz;
    }
  };
  for (let j = 0; j < S; j++) {
    const z = coords[j];
    const fz = Math.min(N0 - 1.0001, Math.max(0, ((z - x0) / span) * (N0 - 1)));
    const j0 = Math.floor(fz), tz = fz - j0;
    for (let i = 0; i < S; i++) {
      const x = coords[i];
      const fx = Math.min(N0 - 1.0001, Math.max(0, ((x - x0) / span) * (N0 - 1)));
      const i0 = Math.floor(fx), tx = fx - i0;
      const k = j * S + i;
      const y00 = gY[j0 * N0 + i0], y10 = gY[j0 * N0 + i0 + 1], y01 = gY[(j0 + 1) * N0 + i0], y11 = gY[(j0 + 1) * N0 + i0 + 1];
      pos[k * 3] = x;
      pos[k * 3 + 1] = (y00 * (1 - tx) + y10 * tx) * (1 - tz) + (y01 * (1 - tx) + y11 * tx) * tz;
      pos[k * 3 + 2] = z;
      bilerp(gN, 3, i0, j0, tx, tz, nrm, k * 3);
      const nl = Math.hypot(nrm[k * 3], nrm[k * 3 + 1], nrm[k * 3 + 2]) || 1;
      nrm[k * 3] /= nl; nrm[k * 3 + 1] /= nl; nrm[k * 3 + 2] /= nl;
      bilerp(gW, 3, i0, j0, tx, tz, wcd, k * 3);
      bilerp(gC, 3, i0, j0, tx, tz, ccd, k * 3);
      bilerp(gD, 4, i0, j0, tx, tz, dat, k * 4);
    }
  }
  const nTris = (S - 1) * (S - 1) * 2;
  const idx = new Uint32Array(nTris * 3);
  let ptr = 0;
  for (let j = 0; j < S - 1; j++) {
    for (let i = 0; i < S - 1; i++) {
      const a = j * S + i, b = a + 1, c = a + S, d = c + 1;
      // Clockwise seen from +Y (cross(b-a, c-a).y < 0) to match Houdini's THREE.BackSide culling
      idx[ptr++] = a; idx[ptr++] = b; idx[ptr++] = c;
      idx[ptr++] = b; idx[ptr++] = d; idx[ptr++] = c;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
  g.setAttribute("_wall_cd", new THREE.BufferAttribute(wcd, 3));
  g.setAttribute("_cavity_cd", new THREE.BufferAttribute(ccd, 3));
  g.setAttribute("_coral_data", new THREE.BufferAttribute(dat, 4));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeBoundingSphere();
  const oldVerts = pos0.count;
  const oldTris = old.index ? old.index.count / 3 : 0;
  seabedMesh.geometry = g;
  old.dispose();
  return { verts: V - oldVerts, tris: nTris - oldTris };
}

// Load the Houdini-exported 3D-displaced GLB model with _WALL_CD, _CAVITY_CD, and _CORAL_DATA attributes
// Remove every triangle of the mesh pieces that touch the box [lo, hi] (whole connected pieces: the vertex
// marks are grown through shared indices, so polyps straddling the box go too). Verified offline that only
// the floating fragment and its polyps have vertices in the box.
function removeFloatingFragment(geom, lo, hi) {
  const pos = geom.attributes.position;
  const idx = geom.index.array;
  const mark = new Uint8Array(pos.count);
  let any = 0;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    if (x >= lo[0] && x <= hi[0] && y >= lo[1] && y <= hi[1] && z >= lo[2] && z <= hi[2]) (mark[i] = 1), any++;
  }
  if (!any) return;
  for (let pass = 0; pass < 6; pass++) {
    let changed = false;
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t], b = idx[t + 1], c = idx[t + 2];
      if ((mark[a] | mark[b] | mark[c]) && !(mark[a] & mark[b] & mark[c])) (mark[a] = mark[b] = mark[c] = 1), (changed = true);
    }
    if (!changed) break;
  }
  const kept = [];
  for (let t = 0; t < idx.length; t += 3) if (!(mark[idx[t]] | mark[idx[t + 1]] | mark[idx[t + 2]])) kept.push(idx[t], idx[t + 1], idx[t + 2]);
  geom.setIndex(new THREE.BufferAttribute(idx.length > 65535 * 3 || pos.count > 65535 ? new Uint32Array(kept) : new Uint16Array(kept), 1));
}
const gltfLoader = new GLTFLoader();
loadGlbWithFallback(
  gltfLoader,
  "./models/coral_reef.glb",
  (gltf) => {
    const root = gltf.scene;
    let totalVerts = 0;
    let totalTris = 0;

    root.traverse((child) => {
      if (child.isMesh) {
        const geom = child.geometry;
        if (!geom.attributes.normal) {
          geom.computeVertexNormals();
        }
        // Ensure fallback attributes exist if any submesh lacks custom attributes
        const vCount = geom.attributes.position.count;
        if (!geom.attributes._wall_cd && geom.attributes.color) {
          geom.setAttribute("_wall_cd", geom.attributes.color);
        }
        if (!geom.attributes._cavity_cd) {
          const cavArr = new Float32Array(vCount * 3);
          for (let i = 0; i < vCount; i++) {
            cavArr[i * 3 + 0] = 0.28;
            cavArr[i * 3 + 1] = 0.10;
            cavArr[i * 3 + 2] = 0.24;
          }
          geom.setAttribute("_cavity_cd", new THREE.BufferAttribute(cavArr, 3));
        }
        if (!geom.attributes._coral_data) {
          const cdArr = new Float32Array(vCount * 4);
          for (let i = 0; i < vCount; i++) {
            cdArr[i * 4 + 0] = 1.0;
            cdArr[i * 4 + 1] = 0.0;
            cdArr[i * 4 + 2] = 0.2;
            cdArr[i * 4 + 3] = 0.0;
          }
          geom.setAttribute("_coral_data", new THREE.BufferAttribute(cdArr, 4));
        }

        // Tangents, UVs and COLOR_0 are not used by the procedural TSL shading: skip the
        // (CPU-heavy, ~1M triangle) tangent generation and drop the attributes before GPU upload.
        geom.deleteAttribute("uv");
        geom.deleteAttribute("color");
        if (geom.attributes.position) {
          totalVerts += geom.attributes.position.count;
        }
        if (geom.index) {
          totalTris += geom.index.count / 3;
        }
        const name = (child.name || "").toLowerCase();
        // Both expert reviews flagged a disconnected Acropora branch tip floating ~1.5 m up (hero upper
        // right). It is a separate 144-vertex piece plus the polyps sitting on it; drop those triangles.
        if (name.includes("sponge") && geom.index) removeFloatingFragment(geom, [0.645, 1.505, 0.218], [0.769, 1.632, 0.316]);
        for (const sp of speciesList) {
          if (name.includes(sp)) {
            reefMeshes[sp] = child;
            child.material = activeVariant.materials[sp];
            break;
          }
        }
      }
    });

    scene.add(root);
    // CPU terrain/occupancy for placing life (from the original 180x180 Houdini seabed), then upgrade the
    // seabed to a graded 360x360 mesh (~2.7 cm cells across the reef & foreground) for real 3D GPU vertex displacement!
    const terrainInfo = env.setReef(reefMeshes);
    fillFootprintTexture(footprintTex, terrainInfo.footprint);
    if (reefMeshes.seabed) {
      const added = subdivideSeabedForDisplacement(reefMeshes.seabed, 300);
      totalVerts += added.verts;
      totalTris += added.tris;
    }
    for (const sp of speciesList) if (sp !== "seabed" && reefMeshes[sp]) env.addShadowCaster(reefMeshes[sp]);
    env.bakeShadows();
    document.getElementById("stat-verts").textContent = totalVerts.toLocaleString();
    document.getElementById("stat-tris").textContent = Math.round(totalTris).toLocaleString();
    if (BENCH) {
      loadingOverlay.classList.add("hidden");
      runBenchmark(performance.now() - T_PAGE_START);
    } else {
      // Compile every pipeline asynchronously behind the loading overlay (the life modules add their meshes
      // a microtask later), so the first visible frame doesn't freeze on synchronous pipeline creation.
      setTimeout(async () => {
        document.getElementById("loading-text").textContent = "Compiling reef shaders…";
        try {
          await houdiniSandReady;
          // warm the ripple-simulation pipelines too (idle until the first click / rain), so the first
          // ripple doesn't stall a frame on pipeline creation
          ocean.addDrop(0.5, 0.5, 0.01, 0.0);
          ocean.update(0.05);
          ocean.update(0.1);
          ocean.reset();
          const hiddenMeshes = lifeModules.filter((m) => m && m.mesh && !m.mesh.visible).map((m) => m.mesh);
          for (const hm of hiddenMeshes) hm.visible = true;
          await renderer.compileAsync(scene, camera);
          for (const hm of hiddenMeshes) hm.visible = false;
        } catch (e) {
          console.warn("compileAsync:", e);
        }
        sceneReady = true;
        loadingOverlay.classList.add("hidden");
      }, 0);
    }
  },
  undefined,
  (err) => {
    console.error("Failed to load coral_reef.glb:", err);
    document.getElementById("loading-text").textContent = "Error loading GLB: " + err.message;
    if (BENCH) window.__BENCH_RESULT__ = { label: BENCH, error: "GLB load failed: " + err.message };
  }
);

// Reef life: each module owns one file and one group (see js/reef-env.js for the shared contract)
const lifeModules = [];
for (const [name, create] of [["fish", createReefFish], ["flora", createReefFlora], ["props", createReefProps], ["snow", createMarineSnow], ["surface", createWaterSurface]]) {
  try {
    lifeModules.push(create(env));
  } catch (e) {
    console.error(`reef life module "${name}" failed:`, e);
  }
}

// UI Controls Wiring
document.getElementById("slider-normal").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  document.getElementById("val-normal").textContent = v.toFixed(2);
  uPoreStrength.value = v;
});

document.getElementById("slider-disp").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  document.getElementById("val-disp").textContent = v.toFixed(2);
  uBumpStrength.value = v;
});

document.getElementById("slider-ao").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  document.getElementById("val-ao").textContent = v.toFixed(2);
  uAoIntensity.value = v;
});

document.getElementById("slider-bump").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  document.getElementById("val-bump").textContent = v.toFixed(2);
  uWaterAtmosphere.value = v;
});

document.getElementById("slider-emissive").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  document.getElementById("val-emissive").textContent = v.toFixed(2);
  uSssIntensity.value = v;
});

document.getElementById("slider-rough").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  document.getElementById("val-rough").textContent = v.toFixed(2);
  uRoughScale.value = v;
});

document.getElementById("slider-caustics").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  document.getElementById("val-caustics").textContent = v.toFixed(2);
  uCausticsScale.value = v;
});

document.getElementById("btn-wireframe").addEventListener("click", (e) => {
  wireframeOn = !wireframeOn;
  for (const sp of speciesList) {
    activeVariant.materials[sp].wireframe = wireframeOn;
  }
  e.target.classList.toggle("active", wireframeOn);
});

if (urlParams.get("rain") === "1") {
  ocean.rainRate = 70;
  document.getElementById("btn-rain").classList.add("active");
  document.getElementById("btn-rain").textContent = "Rain: ON";
}
document.getElementById("btn-rain").addEventListener("click", (e) => {
  const on = !(ocean.rainRate > 0);
  ocean.rainRate = on ? 70 : 0;
  e.target.classList.toggle("active", on);
  e.target.textContent = on ? "Rain: ON" : "Rain: OFF";
});

document.getElementById("btn-autorotate").addEventListener("click", (e) => {
  controls.autoRotate = !controls.autoRotate;
  e.target.classList.toggle("active", controls.autoRotate);
  e.target.textContent = controls.autoRotate ? "Auto-Orbit: ON" : "Auto-Orbit: OFF";
});

// Camera Focus Presets
const LEGACY_CAMS = urlParams.get("legacyCams") === "1";
function setCameraView(eye, target, btnId, fov = 33) {
  if (camera.fov !== fov) {
    camera.fov = fov;
    camera.updateProjectionMatrix();
  }
  document.querySelectorAll(".cam-preset-bar .btn:not(.houdini-btn)").forEach((b) => b.classList.remove("active"));
  if (btnId) document.getElementById(btnId).classList.add("active");
  // only the Look Up shot may orbit below the target (to see the surface); the camera is kept above the sand
  controls.maxPolarAngle = btnId === "cam-up" ? Math.PI * 0.8 : Math.PI * 0.485;
  uDofAmt.value = btnId === "cam-sponge" ? 1.0 : btnId === "cam-wide" ? 0.5 : 0.0;
  camera.position.set(eye[0], eye[1], eye[2]);
  controls.target.set(target[0], target[1], target[2]);
  controls.update();
}

document.getElementById("cam-wide").addEventListener("click", () => {
  if (LEGACY_CAMS) setCameraView([-0.02, 1.32, 2.82], [0.02, 1.08, 0.04], "cam-wide", 35);
  else setCameraView([-0.2, 1.12, 2.95], [0.05, 1.16, 0.02], "cam-wide", 33); // art director: horizon lower, more water above
});
document.getElementById("cam-brain").addEventListener("click", () => {
  if (LEGACY_CAMS) setCameraView([0.08, 2.05, 4.45], [0.0, 0.85, 0.05], "cam-brain", 35);
  else setCameraView([0.35, 1.45, 5.2], [0.0, 0.95, 0.0], "cam-brain", 35); // sand ~55% -> ~40% of the frame
});
document.getElementById("cam-sponge").addEventListener("click", () => {
  setCameraView([-0.08, 1.24, 1.48], [-0.04, 1.08, 0.16], "cam-sponge", 35);
});
// Look Up (undersea expert P11): table & fan silhouettes against Snell's window and the rippling surface
document.getElementById("cam-up").addEventListener("click", () => {
  // aimed into the refracted sun (undersea round 1): the sun sits ~8 deg above centre, corals backlit
  setCameraView([-1.6, 0.35, -2.4], [0.2, 3.4, 0.1], "cam-up", 40);
});

// Click (without dragging) sends a ripple across the simulated surface (webgpu-water style drop):
// aimed at the surface when looking up, otherwise centred on the caustics where you clicked.
{
  const down = new THREE.Vector2();
  let downT = 0;
  const ray = new THREE.Raycaster();
  renderer.domElement.addEventListener("pointerdown", (e) => {
    down.set(e.clientX, e.clientY);
    downT = performance.now();
  });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6 || performance.now() - downT > 350) return;
    const r = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    const o = ray.ray.origin, d = ray.ray.direction;
    if (d.y > 0.05) {
      const t = (uWaterSurfaceY.value - o.y) / d.y;
      const q = o.clone().addScaledVector(d, t);
      const c = Math.cos(ocean.tileRot), sn = Math.sin(ocean.tileRot);
      ocean.addDrop((c * q.x - sn * q.z) / ocean.tileSize, (sn * q.x + c * q.z) / ocean.tileSize, 0.16 / ocean.tileSize, 1.2);
    } else if (d.y < -0.02) {
      const t = (0.6 - o.y) / d.y;
      ocean.dropAt(o.clone().addScaledVector(d, t), 0.14, 1.2);
    }
  });
}

// Houdini Render Checkpoint Modal
const modal = document.getElementById("checkpoint-modal");
document.getElementById("btn-open-modal").addEventListener("click", () => {
  modal.classList.remove("hidden");
});
document.getElementById("btn-close-modal").addEventListener("click", () => {
  modal.classList.add("hidden");
});
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") modal.classList.add("hidden");
  if ((e.key === "h" || e.key === "H") && !e.metaKey && !e.ctrlKey && !(e.target && /input|textarea/i.test(e.target.tagName))) {
    document.body.classList.toggle("hud-hidden");
  }
});
if (!BENCH) {
  const hint = document.createElement("div");
  hint.className = "hud-hint";
  hint.textContent = "H hides the panels · click the water or sand to send a ripple";
  document.body.appendChild(hint);
  setTimeout(() => (hint.style.transition = "opacity 1.2s", (hint.style.opacity = "0")), 9000);
}

function setMoodPreset(mode) {
  document.querySelectorAll(".preset-row .btn[id^='preset-']").forEach((b) => b.classList.remove("active"));
  document.getElementById(`preset-${mode}`).classList.add("active");
  env.applyMood(mode);
  uNightGlow.value = mode === "night" ? 1.0 : 0.0;

  if (mode === "day") {
    // Pastel Reference Reef with luminous aqua-lavender sun shafts & golden-cyan caustics
    uBgDeep.value.set(0.24, 0.15, 0.44);
    uBgMid.value.set(0.48, 0.36, 0.72);
    uBgShallow.value.set(0.68, 0.64, 0.90);
    uWaterGlow.value.set(0.46, 0.34, 0.68);
    uShaftTint.value.set(0.82, 0.98, 1.00);
    uWaterExt.value.set(0.32, 0.18, 0.10);
    uKeyColor.value.set(1.05, 0.99, 0.96);
    uFillColor.value.set(0.78, 0.90, 0.99);
    uSkyLow.value.set(0.60, 0.42, 0.78);
    uSkyHigh.value.set(0.98, 0.95, 1.00);
    document.getElementById("slider-emissive").value = 0.28;
    document.getElementById("slider-emissive").dispatchEvent(new Event("input"));
  } else if (mode === "sunset") {
    // Tropical Marine (exact crystal-clear turquoise/cerulean water column from willwaltz/globes!)
    uBgDeep.value.set(0.03, 0.16, 0.32);
    uBgMid.value.set(0.08, 0.44, 0.66);
    uBgShallow.value.set(0.42, 0.86, 0.96);
    uWaterGlow.value.set(0.10, 0.48, 0.70);
    uShaftTint.value.set(0.68, 0.96, 1.00);
    uWaterExt.value.set(0.85, 0.28, 0.14);
    uKeyColor.value.set(1.08, 0.98, 0.88);
    uFillColor.value.set(0.42, 0.86, 0.98);
    uSkyLow.value.set(0.12, 0.38, 0.58);
    uSkyHigh.value.set(0.72, 0.96, 1.00);
    document.getElementById("slider-emissive").value = 0.34;
    document.getElementById("slider-emissive").dispatchEvent(new Event("input"));
  } else if (mode === "night") {
    // Bio-Night (deep abyssal bioluminescence with electric cyan/indigo moonlit caustics)
    uBgDeep.value.set(0.02, 0.02, 0.08);
    uBgMid.value.set(0.05, 0.08, 0.22);
    uBgShallow.value.set(0.14, 0.28, 0.52);
    uWaterGlow.value.set(0.06, 0.12, 0.28);
    uShaftTint.value.set(0.32, 0.78, 1.00);
    uWaterExt.value.set(0.65, 0.38, 0.18);
    uKeyColor.value.set(0.42, 0.56, 0.95);
    uFillColor.value.set(0.24, 0.64, 0.86);
    uSkyLow.value.set(0.06, 0.05, 0.20);
    uSkyHigh.value.set(0.32, 0.42, 0.78);
    document.getElementById("slider-emissive").value = 0.46;
    document.getElementById("slider-emissive").dispatchEvent(new Event("input"));
  }
}

document.getElementById("preset-day").addEventListener("click", () => setMoodPreset("day"));
document.getElementById("preset-sunset").addEventListener("click", () => setMoodPreset("sunset"));
document.getElementById("preset-night").addEventListener("click", () => setMoodPreset("night"));

// Optional URL query params (?mood=day|sunset|night&cam=wide|brain|sponge)
const initMood = urlParams.get("mood");
if (initMood === "day" || initMood === "sunset" || initMood === "night") {
  setMoodPreset(initMood);
}
const initCam = urlParams.get("cam");
if (initCam === "wide" || initCam === "brain" || initCam === "sponge" || initCam === "up") {
  document.getElementById(`cam-${initCam}`).click();
}

// Species Texture Swatch Switcher
document.querySelectorAll(".species-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".species-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const sp = tab.dataset.species;
    document.getElementById("sw-albedo").src = `./textures/${sp}_albedo.png?v=10`;
    document.getElementById("sw-bump").src = `./textures/${sp}_bump.png?v=10`;
    document.getElementById("sw-normal").src = `./textures/${sp}_normal.png?v=10`;
    document.getElementById("sw-rough").src = `./textures/${sp}_roughness.png?v=10`;
    document.getElementById("sw-emissive").src = `./textures/${sp}_emissive.png?v=10`;
  });
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// -------------------------------------------------------------------------
// Performance monitor + adaptive resolution.
// FPS comes from the animation-loop interval. "GPU latency" is submit -> GPU completion
// (queue.onSubmittedWorkDone): it includes queueing behind other GPU work and the GPU's power
// state, so it is shown for reference only. Resolution decisions use actual frame pacing: the
// internal render scale drops one step (100% -> 50% of the display pixel ratio) when frames are
// being missed (< ~50 fps) and probes back up after a stable period, backing off exponentially
// whenever the higher resolution can't hold 60 fps.
// -------------------------------------------------------------------------
const perfEls = {
  fps: document.getElementById("perf-fps"),
  gpu: document.getElementById("perf-gpu"),
  scale: document.getElementById("perf-scale"),
  adaptiveBtn: document.getElementById("btn-adaptive"),
};
const RENDER_SCALE_STEPS = [1.0, 0.875, 0.75, 0.625, 0.5];
const SLOW_FRAME_MS = 20.0; // < 50 fps: frames are being missed -> lower the render scale
const STABLE_FRAME_MS = 17.5; // holding 60 fps (or better) -> may probe a higher render scale
const basePixelRatio = renderer.getPixelRatio();
const perf = {
  adaptive: !BENCH && !Number.isFinite(dprOverride) && urlParams.get("adaptive") !== "0",
  scaleIndex: 0,
  frameMs: 1000 / 60,
  gpuMs: 0,
  gpuPending: false,
  lastFrame: 0,
  lastHud: 0,
  settleUntil: performance.now() + 2500, // ignore shader-warmup hitches right after load
  lastChange: performance.now(),
  lastUpgrade: -Infinity,
  upgradeDelay: 4000,
};

function setRenderScale(index) {
  const next = Math.max(0, Math.min(RENDER_SCALE_STEPS.length - 1, index));
  if (next === perf.scaleIndex) return;
  const now = performance.now();
  if (next > perf.scaleIndex && now - perf.lastUpgrade < 4000) {
    // The last step up could not hold the frame rate: wait longer before probing again
    perf.upgradeDelay = Math.min(perf.upgradeDelay * 2, 64000);
  }
  if (next < perf.scaleIndex) perf.lastUpgrade = now;
  perf.scaleIndex = next;
  renderer.setPixelRatio(basePixelRatio * RENDER_SCALE_STEPS[next]);
  perf.settleUntil = now + 1500;
  perf.lastChange = now;
}

function perfFrame() {
  const now = performance.now();
  if (perf.lastFrame) {
    const dt = now - perf.lastFrame;
    if (dt < 250) perf.frameMs += (dt - perf.frameMs) * 0.08;
  }
  perf.lastFrame = now;

  const device = renderer.backend && renderer.backend.device;
  if (device && !perf.gpuPending) {
    perf.gpuPending = true;
    device.queue.onSubmittedWorkDone().then(() => {
      const ms = performance.now() - now;
      perf.gpuMs = perf.gpuMs === 0 ? ms : perf.gpuMs + (ms - perf.gpuMs) * 0.12;
      perf.gpuPending = false;
    });
  }

  if (perf.adaptive && now > perf.settleUntil) {
    if (perf.frameMs > SLOW_FRAME_MS && perf.scaleIndex < RENDER_SCALE_STEPS.length - 1) {
      setRenderScale(perf.scaleIndex + 1);
    } else if (perf.frameMs < STABLE_FRAME_MS && perf.scaleIndex > 0 && now - perf.lastChange > perf.upgradeDelay) {
      setRenderScale(perf.scaleIndex - 1);
    }
  }

  if (now - perf.lastHud > 400) {
    perf.lastHud = now;
    perfEls.fps.textContent = Math.round(1000 / perf.frameMs).toString();
    perfEls.gpu.textContent = perf.gpuMs ? `${perf.gpuMs.toFixed(1)} ms` : "n/a";
    perfEls.scale.textContent = `${Math.round(RENDER_SCALE_STEPS[perf.scaleIndex] * 100)}%`;
  }
}

perfEls.adaptiveBtn.classList.toggle("active", perf.adaptive);
perfEls.adaptiveBtn.textContent = perf.adaptive ? "Adaptive Res: ON" : "Adaptive Res: OFF";
perfEls.adaptiveBtn.addEventListener("click", () => {
  perf.adaptive = !perf.adaptive;
  perfEls.adaptiveBtn.classList.toggle("active", perf.adaptive);
  perfEls.adaptiveBtn.textContent = perf.adaptive ? "Adaptive Res: ON" : "Adaptive Res: OFF";
  if (!perf.adaptive) setRenderScale(0);
});

const clock = new THREE.Clock();
function updateScene(t) {
  uTime.value = t;
  const VF = activeVariant ? activeVariant.F : {};
  if (!AB.has("noocean") && !VF.noOcean) ocean.update(t, { caustics: !VF.noOceanCaustics, mips: !VF.noOceanMips });
  for (const m of lifeModules) m.update(t);
  const focusD = camera.position.distanceTo(controls.target);
  env.u.fogStart.value = THREE.MathUtils.clamp(focusD - 0.8, env.moodFogStart ?? 1.6, 4.0);
  uFocusDist.value = focusD;
  env.preRender();

}

// Dev views: ?debug=caustics (caustic tile, 2x2 repeats) | ?debug=surface (surface normals)
const DEBUG_VIEW = urlParams.get("debug");
let debugQuad = null;
if (DEBUG_VIEW === "caustics" || DEBUG_VIEW === "surface") {
  const m = new THREE.MeshBasicNodeMaterial();
  const tex = DEBUG_VIEW === "caustics" ? ocean.causticRT.texture : ocean.surfaceRT.texture;
  const smp = texture(tex, uv().mul(2.0));
  m.fragmentNode =
    DEBUG_VIEW === "caustics"
      ? vec4(sRGBTransferOETF(vec3(smp.x, smp.y, smp.x).mul(0.25)), 1.0)
      : vec4(sRGBTransferOETF(vec3(smp.y.mul(4.0).add(0.5), smp.z.mul(4.0).add(0.5), smp.x.mul(4.0).add(0.5))), 1.0);
  debugQuad = new THREE.QuadMesh(m);
}
function renderFrame() {
  if (debugQuad) {
    renderer.setRenderTarget(null);
    debugQuad.render(renderer);
  } else {
    renderer.render(scene, camera);
  }
}

// keep the camera inside the water: above the sand (the Look Up shot orbits low) and below the surface
function clampCamera() {
  const t = env.terrainInfo;
  const floor = (t ? t.heightAt(camera.position.x, camera.position.z) : 0.0) + 0.22;
  if (camera.position.y < floor) camera.position.y = floor;
  if (camera.position.y > uWaterSurfaceY.value - 0.3) camera.position.y = uWaterSurfaceY.value - 0.3;
}

if (!BENCH) {
  renderer.setAnimationLoop(() => {
    if (!sceneReady) return; // the loading overlay is up until the reef and its pipelines are ready
    controls.update();
    clampCamera();
    const t = clock.getElapsedTime();
    updateScene(t);
    // diver bob (art director): +-1.5 cm, 7 s period, +-0.3 deg roll, removed again after the frame
    const bob = Math.sin((t * 2 * Math.PI) / 7) * 0.015;
    const roll = Math.sin((t * 2 * Math.PI) / 9.3 + 1.1) * 0.0052;
    camera.position.y += bob;
    camera.rotateZ(roll);
    renderFrame();
    camera.rotateZ(-roll);
    camera.position.y -= bob;
    perfFrame();
  });
}

// -------------------------------------------------------------------------
// Dev benchmark harness (?bench=<label>[&abv=specA|specB...]): deterministic camera + time,
// GPU-synchronized frame timing. Multiple render variants are interleaved frame-by-frame so
// background GPU load (other apps) affects every variant equally. Results land on
// window.__BENCH_RESULT__ and are POSTed to /__bench (see serve.py + tools/bench.mjs).
// -------------------------------------------------------------------------
function benchStats(arr) {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
  const r3 = (v) => Math.round(v * 1000) / 1000;
  return { avg: r3(s.reduce((a, b) => a + b, 0) / s.length), p10: r3(q(0.1)), p50: r3(q(0.5)), p90: r3(q(0.9)), min: r3(s[0]), max: r3(s[s.length - 1]), n: s.length };
}

async function runBenchmark(reefLoadMs) {
  const result = { label: BENCH, query: location.search, ab: [...AB] };
  try {
    await new Promise((r) => setTimeout(r, 0));
    await houdiniSandReady;
    const rounds = parseInt(urlParams.get("frames") || "150", 10);
    const warmup = parseInt(urlParams.get("warmup") || "30", 10);
    const blockRounds = parseInt(urlParams.get("blocks") || "8", 10);
    const blockFrames = 20;
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    result.dpr = renderer.getPixelRatio();
    result.drawingBuffer = [size.x, size.y];
    result.megapixels = Math.round((size.x * size.y) / 1e4) / 100;
    result.reefLoadMs = Math.round(reefLoadMs);
    result.honeycombVolume = useHoneycombVolume;

    const specs = (urlParams.get("abv") || "").split("|").filter(Boolean);
    const variants = specs.length
      ? specs.map((spec) => (spec === defaultVariant.spec ? defaultVariant : buildRenderVariant(spec, resolveFeatures(spec))))
      : [defaultVariant];

    for (const v of variants) {
      applyRenderVariant(v);
      const tc0 = performance.now();
      await renderer.compileAsync(scene, camera);
      v.compileMs = Math.round(performance.now() - tc0);
      v.samples = [];
      v.blocks = [];
    }

    applyRenderVariant(variants[0]);
    const tf0 = performance.now();
    controls.update();
    updateScene(4.0);
    renderer.render(scene, camera);
    await renderer.waitForGPU();
    result.firstFrameMs = Math.round(performance.now() - tf0);
    result.timeToFirstFrameMs = Math.round(performance.now() - T_PAGE_START);
    const camPos0 = camera.position.clone();
    const target0 = controls.target.clone();

    // 1) Latency: single GPU-synchronized frames, variants interleaved per frame
    for (let r = 0; r < warmup + rounds; r++) {
      controls.update();
      updateScene(4.0 + r / 60);
      for (const v of variants) {
        applyRenderVariant(v);
        const t0 = performance.now();
        renderer.render(scene, camera);
        await renderer.waitForGPU();
        if (r >= warmup) v.samples.push(performance.now() - t0);
      }
    }
    // 2) Throughput: short blocks of back-to-back frames per variant, interleaved
    for (let b = 0; b < blockRounds; b++) {
      for (const v of variants) {
        applyRenderVariant(v);
        const t0 = performance.now();
        for (let i = 0; i < blockFrames; i++) {
          updateScene(9.0 + (b * blockFrames + i) / 60);
          renderer.render(scene, camera);
        }
        const tCpu = performance.now();
        await renderer.waitForGPU();
        v.blocks.push((performance.now() - t0) / blockFrames);
        (v.cpuBlocks || (v.cpuBlocks = [])).push((tCpu - t0) / blockFrames);
      }
    }

    result.variants = variants.map((v) => ({
      spec: v.spec,
      compileMs: v.compileMs,
      frameMs: benchStats(v.samples),
      throughputMs: benchStats(v.blocks),
      cpuMs: benchStats(v.cpuBlocks || []),
    }));
    result.frameMs = result.variants[0].frameMs;
    result.throughputMs = result.variants[0].throughputMs.p50;
    result.compileMs = result.variants[0].compileMs;

    if (urlParams.has("dumpShaders")) {
      result.shaders = {};
      for (const v of variants) {
        applyRenderVariant(v);
        for (const sp of urlParams.get("dumpShaders").split(",").filter(Boolean)) {
          if (reefMeshes[sp]) result.shaders[`${v.spec}:${sp}`] = await renderer.debug.getShaderAsync(scene, camera, reefMeshes[sp]);
        }
      }
    }

    // Deterministic still frame for screenshots / visual A-B comparisons (?shotVariant=<index>)
    controls.autoRotate = false;
    controls.enableDamping = false;
    const shotT = parseFloat(urlParams.get("shotT") || "12.0");
    if (urlParams.get("shots") === "all") {
      // Canvas-only stills of every variant (no HUD), POSTed to /__shot as <label>_v<index>.png
      for (let i = 0; i < variants.length; i++) {
        applyRenderVariant(variants[i]);
        camera.position.copy(camPos0);
        camera.lookAt(target0);
        ocean.warmTo(shotT);
        for (const m of lifeModules) m.warmTo(shotT);
        updateScene(shotT);
        renderFrame();
        const png = await (await fetch(renderer.domElement.toDataURL("image/png"))).blob();
        await fetch(`/__shot?name=${encodeURIComponent(`${BENCH}_v${i}`)}`, { method: "POST", body: png });
      }
    }
    applyRenderVariant(variants[parseInt(urlParams.get("shotVariant") || "0", 10)] || variants[0]);
    camera.position.copy(camPos0);
    camera.lookAt(target0);
    ocean.warmTo(shotT);
    for (const m of lifeModules) m.warmTo(shotT);
    updateScene(shotT);
    renderFrame();
    await renderer.waitForGPU();
    if (urlParams.has("causticStats")) {
      // Dev: distribution of the simulated caustic irradiance (calibrates uSimCausticFloor/Gain)
      const n = ocean.causticSize;
      const raw = await renderer.readRenderTargetPixelsAsync(ocean.causticRT, 0, 0, n, n);
      const h2f = (h) => {
        const e = (h >> 10) & 0x1f, f = h & 0x3ff, s = h & 0x8000 ? -1 : 1;
        return e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? NaN : s * 2 ** (e - 15) * (1 + f / 1024);
      };
      const stride = Math.round(raw.length / (n * n));
      const chans = [[], [], [], []];
      for (let i = 0; i < n * n; i++)
        for (let c = 0; c < stride; c++) chans[c].push(raw instanceof Uint16Array ? h2f(raw[i * stride + c]) : raw[i * stride + c]);
      const stat = (vals) => {
        const sq = vals.map((v) => Math.sqrt(Math.max(v, 0)));
        vals.sort((a, b) => a - b);
        const q = (p) => Math.round(vals[Math.min(vals.length - 1, Math.floor(p * vals.length))] * 1e4) / 1e4;
        return {
          mean: Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 1e4) / 1e4,
          meanSqrt: Math.round((sq.reduce((a, b) => a + b, 0) / sq.length) * 1e4) / 1e4,
          q: Object.fromEntries([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999].map((p) => [p, q(p)])),
        };
      };
      result.causticStats = {
        type: raw.constructor.name,
        deep: stat(chans[0]),
        shallow: stat(chans[1]),
        waves: ocean.waves.length,
      };
    }
  } catch (e) {
    result.error = String((e && e.stack) || e);
  }
  window.__BENCH_RESULT__ = result;
  document.title = "BENCH_DONE";
  try {
    await fetch(`/__bench?name=${encodeURIComponent(BENCH)}`, { method: "POST", body: JSON.stringify(result) });
  } catch (e) {
    /* runner also reads window.__BENCH_RESULT__ via CDP */
  }
}

})();
