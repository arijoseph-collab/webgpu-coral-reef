// -----------------------------------------------------------------------------
// reef-props.js — static reef life, rubble halo and far bommie silhouettes (props builder)
//
// Contract: see js/reef-env.js. createReefProps(env) adds its objects under one THREE.Group registered as
// env.groups.props and returns { group, update(t), warmTo(t) }. Animation is deterministic in t.
//
// Everything static is generated procedurally on the CPU once the Houdini GLB has loaded (await env.terrain)
// and baked into ONE merged world-space geometry drawn with ONE material (one draw, one pipeline, one
// shadow caster):
//   - rubble halo (undersea expert P5, art director item 9): 1,400 half-buried coral fragments built from
//     11 procedural base pieces (knobbly broken staghorn branch / notched table-plate shard / porous chunk,
//     50-110 triangles each), scattered by rejection sampling with density exp(-d / 0.36 m) from the reef
//     edge (env.terrain.coralDist), patchy, bigger pieces piled at the base, none beyond ~1.5 m; small and
//     sparse wherever it would land in the hero frame; bleached bone / pale lilac / lilac-grey / rose /
//     mauve with ~30% pink crustose-coralline crust on the exposed faces and per-pixel corallite pits
//   - static life (P9): 10 dead-coral boulders, 4 tube-sponge clusters with rolled lips and dark atria, a
//     giant barrel sponge (Xestospongia) in the background, 10 Diadema urchins with 120 needle spines each,
//     two Linckia and one cushion sea star, 2 giant clams (Tridacna) with ruffled iridescent mantles
//   - far bommie silhouettes (P8 / item 9): 16 multi-lobed, noise-displaced flattened icospheres with
//     finger-coral clumps and the odd table plate on a 6-8.4 m ring, partly sunk into the sand, dark so
//     they read at 60-75% fog on the horizon band: the reef continues into the haze
// Shading goes through env.litProp (wrap key x baked sun shadow x caustics, sky, fill, rim) and env.finish
// (water-column fog, highlight shoulder, vignette, dither, encode). Per-vertex attributes carry the albedo,
// a baked AO and the material knobs (sheen, subsurface glow, caustic gain, iridescence / pores).
// The only moving props are the crinoids (feather stars) perched on the exposed brain-mound tops: a second
// small alpha-to-coverage material whose feathery arms are fanned across the surge and sway with env.surge
// (GPU-animated from env.u.time, so they are not shadow casters).
// env.sites.obstacles receives coarse capsules (sponges, barrel, crinoids, tall boulders) for fish avoidance.
// Dev A/B (?ab=...): propsNoRubble, propsNoLife, propsNoCrinoids, propsNoBommies skip one element family.
// -----------------------------------------------------------------------------
import * as THREE from "three/webgpu";
import {
  Fn,
  vec2,
  vec3,
  vec4,
  float,
  attribute,
  varying,
  positionGeometry,
  normalGeometry,
  cameraPosition,
  cameraViewMatrix,
  cameraProjectionMatrix,
  faceDirection,
  normalize,
  mix,
  clamp,
  smoothstep,
  max,
  dot,
  sin,
  cos,
  fract,
  floor,
  abs,
  step,
  length,
  fwidth,
} from "three/tsl";

const UP = new THREE.Vector3(0, 1, 0);
const ONE = new THREE.Vector3(1, 1, 1);

// ---- palette (display-referred albedo, like the coral vertex colours) ----
const C = {
  bone: [0.93, 0.88, 0.82], // bleached skeleton
  paleLilac: [0.83, 0.79, 0.89],
  lilacGrey: [0.72, 0.67, 0.8],
  mauve: [0.63, 0.57, 0.66], // older rubble
  paleRose: [0.9, 0.78, 0.8],
  cca: [0.93, 0.58, 0.7], // crustose coralline algae
  turf: [0.64, 0.62, 0.54], // turf / biofilm on old rubble
  lavender: [0.62, 0.48, 0.84], // #9E7BD6
  periwinkle: [0.55, 0.62, 0.93],
  lemon: [0.95, 0.88, 0.5],
  barrel: [0.68, 0.5, 0.52], // rose-brown giant barrel sponge (pastel)
  urchin: [0.15, 0.12, 0.25], // indigo-black Diadema
  linckia: [0.5, 0.58, 0.93], // periwinkle-blue Linckia laevigata
  cushion: [0.95, 0.68, 0.5], // apricot Culcita
  clamShell: [0.88, 0.84, 0.82],
  bommie: [0.3, 0.27, 0.38], // dark: they must read against 60-75% fog
};

export function createReefProps(env) {
  const group = new THREE.Group();
  group.name = "props";
  env.groups.props = group;
  env.scene.add(group);
  const AB = env.AB || new Set();

  const staticMat = makeStaticMaterial(env);
  const crinoidMat = makeCrinoidMaterial(env);

  // Placeholder geometry with the final attribute layout: renderer.compileAsync() runs before the
  // GLB-dependent placement below, so this lets it build both pipelines up front.
  const staticMesh = new THREE.Mesh(placeholderGeometry("aPar"), staticMat);
  staticMesh.name = "props_static";
  staticMesh.frustumCulled = false;
  group.add(staticMesh);
  const crinoidMesh = new THREE.Mesh(placeholderGeometry("aArm"), crinoidMat);
  crinoidMesh.name = "props_crinoids";
  crinoidMesh.frustumCulled = false;
  group.add(crinoidMesh);

  // Warm both pipelines right away, while the 60 MB Houdini GLB is still downloading: the WGSL is identical
  // for the main scene, so the bench's compileAsync / the first frame find the GPU pipelines cached.
  let warmMs = null;
  try {
    const warm = new THREE.Scene();
    for (const src of [staticMesh, crinoidMesh]) {
      const w = new THREE.Mesh(src.geometry, src.material);
      w.frustumCulled = false;
      warm.add(w);
    }
    const tw = performance.now();
    env.renderer
      .compileAsync(warm, env.camera)
      .then(() => (warmMs = Math.round(performance.now() - tw)))
      .catch((e) => console.error("reef-props: pipeline warm-up failed", e));
  } catch (e) {
    console.error("reef-props: pipeline warm-up failed", e);
  }

  const api = {
    group,
    staticMesh,
    crinoidMesh,
    stats: null,
    update() {}, // everything is static or GPU-animated from env.u.time (crinoid sway)
    warmTo() {},
  };

  env.terrain
    .then((T) => {
      try {
        const t0 = performance.now();
        const built = buildAll(env, T, AB);
        swapGeometry(staticMesh, built.stat.geometry("aPar"));
        swapGeometry(crinoidMesh, built.crin.geometry("aArm"));
        env.addShadowCaster(staticMesh); // static only: the swaying crinoids are not casters
        const obs = env.sites.obstacles || (env.sites.obstacles = []);
        obs.push(...built.obstacles);
        env.sites.props = built.sites;
        api.stats = {
          ms: Math.round(performance.now() - t0),
          staticVerts: built.stat.n,
          staticTris: built.stat.idx.length / 3,
          crinoidTris: built.crin.idx.length / 3,
          rubble: built.counts.rubble,
          warmMs,
        };
        console.log(
          `[props] pipelines warmed in ${warmMs ?? "?"} ms; built in ${api.stats.ms} ms ${JSON.stringify(built.tm)}: ${api.stats.staticTris} static tris (${built.counts.rubble} rubble, ` +
            `${built.counts.bommies} bommies), ${api.stats.crinoidTris} crinoid tris, sites ${JSON.stringify({ crinoids: built.sites.crinoids.length, sponges: built.sites.sponges.length, urchins: built.sites.urchins.length, clams: built.sites.clams.length, stars: built.sites.stars.length, boulders: built.sites.boulders.length })}`
        );
      } catch (e) {
        console.error("reef-props: build failed", e);
      }
    })
    .catch((e) => console.error("reef-props: terrain unavailable", e));

  return api;
}

// -----------------------------------------------------------------------------
// Materials
// -----------------------------------------------------------------------------
/** One shared material for every static prop (merged world-space geometry, identity model matrix). */
function makeStaticMaterial(env) {
  const u = env.u;
  const mat = new THREE.MeshBasicNodeMaterial({ name: "props_static", side: THREE.FrontSide });
  // world-space vertices: skip three's model/normal-matrix varyings (the shadow bake's depth-only
  // override uses the standard transform, which is identical for this identity-matrix mesh)
  mat.vertexNode = cameraProjectionMatrix.mul(cameraViewMatrix).mul(vec4(positionGeometry, 1.0));
  const vPos = varying(positionGeometry);
  const vNrm = varying(normalGeometry);
  // water-column fog per vertex (smooth over these small / far triangles; same terms as env.finish)
  const fogV = env.fogTerms(positionGeometry);
  const vFogT = varying(fogV.T);
  const vFogC = varying(fogV.color);
  const vCol = varying(attribute("aCol", "vec4")); // albedo rgb, baked ao
  // x sheen, y subsurface glow, z caustic gain, w: > 0 iridescence (clam mantle), < 0 corallite pores (rubble)
  const vPar = varying(attribute("aPar", "vec4"));
  mat.fragmentNode = Fn(() => {
    const P = vPos.toVar();
    const Nmacro = normalize(vNrm).toVar();
    const V = normalize(cameraPosition.sub(P)).toVar();
    const nv = clamp(dot(Nmacro, V), 0.0, 1.0);
    const fr = float(1.0).sub(nv);
    // dead-coral corallite pits + 3D calcareous rock/sponge bump normals
    const q = P.mul(240.0).toVar();
    const cell = floor(q);
    const h3 = fract(sin(vec3(dot(cell, vec3(127.1, 311.7, 74.7)), dot(cell, vec3(269.5, 183.3, 246.1)), dot(cell, vec3(113.5, 271.9, 124.6)))).mul(43758.547));
    const pdVec = fract(q).sub(h3.mul(0.4).add(0.3)).toVar();
    const pd = length(pdVec);
    const hasPit = step(0.18, h3.x.add(h3.y).mul(0.5));
    const pit = float(1.0).sub(smoothstep(0.13, 0.30, pd)).mul(hasPit);
    const fadePx = float(1.0).sub(smoothstep(0.32, 0.85, length(fwidth(q))));
    const poreAmt = max(vPar.w.negate(), 0.0).mul(fadePx);
    // multi-octave 3D limestone/sponge micro-relief gradient + inward crater rim slope on corallite pits
    const rockBump = (pt) =>
      sin(pt.x.mul(34.0).add(sin(pt.y.mul(23.0).add(pt.z.mul(29.0))).mul(1.4))).mul(0.55)
        .add(sin(pt.z.mul(83.0).sub(pt.x.mul(67.0)).add(cos(pt.y.mul(71.0)).mul(1.2))).mul(0.32));
    const eB = 0.0035;
    const b0 = rockBump(P).toVar();
    const gRock = vec3(
      rockBump(P.add(vec3(eB, 0.0, 0.0))).sub(b0),
      rockBump(P.add(vec3(0.0, eB, 0.0))).sub(b0),
      rockBump(P.add(vec3(0.0, 0.0, eB))).sub(b0)
    ).mul(1.0 / (eB * 48.0));
    const pitSlope = pdVec.div(max(pd, float(1e-3))).mul(sin(clamp(pd.div(0.30), 0.0, 1.0).mul(3.14159)).mul(hasPit).mul(poreAmt).mul(0.85));
    const gTot = gRock.mul(mix(float(0.55), float(0.18), max(vPar.w, 0.0))).add(pitSlope);
    const gSurf = gTot.sub(Nmacro.mul(dot(gTot, Nmacro)));
    const camFade = float(1.0).sub(smoothstep(2.5, 6.5, length(cameraPosition.sub(P))));
    const N = normalize(Nmacro.sub(gSurf.mul(u.bumpStrength.mul(camFade)))).toVar();
    // giant-clam mantle: iridophore structural colour that slides blue -> teal -> violet with view angle
    const film = cos(vec3(0.0, 2.1, 4.2).add(nv.mul(3.2)).add(P.x.add(P.z).mul(7.0))).mul(0.5).add(0.5);
    const iridCol = vCol.xyz.mul(0.8).add(film.mul(vec3(0.14, 0.18, 0.2)));
    const base = mix(vCol.xyz, iridCol, max(vPar.w, 0.0));
    const albedo = base.mul(float(1.0).sub(pit.mul(poreAmt).mul(0.32))).toVar();
    const ao = vCol.w.mul(float(1.0).sub(pit.mul(poreAmt).mul(0.28)));
    const shaded = env.litProp({ albedo, N, Nmacro, posW: P, ao, sheen: vPar.x, causticGain: vPar.z });
    // calcium-carbonate / tissue subsurface glow, same form and mood scaling as the coral shader's sssGlow
    const sssTint = mix(albedo, vec3(1.0, 0.96, 0.98), 0.34);
    const glow = sssTint.mul(vPar.y).mul(float(0.28).add(fr.mul(fr).mul(0.28))).mul(u.sssIntensity.div(0.28)).mul(ao.mul(0.4).add(0.6));
    const fogged = shaded.add(glow).mul(vFogT).add(vFogC.mul(vec3(1.0).sub(vFogT)));
    return env.output(fogged);
  })();
  return mat;
}

/** Feather-star arms: crossed ribbons with alpha-to-coverage pinnules, swaying with the shared surge. */
function makeCrinoidMaterial(env) {
  const u = env.u;
  const mat = new THREE.MeshBasicNodeMaterial({ name: "props_crinoid", side: THREE.DoubleSide });
  mat.alphaToCoverage = true;
  // Alpha-to-coverage needs the pinnule alpha in the fragment output, but that alpha must not reach the
  // (premultiplied) canvas: an opaque-style blend (colour One/Zero) that keeps the destination alpha.
  // `transparent` only enables the blend state; depth writes stay on and nothing is alpha-sorted/blended.
  mat.transparent = true;
  mat.depthWrite = true;
  mat.blending = THREE.CustomBlending;
  mat.blendEquation = THREE.AddEquation;
  mat.blendSrc = THREE.OneFactor;
  mat.blendDst = THREE.ZeroFactor;
  mat.blendEquationAlpha = THREE.AddEquation;
  mat.blendSrcAlpha = THREE.ZeroFactor;
  mat.blendDstAlpha = THREE.OneFactor;
  const aArm = attribute("aArm", "vec4"); // x: s along the arm (0 calyx .. 1 tip), y: across (-1..1), z: phase, w: sway gain
  const aCol = attribute("aCol", "vec4");
  const p0 = positionGeometry;
  const s = aArm.x;
  const t = u.time;
  // drag-driven bend: the shared surge, lagging toward the tip, growing with s^2 (zero at the calyx)
  const sw = env.surge(vec2(p0.x, p0.z), t, s.mul(0.55)).mul(aArm.w).mul(s.mul(s)).toVar();
  const flut = sin(t.mul(2.3).add(aArm.z.mul(6.2832)).add(s.mul(4.0))).mul(0.0035).mul(s);
  const posW = p0.add(vec3(sw.x.add(flut), dot(sw, sw).mul(-3.0), sw.y.sub(flut.mul(0.6)))).toVar();
  mat.vertexNode = cameraProjectionMatrix.mul(cameraViewMatrix).mul(vec4(posW, 1.0));
  const vPos = varying(posW);
  const vNrm = varying(normalGeometry);
  const vArm = varying(aArm.xy);
  const vCol = varying(aCol);
  const fogV = env.fogTerms(posW);
  const vFogT = varying(fogV.T);
  const vFogC = varying(fogV.color);
  mat.fragmentNode = Fn(() => {
    const P = vPos.toVar();
    const N = normalize(vNrm).mul(faceDirection).toVar();
    const sA = vArm.x;
    const av = abs(vArm.y);
    // pinnules: fine side branches angled toward the arm tip; fade to their mean coverage when they
    // get finer than a pixel (no shimmer at wide-view distances)
    const ph = sA.mul(30.0).sub(av.mul(1.15));
    const fw = fwidth(ph);
    const line = abs(fract(ph).sub(0.5));
    const wv = mix(float(0.34), float(0.14), av);
    const pin = float(1.0).sub(smoothstep(wv.sub(fw), wv.add(fw), line)).mul(float(1.0).sub(smoothstep(0.72, 1.0, av)));
    const pinAA = mix(pin, float(0.5), smoothstep(0.35, 0.9, fw));
    const rachis = float(1.0).sub(smoothstep(0.14, 0.3, av));
    const alpha = max(rachis, pinAA);
    const tip = smoothstep(0.55, 1.0, sA);
    const albedo = mix(vCol.xyz, vCol.xyz.mul(0.55).add(0.42), tip.mul(0.45).add(av.mul(0.18)));
    const shaded = env.litProp({ albedo, N, posW: P, ao: vCol.w, trans: 0.45, rim: 0.6 });
    return env.output(shaded.mul(vFogT).add(vFogC.mul(vec3(1.0).sub(vFogT))), alpha);
  })();
  return mat;
}

function placeholderGeometry(extra) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(9), 3));
  g.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(9), 3));
  g.setAttribute("aCol", new THREE.BufferAttribute(new Float32Array(12), 4));
  g.setAttribute(extra, new THREE.BufferAttribute(new Float32Array(12), 4));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array([0, 1, 2]), 1));
  return g;
}

function swapGeometry(mesh, geo) {
  const old = mesh.geometry;
  mesh.geometry = geo;
  old.dispose();
}

// -----------------------------------------------------------------------------
// CPU geometry helpers
// -----------------------------------------------------------------------------
const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const lerp = (a, b, t) => a + (b - a) * t;

/** Seeded 3D gradient noise (Perlin), range about [-1, 1]. */
function makeNoise3(rng) {
  const p = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = p[i];
    p[i] = p[j];
    p[j] = tmp;
  }
  const perm = new Uint8Array(512);
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  const G = [1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0, 1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1, 0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1];
  const g = (h, x, y, z) => {
    const k = (h % 12) * 3;
    return G[k] * x + G[k + 1] * y + G[k + 2] * z;
  };
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  return (x, y, z) => {
    const X = Math.floor(x), Y = Math.floor(y), Z = Math.floor(z);
    x -= X; y -= Y; z -= Z;
    const xi = X & 255, yi = Y & 255, zi = Z & 255;
    const u = fade(x), v = fade(y), w = fade(z);
    const A = perm[xi] + yi, AA = perm[A] + zi, AB = perm[A + 1] + zi;
    const B = perm[xi + 1] + yi, BA = perm[B] + zi, BB = perm[B + 1] + zi;
    return lerp(
      lerp(lerp(g(perm[AA], x, y, z), g(perm[BA], x - 1, y, z), u), lerp(g(perm[AB], x, y - 1, z), g(perm[BB], x - 1, y - 1, z), u), v),
      lerp(
        lerp(g(perm[AA + 1], x, y, z - 1), g(perm[BA + 1], x - 1, y, z - 1), u),
        lerp(g(perm[AB + 1], x, y - 1, z - 1), g(perm[BB + 1], x - 1, y - 1, z - 1), u),
        v
      ),
      w
    );
  };
}
function fbm(noise, x, y, z, oct) {
  let s = 0, a = 0.5, f = 1;
  for (let o = 0; o < oct; o++) {
    s += a * noise(x * f, y * f, z * f);
    f *= 2.03;
    a *= 0.5;
  }
  return s;
}

function computeNormals(p, idx) {
  const n = new Float32Array(p.length);
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    n[a] += nx; n[a + 1] += ny; n[a + 2] += nz;
    n[b] += nx; n[b + 1] += ny; n[b + 2] += nz;
    n[c] += nx; n[c + 1] += ny; n[c + 2] += nz;
  }
  for (let i = 0; i < n.length; i += 3) {
    const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1;
    n[i] /= l; n[i + 1] /= l; n[i + 2] /= l;
  }
  return n;
}
/** Finalise a local mesh: typed positions + smooth area-weighted normals (CCW = outward). */
function fin(m) {
  m.p = m.p instanceof Float32Array ? m.p : Float32Array.from(m.p);
  m.n = computeNormals(m.p, m.i);
  return m;
}
/** Concatenate local meshes. */
function merge(...ms) {
  const p = [], i = [];
  for (const m of ms) {
    const base = p.length / 3;
    for (const v of m.p) p.push(v);
    for (const k of m.i) i.push(base + k);
  }
  return { p, i };
}

/** Indexed unit icosphere (CCW, outward). sub=1: 80 tris, 2: 320, 3: 1280. Memoised: read-only result. */
const _ico = new Map();
function icosphere(sub) {
  if (!_ico.has(sub)) _ico.set(sub, buildIcosphere(sub));
  return _ico.get(sub);
}
function buildIcosphere(sub) {
  const t = (1 + Math.sqrt(5)) / 2;
  const verts = [];
  const add = (x, y, z) => {
    const l = Math.hypot(x, y, z);
    verts.push(x / l, y / l, z / l);
    return verts.length / 3 - 1;
  };
  for (const v of [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]) add(...v);
  let faces = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1];
  for (let s = 0; s < sub; s++) {
    const cache = new Map();
    const mid = (a, b) => {
      const key = a < b ? a * 1e6 + b : b * 1e6 + a;
      let m = cache.get(key);
      if (m === undefined) {
        m = add((verts[a * 3] + verts[b * 3]) / 2, (verts[a * 3 + 1] + verts[b * 3 + 1]) / 2, (verts[a * 3 + 2] + verts[b * 3 + 2]) / 2);
        cache.set(key, m);
      }
      return m;
    };
    const nf = [];
    for (let i = 0; i < faces.length; i += 3) {
      const a = faces[i], b = faces[i + 1], c = faces[i + 2];
      const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
      nf.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    faces = nf;
  }
  return { p: Float32Array.from(verts), i: faces };
}

/**
 * Closed tube along a polyline P (array of [x,y,z]) with radii R. Caps are pushed out along the end
 * tangents by capStart / capEnd x radius (false = open). radial(k, j, dx, dy, dz) scales each ring vertex.
 */
function tube(P, R, sides, o = {}) {
  const p = [], idx = [];
  const n = P.length;
  let ref = null;
  let T0 = null, T1 = null;
  for (let k = 0; k < n; k++) {
    const a = P[Math.max(0, k - 1)], b = P[Math.min(n - 1, k + 1)];
    let tx = b[0] - a[0], ty = b[1] - a[1], tz = b[2] - a[2];
    const tl = Math.hypot(tx, ty, tz) || 1;
    tx /= tl; ty /= tl; tz /= tl;
    if (k === 0) T0 = [tx, ty, tz];
    if (k === n - 1) T1 = [tx, ty, tz];
    if (!ref) ref = Math.abs(ty) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    // B = normalize(T x ref), N = B x T (ref carried along: approximate parallel transport)
    let bx = ty * ref[2] - tz * ref[1], by = tz * ref[0] - tx * ref[2], bz = tx * ref[1] - ty * ref[0];
    const bl = Math.hypot(bx, by, bz) || 1;
    bx /= bl; by /= bl; bz /= bl;
    const nx = by * tz - bz * ty, ny = bz * tx - bx * tz, nz = bx * ty - by * tx;
    ref = [nx, ny, nz];
    for (let j = 0; j < sides; j++) {
      const ang = (j / sides) * Math.PI * 2 + (o.twist || 0) * k;
      const c = Math.cos(ang), s = Math.sin(ang);
      const dx = nx * c + bx * s, dy = ny * c + by * s, dz = nz * c + bz * s;
      const r = R[k] * (o.radial ? o.radial(k, j, dx, dy, dz) : 1);
      p.push(P[k][0] + dx * r, P[k][1] + dy * r, P[k][2] + dz * r);
    }
  }
  for (let k = 0; k < n - 1; k++)
    for (let j = 0; j < sides; j++) {
      const a = k * sides + j, b = k * sides + ((j + 1) % sides), c = (k + 1) * sides + j, d = (k + 1) * sides + ((j + 1) % sides);
      idx.push(a, b, c, c, b, d);
    }
  if (o.capStart !== false) {
    const out = (o.capStart ?? 0.25) * R[0];
    const ci = p.length / 3;
    p.push(P[0][0] - T0[0] * out, P[0][1] - T0[1] * out, P[0][2] - T0[2] * out);
    for (let j = 0; j < sides; j++) idx.push(ci, (j + 1) % sides, j);
  }
  if (o.capEnd !== false) {
    const out = (o.capEnd ?? 0.25) * R[n - 1];
    const ci = p.length / 3, base = (n - 1) * sides;
    p.push(P[n - 1][0] + T1[0] * out, P[n - 1][1] + T1[1] * out, P[n - 1][2] + T1[2] * out);
    for (let j = 0; j < sides; j++) idx.push(ci, base + j, base + ((j + 1) % sides));
  }
  return { p, i: idx };
}

/**
 * Surface of revolution around +y. The profile [[r, y], ...] must run counter-clockwise in the (r, y)
 * half-plane (up the outside, inward across the top, down the inside) for outward-facing triangles.
 */
function lathe(prof, seg, radial) {
  const p = [], idx = [];
  const K = prof.length;
  for (let k = 0; k < K; k++)
    for (let j = 0; j < seg; j++) {
      const a = (j / seg) * Math.PI * 2;
      const r = radial ? radial(k, j, a, prof[k][0], prof[k][1]) : prof[k][0];
      p.push(Math.cos(a) * r, prof[k][1], Math.sin(a) * r);
    }
  for (let k = 0; k < K - 1; k++)
    for (let j = 0; j < seg; j++) {
      const a = k * seg + j, b = (k + 1) * seg + j, c = k * seg + ((j + 1) % seg), d = (k + 1) * seg + ((j + 1) % seg);
      idx.push(a, b, c, b, d, c);
    }
  return { p, i: idx };
}

/** Growable merged vertex stream: position, normal, aCol (albedo, ao) and one extra vec4. */
class Accum {
  constructor() {
    this.cap = 1 << 15;
    this.n = 0;
    this.pos = new Float32Array(this.cap * 3);
    this.nrm = new Float32Array(this.cap * 3);
    this.col = new Float32Array(this.cap * 4);
    this.ext = new Float32Array(this.cap * 4);
    this.idx = [];
  }
  grow(k) {
    if (this.n + k <= this.cap) return;
    while (this.n + k > this.cap) this.cap *= 2;
    const g = (a, s) => {
      const b = new Float32Array(this.cap * s);
      b.set(a);
      return b;
    };
    this.pos = g(this.pos, 3);
    this.nrm = g(this.nrm, 3);
    this.col = g(this.col, 4);
    this.ext = g(this.ext, 4);
  }
  vertex(x, y, z, nx, ny, nz, o) {
    this.grow(1);
    const j = this.n++;
    this.pos[j * 3] = x; this.pos[j * 3 + 1] = y; this.pos[j * 3 + 2] = z;
    this.nrm[j * 3] = nx; this.nrm[j * 3 + 1] = ny; this.nrm[j * 3 + 2] = nz;
    for (let c = 0; c < 4; c++) {
      this.col[j * 4 + c] = o[c];
      this.ext[j * 4 + c] = o[4 + c];
    }
    return j;
  }
  /** Append a finalised local mesh under matrix m; shade(i, worldPos, worldNormal, out8) sets the attributes. */
  addMesh(mesh, m, shade) {
    const k = mesh.p.length / 3;
    this.grow(k);
    const nm = _nm3.getNormalMatrix(m);
    const base = this.n;
    for (let i = 0; i < k; i++) {
      _v.set(mesh.p[i * 3], mesh.p[i * 3 + 1], mesh.p[i * 3 + 2]).applyMatrix4(m);
      _w.set(mesh.n[i * 3], mesh.n[i * 3 + 1], mesh.n[i * 3 + 2]).applyMatrix3(nm).normalize();
      shade(i, _v, _w, _o);
      this.vertex(_v.x, _v.y, _v.z, _w.x, _w.y, _w.z, _o);
    }
    for (const t of mesh.i) this.idx.push(base + t);
  }
  geometry(extraName) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(this.pos.slice(0, this.n * 3), 3));
    g.setAttribute("normal", new THREE.BufferAttribute(this.nrm.slice(0, this.n * 3), 3));
    g.setAttribute("aCol", new THREE.BufferAttribute(this.col.slice(0, this.n * 4), 4));
    g.setAttribute(extraName, new THREE.BufferAttribute(this.ext.slice(0, this.n * 4), 4));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(this.idx.length ? this.idx : [0, 0, 0]), 1));
    if (!this.n) {
      // nothing built: keep a valid degenerate triangle
      for (const name of ["position", "normal"]) g.setAttribute(name, new THREE.BufferAttribute(new Float32Array(9), 3));
      for (const name of ["aCol", extraName]) g.setAttribute(name, new THREE.BufferAttribute(new Float32Array(12), 4));
    }
    g.computeBoundingSphere();
    return g;
  }
}
const _v = new THREE.Vector3();
const _w = new THREE.Vector3();
const _nm3 = new THREE.Matrix3();
const _o = new Float32Array(8);

/** Matrix for a prop standing on the seabed: aligned to the terrain normal (blend), yaw, extra tilt. */
function groundMatrix(T, x, y, z, yaw, tilt, tiltDir, scale, normalBlend = 1) {
  const n = T.normalAt(x, z);
  const nb = new THREE.Vector3().copy(UP).lerp(n, normalBlend).normalize();
  const q = new THREE.Quaternion().setFromUnitVectors(UP, nb);
  if (tilt) q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(Math.cos(tiltDir), 0, Math.sin(tiltDir)), tilt));
  q.multiply(new THREE.Quaternion().setFromAxisAngle(UP, yaw));
  const s = typeof scale === "number" ? new THREE.Vector3(scale, scale, scale) : scale;
  return new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), q, s);
}

// -----------------------------------------------------------------------------
// Occupancy grids from the Houdini corals (CPU, once)
// -----------------------------------------------------------------------------
function makeOccupancy(T) {
  const CS = 0.04, X0 = -4.4, Z0 = -3.4, NX = 220, NZ = 170;
  const minY = new Float32Array(NX * NZ).fill(1e9); // lowest coral point per cell (all species)
  const otherMinY = new Float32Array(NX * NZ).fill(1e9); // lowest non-brain coral point per cell
  const otherMaxY = new Float32Array(NX * NZ).fill(-1e9); // highest non-brain coral point per cell
  const brainTop = new Float32Array(NX * NZ).fill(-1e9);
  for (const g of T.coralGeometry || []) {
    const P = g.positions;
    const isBrain = g.name === "brain";
    for (let i = 0; i < P.length; i += 6) {
      const gi = Math.floor((P[i] - X0) / CS), gj = Math.floor((P[i + 2] - Z0) / CS);
      if (gi < 0 || gj < 0 || gi >= NX || gj >= NZ) continue;
      const c = gj * NX + gi, y = P[i + 1];
      if (y < minY[c]) minY[c] = y;
      if (isBrain) {
        if (y > brainTop[c]) brainTop[c] = y;
      } else {
        if (y < otherMinY[c]) otherMinY[c] = y;
        if (y > otherMaxY[c]) otherMaxY[c] = y;
      }
    }
  }
  const sand = new Float32Array(NX * NZ);
  for (let j = 0; j < NZ; j++) for (let i = 0; i < NX; i++) sand[j * NX + i] = T.heightAt(X0 + (i + 0.5) * CS, Z0 + (j + 0.5) * CS);
  const range = (x, z, r) => [
    Math.max(0, Math.floor((x - r - X0) / CS)),
    Math.min(NX - 1, Math.floor((x + r - X0) / CS)),
    Math.max(0, Math.floor((z - r - Z0) / CS)),
    Math.min(NZ - 1, Math.floor((z + r - Z0) / CS)),
  ];
  /** true when no coral comes within `clear` m above the sand inside radius r of (x, z) */
  const free = (x, z, r, clear) => {
    const [i0, i1, j0, j1] = range(x, z, r);
    for (let j = j0; j <= j1; j++)
      for (let i = i0; i <= i1; i++) {
        const c = j * NX + i;
        if (minY[c] < sand[c] + clear) return false;
      }
    return true;
  };
  /** true when no non-brain coral overlaps the height band [y0, y1] inside radius r of (x, z) */
  const freeOfOthers = (x, z, r, y0, y1) => {
    const [i0, i1, j0, j1] = range(x, z, r);
    for (let j = j0; j <= j1; j++)
      for (let i = i0; i <= i1; i++) {
        const c = j * NX + i;
        if (otherMinY[c] < y1 && otherMaxY[c] > y0) return false;
      }
    return true;
  };
  return { CS, X0, Z0, NX, NZ, minY, otherMinY, otherMaxY, brainTop, sand, free, freeOfOthers };
}

// -----------------------------------------------------------------------------
// Build everything (CPU, once the reef is loaded)
// -----------------------------------------------------------------------------
function buildAll(env, T, AB) {
  const tm = {};
  let tPrev = performance.now();
  const lap = (k) => {
    const t = performance.now();
    tm[k] = Math.round(t - tPrev);
    tPrev = t;
  };
  const rng = env.makeRng(0x5eed7);
  const noise = makeNoise3(env.makeRng(0x2b1d));
  const occ = makeOccupancy(T);
  lap("occ");
  const stat = new Accum();
  const crin = new Accum();
  const reserved = []; // {x, z, r}: life footprints (rubble keeps out of their cores)
  if (Array.isArray(env.sites && env.sites.anemones)) {
    for (const a of env.sites.anemones) {
      if (a && a.position) reserved.push({ x: a.position.x, z: a.position.z, r: (a.radius || 0.18) + 0.04 });
    }
  }
  const obstacles = [];
  const sites = { sponges: [], urchins: [], stars: [], clams: [], crinoids: [], boulders: [], bommies: [] };
  const counts = { rubble: 0, bommies: 0 };
  const isReserved = (x, z, r) => reserved.some((s) => Math.hypot(s.x - x, s.z - z) < s.r + r);
  // the sand between the hero camera and the bouquet: anything that lands in the hero frame within
  // ~3.4 m of the lens is kept small and sparse (hero composition: eye (-0.2, 1.12, 2.95) -> (0.05, 1.16, 0.02), FOV 33)
  const heroCam = new THREE.PerspectiveCamera(33, 2940 / 1660, 0.08, 120);
  heroCam.position.set(-0.2, 1.12, 2.95);
  heroCam.lookAt(0.05, 1.16, 0.02);
  heroCam.updateMatrixWorld();
  heroCam.updateProjectionMatrix();
  const _hp = new THREE.Vector3();
  const heroCorridor = (x, z) => {
    const y = T.heightAt(x, z);
    const dist = Math.hypot(x + 0.2, y - 1.12, z - 2.95);
    _hp.set(x, y, z).project(heroCam);
    const inFrame = smooth(1.12, 0.98, Math.abs(_hp.x)) * smooth(1.12, 0.98, Math.abs(_hp.y)) * (_hp.z < 1 ? 1 : 0);
    const corridor = smooth(1.5, 1.1, Math.abs(x)) * smooth(0.25, 0.6, z); // orbiting hero distance
    return Math.max(inFrame * smooth(3.6, 3.0, dist), corridor);
  };
  // keep the low "Look Up" cameras free of props (env.cameraPresets.up when available)
  const upEye = (env.cameraPresets && env.cameraPresets.up && env.cameraPresets.up.eye) || [-1.6, 0.35, -2.4];
  const keepOut = (x, z) => Math.hypot(x - 0.9, z - 1.9) < 0.45 || Math.hypot(x - upEye[0], z - upEye[2]) < 0.55;

  /** Nearest valid spot to (ax, az): coral distance in [dMin, dMax], no coral base within r. */
  const findSpot = (ax, az, o) => {
    const step = 0.04;
    const maxR = o.search ?? 0.6;
    for (let rad = 0; rad <= maxR + 1e-6; rad += step) {
      const nA = rad === 0 ? 1 : Math.max(6, Math.round((2 * Math.PI * rad) / step));
      const a0 = rng() * Math.PI * 2;
      for (let k = 0; k < nA; k++) {
        const a = a0 + (k / nA) * Math.PI * 2;
        const x = ax + Math.cos(a) * rad, z = az + Math.sin(a) * rad;
        const d = T.coralDist(x, z);
        if (d < (o.dMin ?? 0) || d > (o.dMax ?? 9)) continue;
        if (!occ.free(x, z, o.r, o.clear ?? 0.25)) continue;
        if (isReserved(x, z, o.r * 0.8) || keepOut(x, z)) continue;
        return { x, z, y: T.heightAt(x, z), d };
      }
    }
    return null;
  };

  // ---------------------------------------------------------------- life ---
  if (!AB.has("propsNoLife")) {
    // dead-coral boulders at the reef edge (perches for sponges and urchins; structure in the rubble)
    const boulderAt = [
      [-2.5, 0.15], [2.48, 0.0], [-1.28, 1.18], [2.15, 1.08], [-2.42, -1.35], [2.38, -1.32],
      [-0.55, -1.32], [0.9, -1.28], [-2.95, -0.55], [2.95, -0.65],
    ];
    const boulderMesh = [0, 1, 2, 3].map((v) => boulderPiece(env.makeRng(500 + v), noise, v * 17.3));
    for (const [ax, az] of boulderAt) {
      const S = 0.2 + rng() * 0.2;
      const spot = findSpot(ax, az, { r: S * 0.45, dMin: 0.0, dMax: 0.7, clear: 0.2 });
      if (!spot) continue;
      const mesh = boulderMesh[Math.floor(rng() * boulderMesh.length)];
      const hy = mesh.hy * S;
      const y = spot.y - hy * 0.35;
      const m = groundMatrix(T, spot.x, y, spot.z, rng() * 6.283, (rng() - 0.5) * 0.3, rng() * 6.283, S, 0.7);
      addRubbleLike(stat, mesh, m, T, noise, pick(rng, [C.lilacGrey, C.mauve, C.paleLilac]), 0.7 + rng() * 0.25, rng() < 0.6 ? 0.55 : 0, hy * 1.6, rng() * 50, rng() < 0.4 ? 0.4 : 0, 0.18, 0.7);
      reserved.push({ x: spot.x, z: spot.z, r: S * 0.42 });
      sites.boulders.push({ x: spot.x, z: spot.z, r: S * 0.5, top: y + hy });
      if (y + hy - spot.y > 0.12) obstacles.push({ x: spot.x, z: spot.z, y0: spot.y, y1: y + hy, r: S * 0.5 });
    }

    // tube-sponge clusters (Callyspongia / Haliclona): lathe tubes with rolled lips on hard substrate
    const spongeAt = [
      { at: [-2.28, 1.02], col: C.lemon, n: 5, h: 0.38 },
      { at: [2.42, -0.28], col: C.lavender, n: 6, h: 0.42 },
      { at: [-0.78, -1.3], col: C.periwinkle, n: 4, h: 0.36 },
      { at: [1.12, -1.36], col: C.lavender, n: 7, h: 0.3 },
    ];
    for (const sp of spongeAt) {
      const spot = findSpot(sp.at[0], sp.at[1], { r: 0.12, dMin: 0.0, dMax: 0.8, clear: 0.3 });
      if (!spot) continue;
      const top = addTubeSponge(stat, T, spot, sp, env.makeRng(Math.floor(spot.x * 1000 + spot.z * 7919) >>> 0), noise);
      reserved.push({ x: spot.x, z: spot.z, r: 0.14 });
      sites.sponges.push({ x: spot.x, z: spot.z, top });
      obstacles.push({ x: spot.x, z: spot.z, y0: spot.y, y1: top, r: 0.17 });
    }

    // one giant barrel sponge (Xestospongia testudinaria), background
    {
      const spot = findSpot(-3.1, -2.1, { r: 0.42, dMin: 0.3, dMax: 2.5, clear: 0.6, search: 0.8 });
      if (spot) {
        const top = addBarrelSponge(stat, T, spot, env.makeRng(777), noise);
        reserved.push({ x: spot.x, z: spot.z, r: 0.45 });
        sites.sponges.push({ x: spot.x, z: spot.z, top, barrel: true });
        obstacles.push({ x: spot.x, z: spot.z, y0: spot.y, y1: top, r: 0.46 });
      }
    }

    // giant clams (Tridacna) nestled at coral bases
    const clamAt = [
      { at: [1.22, 0.92], mantle: [0.52, 0.72, 0.86], spot2: [0.78, 0.9, 0.84] },
      { at: [1.25, -0.98], mantle: [0.62, 0.58, 0.88], spot2: [0.58, 0.84, 0.8] },
    ];
    for (const cl of clamAt) {
      const spot = findSpot(cl.at[0], cl.at[1], { r: 0.15, dMin: 0.0, dMax: 0.45, clear: 0.22 });
      if (!spot) continue;
      addClam(stat, T, spot, cl, env.makeRng(Math.floor(spot.x * 991 + 13) >>> 0), noise);
      reserved.push({ x: spot.x, z: spot.z, r: 0.16 });
      sites.clams.push({ x: spot.x, z: spot.z });
    }

    // sea stars: two Linckia (periwinkle-blue), one cushion star (apricot)
    const starAt = [
      { type: "linckia", at: [-0.62, 1.5] },
      { type: "linckia", at: [2.95, -0.22] },
      { type: "cushion", at: [1.62, 1.38] },
    ];
    for (const st of starAt) {
      const spot = findSpot(st.at[0], st.at[1], { r: 0.13, dMin: 0.02, dMax: 1.2, clear: 0.15 });
      if (!spot) continue;
      const srng = env.makeRng(Math.floor(spot.z * 3301 + 7) >>> 0);
      if (st.type === "linckia") addLinckia(stat, T, spot, srng, noise);
      else addCushionStar(stat, T, spot, srng, noise);
      reserved.push({ x: spot.x, z: spot.z, r: 0.14 });
      sites.stars.push({ x: spot.x, z: spot.z, type: st.type });
    }

    // Diadema urchins in the rubble (sheltering at the reef edge)
    const urchinAt = [
      [-1.22, 1.26], [0.92, 1.32], [2.18, 1.22], [-2.62, 0.62], [2.72, 0.36],
      [-1.38, -1.42], [0.24, -1.26], [1.52, -1.52], [-2.76, -0.96], [2.72, -0.98],
    ];
    for (const [ax, az] of urchinAt) {
      const spot = findSpot(ax, az, { r: 0.07, dMin: 0.0, dMax: 0.7, clear: 0.22 });
      if (!spot) continue;
      addUrchin(stat, T, spot, env.makeRng(Math.floor(spot.x * 4049 + spot.z * 271) >>> 0));
      reserved.push({ x: spot.x, z: spot.z, r: 0.06 });
      sites.urchins.push({ x: spot.x, z: spot.z });
    }

    lap("life");
    // crinoids perched on the brain-mound tops, arms fanned across the surge
    if (!AB.has("propsNoCrinoids")) {
      const perches = findBrainPerches(T, occ);
      const crinoidCols = [[0.96, 0.85, 0.52], [0.98, 0.66, 0.5], [0.62, 0.42, 0.74], [0.93, 0.89, 0.8]];
      perches.slice(0, 4).forEach((pp, k) => {
        addCrinoid(crin, stat, pp, env, crinoidCols[k % crinoidCols.length], env.makeRng(9100 + k));
        sites.crinoids.push({ x: pp.x, y: pp.y, z: pp.z });
        obstacles.push({ x: pp.x, z: pp.z, y0: pp.y, y1: pp.y + 0.17, r: 0.14 });
      });
    }
  }

  lap("crinoids");
  // -------------------------------------------------------------- rubble ---
  if (!AB.has("propsNoRubble")) counts.rubble = addRubbleHalo(stat, T, occ, env, noise, isReserved, heroCorridor, keepOut);

  lap("rubble");
  // ------------------------------------------------------------- bommies ---
  if (!AB.has("propsNoBommies")) counts.bommies = addBommies(stat, T, env.makeRng(4242), noise, sites);
  lap("bommies");

  return { stat, crin, obstacles, sites, counts, tm };
}

const pick = (rng, arr) => arr[Math.floor(rng() * arr.length) % arr.length];

// -----------------------------------------------------------------------------
// Rubble
// -----------------------------------------------------------------------------
/** Corallite knobs: radial noise for tube rings (seeded, closure-local). */
const knobby = (noise, seed, amp = 1) => (k, j, dx, dy, dz) =>
  1 + amp * (0.15 * noise(k * 0.9 + dx * 1.3 + seed, dy * 1.3, dz * 1.3) + 0.13 * Math.abs(noise(k * 2.7 + dx * 3.1 - seed, dy * 3.1, dz * 3.1)));

/** Broken staghorn branch: knobbly bent tapered tube, ragged ends, 0-2 broken side branches (~50-110 tris). */
function branchPiece(rng, noise, seed) {
  const rings = 5, sides = 6;
  const r0 = 0.1 + rng() * 0.06, taper = 0.15 + rng() * 0.3;
  const bendY = (rng() - 0.5) * 0.25, bendZ = (rng() - 0.5) * 0.45;
  const cl = (s) => [-0.5 + s, bendY * Math.sin(Math.PI * s), bendZ * Math.sin(Math.PI * s)];
  const P = [], R = [];
  for (let k = 0; k < rings; k++) {
    const s = k / (rings - 1);
    P.push(cl(s));
    R.push(r0 * (1 - taper * s) * (0.85 + rng() * 0.3));
  }
  const parts = [tube(P, R, sides, { radial: knobby(noise, seed, 1.35), capStart: 0.1 + rng() * 0.6, capEnd: 0.1 + rng() * 0.6, twist: 0.35 })];
  const nSide = rng() < 0.3 ? 0 : rng() < 0.65 ? 1 : 2;
  for (let b = 0; b < nSide; b++) {
    const s0 = 0.2 + rng() * 0.55;
    const c0 = cl(s0);
    const side = (b === 0 ? 1 : -1) * (rng() < 0.5 ? -1 : 1);
    const dir = new THREE.Vector3(0.45 + rng() * 0.35, (rng() - 0.25) * 0.6, side * (0.7 + rng() * 0.3)).normalize();
    const L = 0.16 + rng() * 0.24;
    const rr = r0 * (1 - taper * s0) * (0.55 + rng() * 0.15);
    const P2 = [0, 0.5, 1].map((t) => [c0[0] + dir.x * L * t, c0[1] + dir.y * L * t, c0[2] + dir.z * L * t]);
    parts.push(tube(P2, [rr, rr * 0.88, rr * 0.74], 5, { capStart: false, capEnd: 0.2 + rng() * 0.5, radial: knobby(noise, seed + 9 * b + 3) }));
  }
  const m = fin(merge(...parts));
  m.hy = r0 * 1.1;
  m.tag = null;
  return m;
}

/** Broken table / plate fragment: notched irregular, cupped thin slab with radial corallite ridges (~80 tris). */
function platePiece(rng, noise, seed) {
  const K = 12;
  const thick = 0.035 + rng() * 0.04, cup = (rng() - 0.35) * 0.3;
  const ang = [], rad = [];
  const notch = Math.floor(rng() * K);
  for (let k = 0; k < K; k++) {
    ang.push((k / K) * Math.PI * 2 + (rng() - 0.5) * 0.35);
    let r = 0.5 * (0.5 + 0.5 * rng());
    if (k === notch || (k === (notch + 5) % K && rng() < 0.6)) r *= 0.45; // broken bites
    rad.push(r);
  }
  const yS = (x, z, rr) => cup * rr * rr * 4 + 0.02 * noise(x * 6 + seed, 0.5, z * 6);
  const p = [], i = [], tag = [];
  const v = (x, y, z, t) => (p.push(x, y, z), tag.push(t), p.length / 3 - 1);
  const ridge = (a) => 0.5 + 0.5 * Math.cos(a * 15 + seed);
  const ring = (f, off, shrink, top) =>
    ang.map((a, k) => {
      const rr = rad[k] * f * shrink;
      const x = Math.cos(a) * rr, z = Math.sin(a) * rr;
      return v(x, yS(x, z, rr) + off, z, top ? ridge(a) : 0.4);
    });
  const h = thick / 2;
  const ct = v(0, yS(0, 0, 0) + h, 0, 0.5), mt = ring(0.55, h, 1, true), ot = ring(1, h, 1, true);
  const cb = v(0, yS(0, 0, 0) - h, 0, 0.4), mb = ring(0.55, -h, 0.97, false), ob = ring(1, -h, 0.94, false);
  for (let k = 0; k < K; k++) {
    const k1 = (k + 1) % K;
    i.push(ct, mt[k1], mt[k], mt[k], mt[k1], ot[k], ot[k], mt[k1], ot[k1]); // top (+y)
    i.push(cb, mb[k], mb[k1], mb[k1], mb[k], ob[k], mb[k1], ob[k], ob[k1]); // bottom (-y)
  }
  // side walls with their own vertices (crisp broken edge, paler fresh break)
  for (let k = 0; k < K; k++) {
    const k1 = (k + 1) % K;
    const g = (idx) => v(p[idx * 3], p[idx * 3 + 1], p[idx * 3 + 2], 0.9);
    const tk = g(ot[k]), tk1 = g(ot[k1]), bk = g(ob[k]), bk1 = g(ob[k1]);
    i.push(tk, tk1, bk, bk, tk1, bk1);
  }
  const m = fin({ p, i });
  m.hy = h + Math.abs(cup) * 0.6;
  m.tag = Float32Array.from(tag);
  return m;
}

/** Porous coral chunk: angular displaced, flattened icosphere (80 tris). */
function cobblePiece(rng, noise, seed) {
  const s = icosphere(1);
  const ys = 0.55 + rng() * 0.25, zs = 0.7 + rng() * 0.3;
  const p = new Float32Array(s.p.length);
  for (let k = 0; k < s.p.length; k += 3) {
    const x = s.p[k], y = s.p[k + 1], z = s.p[k + 2];
    const r = 0.5 * (1 + 0.3 * noise(x * 2.1 + seed, y * 2.1, z * 2.1 - seed) + 0.16 * noise(x * 5.3 - seed, y * 5.3, z * 5.3));
    p[k] = x * r;
    p[k + 1] = y * r * ys;
    p[k + 2] = z * r * zs;
  }
  const m = fin({ p, i: s.i });
  m.hy = 0.5 * ys;
  m.tag = null;
  return m;
}

/** Dead-coral boulder: lumpy displaced icosphere (320 tris). */
function boulderPiece(rng, noise, seed) {
  const s = icosphere(3);
  const ys = 0.55 + rng() * 0.25, zs = 0.75 + rng() * 0.25;
  const p = new Float32Array(s.p.length);
  for (let k = 0; k < s.p.length; k += 3) {
    const x = s.p[k], y = s.p[k + 1], z = s.p[k + 2];
    const r = 0.5 * (1 + 0.3 * fbm(noise, x * 1.4 + seed, y * 1.4, z * 1.4 + seed, 3) + 0.12 * Math.abs(noise(x * 4 + seed, y * 4, z * 4)) + 0.05 * noise(x * 11 - seed, y * 11, z * 11));
    p[k] = x * r;
    p[k + 1] = y * r * ys;
    p[k + 2] = z * r * zs;
  }
  const m = fin({ p, i: s.i });
  m.hy = 0.5 * ys;
  m.tag = null;
  return m;
}

/** Append a rubble-like mesh with coralline / turf mottling, corallite ridges and contact AO against the sand. */
function addRubbleLike(acc, mesh, m, T, noise, base, bright, cca, hExt, seed, turf = 0, glow = 0.2, pores = 0.9) {
  const tag = mesh.tag;
  const sandY = T.heightAt(m.elements[12], m.elements[14]); // pieces are small: one sand height per piece
  const mottled = cca > 0 || turf > 0;
  acc.addMesh(mesh, m, (i, wp, wn, o) => {
    const mk = mottled ? noise(wp.x * 24 + seed, wp.y * 24, wp.z * 24 - seed) : 0;
    const up = smooth(-0.25, 0.55, wn.y);
    let r = base[0], g = base[1], b = base[2];
    const wc = cca * smooth(-0.12, 0.3, mk) * (0.3 + 0.7 * up);
    r = lerp(r, C.cca[0], wc); g = lerp(g, C.cca[1], wc); b = lerp(b, C.cca[2], wc);
    const wt = turf * smooth(0.05, 0.45, -mk) * up;
    r = lerp(r, C.turf[0], wt); g = lerp(g, C.turf[1], wt); b = lerp(b, C.turf[2], wt);
    const hs = Math.sin(i * 12.9898 + seed * 78.233) * 43758.5453;
    const grain = 0.94 + 0.12 * (hs - Math.floor(hs));
    const rib = tag ? 0.9 + 0.2 * tag[i] : 1;
    const h = wp.y - sandY;
    const ao = 0.6 + 0.4 * smooth(-0.004, Math.max(0.012, hExt * 0.8), h);
    const v = bright * grain * rib;
    o[0] = r * v; o[1] = g * v; o[2] = b * v; o[3] = ao;
    o[4] = 0.05; o[5] = glow; o[6] = 1.0; o[7] = -pores * (1 - 0.6 * wc);
  });
}

function addRubbleHalo(acc, T, occ, env, noise, isReserved, heroCorridor, keepOut) {
  const rng = env.makeRng(0xc0ffee);
  const pieces = {
    branch: [0, 1, 2, 3, 4].map((k) => branchPiece(env.makeRng(100 + k), noise, k * 13.1)),
    plate: [0, 1, 2].map((k) => platePiece(env.makeRng(200 + k), noise, k * 7.7)),
    cobble: [0, 1, 2].map((k) => cobblePiece(env.makeRng(300 + k), noise, k * 3.3)),
  };
  const N = 1400;
  const pal = [
    [C.bone, 0.32],
    [C.paleLilac, 0.24],
    [C.lilacGrey, 0.2],
    [C.paleRose, 0.14],
    [C.mauve, 0.1],
  ];
  const pickPal = () => {
    let r = rng();
    for (const [c, w] of pal) if ((r -= w) <= 0) return c;
    return pal[0][0];
  };
  let placed = 0;
  for (let tries = 0; placed < N && tries < 90000; tries++) {
    const x = -4.0 + rng() * 8.0, z = -3.0 + rng() * 5.9;
    const d = T.coralDist(x, z);
    if (d > 1.6) continue;
    let dens = d <= 0.001 ? 0.7 : Math.exp(-d / 0.36);
    dens *= 0.25 + 0.75 * smooth(-0.3, 0.4, noise(x * 1.3 + 3.7, 11.3, z * 1.3 - 2.1)); // patchy rubble fields
    dens *= smooth(1.5, 1.1, d); // none beyond ~1.5 m
    const hero = heroCorridor(x, z);
    dens *= 1 - 0.6 * hero; // keep the hero foreground sparse
    if (rng() > dens) continue;
    if (keepOut(x, z)) continue;
    const r = rng();
    let S = (0.03 + 0.085 * r * r) * (1 + 0.7 * Math.exp(-d / 0.22)); // bigger fragments pile up at the base
    if (rng() < 0.07 && d < 0.6) S = 0.12 + 0.12 * rng();
    S *= 1 - 0.5 * hero; // and small
    const tr = rng();
    const type = tr < 0.62 ? "branch" : tr < 0.85 ? "plate" : "cobble";
    if (type === "branch") S *= 1.25; // branch length
    if (!occ.free(x, z, S * 0.4, 0.2)) continue;
    if (isReserved(x, z, S * 0.35)) continue;
    const mesh = pick(rng, pieces[type]);
    const sand = T.heightAt(x, z);
    const hy = mesh.hy * S;
    const bury = 0.25 + rng() * 0.35; // fraction of the thickness below the sand
    const y = sand + hy * (1 - 2 * bury);
    const tilt = type === "plate" ? (rng() - 0.5) * 0.5 : (rng() - 0.5) * 0.6;
    const asp = type === "cobble" ? new THREE.Vector3(S, S * (0.85 + rng() * 0.3), S * (0.85 + rng() * 0.3)) : S;
    const m = groundMatrix(T, x, y, z, rng() * Math.PI * 2, tilt, rng() * Math.PI * 2, asp, 0.85);
    const base = pickPal();
    const cca = rng() < 0.3 ? 0.45 + 0.45 * rng() : 0;
    const turf = rng() < 0.2 ? 0.25 + 0.35 * rng() : 0;
    addRubbleLike(acc, mesh, m, T, noise, base, 0.9 + rng() * 0.18, cca, hy * 2.0, rng() * 100, turf);
    placed++;
  }
  return placed;
}

// -----------------------------------------------------------------------------
// Static life
// -----------------------------------------------------------------------------
function addTubeSponge(acc, T, spot, sp, rng, noise) {
  const n = sp.n;
  let top = spot.y;
  // encrusting base pad
  const pad = fin(lathe([[0.0, -0.03], [0.1, -0.03], [0.12, 0.0], [0.1, 0.02], [0.06, 0.035], [0.0, 0.04]], 12, (k, j, a, r) => r * (1 + 0.18 * Math.sin(a * 3 + 1.3))));
  const mp = groundMatrix(T, spot.x, spot.y - 0.005, spot.z, rng() * 6.28, 0, 0, 1, 0.8);
  acc.addMesh(pad, mp, (i, wp, wn, o) => {
    o[0] = sp.col[0] * 0.8; o[1] = sp.col[1] * 0.8; o[2] = sp.col[2] * 0.8; o[3] = 0.65;
    o[4] = 0.12; o[5] = 0.22; o[6] = 1; o[7] = 0;
  });
  for (let k = 0; k < n; k++) {
    const h = sp.h * (0.55 + 0.45 * rng()) * (k === 0 ? 1.15 : 1);
    const r = 0.02 + 0.012 * rng() + h * 0.035;
    // profile: up the outside (slight flare), over the rolled lip, down the inside to a dark floor
    const prof = [
      [r * 1.06, -0.06], [r * 1.0, 0.0], [r * 0.97, 0.22 * h], [r * 1.0, 0.5 * h], [r * 1.05, 0.75 * h], [r * 1.12, 0.92 * h],
      [r * 1.16, 0.972 * h], [r * 1.08, h], [r * 0.9, 0.992 * h], [r * 0.8, 0.955 * h],
      [r * 0.76, 0.86 * h], [r * 0.74, 0.62 * h], [r * 0.72, 0.42 * h], [0.0, 0.4 * h],
    ];
    const seed = rng() * 40;
    const mesh = fin(lathe(prof, 12, (kk, j, a, rr, y) => rr * (1 + 0.07 * noise(Math.cos(a) * 1.5 + seed, y * 9, Math.sin(a) * 1.5))));
    const ang = (k / n) * Math.PI * 2 + rng() * 0.6;
    const off = k === 0 ? 0 : 0.035 + 0.05 * rng();
    const x = spot.x + Math.cos(ang) * off, z = spot.z + Math.sin(ang) * off;
    const tilt = k === 0 ? rng() * 0.12 : 0.12 + rng() * 0.22;
    const m = groundMatrix(T, x, T.heightAt(x, z) - 0.01, z, rng() * 6.28, tilt, ang, 1, 0.5);
    const lipBright = 1.12;
    acc.addMesh(mesh, m, (i, wp, wn, o) => {
      const kk = Math.floor(i / 12);
      const inner = kk >= 9;
      const yl = prof[kk][1] / h;
      let ao = inner ? 0.22 + 0.55 * smooth(0.4, 1.0, yl) : 0.62 + 0.38 * smooth(0.0, 0.5, yl);
      if (kk === 13) ao = 0.16;
      const lip = kk >= 6 && kk <= 8 ? lipBright : 1;
      const v = (1 + 0.06 * noise(wp.x * 40, wp.y * 40, wp.z * 40)) * lip * (inner ? 0.85 : 1);
      o[0] = sp.col[0] * v; o[1] = sp.col[1] * v; o[2] = sp.col[2] * v; o[3] = ao;
      o[4] = 0.16; o[5] = 0.3; o[6] = 1.0; o[7] = inner ? 0 : -0.35;
    });
    top = Math.max(top, T.heightAt(x, z) + h);
  }
  return top;
}

function addBarrelSponge(acc, T, spot, rng, noise) {
  const H = 0.74;
  const prof = [
    [0.21, -0.1], [0.2, 0.0], [0.27, 0.14 * H], [0.34, 0.34 * H], [0.38, 0.56 * H], [0.385, 0.78 * H],
    [0.37, 0.94 * H], [0.35, H], [0.31, H], // rolled rim
    [0.3, 0.97 * H], [0.29, 0.85 * H], [0.26, 0.62 * H], [0.2, 0.42 * H], [0.0, 0.38 * H], // deep atrium
  ];
  const seed = rng() * 30;
  const NR = 17, SEG = 68;
  const ridgePh = rng() * 6.28;
  const ridge = new Float32Array(prof.length * SEG).fill(1);
  const mesh = fin(
    lathe(prof, SEG, (k, j, a, r, y) => {
      if (k >= 9) return r * (1 + 0.03 * noise(Math.cos(a) * 2 + seed, y * 4, Math.sin(a) * 2));
      const rid = Math.pow(Math.abs(Math.cos((a * NR) / 2 + ridgePh + 0.3 * noise(y * 3 + seed, a, 1.7))), 3);
      ridge[k * SEG + j] = rid;
      return r * (1 + 0.2 * rid * smooth(0.0, 0.25, y) + 0.05 * fbm(noise, Math.cos(a) * 2 + seed, y * 3, Math.sin(a) * 2, 2));
    })
  );
  const m = groundMatrix(T, spot.x, spot.y - 0.05, spot.z, rng() * 6.28, 0.06, rng() * 6.28, 1, 0.6);
  acc.addMesh(mesh, m, (i, wp, wn, o) => {
    const k = Math.floor(i / SEG);
    const inner = k >= 9;
    const yl = prof[k][1] / H;
    const valley = inner ? 1 : 0.62 + 0.38 * ridge[i]; // darker grooves between the vertical ridges
    const ao = (inner ? 0.2 + 0.5 * smooth(0.36, 1.0, yl) : 0.55 + 0.45 * smooth(0.0, 0.35, yl)) * valley;
    const v = (1 + 0.08 * noise(wp.x * 18, wp.y * 18, wp.z * 18)) * (inner ? 0.8 : 1) * (k >= 6 && k <= 8 ? 1.1 : 1) * (0.9 + 0.1 * valley);
    o[0] = C.barrel[0] * v; o[1] = C.barrel[1] * v; o[2] = C.barrel[2] * v; o[3] = ao;
    o[4] = 0.12; o[5] = 0.16; o[6] = 0.9; o[7] = inner ? 0 : -0.45;
  });
  return spot.y + H - 0.05;
}

function addUrchin(acc, T, spot, rng) {
  const sc = 0.85 + rng() * 0.35;
  const cy = spot.y + 0.012 * sc;
  const test = icosphere(2);
  const tm = fin({ p: Float32Array.from(test.p), i: test.i });
  const m = new THREE.Matrix4().compose(new THREE.Vector3(spot.x, cy, spot.z), new THREE.Quaternion(), new THREE.Vector3(0.034 * sc, 0.024 * sc, 0.034 * sc));
  acc.addMesh(tm, m, (i, wp, wn, o) => {
    const top = smooth(0.012, 0.022, wp.y - cy);
    // tiny orange anal ring on top, iridescent-blue spots are below the spines' scale
    o[0] = lerp(C.urchin[0], 0.85, top * 0.7); o[1] = lerp(C.urchin[1], 0.45, top * 0.7); o[2] = lerp(C.urchin[2], 0.3, top * 0.7);
    o[3] = 0.55 + 0.45 * smooth(-0.02, 0.02, wp.y - cy);
    o[4] = 0.45; o[5] = 0; o[6] = 0.8; o[7] = 0;
  });
  // spines: long thin 3-sided needles radiating over the upper ~80% of the test
  const NS = 120;
  const ga = Math.PI * (3 - Math.sqrt(5));
  const p = [], idx = [];
  for (let k = 0; k < NS; k++) {
    const yy = 1 - (k + 0.5) / NS * 1.6; // down to y = -0.6
    const rr = Math.sqrt(Math.max(0, 1 - yy * yy));
    const th = k * ga + rng() * 0.3;
    let dx = Math.cos(th) * rr, dy = yy, dz = Math.sin(th) * rr;
    dy += 0.15; // spines splay slightly upward
    const dl = Math.hypot(dx, dy, dz);
    dx /= dl; dy /= dl; dz /= dl;
    const L = (0.07 + 0.08 * rng()) * sc * (0.75 + 0.35 * Math.max(0, dy));
    const bx = spot.x + dx * 0.03 * sc, by = cy + dy * 0.021 * sc, bz = spot.z + dz * 0.03 * sc;
    // frame
    const ref = Math.abs(dy) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    let ex = dy * ref[2] - dz * ref[1], ey = dz * ref[0] - dx * ref[2], ez = dx * ref[1] - dy * ref[0];
    const el = Math.hypot(ex, ey, ez);
    ex /= el; ey /= el; ez /= el;
    const fx = ey * dz - ez * dy, fy = ez * dx - ex * dz, fz = ex * dy - ey * dx;
    const rb = 0.0019 * sc;
    const base = p.length / 3;
    for (let j = 0; j < 3; j++) {
      const a = (j / 3) * Math.PI * 2;
      const c = Math.cos(a) * rb, s = Math.sin(a) * rb;
      p.push(bx + fx * c + ex * s, by + fy * c + ey * s, bz + fz * c + ez * s);
    }
    p.push(bx + dx * L, by + dy * L - L * L * 0.6 * (1 - dy), bz + dz * L);
    for (let j = 0; j < 3; j++) idx.push(base + j, base + ((j + 1) % 3), base + 3);
  }
  const sm = fin({ p, i: idx });
  acc.addMesh(sm, new THREE.Matrix4(), (i, wp, wn, o) => {
    const tip = i % 4 === 3 ? 1 : 0;
    o[0] = C.urchin[0] * (1 + 0.5 * tip); o[1] = C.urchin[1] * (1 + 0.5 * tip); o[2] = C.urchin[2] * (1 + 0.6 * tip);
    o[3] = 0.8 + 0.2 * tip;
    o[4] = 0.35; o[5] = 0; o[6] = 0.7; o[7] = 0;
  });
}

function addLinckia(acc, T, spot, rng, noise) {
  const yaw = rng() * 6.283;
  const parts = [];
  for (let k = 0; k < 5; k++) {
    const a0 = yaw + (k / 5) * Math.PI * 2 + (rng() - 0.5) * 0.15;
    const L = 0.12 + rng() * 0.035;
    const bend = (rng() - 0.5) * 0.7;
    const P = [], R = [];
    let x = 0, z = 0;
    const NSEG = 7;
    for (let s = 0; s <= NSEG; s++) {
      const t = s / NSEG;
      const a = a0 + bend * t * t;
      const r = 0.0115 * (1 - 0.38 * t) * (s === NSEG ? 0.6 : 1);
      P.push([x, r * 0.75, z]);
      R.push(r);
      x += (Math.cos(a) * L) / NSEG;
      z += (Math.sin(a) * L) / NSEG;
    }
    parts.push(tube(P, R, 7, { capStart: false, capEnd: 0.8, radial: (kk, j, dx, dy) => (dy < 0 ? 0.8 : 1) }));
  }
  const disc = icosphere(1);
  const dp = [];
  for (let i = 0; i < disc.p.length; i += 3) dp.push(disc.p[i] * 0.02, disc.p[i + 1] * 0.009 + 0.008, disc.p[i + 2] * 0.02);
  parts.push({ p: dp, i: disc.i });
  const mesh = fin(merge(...parts));
  const m = groundMatrix(T, spot.x, spot.y - 0.002, spot.z, 0, 0, 0, 1, 1);
  acc.addMesh(mesh, m, (i, wp, wn, o) => {
    const dist = Math.hypot(wp.x - spot.x, wp.z - spot.z);
    const tip = smooth(0.08, 0.15, dist);
    const v = 1 + 0.07 * noise(wp.x * 90, wp.y * 90, wp.z * 90);
    o[0] = lerp(C.linckia[0], 0.72, tip * 0.5) * v; o[1] = lerp(C.linckia[1], 0.76, tip * 0.5) * v; o[2] = lerp(C.linckia[2], 0.96, tip * 0.5) * v;
    o[3] = 0.6 + 0.4 * smooth(-0.15, 0.6, wn.y);
    o[4] = 0.25; o[5] = 0.18; o[6] = 1; o[7] = 0;
  });
}

function addCushionStar(acc, T, spot, rng, noise) {
  const s = icosphere(3);
  const R = 0.085 + rng() * 0.02;
  const ph = rng() * 6.28;
  const p = new Float32Array(s.p.length);
  for (let k = 0; k < s.p.length; k += 3) {
    const x = s.p[k], y = s.p[k + 1], z = s.p[k + 2];
    const th = Math.atan2(z, x);
    const rr = (1 + 0.11 * Math.cos(5 * (th - ph))) * R;
    p[k] = x * rr;
    p[k + 1] = (y > 0 ? y * 0.5 : y * 0.12) * R * (1 + 0.04 * Math.cos(5 * (th - ph)));
    p[k + 2] = z * rr;
  }
  const mesh = fin({ p, i: s.i });
  const m = groundMatrix(T, spot.x, spot.y + 0.004, spot.z, 0, 0, 0, 1, 1);
  acc.addMesh(mesh, m, (i, wp, wn, o) => {
    const n1 = noise(wp.x * 45 + 3.1, wp.y * 45, wp.z * 45);
    const net = 1 - smooth(0.0, 0.12, Math.abs(n1)); // reticulated darker grooves
    const dots = smooth(0.35, 0.55, noise(wp.x * 110, wp.y * 110 + 7, wp.z * 110)); // pale tubercles
    let r = C.cushion[0], g = C.cushion[1], b = C.cushion[2];
    r = lerp(r, 0.78, net * 0.55); g = lerp(g, 0.42, net * 0.55); b = lerp(b, 0.34, net * 0.55);
    r = lerp(r, 1.0, dots * 0.35); g = lerp(g, 0.86, dots * 0.35); b = lerp(b, 0.72, dots * 0.35);
    o[0] = r; o[1] = g; o[2] = b;
    o[3] = 0.55 + 0.45 * smooth(-0.3, 0.5, wn.y);
    o[4] = 0.2; o[5] = 0.22; o[6] = 1; o[7] = 0;
  });
}

function addClam(acc, T, spot, cl, rng, noise) {
  const L = 0.27 + rng() * 0.06, W = 0.15 + rng() * 0.03, D = 0.11;
  const NU = 30, NV = 12;
  const f = (x) => Math.pow(Math.max(0, 1 - Math.pow(Math.abs((2 * x) / L), 2.2)), 0.5);
  const fold = (x) => Math.cos((x / L) * 5 * Math.PI * 2);
  const A = 0.014;
  // shell: two fluted valves (hinge along local x at the bottom, scalloped rim at y ~ 0)
  const sp = [], si = [];
  for (let iu = 0; iu <= NU; iu++) {
    const x = -L / 2 + (iu / NU) * L;
    const fx = Math.max(f(x), 0.04);
    for (let iv = 0; iv <= NV; iv++) {
      const phi = -Math.PI / 2 + (iv / NV) * Math.PI;
      const rim = Math.pow(Math.abs(Math.sin(phi)), 2);
      const fl = fold(x) * rim * A;
      sp.push(x, -D * Math.cos(phi) * Math.pow(fx, 0.6) + fl, (W / 2) * Math.sin(phi) * fx * (1 + 0.25 * fl / A * 0.15));
    }
  }
  for (let iu = 0; iu < NU; iu++)
    for (let iv = 0; iv < NV; iv++) {
      const a = iu * (NV + 1) + iv, b = (iu + 1) * (NV + 1) + iv, c = a + 1, d = b + 1;
      si.push(a, b, c, c, b, d);
    }
  const shell = fin({ p: sp, i: si });
  // mantle: fleshy dome filling the gape, overhanging the rim, tucked into the shell at its edges
  const NT = 14;
  const mp = [], mi = [], mt = [];
  for (let iu = 0; iu <= NU; iu++) {
    const x = -L / 2 + (iu / NU) * L * 0.98 + L * 0.01;
    const fx = f(x);
    for (let it = 0; it <= NT; it++) {
      const t = -1 + (it / NT) * 2;
      const edge = Math.abs(t);
      const tuck = it === 0 || it === NT;
      const z = t * (W / 2) * fx * (tuck ? 0.9 : 1.07);
      const wave = (0.011 * Math.sin(x * 62 + t * 2) + 0.004 * Math.sin(x * 150)) * Math.pow(edge, 2.5); // ruffled mantle edge
      const y = tuck ? -0.03 : 0.012 + 0.016 * (1 - t * t) * fx + wave + fold(x) * A * 0.5 * edge * edge;
      mp.push(x, y, z);
      mt.push(t);
    }
  }
  for (let iu = 0; iu < NU; iu++)
    for (let it = 0; it < NT; it++) {
      const a = iu * (NT + 1) + it, b = a + 1, c = (iu + 1) * (NT + 1) + it, d = c + 1;
      mi.push(a, b, c, c, b, d);
    }
  const mantle = fin({ p: mp, i: mi });
  const yaw = (rng() - 0.5) * 0.7; // long axis roughly across the default views (reads as a clam, not a blob)
  const m = groundMatrix(T, spot.x, spot.y + 0.045, spot.z, yaw, (rng() - 0.5) * 0.2, rng() * 6.28, 1, 0.6);
  acc.addMesh(shell, m, (i, wp, wn, o) => {
    const iu = Math.floor(i / (NV + 1));
    const x = -L / 2 + (iu / NU) * L;
    const rib = 0.5 + 0.5 * fold(x);
    const v = 0.9 + 0.12 * rib + 0.05 * noise(wp.x * 60, wp.y * 60, wp.z * 60);
    o[0] = C.clamShell[0] * v; o[1] = C.clamShell[1] * v; o[2] = C.clamShell[2] * v;
    o[3] = 0.5 + 0.5 * smooth(-0.04, 0.04, wp.y - spot.y);
    o[4] = 0.1; o[5] = 0.26; o[6] = 1; o[7] = 0;
  });
  acc.addMesh(mantle, m, (i, wp, wn, o) => {
    const t = mt[i];
    const edge = Math.abs(t);
    const spots = smooth(0.12, 0.42, noise(wp.x * 38 + 1.7, wp.y * 38, wp.z * 38)); // mottled iridophore blotches
    const siphon = 1 - smooth(0.04, 0.16, edge); // dark incurrent slit along the middle
    let r = cl.mantle[0], g = cl.mantle[1], b = cl.mantle[2];
    r = lerp(r, cl.spot2[0], spots * 0.6); g = lerp(g, cl.spot2[1], spots * 0.6); b = lerp(b, cl.spot2[2], spots * 0.6);
    const lift = smooth(0.7, 1.0, edge) * 0.25; // paler, fleshy mantle edge
    r = lerp(r, 0.85, lift); g = lerp(g, 0.92, lift); b = lerp(b, 0.96, lift);
    const dk = 1 - 0.55 * siphon;
    o[0] = r * dk; o[1] = g * dk; o[2] = b * dk;
    o[3] = 0.75 + 0.25 * (1 - siphon);
    o[4] = 0.45; o[5] = 0.3; o[6] = 0.8; o[7] = 1.0;
  });
}

/** Exposed high points of the brain mounds that a crinoid can grip (clear of the other corals). */
function findBrainPerches(T, occ) {
  const brain = (T.coralGeometry || []).find((g) => g.name === "brain");
  if (!brain) return [];
  const P = brain.positions;
  // coarse 0.25 m peak grid
  const CS = 0.25, X0 = -4, Z0 = -3, NX = 32, NZ = 24;
  const top = new Float32Array(NX * NZ).fill(-1e9);
  for (let i = 0; i < P.length; i += 3) {
    const gi = Math.floor((P[i] - X0) / CS), gj = Math.floor((P[i + 2] - Z0) / CS);
    if (gi < 0 || gj < 0 || gi >= NX || gj >= NZ) continue;
    const c = gj * NX + gi;
    if (P[i + 1] > top[c]) top[c] = P[i + 1];
  }
  // high brain vertices only (candidates for perches and their normals)
  const hiIdx = [];
  for (let i = 0; i < P.length; i += 3) if (P[i + 1] > 0.32) hiIdx.push(i);
  const peaks = [];
  for (let j = 0; j < NZ; j++)
    for (let i = 0; i < NX; i++) {
      const v = top[j * NX + i];
      if (v < 0.3) continue;
      let isMax = true;
      for (let b = -2; b <= 2 && isMax; b++)
        for (let a = -2; a <= 2; a++) {
          if (!a && !b) continue;
          const ii = i + a, jj = j + b;
          if (ii < 0 || jj < 0 || ii >= NX || jj >= NZ) continue;
          if (top[jj * NX + ii] > v) {
            isMax = false;
            break;
          }
        }
      if (isMax) peaks.push({ x: X0 + (i + 0.5) * CS, z: Z0 + (j + 0.5) * CS, top: v });
    }
  peaks.sort((a, b) => b.top - a.top);
  const out = [];
  for (const pk of peaks) {
    if (out.length >= 4) break;
    // highest brain vertex near the peak with no other coral within 0.16 m (band up to +0.35 m)
    const cand = [];
    for (const i of hiIdx) {
      if (Math.abs(P[i] - pk.x) > 0.5 || Math.abs(P[i + 2] - pk.z) > 0.5 || P[i + 1] < pk.top - 0.12) continue;
      cand.push(i);
    }
    cand.sort((a, b) => P[b + 1] - P[a + 1]);
    let best = null;
    for (const i of cand) {
      if (!occ.freeOfOthers(P[i], P[i + 2], 0.16, P[i + 1] - 0.05, P[i + 1] + 0.35)) continue;
      best = { x: P[i], y: P[i + 1], z: P[i + 2] };
      break;
    }
    if (best && !out.some((o) => Math.hypot(o.x - best.x, o.z - best.z) < 0.6)) {
      // local surface normal from nearby brain vertices (centroid offset)
      let sx = 0, sy = 0, sz = 0, n = 0;
      for (const i of hiIdx) {
        const dx = P[i] - best.x, dy = P[i + 1] - best.y, dz = P[i + 2] - best.z;
        if (dx * dx + dy * dy + dz * dz > 0.0036) continue;
        sx += dx; sy += dy; sz += dz; n++;
      }
      const nrm = new THREE.Vector3(-sx, -sy + n * 0.02, -sz);
      if (nrm.lengthSq() < 1e-8 || nrm.y < 0.3 * nrm.length()) nrm.set(0, 1, 0);
      nrm.normalize().lerp(UP, 0.5).normalize();
      out.push({ x: best.x, y: best.y, z: best.z, n: nrm });
    }
  }
  return out;
}

function addCrinoid(crin, stat, pp, env, col, rng) {
  const S = new THREE.Vector3(env.surgeDir.x, 0, env.surgeDir.y).normalize(); // surge axis (fan normal)
  const F = new THREE.Vector3().crossVectors(UP, S).normalize(); // across-current axis in the fan plane
  const base = new THREE.Vector3(pp.x, pp.y, pp.z).addScaledVector(pp.n, 0.012);
  const sc = 0.9 + rng() * 0.25;
  // calyx knob + cirri (static, in the merged mesh)
  const knob = icosphere(1);
  const km = fin({ p: Float32Array.from(knob.p), i: knob.i });
  const mk = new THREE.Matrix4().compose(new THREE.Vector3(pp.x, pp.y + 0.006, pp.z), new THREE.Quaternion(), new THREE.Vector3(0.013 * sc, 0.011 * sc, 0.013 * sc));
  stat.addMesh(km, mk, (i, wp, wn, o) => {
    o[0] = col[0] * 0.55; o[1] = col[1] * 0.5; o[2] = col[2] * 0.55; o[3] = 0.7;
    o[4] = 0.2; o[5] = 0.1; o[6] = 1; o[7] = 0;
  });
  const NA = 36, NSEG = 13;
  const o8 = new Float32Array(8);
  for (let a = 0; a < NA; a++) {
    const th = -1.45 + (a / (NA - 1)) * 2.9 + (rng() - 0.5) * 0.12; // spread across the fan (rad from vertical)
    const sgn = th === 0 ? 1 : Math.sign(th);
    const L = (0.12 + rng() * 0.06) * sc;
    const cup = (rng() - 0.3) * 0.7; // shallow bowl into the current
    const bend = 0.35 + rng() * 0.45, curl = 2.2 + rng() * 1.6;
    const phase = rng();
    const pts = [], tans = [];
    let p = base.clone();
    for (let k = 0; k <= NSEG; k++) {
      const s = k / NSEG;
      const beta = th + sgn * (bend * s + curl * Math.pow(s, 4)); // outward bend + curled tip
      const t = new THREE.Vector3().addScaledVector(UP, Math.cos(beta)).addScaledVector(F, Math.sin(beta)).addScaledVector(S, cup * (0.4 + s)).normalize();
      pts.push(p.clone());
      tans.push(t);
      p = p.clone().addScaledVector(t, L / NSEG);
    }
    for (let rib = 0; rib < 2; rib++) {
      const startIdx = crin.n;
      for (let k = 0; k <= NSEG; k++) {
        const s = k / NSEG;
        const t = tans[k];
        const side = rib === 0 ? new THREE.Vector3().crossVectors(t, S).normalize() : S.clone().addScaledVector(t, -S.dot(t)).normalize();
        const nrm = new THREE.Vector3().crossVectors(t, side).normalize();
        const w = (rib === 0 ? 0.013 : 0.008) * sc * (1 - 0.5 * s) * smooth(0.0, 0.12, s + 0.04);
        for (let e = -1; e <= 1; e += 2) {
          const q = pts[k].clone().addScaledVector(side, w * e);
          o8[0] = col[0]; o8[1] = col[1]; o8[2] = col[2]; o8[3] = 0.75 + 0.25 * s;
          o8[4] = s; o8[5] = e; o8[6] = phase; o8[7] = 0.32;
          crin.vertex(q.x, q.y, q.z, nrm.x, nrm.y, nrm.z, o8);
        }
      }
      for (let k = 0; k < NSEG; k++) {
        const a0 = startIdx + k * 2;
        crin.idx.push(a0, a0 + 1, a0 + 2, a0 + 2, a0 + 1, a0 + 3);
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Far bommie silhouettes
// -----------------------------------------------------------------------------
function addBommies(acc, T, rng, noise, sites) {
  const sphere3 = icosphere(3), sphere2 = icosphere(2);
  // 16 slots every 22.5 deg starting behind-left of the default views (-150 deg), so the horizon band
  // beside and between the reef silhouettes always has a few heads in the haze
  const K = 16;
  let made = 0;
  for (let k = 0; k < K; k++) {
    const ang = ((-150 + k * 22.5 + (rng() - 0.5) * 10) * Math.PI) / 180;
    const rad = 6.0 + rng() * 2.4;
    const cx = Math.cos(ang) * rad, cz = Math.sin(ang) * rad;
    const W = 1.2 + rng() * 1.6, H = 0.65 + rng() * 1.0;
    const seed = rng() * 100;
    const sandY = T.heightAt(cx, cz);
    const yaw = rng() * Math.PI * 2;
    const parts = [];
    // 1-3 coral-head lobes: billowy (abs-noise) "cauliflower" outline, flattened base
    const nLobe = 1 + Math.floor(rng() * 3);
    const lobes = [];
    for (let l = 0; l < nLobe; l++) {
      const main = l === 0;
      const lw = main ? W : W * (0.45 + rng() * 0.3);
      const ld = lw * (0.65 + rng() * 0.45);
      const lh = main ? H : H * (0.45 + rng() * 0.35);
      const la = rng() * Math.PI * 2, lr = main ? 0 : W * (0.32 + rng() * 0.18);
      const ox = Math.cos(la) * lr, oz = Math.sin(la) * lr;
      const ls = seed + l * 31.7;
      const surf = (x, y, z) => {
        const r =
          1 +
          0.2 * fbm(noise, x * 1.3 + ls, y * 1.3, z * 1.3 - ls, 3) +
          0.16 * (Math.abs(noise(x * 3.4 - ls, y * 3.4, z * 3.4)) - 0.25) +
          0.05 * Math.abs(noise(x * 8 + ls, y * 8, z * 8));
        return [ox + x * r * lw * 0.5, (y > 0 ? y * lh : y * lh * 0.35) * r, oz + z * r * ld * 0.5];
      };
      const sph = main ? sphere3 : sphere2;
      const p = new Float32Array(sph.p.length);
      for (let i = 0; i < sph.p.length; i += 3) {
        const q = surf(sph.p[i], sph.p[i + 1], sph.p[i + 2]);
        p[i] = q[0]; p[i + 1] = q[1]; p[i + 2] = q[2];
      }
      parts.push({ p, i: sph.i });
      lobes.push(surf);
    }
    // finger-coral clumps (Porites cylindrica style): short knobbly fingers fanning up from one spot
    const nClump = rng() < 0.65 ? 1 + Math.floor(rng() * 2) : 0;
    for (let c = 0; c < nClump; c++) {
      const surf = lobes[Math.floor(rng() * lobes.length)];
      const th = rng() * Math.PI * 2, ey = 0.55 + rng() * 0.4;
      const er = Math.sqrt(1 - ey * ey);
      const b = surf(Math.cos(th) * er, ey, Math.sin(th) * er);
      const nF = 4 + Math.floor(rng() * 5);
      for (let f = 0; f < nF; f++) {
        const fa = rng() * Math.PI * 2, spread = 0.25 + rng() * 0.5;
        const len = 0.12 + rng() * 0.22, r0 = 0.03 + rng() * 0.025;
        const dx = Math.cos(fa) * spread, dz = Math.sin(fa) * spread;
        const bx = b[0] + Math.cos(fa) * r0 * 1.5, bz = b[2] + Math.sin(fa) * r0 * 1.5;
        const P = [0, 0.5, 1].map((t) => [bx + dx * len * t * (0.5 + 0.5 * t), b[1] - 0.04 + len * t, bz + dz * len * t * (0.5 + 0.5 * t)]);
        parts.push(tube(P, [r0, r0 * 0.92, r0 * 0.8], 5, { capStart: false, capEnd: 0.8, radial: knobby(noise, seed + f * 3.1 + c, 0.8) }));
      }
    }
    // an irregular, slightly tilted table-coral plate on some
    if (rng() < 0.22) {
      const b = lobes[0](0.0, 1.0, 0.0); // crown of the main head
      const R = 0.35 + rng() * 0.3, st = 0.06 + rng() * 0.06;
      const ph = rng() * 6.28;
      // thick stalk rooted well inside the head, so the plate never floats
      const stalk = tube([[b[0], b[1] - 0.3, b[2]], [b[0], b[1] + st, b[2]]], [0.14, 0.1], 6, { capStart: false, capEnd: false });
      const disc = lathe(
        [[0.12, -0.13], [R * 0.45, -0.05], [R, -0.012], [R * 1.03, 0.0], [R, 0.016], [R * 0.5, 0.034], [0.0, 0.04]], // conical underside: reads attached
        24,
        (kk, j, a, r) => r * (1 + 0.22 * noise(Math.cos(a) * 1.6 + seed, Math.sin(a) * 1.6, 3.3) + 0.1 * Math.cos(a * 2 + ph))
      );
      const tq = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(Math.cos(ph), 0, Math.sin(ph)), 0.12 + rng() * 0.15);
      const tv = new THREE.Vector3();
      const dp = [];
      for (let i = 0; i < disc.p.length; i += 3) {
        tv.set(disc.p[i], disc.p[i + 1] + 0.03 * Math.pow(Math.hypot(disc.p[i], disc.p[i + 2]) / R, 2), disc.p[i + 2]).applyQuaternion(tq);
        dp.push(tv.x + b[0], tv.y + b[1] + st, tv.z + b[2]);
      }
      parts.push(stalk, { p: dp, i: disc.i });
    }
    const mesh = fin(merge(...parts));
    const m = new THREE.Matrix4().compose(new THREE.Vector3(cx, sandY - 0.18 * H, cz), new THREE.Quaternion().setFromAxisAngle(UP, yaw), ONE);
    const tint = pick(rng, [
      [1.0, 1.0, 1.0],
      [1.1, 0.93, 0.96],
      [0.93, 1.06, 1.04],
      [0.98, 0.94, 1.1],
    ]);
    const br = 0.88 + rng() * 0.24;
    acc.addMesh(mesh, m, (i, wp, wn, o) => {
      const h = wp.y - sandY;
      const cav = 0.75 + 0.25 * smooth(-0.3, 0.3, noise(wp.x * 1.7 + seed, wp.y * 1.7, wp.z * 1.7));
      const ao = (0.4 + 0.6 * smooth(-0.05, 0.75 * H, h)) * cav;
      const v = br * (1 + 0.08 * noise(wp.x * 6, wp.y * 6, wp.z * 6));
      o[0] = C.bommie[0] * tint[0] * v; o[1] = C.bommie[1] * tint[1] * v; o[2] = C.bommie[2] * tint[2] * v;
      o[3] = ao;
      o[4] = 0.0; o[5] = 0.0; o[6] = 0.15; o[7] = 0.0;
    });
    sites.bommies.push({ x: cx, z: cz, r: W * 0.5, h: H * 0.8 });
    made++;
  }
  return made;
}
