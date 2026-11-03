// -----------------------------------------------------------------------------
// reef-fish.js — procedural Indo-Pacific reef fish (undersea expert P4, art director item 6)
//
// Replaces the 45 orange cones. Seven species, each a JS-lofted, laterally compressed mesh
// (14 rings x 10 segments + forked/lunate/rounded caudal, dorsal, anal and hinged pectoral fins, 262 verts).
// All species share one topology, so every fish (266) is drawn by ONE instanced draw with ONE pipeline:
// per-species vertices are pulled from a read-only storage buffer by (species, vertex_index).
//   anthias (peach females, lilac males)  hovering cloud above the bouquet, facing into the current, darts
//   blue-green chromis                     low aggregations over the coral heads, tuck into the coral
//   humbug damsels                         within 0.3 m of their coral head, dart in/out of the branches
//   lemon butterflyfish (2 pairs)          slow loop around the reef, partner 0.25 m behind, pecking stops
//   moon wrasse                            weave 0.1-0.3 m over the coral, rowing pectorals
//   clownfish                              live in the anemones published by the flora module
//   fusiliers                              tight school passing 6-10 m out in the haze every ~40 s
//
// Motion
//  - Hoverers (anthias, chromis, damsels, clownfish) are fully analytic in the vertex shader and
//    deterministic in t: a home anchor placed in free water (3D clearance field baked once from the coral
//    vertices), slow 3D wander, slot-scheduled darts (hash per slot: turn, burst, glide, turn back),
//    sloshing with the shared surge field (env.surge) and heading into the current (env.surgeVel).
//    Excursions are bounded by the anchor's clearance, so fish never swim through the coral; fish with a
//    shelter (chromis, damsels, clownfish) tuck into it when the camera comes within ~1.2 m.
//  - Butterflyfish and wrasse follow arc-length paths built once around/over the reef; the CPU evaluates
//    7 fish per frame analytically from t (no state) and uploads a 7-instance range of the instance buffer.
//  - The fusilier school moves as a unit (school frame uniforms, one transform per frame).
//  - Swim animation in the vertex shader: travelling body wave x += L*A*(0.08+0.92u^2)*sin(2pi(u/0.95-ft))
//    with the tail-beat phase integrated analytically across darts (8-12 Hz bursts, 2-4 Hz hover), a C-curl
//    while turning, normals bent with the wave, pectorals flapped about their hinge (rowing for the wrasse).
// Shading: env.litProp + env.fogTerms/env.output (caustics, baked sun shadow, fog, grade) with a procedural
// pattern atlas (countershading, bars/stripes from s, eye, fin membranes that show the water behind them),
// a mirror term reflecting the water gradient on silvery flanks with a subtle thin-film tint, wet gloss.
// Fish within ~0.8 m of the camera shrink away so nothing big pops in front of the lens.
// Dev URL flags: fishDebug=1 (logs), fishOnly=anthias,chromis,... , fishLineup=1 (one of each species side-on
// in front of the hero camera), fishDump=1 (logs the generated WGSL).
// -----------------------------------------------------------------------------
import * as THREE from "three/webgpu";
import {
  Fn,
  uniform,
  attribute,
  varying,
  texture,
  vec2,
  vec3,
  vec4,
  float,
  uint,
  floor,
  fract,
  sin,
  cos,
  min,
  max,
  mix,
  clamp,
  smoothstep,
  step,
  dot,
  cross,
  normalize,
  length,
  reflect,
  storage,
  vertexIndex,
  cameraPosition,
  cameraViewMatrix,
  cameraProjectionMatrix,
} from "three/tsl";

const TAU = Math.PI * 2;
const XC = 0.55; // local x = s - XC (fish centre slightly behind the head)
const ATLAS_W = 128;
const BLOCK_ROWS = 64;

// ---------------------------------------------------------------------------------------------
// Species table. Profiles: [s, dorsal outline y, ventral outline y, half width] in body lengths
// (s = 0 tail tip .. 1 snout). Colours are display-referred albedo like the coral attributes.
// ---------------------------------------------------------------------------------------------
const SPECIES = {
  anthias: {
    block: 0,
    variants: 2,
    mode: 0,
    prof: [[0.2, 0.04, -0.04, 0.017], [0.3, 0.074, -0.068, 0.028], [0.45, 0.11, -0.1, 0.042], [0.6, 0.126, -0.114, 0.05], [0.74, 0.118, -0.104, 0.049], [0.85, 0.09, -0.078, 0.04], [0.93, 0.052, -0.046, 0.028], [0.985, 0.016, -0.018, 0.01], [1.0, 0.004, -0.008, 0.004]],
    tail: { type: "lunate", notch: 0.085, span: 0.13 },
    dorsal: { s0: 0.8, s1: 0.27, h: [[0.27, 0.0], [0.3, 0.07], [0.45, 0.064], [0.62, 0.05], [0.74, 0.066], [0.78, 0.08], [0.8, 0.0]], rake: 0.35 },
    anal: { s0: 0.47, s1: 0.28, h: [[0.28, 0.0], [0.31, 0.062], [0.42, 0.05], [0.47, 0.0]], rake: 0.4 },
    pec: { s: 0.7, y: 0.36, len: 0.15, wid: 0.028 },
    swim: { fH: 2.4, aH: 0.03, fD: 10.0, aD: 0.1, fP: 5.0, pecAmp: 0.3, pecRest: 0.5, dartDur: 0.32, faceGain: 2.6 },
    shade: { mirror: 0.25, finTrans: 0.5, sheen: 0.12 },
  },
  chromis: {
    block: 2,
    variants: 1,
    mode: 0,
    prof: [[0.22, 0.04, -0.04, 0.018], [0.32, 0.088, -0.083, 0.031], [0.48, 0.142, -0.128, 0.047], [0.62, 0.152, -0.138, 0.051], [0.76, 0.128, -0.114, 0.048], [0.88, 0.085, -0.07, 0.036], [0.95, 0.042, -0.036, 0.022], [0.988, 0.012, -0.012, 0.008], [1.0, 0.004, -0.006, 0.004]],
    tail: { type: "fork", notch: 0.13, span: 0.15 },
    dorsal: { s0: 0.74, s1: 0.31, h: [[0.31, 0.0], [0.34, 0.085], [0.42, 0.07], [0.6, 0.06], [0.72, 0.07], [0.74, 0.0]], rake: 0.4 },
    anal: { s0: 0.5, s1: 0.3, h: [[0.3, 0.0], [0.33, 0.075], [0.44, 0.06], [0.5, 0.0]], rake: 0.4 },
    pec: { s: 0.7, y: 0.4, len: 0.14, wid: 0.03 },
    swim: { fH: 2.8, aH: 0.035, fD: 10.0, aD: 0.1, fP: 4.5, pecAmp: 0.32, pecRest: 0.45, dartDur: 0.3, faceGain: 2.6 },
    shade: { mirror: 0.55, finTrans: 0.55, sheen: 0.15 },
  },
  humbug: {
    block: 3,
    variants: 1,
    mode: 0,
    prof: [[0.18, 0.05, -0.05, 0.02], [0.28, 0.118, -0.108, 0.035], [0.42, 0.195, -0.176, 0.051], [0.58, 0.22, -0.196, 0.057], [0.72, 0.205, -0.176, 0.054], [0.84, 0.155, -0.118, 0.045], [0.93, 0.083, -0.06, 0.03], [0.985, 0.025, -0.02, 0.012], [1.0, 0.008, -0.008, 0.004]],
    tail: { type: "trunc", notch: 0.02, span: 0.13 },
    dorsal: { s0: 0.8, s1: 0.29, h: [[0.29, 0.0], [0.32, 0.075], [0.45, 0.08], [0.6, 0.08], [0.74, 0.1], [0.8, 0.0]], rake: 0.3 },
    anal: { s0: 0.48, s1: 0.28, h: [[0.28, 0.0], [0.31, 0.08], [0.42, 0.075], [0.48, 0.0]], rake: 0.35 },
    pec: { s: 0.7, y: 0.4, len: 0.13, wid: 0.032 },
    swim: { fH: 3.0, aH: 0.04, fD: 9.0, aD: 0.09, fP: 4.0, pecAmp: 0.35, pecRest: 0.45, dartDur: 0.28, faceGain: 2.0 },
    shade: { mirror: 0.08, finTrans: 0.5, sheen: 0.1 },
  },
  butterfly: {
    block: 4,
    variants: 1,
    mode: 1,
    prof: [[0.16, 0.05, -0.05, 0.016], [0.26, 0.165, -0.148, 0.028], [0.4, 0.245, -0.218, 0.04], [0.55, 0.265, -0.232, 0.044], [0.7, 0.23, -0.188, 0.04], [0.82, 0.148, -0.108, 0.032], [0.9, 0.062, -0.05, 0.02], [0.955, 0.024, -0.02, 0.012], [1.0, 0.007, -0.006, 0.005]],
    tail: { type: "round", notch: 0.035, span: 0.11 },
    dorsal: { s0: 0.76, s1: 0.22, h: [[0.22, 0.0], [0.25, 0.07], [0.35, 0.085], [0.5, 0.06], [0.66, 0.05], [0.76, 0.0]], rake: 0.25 },
    anal: { s0: 0.5, s1: 0.22, h: [[0.22, 0.0], [0.25, 0.065], [0.36, 0.075], [0.5, 0.0]], rake: 0.25 },
    pec: { s: 0.72, y: 0.42, len: 0.12, wid: 0.03 },
    swim: { fH: 1.5, aH: 0.02, fD: 6.0, aD: 0.06, fP: 2.6, pecAmp: 0.38, pecRest: 0.45, dartDur: 0.3, faceGain: 0.0 },
    shade: { mirror: 0.06, finTrans: 0.45, sheen: 0.14 },
  },
  wrasse: {
    block: 5,
    variants: 1,
    mode: 1,
    prof: [[0.18, 0.038, -0.038, 0.02], [0.3, 0.066, -0.062, 0.033], [0.45, 0.09, -0.082, 0.044], [0.6, 0.1, -0.088, 0.049], [0.75, 0.092, -0.078, 0.046], [0.87, 0.064, -0.054, 0.036], [0.95, 0.03, -0.028, 0.02], [0.988, 0.01, -0.012, 0.008], [1.0, 0.004, -0.006, 0.004]],
    tail: { type: "lunate", notch: 0.09, span: 0.13 },
    dorsal: { s0: 0.8, s1: 0.27, h: [[0.27, 0.0], [0.3, 0.04], [0.5, 0.035], [0.78, 0.035], [0.8, 0.0]], rake: 0.3 },
    anal: { s0: 0.5, s1: 0.27, h: [[0.27, 0.0], [0.3, 0.035], [0.48, 0.03], [0.5, 0.0]], rake: 0.3 },
    pec: { s: 0.72, y: 0.42, len: 0.17, wid: 0.034 },
    swim: { fH: 1.5, aH: 0.02, fD: 6.0, aD: 0.06, fP: 2.2, pecAmp: 0.6, pecRest: 0.75, dartDur: 0.3, faceGain: 0.0 },
    shade: { mirror: 0.12, finTrans: 0.45, sheen: 0.12 },
  },
  clown: {
    block: 6,
    variants: 1,
    mode: 0,
    prof: [[0.2, 0.05, -0.05, 0.022], [0.3, 0.098, -0.093, 0.038], [0.45, 0.147, -0.137, 0.052], [0.6, 0.162, -0.147, 0.058], [0.75, 0.148, -0.128, 0.055], [0.86, 0.108, -0.088, 0.045], [0.94, 0.06, -0.05, 0.03], [0.988, 0.02, -0.02, 0.012], [1.0, 0.008, -0.01, 0.006]],
    tail: { type: "round", notch: 0.03, span: 0.11 },
    dorsal: { s0: 0.78, s1: 0.3, h: [[0.3, 0.0], [0.33, 0.07], [0.44, 0.075], [0.53, 0.045], [0.62, 0.06], [0.76, 0.05], [0.78, 0.0]], rake: 0.25 },
    anal: { s0: 0.47, s1: 0.29, h: [[0.29, 0.0], [0.32, 0.065], [0.42, 0.06], [0.47, 0.0]], rake: 0.3 },
    pec: { s: 0.71, y: 0.4, len: 0.13, wid: 0.034 },
    swim: { fH: 3.6, aH: 0.05, fD: 7.0, aD: 0.08, fP: 4.2, pecAmp: 0.42, pecRest: 0.45, dartDur: 0.3, faceGain: 2.0 },
    shade: { mirror: 0.06, finTrans: 0.35, sheen: 0.12 },
  },
  fusilier: {
    block: 7,
    variants: 1,
    mode: 2,
    prof: [[0.22, 0.03, -0.03, 0.016], [0.34, 0.064, -0.06, 0.03], [0.5, 0.098, -0.088, 0.043], [0.64, 0.108, -0.096, 0.047], [0.78, 0.094, -0.084, 0.044], [0.89, 0.06, -0.055, 0.032], [0.96, 0.028, -0.026, 0.018], [0.99, 0.01, -0.01, 0.007], [1.0, 0.004, -0.006, 0.004]],
    tail: { type: "fork", notch: 0.14, span: 0.14 },
    dorsal: { s0: 0.75, s1: 0.38, h: [[0.38, 0.0], [0.41, 0.045], [0.6, 0.05], [0.72, 0.055], [0.75, 0.0]], rake: 0.35 },
    anal: { s0: 0.46, s1: 0.32, h: [[0.32, 0.0], [0.35, 0.04], [0.44, 0.035], [0.46, 0.0]], rake: 0.35 },
    pec: { s: 0.72, y: 0.4, len: 0.13, wid: 0.024 },
    swim: { fH: 4.2, aH: 0.065, fD: 4.2, aD: 0.065, fP: 3.0, pecAmp: 0.15, pecRest: 0.3, dartDur: 0.3, faceGain: 0.0 },
    shade: { mirror: 0.85, finTrans: 0.45, sheen: 0.1 },
  },
};
const ATLAS_BLOCKS = 8;

// ---------------------------------------------------------------------------------------------
// small math helpers (CPU)
// ---------------------------------------------------------------------------------------------
const sstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
const mixc = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const hex = (h) => [((h >> 16) & 255) / 255, ((h >> 8) & 255) / 255, (h & 255) / 255];

/** Monotone cubic (Fritsch-Carlson) interpolant through (xs, ys). */
function makeMono(xs, ys) {
  const n = xs.length;
  const d = [];
  const m = new Array(n).fill(0);
  for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) {
      m[i] = 0;
      m[i + 1] = 0;
      continue;
    }
    const a = m[i] / d[i], b = m[i + 1] / d[i], s = a * a + b * b;
    if (s > 9) {
      const tt = 3 / Math.sqrt(s);
      m[i] = tt * a * d[i];
      m[i + 1] = tt * b * d[i];
    }
  }
  return (x) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (x > xs[i + 1]) i++;
    const h = xs[i + 1] - xs[i], t = (x - xs[i]) / h, t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1];
  };
}
/** Piecewise-linear interpolant through [[x, y], ...]. */
function makeLin(pts) {
  return (x) => {
    if (x <= pts[0][0]) return pts[0][1];
    for (let i = 0; i < pts.length - 1; i++) {
      if (x <= pts[i + 1][0]) {
        const t = (x - pts[i][0]) / (pts[i + 1][0] - pts[i][0]);
        return pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t;
      }
    }
    return pts[pts.length - 1][1];
  };
}
function profileFns(sp) {
  const S = sp.prof.map((p) => p[0]);
  return {
    sPed: S[0],
    top: makeMono(S, sp.prof.map((p) => p[1])),
    bot: makeMono(S, sp.prof.map((p) => p[2])),
    hw: makeMono(S, sp.prof.map((p) => p[3])),
  };
}

// ---------------------------------------------------------------------------------------------
// Geometry: lofted body + fins. Attributes: position, normal (local fish frame: +x forward/snout,
// +y dorsal, +z lateral; total length 1), aFish = (s, h, finId, side), aHinge = (hinge.xyz, texU).
// finId: 0 body (+ caudal fin, painted in the body map), 1 dorsal, 2 anal, 3 pectoral.
// ---------------------------------------------------------------------------------------------
function buildFishGeometry(sp) {
  const P = profileFns(sp);
  const pos = [], fishA = [], hingeA = [], idx = [];
  const vert = (x, y, z, s, h, fin, side, hx = 0, hy = 0, hz = 0, tu = s) => {
    pos.push(x, y, z);
    fishA.push(s, h, fin, side);
    hingeA.push(hx, hy, hz, tu);
    return pos.length / 3 - 1;
  };

  // ---- body: NR rings x NS segments, snout tip, peduncle cap ----
  const NR = 14, NS = 10, sEnd = 0.985;
  const ring0 = pos.length / 3;
  for (let i = 0; i < NR; i++) {
    const q = i / (NR - 1);
    const s = P.sPed + (sEnd - P.sPed) * Math.pow(q, 0.8);
    const T = P.top(s), B = P.bot(s), W = P.hw(s);
    for (let j = 0; j < NS; j++) {
      const th = Math.PI / 2 + (TAU * j) / NS;
      const c = Math.cos(th), sn = Math.sin(th);
      const y = sn >= 0 ? T * sn : -B * sn;
      vert(s - XC, y, W * c, s, sn, 0, Math.sign(c));
    }
  }
  for (let i = 0; i < NR - 1; i++)
    for (let j = 0; j < NS; j++) {
      const a = ring0 + i * NS + j, b = ring0 + (i + 1) * NS + j;
      const c = ring0 + i * NS + ((j + 1) % NS), d = ring0 + (i + 1) * NS + ((j + 1) % NS);
      idx.push(a, b, c, b, d, c);
    }
  const tip = vert(1 - XC, (P.top(1) + P.bot(1)) * 0.5, 0, 1, 0, 0, 0);
  const last = ring0 + (NR - 1) * NS;
  for (let j = 0; j < NS; j++) idx.push(last + j, tip, last + ((j + 1) % NS));
  const cap = vert(P.sPed - 0.006 - XC, (P.top(P.sPed) + P.bot(P.sPed)) * 0.5, 0, P.sPed, 0, 0, 0);
  for (let j = 0; j < NS; j++) idx.push(cap, ring0 + j, ring0 + ((j + 1) % NS));

  // ---- double-sided sheet helper: grid G[a][b] of {p:[x,y,z], s, h, fin, hinge, tu, side} ----
  const sheet = (G) => {
    const na = G.length, nb = G[0].length;
    // reference normal of the "+" copy from the grid diagonals
    const p00 = G[0][0].p, p11 = G[na - 1][nb - 1].p, p10 = G[na - 1][0].p, p01 = G[0][nb - 1].p;
    const e1 = [p11[0] - p00[0], p11[1] - p00[1], p11[2] - p00[2]];
    const e2 = [p01[0] - p10[0], p01[1] - p10[1], p01[2] - p10[2]];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    for (const sideSign of [1, -1]) {
      const base = pos.length / 3;
      for (let a = 0; a < na; a++)
        for (let b = 0; b < nb; b++) {
          const g = G[a][b];
          const hv = g.hinge || [0, 0, 0];
          vert(g.p[0], g.p[1], g.p[2], g.s, g.h, g.fin, g.side ?? 0, hv[0], hv[1], hv[2], g.tu ?? g.s);
        }
      for (let a = 0; a < na - 1; a++)
        for (let b = 0; b < nb - 1; b++) {
          const i00 = base + a * nb + b, i10 = base + (a + 1) * nb + b, i01 = base + a * nb + b + 1, i11 = base + (a + 1) * nb + b + 1;
          // geometric normal of (i00, i10, i01) is (+a dir) x (+b dir); flip winding to match the side
          const pa = G[a + 1][b].p, pb = G[a][b + 1].p, p0 = G[a][b].p;
          const u1 = [pa[0] - p0[0], pa[1] - p0[1], pa[2] - p0[2]], u2 = [pb[0] - p0[0], pb[1] - p0[1], pb[2] - p0[2]];
          const gn = [u1[1] * u2[2] - u1[2] * u2[1], u1[2] * u2[0] - u1[0] * u2[2], u1[0] * u2[1] - u1[1] * u2[0]];
          const agree = gn[0] * n[0] + gn[1] * n[1] + gn[2] * n[2] >= 0;
          if (agree === sideSign > 0) idx.push(i00, i10, i01, i10, i11, i01);
          else idx.push(i00, i01, i10, i10, i01, i11);
        }
    }
  };

  // ---- caudal fin (body map, finId 0, alpha marks it as fin in the atlas) ----
  {
    const t = sp.tail;
    const pedT = P.top(P.sPed), pedB = P.bot(P.sPed);
    const NC = 4, NQ = 5;
    const ste = (q) => {
      const aq = Math.abs(q);
      if (t.type === "fork") return t.notch * Math.pow(1 - aq, 1.25);
      if (t.type === "lunate") return t.notch * Math.pow(1 - aq * aq, 1.2);
      if (t.type === "round") return t.notch * aq * aq;
      return t.notch * (1 - aq * aq) * 0.5 + 0.004; // truncate / emarginate
    };
    const G = [];
    for (let c = 0; c < NC; c++) {
      const a = c / (NC - 1);
      const col = [];
      for (let r = 0; r < NQ; r++) {
        const q = -1 + (2 * r) / (NQ - 1);
        const s = P.sPed + 0.012 + (ste(q) - P.sPed - 0.012) * a;
        const yB = q >= 0 ? q * pedT : -q * pedB;
        const lobe = t.type === "round" || t.type === "trunc" ? 1 - 0.18 * q * q * a : 1;
        const yE = q * t.span * lobe;
        const y = yB + (yE - yB) * Math.pow(a, 0.75);
        col.push({ p: [s - XC, y, 0], s, h: q, fin: 0 });
      }
      G.push(col);
    }
    sheet(G);
  }

  // ---- dorsal / anal fins ----
  const fin = (F, finId, dorsal) => {
    const hgt = makeLin(F.h);
    const NC = 7;
    const G = [];
    for (let c = 0; c < NC; c++) {
      const s = F.s1 + ((F.s0 - F.s1) * c) / (NC - 1);
      const hh = hgt(s);
      const yb = dorsal ? P.top(s) - 0.006 : P.bot(s) + 0.006;
      const yt = dorsal ? P.top(s) + hh : P.bot(s) - hh;
      const st = s - F.rake * hh;
      G.push([
        { p: [s - XC, yb, 0], s, h: 0, fin: finId },
        { p: [st - XC, yt, 0], s: st, h: 1, fin: finId },
      ]);
    }
    sheet(G);
  };
  fin(sp.dorsal, 1, true);
  fin(sp.anal, 2, false);

  // ---- pectoral fins (hinged paddles on both flanks) ----
  {
    const pc = sp.pec;
    const yP = P.bot(pc.s) + (P.top(pc.s) - P.bot(pc.s)) * pc.y;
    for (const side of [1, -1]) {
      const hz = side * (P.hw(pc.s) * 0.92 + 0.002);
      const hinge = [pc.s - XC, yP, hz];
      const dir = [-0.96, -0.24, side * 0.14];
      const across = [0.24, 0.97, 0];
      const G = [];
      for (const a of [0, 0.55, 1]) {
        const w = pc.wid * (a < 1 ? 1 - 0.1 * a : 0.55);
        const cx = hinge[0] + dir[0] * pc.len * a, cy = hinge[1] + dir[1] * pc.len * a, cz = hinge[2] + dir[2] * pc.len * a;
        const col = [];
        for (const b of [-1, 1]) {
          const x = cx + across[0] * w * b, y = cy + across[1] * w * b;
          col.push({ p: [x, y, cz], s: x + XC, h: (b + 1) * 0.5, fin: 3, side, hinge, tu: a });
        }
        G.push(col);
      }
      sheet(G);
    }
  }

  // ---- normals: area-weighted face normals (sheet sides have separate vertices) ----
  const nrm = new Float32Array(pos.length);
  for (let i = 0; i < idx.length; i += 3) {
    const a = idx[i] * 3, b = idx[i + 1] * 3, c = idx[i + 2] * 3;
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const k of [a, b, c]) {
      nrm[k] += nx;
      nrm[k + 1] += ny;
      nrm[k + 2] += nz;
    }
  }
  for (let i = 0; i < nrm.length; i += 3) {
    const l = Math.hypot(nrm[i], nrm[i + 1], nrm[i + 2]) || 1;
    nrm[i] /= l;
    nrm[i + 1] /= l;
    nrm[i + 2] /= l;
  }
  return { pos, nrm, fishA, hingeA, idx, nv: pos.length / 3 };
}

/**
 * Every species shares one topology (same rings, fin grids and winding), so all fish are drawn by ONE
 * instanced draw: per-species vertices are pulled from a read-only storage buffer by (species, vertex_index).
 * Layout per vertex (4 x vec4): (pos.xyz, s), (normal.xyz, h), (finId, side, texU, 0), (hinge.xyz, 0);
 * followed by 4 vec4 of per-species constants (behaviour / swim / pectoral / shading).
 */
function buildSharedGeometry() {
  const names = Object.keys(SPECIES);
  const geos = names.map((n) => buildFishGeometry(SPECIES[n]));
  const NV = geos[0].nv;
  const idx0 = geos[0].idx;
  for (const g of geos) {
    if (g.nv !== NV || g.idx.length !== idx0.length) throw new Error("reef-fish: species topologies differ");
    for (let i = 0; i < idx0.length; i++) if (g.idx[i] !== idx0[i]) throw new Error("reef-fish: species windings differ");
  }
  const specBase = names.length * NV * 4;
  const data = new Float32Array((specBase + names.length * 4) * 4);
  geos.forEach((g, si) => {
    for (let v = 0; v < NV; v++) {
      const o = (si * NV + v) * 16;
      data.set([g.pos[v * 3], g.pos[v * 3 + 1], g.pos[v * 3 + 2], g.fishA[v * 4]], o);
      data.set([g.nrm[v * 3], g.nrm[v * 3 + 1], g.nrm[v * 3 + 2], g.fishA[v * 4 + 1]], o + 4);
      data.set([g.fishA[v * 4 + 2], g.fishA[v * 4 + 3], g.hingeA[v * 4 + 3], 0], o + 8);
      data.set([g.hingeA[v * 4], g.hingeA[v * 4 + 1], g.hingeA[v * 4 + 2], 0], o + 12);
    }
  });
  names.forEach((n, si) => {
    const sp = SPECIES[n], s = sp.swim, sh = sp.shade;
    const o = (specBase + si * 4) * 4;
    data.set([sp.mode, sp.block, s.faceGain, 0], o);
    data.set([s.fH, s.aH, s.fD, s.aD], o + 4);
    data.set([s.fP, s.pecAmp, s.pecRest, s.dartDur], o + 8);
    data.set([sh.mirror, sh.finTrans, sh.sheen, 0], o + 12);
  });
  const vbuf = storage(new THREE.StorageBufferAttribute(data, 4), "vec4", data.length / 4);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(geos[0].pos, 3)); // unused by the shader
  geo.setIndex(idx0);
  return { vbuf, NV, specBase, geo, speciesIndex: Object.fromEntries(names.map((n, i) => [n, i])) };
}

// ---------------------------------------------------------------------------------------------
// Pattern atlas: one 128 x 64 block per species variant.
//   rows 0-31 body map (h = -1 belly .. +1 back; s < sPed is the caudal fin), rows 32-47 dorsal,
//   48-55 anal, 56-63 pectoral (8-row aligned so the first mips never mix regions).
//   alpha >= 0.5: body, silver amount = (a - 0.5) * 2;  alpha < 0.5: fin, opacity = a * 2.
// ---------------------------------------------------------------------------------------------
const C = {
  peach: hex(0xf6a48b),
  lilac: hex(0xb48cf0),
  aqua: hex(0x8fe0d8),
  lemon: hex(0xf3e08a),
  clown: hex(0xff8a3d),
  white: [0.95, 0.95, 0.97],
  indigo: [0.17, 0.15, 0.33],
  pupil: [0.05, 0.04, 0.08],
};
const bodyA = (silver) => 0.5 + 0.5 * Math.min(1, Math.max(0, silver)) * 0.996;
const finA = (op) => 0.498 * Math.min(1, Math.max(0, op));
const countershade = (back, flank, belly, h) =>
  h >= 0 ? mixc(flank, back, sstep(0.1, 0.95, h)) : mixc(flank, belly, sstep(-0.05, -0.8, h));

/** eye disc in body-map space: returns [pupilMask, irisMask] */
function eyeMask(P, s, h, se, he, re) {
  const T = Math.max(1e-3, he >= 0 ? P.top(se) : -P.bot(se));
  const d = Math.hypot(s - se, (h - he) * T);
  return [1 - sstep(re * 0.5, re * 0.62, d), 1 - sstep(re * 0.9, re * 1.04, d)];
}
const rayLines = (h, n) => Math.pow(Math.abs(Math.cos(h * n * Math.PI)), 10);
function segDist(px, py, ax, ay, bx, by) {
  const vx = bx - ax, vy = by - ay;
  const t = Math.min(1, Math.max(0, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)));
  return Math.hypot(px - ax - vx * t, py - ay - vy * t);
}

const PAINTERS = {
  // Pseudanthias squamipinnis female: peach, lilac eye-streak, yellow-orange fin margins
  anthias0(P, reg, s, v) {
    if (reg === "body") {
      if (s < P.sPed) {
        const lobe = sstep(0.55, 0.95, Math.abs(v));
        let col = mixc([0.99, 0.74, 0.62], [0.97, 0.55, 0.42], lobe * 0.7);
        col = mixc(col, [1.0, 0.86, 0.6], sstep(0.12, 0.2, s) * 0.3);
        return [...col, finA(0.62 + 0.25 * lobe + 0.1 * rayLines(v, 5))];
      }
      let col = countershade([0.92, 0.58, 0.54], C.peach, [1.0, 0.86, 0.8], v);
      const T = P.top(0.82);
      const st = segDist(s, v * T, 0.905, 0.06 * T, 0.745, -0.36 * T);
      col = mixc(col, [0.78, 0.6, 0.97], (1 - sstep(0.008, 0.016, st)) * 0.9);
      const [pu, ir] = eyeMask(P, s, v, 0.905, 0.27, 0.03);
      col = mixc(col, [0.98, 0.76, 0.42], ir);
      col = mixc(col, C.pupil, pu);
      return [...col, bodyA(0.18 + 0.2 * sstep(0.2, -0.8, v))];
    }
    if (reg === "dorsal") return [...mixc([0.98, 0.66, 0.52], [1.0, 0.82, 0.48], sstep(0.4, 1.0, v)), finA(0.72 - 0.2 * v + 0.12 * sstep(0.8, 1, v))];
    if (reg === "anal") return [...mixc([0.98, 0.7, 0.6], [0.86, 0.66, 0.96], sstep(0.3, 1.0, v) * 0.6), finA(0.68 - 0.15 * v)];
    return [...mixc([0.99, 0.78, 0.7], [1.0, 0.84, 0.66], s), finA(0.82 - 0.1 * s)];
  },
  // male: lilac-magenta, rose pectoral spot
  anthias1(P, reg, s, v) {
    if (reg === "body") {
      if (s < P.sPed) {
        const lobe = sstep(0.55, 0.95, Math.abs(v));
        return [...mixc([0.82, 0.7, 0.97], [0.62, 0.42, 0.9], lobe), finA(0.62 + 0.25 * lobe)];
      }
      let col = countershade([0.6, 0.42, 0.86], C.lilac, [0.88, 0.76, 0.99], v);
      const T = P.top(0.82);
      const st = segDist(s, v * T, 0.9, 0.04 * T, 0.76, -0.3 * T);
      col = mixc(col, [0.99, 0.66, 0.6], (1 - sstep(0.007, 0.014, st)) * 0.75);
      const [pu, ir] = eyeMask(P, s, v, 0.905, 0.27, 0.03);
      col = mixc(col, [0.95, 0.72, 0.5], ir);
      col = mixc(col, C.pupil, pu);
      return [...col, bodyA(0.22)];
    }
    if (reg === "dorsal") return [...mixc([0.72, 0.52, 0.94], [0.99, 0.6, 0.66], sstep(0.55, 1.0, v)), finA(0.75 - 0.15 * v)];
    if (reg === "anal") return [...mixc([0.75, 0.58, 0.95], [0.98, 0.72, 0.6], sstep(0.5, 1.0, v)), finA(0.7)];
    const spot = 1 - sstep(0.12, 0.2, Math.hypot(s - 0.35, (v - 0.5) * 0.4));
    return [...mixc([0.8, 0.68, 0.97], [0.95, 0.42, 0.55], spot), finA(0.84 + 0.12 * spot)];
  },
  // Chromis viridis: iridescent aqua, dark snout line, clear forked tail
  chromis0(P, reg, s, v) {
    if (reg === "body") {
      if (s < P.sPed) {
        const lobe = sstep(0.6, 1.0, Math.abs(v));
        return [...mixc([0.72, 0.94, 0.9], [0.5, 0.8, 0.84], lobe), finA(0.46 + 0.2 * lobe)];
      }
      let col = countershade([0.38, 0.7, 0.8], C.aqua, [0.86, 0.97, 0.95], v);
      const T = P.top(0.95);
      const ln = segDist(s, v * T, 0.995, 0.0, 0.915, 0.2 * T);
      col = mixc(col, [0.22, 0.38, 0.5], (1 - sstep(0.004, 0.008, ln)) * 0.7);
      const [pu, ir] = eyeMask(P, s, v, 0.9, 0.24, 0.028);
      col = mixc(col, [0.7, 0.86, 0.9], ir);
      col = mixc(col, C.pupil, pu);
      return [...col, bodyA(0.4 + 0.45 * sstep(0.5, -0.5, v))];
    }
    if (reg === "dorsal") return [...mixc([0.55, 0.84, 0.86], [0.75, 0.95, 0.94], v), finA(0.55 - 0.15 * v)];
    if (reg === "anal") return [...[0.62, 0.88, 0.88], finA(0.52 - 0.15 * v)];
    return [...[0.68, 0.9, 0.88], finA(0.8)];
  },
  // Dascyllus aruanus: white with three slanted indigo bars, dark anal fin, clear tail
  humbug0(P, reg, s, v) {
    const bars = (ss, vv) => {
      const b1 = 1 - sstep(0.035, 0.045, Math.abs(ss - (0.835 - 0.02 * vv)));
      const b2 = 1 - sstep(0.05, 0.06, Math.abs(ss - (0.6 - 0.035 * vv)));
      const b3 = 1 - sstep(0.035, 0.045, Math.abs(ss - (0.37 - 0.02 * vv)));
      return Math.max(b1, b2, b3);
    };
    if (reg === "body") {
      if (s < P.sPed) return [...[0.92, 0.93, 0.95], finA(0.4 + 0.08 * rayLines(v, 4))];
      let col = countershade([0.84, 0.86, 0.9], C.white, [0.97, 0.97, 0.98], v);
      col = mixc(col, [0.78, 0.8, 0.85], sstep(0.9, 0.96, s) * 0.6);
      col = mixc(col, C.indigo, bars(s, v));
      const [pu, ir] = eyeMask(P, s, v, 0.9, 0.22, 0.028);
      col = mixc(col, [0.5, 0.5, 0.62], ir * 0.6);
      col = mixc(col, C.pupil, pu);
      return [...col, bodyA(0.1)];
    }
    if (reg === "dorsal") {
      const b = Math.max(bars(s, 1.0), sstep(0.5, 0.42, s));
      return [...mixc([0.93, 0.94, 0.96], C.indigo, b), finA(0.55 + 0.4 * b - 0.1 * v)];
    }
    if (reg === "anal") return [...mixc(C.indigo, [0.3, 0.28, 0.45], v * 0.5), finA(0.85)];
    return [...[0.88, 0.9, 0.93], finA(0.8)];
  },
  // Chaetodon (lemon) with a dark eye band edged in cream, faint oblique lines, dorsal ocellus
  butterfly0(P, reg, s, v) {
    if (reg === "body") {
      if (s < P.sPed) {
        const base = sstep(0.06, 0.13, s);
        return [...mixc([0.97, 0.95, 0.86], C.lemon, base), finA(0.42 + 0.35 * base)];
      }
      let col = countershade([0.93, 0.8, 0.42], C.lemon, [0.98, 0.95, 0.75], v);
      const lines = Math.pow(Math.abs(Math.sin((s * 30 + Math.abs(v - 0.1) * 4.5) * Math.PI)), 10); // chevrons
      col = mixc(col, [0.88, 0.68, 0.34], lines * 0.5 * sstep(0.25, 0.4, s) * (1 - sstep(0.72, 0.8, s)));
      col = mixc(col, [0.97, 0.8, 0.4], sstep(0.45, 0.2, s) * 0.35); // warmer toward the tail
      const bc = 0.83 + 0.025 * v;
      const band = 1 - sstep(0.026, 0.032, Math.abs(s - bc));
      const edge = 1 - sstep(0.032, 0.04, Math.abs(s - bc));
      col = mixc(col, [0.99, 0.97, 0.9], (edge - band) * 0.8);
      col = mixc(col, [0.23, 0.21, 0.33], band);
      col = mixc(col, [0.99, 0.96, 0.84], sstep(0.9, 0.97, s) * 0.55); // pale snout
      const [pu] = eyeMask(P, s, v, 0.835, 0.22, 0.026);
      col = mixc(col, C.pupil, pu);
      return [...col, bodyA(0.08)];
    }
    if (reg === "dorsal") {
      let col = mixc(C.lemon, [0.97, 0.9, 0.62], v * 0.4);
      col = mixc(col, [0.3, 0.26, 0.36], (1 - sstep(0.04, 0.08, Math.abs(v - 0.8))) * 0.85);
      const oc = Math.hypot((s - 0.33) * 1.0, (v - 0.42) * 0.09);
      col = mixc(col, [0.99, 0.97, 0.9], 1 - sstep(0.016, 0.02, oc));
      col = mixc(col, [0.12, 0.1, 0.18], 1 - sstep(0.011, 0.014, oc));
      return [...col, finA(0.85 - 0.25 * sstep(0.85, 1, v))];
    }
    if (reg === "anal") {
      let col = mixc(C.lemon, [0.97, 0.9, 0.62], v * 0.4);
      col = mixc(col, [0.3, 0.26, 0.36], (1 - sstep(0.04, 0.08, Math.abs(v - 0.78))) * 0.8);
      return [...col, finA(0.85 - 0.25 * sstep(0.85, 1, v))];
    }
    return [...mixc(C.lemon, [1, 1, 0.92], 0.18), finA(0.84)];
  },
  // Thalassoma lunare: mint body, rose scale lines, banded head, cream-and-rose lunate tail
  wrasse0(P, reg, s, v) {
    const rose = [0.93, 0.56, 0.7];
    if (reg === "body") {
      if (s < P.sPed) {
        const lobe = sstep(0.4, 0.75, Math.abs(v));
        return [...mixc([0.97, 0.9, 0.58], rose, lobe), finA(0.6 + 0.25 * lobe)];
      }
      let col = countershade([0.44, 0.76, 0.74], [0.62, 0.88, 0.74], [0.84, 0.95, 0.86], v);
      const lines = Math.pow(Math.abs(Math.sin(s * 44 * Math.PI)), 10) * sstep(0.26, 0.32, s) * (1 - sstep(0.76, 0.8, s));
      col = mixc(col, rose, lines * 0.55);
      const T = P.top(0.88);
      const d = Math.hypot(s - 0.9, (v - 0.22) * T);
      const hb = sstep(0.79, 0.81, s) * Math.pow(Math.abs(Math.cos(d * 70)), 3);
      col = mixc(col, rose, hb * 0.75);
      const [pu, ir] = eyeMask(P, s, v, 0.9, 0.22, 0.024);
      col = mixc(col, [0.95, 0.6, 0.62], ir);
      col = mixc(col, C.pupil, pu);
      return [...col, bodyA(0.2)];
    }
    if (reg === "dorsal" || reg === "anal") {
      const col = mixc(rose, [0.6, 0.86, 0.78], 1 - sstep(0.12, 0.22, Math.abs(v - 0.45)));
      return [...col, finA(0.72)];
    }
    return [...mixc([0.95, 0.64, 0.75], [0.55, 0.62, 0.9], sstep(0.7, 1.0, s)), finA(0.9)];
  },
  // Amphiprion ocellaris: orange, three white bands with thin black edges, black fin margins
  clown0(P, reg, s, v) {
    const bands = (ss, vv) => {
      const c1 = 0.79 + 0.025 * (1 - vv * vv), c2 = 0.53 + 0.04 * Math.max(0, 1 - Math.abs(vv - 0.1) * 1.8), c3 = 0.245;
      return Math.min(Math.abs(ss - c1) - 0.028, Math.abs(ss - c2) - 0.034, Math.abs(ss - c3) - 0.02); // < 0 inside a band
    };
    if (reg === "body") {
      if (s < P.sPed) {
        let col = C.clown;
        col = mixc(col, [0.07, 0.05, 0.06], sstep(0.03, 0.012, s));
        col = mixc(col, [0.97, 0.96, 0.94], sstep(0.012, 0.004, s));
        return [...col, finA(0.85)];
      }
      let col = countershade([0.97, 0.46, 0.18], C.clown, [1.0, 0.66, 0.4], v);
      const d = bands(s, v);
      col = mixc(col, [0.07, 0.05, 0.06], 1 - sstep(0.0, 0.006, d));
      col = mixc(col, [0.98, 0.97, 0.95], 1 - sstep(-0.006, -0.002, d));
      const [pu, ir] = eyeMask(P, s, v, 0.885, 0.2, 0.029);
      col = mixc(col, [0.98, 0.5, 0.2], ir);
      col = mixc(col, C.pupil, pu);
      return [...col, bodyA(0.06)];
    }
    if (reg === "dorsal" || reg === "anal") {
      let col = mixc(C.clown, [0.07, 0.05, 0.06], sstep(0.72, 0.84, v));
      col = mixc(col, [0.97, 0.96, 0.94], sstep(0.93, 0.99, v) * 0.6);
      return [...col, finA(0.88)];
    }
    return [...mixc(C.clown, [0.07, 0.05, 0.06], sstep(0.82, 0.95, s)), finA(0.92)];
  },
  // Caesio: blue back, silver flanks with a yellow stripe into the upper tail lobe, dark lobe streaks
  fusilier0(P, reg, s, v) {
    if (reg === "body") {
      if (s < P.sPed) {
        const lobe = Math.abs(v);
        let col = mixc([0.7, 0.8, 0.95], [0.4, 0.55, 0.85], lobe);
        col = mixc(col, [0.98, 0.88, 0.45], (1 - sstep(0.08, 0.16, Math.abs(v - 0.62))) * sstep(0.14, 0.04, s) * 0.8);
        col = mixc(col, [0.18, 0.22, 0.38], (1 - sstep(0.06, 0.12, Math.abs(Math.abs(v) - 0.72))) * sstep(0.1, 0.02, s) * 0.8);
        return [...col, finA(0.66)];
      }
      let col = countershade([0.36, 0.52, 0.86], [0.68, 0.79, 0.96], [0.93, 0.95, 0.99], v);
      const st = 1 - sstep(0.07, 0.11, Math.abs(v - (0.42 + 0.1 * sstep(0.5, 0.25, s))));
      col = mixc(col, [0.99, 0.88, 0.42], st * sstep(0.97, 0.9, s));
      const [pu, ir] = eyeMask(P, s, v, 0.915, 0.2, 0.03);
      col = mixc(col, [0.85, 0.88, 0.95], ir);
      col = mixc(col, C.pupil, pu);
      return [...col, bodyA(0.3 + 0.7 * sstep(0.35, -0.3, v) - 0.5 * st)];
    }
    if (reg === "dorsal") return [...[0.55, 0.68, 0.92], finA(0.5)];
    if (reg === "anal") return [...[0.85, 0.9, 0.98], finA(0.42)];
    return [...[0.8, 0.87, 0.97], finA(0.8)];
  },
};

function buildAtlas() {
  const H = ATLAS_BLOCKS * BLOCK_ROWS;
  const data = new Uint8Array(ATLAS_W * H * 4);
  const put = (x, y, c) => {
    const o = (y * ATLAS_W + x) * 4;
    data[o] = Math.round(Math.min(1, Math.max(0, c[0])) * 255);
    data[o + 1] = Math.round(Math.min(1, Math.max(0, c[1])) * 255);
    data[o + 2] = Math.round(Math.min(1, Math.max(0, c[2])) * 255);
    data[o + 3] = Math.round(Math.min(1, Math.max(0, c[3])) * 255);
  };
  for (const [name, sp] of Object.entries(SPECIES)) {
    const P = profileFns(sp);
    const mirror = sp.shade.mirror;
    const bodyPut = (x, y, c) => put(x, y, c[3] >= 0.5 ? [c[0], c[1], c[2], bodyA(((c[3] - 0.5) / 0.498) * mirror)] : c);
    for (let vi = 0; vi < sp.variants; vi++) {
      const paint = PAINTERS[name + vi];
      const y0 = (sp.block + vi) * BLOCK_ROWS;
      for (let x = 0; x < ATLAS_W; x++) {
        const s = x / (ATLAS_W - 1);
        for (let r = 0; r < 32; r++) bodyPut(x, y0 + r, paint(P, "body", s, -1 + (2 * r) / 31));
        for (let r = 0; r < 16; r++) put(x, y0 + 32 + r, paint(P, "dorsal", s, r / 15));
        for (let r = 0; r < 8; r++) put(x, y0 + 48 + r, paint(P, "anal", s, r / 7));
        for (let r = 0; r < 8; r++) put(x, y0 + 56 + r, paint(P, "pec", s, r / 7));
      }
    }
  }
  const tex = new THREE.DataTexture(data, ATLAS_W, H, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.colorSpace = THREE.NoColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return { tex, rows: H, data };
}

// ---------------------------------------------------------------------------------------------
// The shared fish material: ONE pipeline and ONE instanced draw for every fish. Per-vertex data and
// per-species constants come from the storage buffer (vertex pulling), per-fish data from one
// interleaved instance buffer:
//   iA = anchor.xyz (world; school-local for the fusiliers), body length (m)
//   iB = seed, dart slot length (s), dart radius (m), wander amplitude (m)  | roamers: seed, -, 0, tail amp
//   iC = hide offset.xyz, hide probability                                  | roamers: heading.xyz, tail phase
//   iD = surge drift factor, atlas variant, species index, -
// ---------------------------------------------------------------------------------------------
/** PCG hash (pcg-random.org) emitted once as a WGSL function: uint -> uint */
const fishHash = Fn(([x]) => {
  const state = x.mul(uint(747796405)).add(uint(2891336453)).toVar();
  const word = state.shiftRight(state.shiftRight(uint(28)).add(uint(4))).bitXor(state).mul(uint(277803737)).toVar();
  return word.shiftRight(uint(22)).bitXor(word);
}).setLayout({ name: "fishHash", type: "uint", inputs: [{ name: "x", type: "uint" }] });

function buildFishMaterial(env, atlas, G) {
  const u = env.u;
  const uT = uniform(0.0); // local animation time (wrapped hourly to keep f32 phase precision)
  const uFlow = uniform(new THREE.Vector2(0.9, 0.44).normalize()); // direction hoverers face (into the mean current)
  const uNight = uniform(0.0);
  const uSchoolPos = uniform(new THREE.Vector3(0, -50, 0));
  const uSchoolFwd = uniform(new THREE.Vector3(1, 0, 0));

  const mat = new THREE.MeshBasicNodeMaterial({ name: "reef_fish", side: THREE.FrontSide });

  const iA = attribute("iA", "vec4");
  const iB = attribute("iB", "vec4");
  const iC = attribute("iC", "vec4");
  const iD = attribute("iD", "vec4");
  const sidU = iD.z.add(0.5).toUint().toVar();
  const vIdx = sidU.mul(uint(G.NV)).add(vertexIndex).mul(uint(4)).toVar();
  const V0 = G.vbuf.element(vIdx).toVar(); // pos.xyz, s
  const V1 = G.vbuf.element(vIdx.add(uint(1))).toVar(); // normal.xyz, h
  const V2 = G.vbuf.element(vIdx.add(uint(2))).toVar(); // finId, side, texU
  const V3 = G.vbuf.element(vIdx.add(uint(3))).toVar(); // hinge.xyz
  const cIdx = uint(G.specBase).add(sidU.mul(uint(4))).toVar();
  const spA = G.vbuf.element(cIdx).toVar(); // mode, atlas block, face gain
  const spB = G.vbuf.element(cIdx.add(uint(1))).toVar(); // hover f/A, dart f/A
  const spC = G.vbuf.element(cIdx.add(uint(2))).toVar(); // pectoral f, amp, rest; dart duration
  const t = uT;
  const tg = u.time;

  const mode = spA.x;
  const isRoam = step(0.5, mode).mul(step(mode, 1.5)).toVar();
  const isSchool = step(1.5, mode).toVar();
  const isHover = float(1.0).sub(isRoam).sub(isSchool).toVar();

  const seed = iB.x.toVar();
  const seedU = seed.mul(4194301.0).toUint().toVar();
  // one PCG hash per (fish, slot, salt), sliced into three 10-bit uniforms
  const HS = (kU, salt) => fishHash(kU.mul(uint(2654435761)).add(seedU.mul(uint(2246822519))).add(uint(salt))).toVar();
  const bits = (h, shift) => h.shiftRight(uint(shift)).bitAnd(uint(1023)).toFloat().mul(1.0 / 1023.0);
  const hc = fishHash(seedU.mul(uint(2246822519)).add(uint(7933))).toVar(); // per-fish constants
  const slotT = max(iB.y, 0.5).toVar();
  const R = iB.z;
  const wAmp = iB.w;
  const len = iA.w;
  // the fusilier school moves as a unit: school-local anchors in the frame of (uSchoolPos, uSchoolFwd)
  const sF = normalize(uSchoolFwd).toVar();
  const sL = normalize(cross(sF, vec3(0.0, 1.0, 0.0))).toVar();
  const sU = cross(sL, sF).toVar();
  const schoolAnchor = uSchoolPos.add(sF.mul(iA.x)).add(sU.mul(iA.y)).add(sL.mul(iA.z));
  const anchorW = mix(iA.xyz, schoolAnchor, isSchool).toVar();
  const ph = seed.mul(TAU).toVar();
  const fj = bits(hc, 0).mul(0.3).add(0.85).toVar(); // per-fish tempo
  const night = uNight;

  // ---- slow 3D wander (incommensurate sines: no orbits) ----
  const wt = t.mul(fj).mul(float(1.0).sub(night.mul(0.4))).toVar();
  const wander = vec3(
    sin(wt.mul(0.61).add(ph)).add(sin(wt.mul(1.43).add(ph.mul(2.3))).mul(0.45)),
    sin(wt.mul(0.47).add(ph.mul(1.7))).add(sin(wt.mul(1.21).add(ph.mul(3.1))).mul(0.4)).mul(0.55),
    sin(wt.mul(0.53).add(ph.mul(2.9))).add(sin(wt.mul(1.33).add(ph.mul(4.3))).mul(0.45))
  )
    .mul(wAmp)
    .toVar();

  // ---- darts: one per slot; turn toward the target, burst, glide, turn back into the current ----
  const hasDart = step(0.001, R).mul(isHover).toVar();
  const durS = spC.w;
  const tt = t.div(slotT.mul(night.mul(1.5).add(1.0))).add(seed.mul(7.31)).add(16.0).toVar();
  const kF = floor(tt).toVar();
  const fs = tt.sub(kF).toVar();
  const kU = kF.toUint().toVar();
  const kPrevU = kF.sub(1.0).toUint().toVar();
  const durN = clamp(durS.div(slotT), 0.02, 0.4).toVar();
  const leadN = min(float(0.3).div(slotT), 0.15).toVar();
  const settleN = min(float(0.7).div(slotT), 0.25).toVar();
  const hA = HS(kU, 11), hB = HS(kU, 12);
  const start = leadN.add(bits(hB, 20).mul(max(float(0.0), float(1.0).sub(leadN).sub(durN).sub(settleN)))).toVar();
  const tau = clamp(fs.sub(start).div(durN), 0.0, 1.0).toVar();
  const om = float(1.0).sub(tau);
  const ease = float(1.0).sub(om.mul(om).mul(om)); // C-start burst then glide
  const target = (h1, h2) => {
    const r = vec3(bits(h1, 0), bits(h1, 10), bits(h1, 20)).mul(2.0).sub(1.0).toVar();
    const dir = normalize(vec3(r.x, r.y.mul(0.5), r.z).add(vec3(1e-3, 0.0, 0.0)));
    const out = dir.mul(R.mul(bits(h2, 0).mul(0.75).add(0.25)));
    const hideOn = step(bits(h2, 10), iC.w).mul(isHover);
    return mix(out, iC.xyz.add(r.mul(0.008)), hideOn);
  };
  const oPrev = target(HS(kPrevU, 11), HS(kPrevU, 12)).toVar();
  const oCur = target(hA, hB).toVar();
  const dartOff = mix(oPrev, oCur, ease).mul(hasDart).toVar();
  // fish that have a shelter (chromis, damsels, clownfish) drop into it when the camera comes close or at night
  const shy = step(0.001, iC.w).mul(isHover).mul(max(night.mul(0.8), float(1.0).sub(smoothstep(0.85, 1.5, length(anchorW.sub(cameraPosition))))));
  const tuck = iC.xyz.sub(dartOff.add(wander)).mul(shy);

  // ---- position: anchor + wander + dart + surge slosh (roamers: CPU position in iA) ----
  const sd = env.surge(vec2(anchorW.x, anchorW.z), tg, 0.15);
  const center = anchorW
    .add(wander.add(dartOff).add(tuck).mul(float(1.0).sub(isRoam)))
    .add(vec3(sd.x, 0.0, sd.y).mul(iD.x).mul(float(1.0).sub(shy.mul(0.7))))
    .toVar();

  // ---- heading ----
  const sv = env.surgeVel(vec2(anchorW.x, anchorW.z), tg);
  const into = normalize(uFlow.sub(sv.mul(spA.z)));
  const yaw = bits(hc, 10).sub(0.5).mul(0.9).add(sin(t.mul(0.23).mul(fj).add(ph.mul(1.9))).mul(0.22)).add(sin(t.mul(0.61).add(ph.mul(3.3))).mul(0.06));
  const cy = cos(yaw), sy = sin(yaw);
  const intoR = vec2(into.x.mul(cy).sub(into.y.mul(sy)), into.x.mul(sy).add(into.y.mul(cy)));
  const pitch = bits(hc, 20).sub(0.5).mul(0.3).add(sin(t.mul(0.37).add(ph.mul(1.3))).mul(0.1));
  const hoverDir = normalize(vec3(intoR.x, pitch, intoR.y)).toVar();
  const dv = oCur.sub(oPrev);
  const dDir = dv.div(max(length(dv), 1e-4)).toVar();
  const wTurn = smoothstep(start.sub(leadN), start, fs)
    .mul(float(1.0).sub(smoothstep(start.add(durN), start.add(durN).add(settleN), fs)))
    .mul(hasDart)
    .toVar();
  const kick = normalize(cross(hoverDir, vec3(0.0, 1.0, 0.0)).add(vec3(0.0, 0.0, 1e-4)))
    .mul(wTurn.mul(float(1.0).sub(wTurn)).mul(float(1.0).sub(dot(hoverDir, dDir))).mul(1.2));
  const headH = mix(hoverDir, dDir, wTurn).add(kick);
  const wob = vec3(1.0, sin(t.mul(0.7).add(ph)).mul(0.05), sin(t.mul(0.53).add(ph.mul(2.0))).mul(0.08));
  const headS = sF.mul(wob.x).add(sU.mul(wob.y)).add(sL.mul(wob.z));
  const headRaw = normalize(headH.mul(isHover).add(iC.xyz.mul(isRoam)).add(headS.mul(isSchool)).add(vec3(1e-4, 0.0, 0.0)));
  const head = normalize(vec3(headRaw.x, clamp(headRaw.y, -0.6, 0.6), headRaw.z)).toVar();

  // ---- tail-beat phase, integrated analytically across the darts ----
  const sinPi = sin(tau.mul(Math.PI));
  const wd = sinPi.mul(sinPi).mul(hasDart).toVar();
  const fH = spB.x.mul(fj).mul(float(1.0).sub(night.mul(0.35)));
  const dCyc = spB.z.sub(fH).mul(durS).mul(hasDart).toVar();
  const phaseHover = fract(fH.mul(t).add(seed.mul(3.7)))
    .add(fract(dCyc.mul(0.5).mul(kF)))
    .add(dCyc.mul(tau.mul(0.5).sub(sin(tau.mul(TAU)).mul(1.0 / (2.0 * TAU)))));
  const phase = mix(phaseHover, iC.w, isRoam).toVar();
  const ampH = mix(spB.y, spB.w, wd).mul(fj);
  const amp = mix(ampH, iB.w, isRoam).toVar();

  // ---- pectoral flap about the hinge (folded during bursts) ----
  const isPec = step(2.5, V2.x);
  const pecPh = t.mul(spC.x).mul(fj).add(seed.mul(5.1)).mul(TAU);
  const ang = spC.z.add(spC.y.mul(sin(pecPh))).mul(float(1.0).sub(wd.mul(0.85))).mul(V2.y).mul(isPec).toVar();
  const axis = vec3(0.24, 0.97, 0.0).normalize();
  const ca = cos(ang), sa = sin(ang);
  const rot = (v) => v.mul(ca).add(cross(axis, v).mul(sa)).add(axis.mul(dot(axis, v).mul(float(1.0).sub(ca))));
  const hinge = V3.xyz;
  const pL0 = hinge.add(rot(V0.xyz.sub(hinge))).toVar();
  const nL0 = rot(V1.xyz).toVar();

  // ---- travelling body wave (s: 0 tail .. 1 head) with bent normals ----
  const uu = float(1.0).sub(V0.w).toVar();
  const envl = uu.mul(uu).mul(0.92).add(0.08);
  const arg = uu.div(0.95).sub(phase).mul(TAU).toVar();
  const sA = sin(arg), cA = cos(arg);
  // C-shaped curl while turning toward a dart target (and a gentler one turning back into the current)
  const sideT = dot(dDir, normalize(cross(hoverDir, vec3(0.0, 1.0, 0.0)).add(vec3(0.0, 0.0, 1e-4))));
  const lead = step(fs, start);
  const curl = wTurn.mul(float(1.0).sub(wTurn)).mul(sideT).mul(lead.mul(1.6).sub(0.6)).mul(0.9).toVar();
  const um = uu.sub(0.45);
  const lat = amp.mul(envl).mul(sA).add(curl.mul(um.mul(um).sub(0.06)));
  const dl = amp.mul(uu.mul(1.84).mul(sA).add(envl.mul(cA).mul(TAU / 0.95))).add(curl.mul(um).mul(2.0));
  const pL = vec3(pL0.x, pL0.y, pL0.z.add(lat)).toVar();
  const nL = vec3(nL0.x.add(dl.mul(nL0.z)), nL0.y, nL0.z).toVar();

  // ---- orient + place; shrink away near the lens; night: diurnal planktivores & roamers shelter ----
  const dc = length(center.sub(cameraPosition));
  const shelter = float(1.0).sub(step(seed, night.mul(0.86)).mul(isHover)).mul(float(1.0).sub(isRoam.mul(night)));
  const scl = len.mul(smoothstep(0.42, 0.85, dc)).mul(shelter);
  const lax = normalize(cross(head, vec3(0.0, 1.0, 0.0))).toVar();
  const upv = cross(lax, head).toVar();
  const posW = center.add(head.mul(pL.x).add(upv.mul(pL.y)).add(lax.mul(pL.z)).mul(scl)).toVar();
  const nW = head.mul(nL.x).add(upv.mul(nL.y)).add(lax.mul(nL.z));

  // ---- pattern atlas coordinates ----
  const fin = V2.x;
  const hh = V1.w;
  const f1 = step(0.5, fin), f2 = step(1.5, fin), f3 = step(2.5, fin);
  const row = mix(mix(mix(hh.add(1.0).mul(15.5).add(0.5), hh.mul(15.0).add(32.5), f1), hh.mul(7.0).add(48.5), f2), hh.mul(7.0).add(56.5), f3);
  const vTex = spA.y.add(iD.y).mul(BLOCK_ROWS).add(row).div(atlas.rows);
  const uTex = V2.z.mul(ATLAS_W - 1).add(0.5).div(ATLAS_W);

  mat.vertexNode = cameraProjectionMatrix.mul(cameraViewMatrix).mul(vec4(posW, 1.0));
  const vA = varying(vec4(posW, uTex));
  const vB = varying(vec4(nW, vTex));

  mat.fragmentNode = Fn(() => {
    const pW = vA.xyz;
    const tex = texture(atlas.tex, vec2(vA.w, vB.w));
    const a = tex.a;
    const isFin = float(1.0).sub(step(0.5, a)).toVar();
    const membrane = clamp(a.mul(2.04), 0.0, 1.0); // authored fin opacity
    const silver = clamp(a.sub(0.5).mul(2.0), 0.0, 1.0).mul(float(1.0).sub(isFin)).toVar(); // species mirror baked in
    const N = normalize(vB.xyz).toVar();
    const lit = env.litProp({
      albedo: tex.rgb,
      N,
      posW: pW,
      wrap: 0.5,
      trans: isFin.mul(0.5).add(0.05),
      causticGain: 0.5,
      rim: 0.8,
      sheen: 0.12,
    });
    const V = normalize(cameraPosition.sub(pW)).toVar();
    const nv = clamp(dot(N, V), 0.0, 1.0);
    const fr = float(1.0).sub(nv);
    const dl2 = env.depthLight(pW);
    // silvery flanks mirror the water column (guanine platelets) with a faint thin-film tint; the
    // reflection uses the cheap base gradient of env.waterColor (no shafts: it is a small, curved mirror)
    const Rr = reflect(V.negate(), N).toVar();
    const grad = mix(mix(u.bgDeep, u.bgMid, smoothstep(-0.6, 0.35, Rr.y)), u.bgShallow, smoothstep(0.04, 0.72, Rr.y).mul(0.72));
    const toSun = dot(Rr, u.sunDir).mul(0.5).add(0.5);
    const film = cos(vec3(9.666, 11.855, 13.963).mul(nv.mul(0.55).add(0.95))).mul(0.5).add(0.5);
    const mirror = grad
      .mul(mix(float(0.99), float(1.24), toSun.mul(toSun)))
      .mul(dl2.amb)
      .mul(mix(vec3(1.0), film, 0.3))
      .mul(fr.mul(fr).mul(0.55).add(0.32))
      .mul(silver);
    // wet gloss: small sun highlight on the scales
    const hv = clamp(dot(N, normalize(u.sunDir.add(V))), 0.0, 1.0);
    const h2 = hv.mul(hv), h4 = h2.mul(h2), h8 = h4.mul(h4), h16 = h8.mul(h8);
    const gloss = u.keyColor.mul(h16.mul(h16).mul(h8)).mul(u.keyGain).mul(dl2.sun).mul(silver.mul(0.25).add(0.08)).mul(float(1.0).sub(isFin));
    const body = lit.mul(float(1.0).sub(silver.mul(0.45))).add(mirror).add(gloss);
    // fog terms computed once: the water colour along the view ray is both the fog colour and what a thin
    // fin membrane shows behind it (opaque coverage: no alpha-to-coverage stipple)
    const fog = env.fogTerms(pW);
    const waterRay = fog.color.toVar();
    const finCol = mix(waterRay.mul(dl2.amb).mul(0.92), lit, membrane.mul(0.72).add(0.28));
    const shaded = mix(body, finCol, isFin).add(u.rimColor.mul(fr.mul(fr).mul(0.52).add(0.09)).mul(uNight));
    return env.output(shaded.mul(fog.T).add(waterRay.mul(vec3(1.0).sub(fog.T))), 1.0);
  })();

  return { mat, uT, uFlow, uNight, uSchoolPos, uSchoolFwd };
}

// ---------------------------------------------------------------------------------------------
// 3D clearance field (5 cm voxels) from the Houdini coral vertices + seabed, interiors filled,
// chamfer distance transform; exact nearest-vertex queries close to the surface.
// ---------------------------------------------------------------------------------------------
function buildClearance(T) {
  const CS = 0.05, X0 = -2.6, Y0 = -0.3, Z0 = -1.6, NX = 104, NY = 66, NZ = 60;
  const NXY = NX * NY, N = NXY * NZ;
  const solid = new Uint8Array(N);
  const cellStart = new Int32Array(N + 1);
  const geos = T.coralGeometry || [];
  const cellOf = (x, y, z) => {
    const i = Math.floor((x - X0) / CS), j = Math.floor((y - Y0) / CS), k = Math.floor((z - Z0) / CS);
    if (i < 0 || j < 0 || k < 0 || i >= NX || j >= NY || k >= NZ) return -1;
    return k * NXY + j * NX + i;
  };
  let total = 0;
  for (const g of geos) {
    const p = g.positions;
    for (let v = 0; v < g.count; v++) {
      const id = cellOf(p[3 * v], p[3 * v + 1], p[3 * v + 2]);
      if (id < 0) continue;
      solid[id] = 1;
      cellStart[id + 1]++;
      total++;
    }
  }
  for (let id = 0; id < N; id++) cellStart[id + 1] += cellStart[id];
  const fill = new Int32Array(N);
  const vx = new Float32Array(total * 3);
  for (const g of geos) {
    const p = g.positions;
    for (let v = 0; v < g.count; v++) {
      const id = cellOf(p[3 * v], p[3 * v + 1], p[3 * v + 2]);
      if (id < 0) continue;
      const o = (cellStart[id] + fill[id]++) * 3;
      vx[o] = p[3 * v];
      vx[o + 1] = p[3 * v + 1];
      vx[o + 2] = p[3 * v + 2];
    }
  }
  // seabed is solid
  for (let k = 0; k < NZ; k++)
    for (let i = 0; i < NX; i++) {
      const h = T.heightAt(X0 + (i + 0.5) * CS, Z0 + (k + 0.5) * CS);
      for (let j = 0; j < NY; j++) {
        if (Y0 + (j + 0.5) * CS < h - 0.01) solid[k * NXY + j * NX + i] = 1;
        else break;
      }
    }
  // exterior flood fill on the 1-voxel dilation (closes vertex-sampling gaps), regrown by one voxel
  const dil = new Uint8Array(N);
  for (let k = 0; k < NZ; k++)
    for (let j = 0; j < NY; j++)
      for (let i = 0; i < NX; i++) {
        const id = k * NXY + j * NX + i;
        if (!solid[id]) continue;
        dil[id] = 1;
        if (i > 0) dil[id - 1] = 1;
        if (i < NX - 1) dil[id + 1] = 1;
        if (j > 0) dil[id - NX] = 1;
        if (j < NY - 1) dil[id + NX] = 1;
        if (k > 0) dil[id - NXY] = 1;
        if (k < NZ - 1) dil[id + NXY] = 1;
      }
  const ext = new Uint8Array(N);
  const queue = new Int32Array(N);
  let qh = 0, qt = 0;
  const seedExt = (id) => {
    if (!dil[id] && !ext[id]) {
      ext[id] = 1;
      queue[qt++] = id;
    }
  };
  for (let k = 0; k < NZ; k++)
    for (let j = 0; j < NY; j++) {
      seedExt(k * NXY + j * NX);
      seedExt(k * NXY + j * NX + NX - 1);
    }
  for (let j = 0; j < NY; j++)
    for (let i = 0; i < NX; i++) {
      seedExt(j * NX + i);
      seedExt((NZ - 1) * NXY + j * NX + i);
    }
  for (let k = 0; k < NZ; k++) for (let i = 0; i < NX; i++) seedExt(k * NXY + (NY - 1) * NX + i);
  while (qh < qt) {
    const id = queue[qh++];
    const i = id % NX, j = ((id - i) / NX) % NY, k = Math.floor(id / NXY);
    if (i > 0) seedExt(id - 1);
    if (i < NX - 1) seedExt(id + 1);
    if (j > 0) seedExt(id - NX);
    if (j < NY - 1) seedExt(id + NX);
    if (k > 0) seedExt(id - NXY);
    if (k < NZ - 1) seedExt(id + NXY);
  }
  for (let k = 1; k < NZ - 1; k++)
    for (let j = 1; j < NY - 1; j++)
      for (let i = 1; i < NX - 1; i++) {
        const id = k * NXY + j * NX + i;
        if (ext[id] || solid[id]) continue;
        if (ext[id - 1] === 1 || ext[id + 1] === 1 || ext[id - NX] === 1 || ext[id + NX] === 1 || ext[id - NXY] === 1 || ext[id + NXY] === 1) ext[id] = 2;
      }
  let inside = 0;
  for (let id = 0; id < N; id++)
    if (!ext[id] && !solid[id]) {
      solid[id] = 2; // enclosed volume (inside a closed coral or under the seabed)
      inside++;
    }
  // chamfer distance transform (metres)
  const dist = new Float32Array(N);
  for (let id = 0; id < N; id++) dist[id] = solid[id] ? 0 : 1e6;
  const a1 = CS, a2 = CS * Math.SQRT2, a3 = CS * Math.sqrt(3);
  const fw = [
    [-1, 0, 0, a1], [-1, -1, 0, a2], [0, -1, 0, a1], [1, -1, 0, a2],
    [-1, -1, -1, a3], [0, -1, -1, a2], [1, -1, -1, a3], [-1, 0, -1, a2], [0, 0, -1, a1], [1, 0, -1, a2], [-1, 1, -1, a3], [0, 1, -1, a2], [1, 1, -1, a3],
  ].map(([di, dj, dk, w]) => [dk * NXY + dj * NX + di, w]);
  const offF = fw.map((o) => o[0]), wF = fw.map((o) => o[1]);
  for (let k = 1; k < NZ - 1; k++)
    for (let j = 1; j < NY - 1; j++)
      for (let i = 1; i < NX - 1; i++) {
        const id = k * NXY + j * NX + i;
        let d = dist[id];
        if (d === 0) continue;
        for (let n = 0; n < 13; n++) {
          const e = dist[id + offF[n]] + wF[n];
          if (e < d) d = e;
        }
        dist[id] = d;
      }
  for (let k = NZ - 2; k >= 1; k--)
    for (let j = NY - 2; j >= 1; j--)
      for (let i = NX - 2; i >= 1; i--) {
        const id = k * NXY + j * NX + i;
        let d = dist[id];
        if (d === 0) continue;
        for (let n = 0; n < 13; n++) {
          const e = dist[id - offF[n]] + wF[n];
          if (e < d) d = e;
        }
        dist[id] = d;
      }

  const coarse = (x, y, z) => {
    const fx = (x - X0) / CS - 0.5, fy = (y - Y0) / CS - 0.5, fz = (z - Z0) / CS - 0.5;
    if (fx < 1 || fy < 1 || fz < 1 || fx > NX - 3 || fy > NY - 3 || fz > NZ - 3) return Math.max(0, y - T.heightAt(x, z)) + (fy > NY - 3 ? 0.5 : 0);
    const i = Math.floor(fx), j = Math.floor(fy), k = Math.floor(fz);
    const tx = fx - i, ty = fy - j, tz = fz - k;
    const id = k * NXY + j * NX + i;
    const l = (o) => dist[id + o];
    const c00 = l(0) * (1 - tx) + l(1) * tx, c10 = l(NX) * (1 - tx) + l(NX + 1) * tx;
    const c01 = l(NXY) * (1 - tx) + l(NXY + 1) * tx, c11 = l(NXY + NX) * (1 - tx) + l(NXY + NX + 1) * tx;
    return (c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz - 0.018;
  };
  const isInside = (x, y, z) => {
    const id = cellOf(x, y, z);
    return id >= 0 && solid[id] === 2;
  };
  /** nearest coral vertex within `rad` cells: { d, x, y, z } (d = Infinity when none) */
  const nearest = (x, y, z, rad = 2) => {
    const i0 = Math.floor((x - X0) / CS), j0 = Math.floor((y - Y0) / CS), k0 = Math.floor((z - Z0) / CS);
    let best = Infinity, bx = 0, by = 0, bz = 0;
    for (let k = Math.max(0, k0 - rad); k <= Math.min(NZ - 1, k0 + rad); k++)
      for (let j = Math.max(0, j0 - rad); j <= Math.min(NY - 1, j0 + rad); j++)
        for (let i = Math.max(0, i0 - rad); i <= Math.min(NX - 1, i0 + rad); i++) {
          const id = k * NXY + j * NX + i;
          for (let o = cellStart[id]; o < cellStart[id + 1]; o++) {
            const dx = vx[3 * o] - x, dy = vx[3 * o + 1] - y, dz = vx[3 * o + 2] - z;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best) {
              best = d2;
              bx = vx[3 * o];
              by = vx[3 * o + 1];
              bz = vx[3 * o + 2];
            }
          }
        }
    return { d: Math.sqrt(best), x: bx, y: by, z: bz };
  };
  /** clearance (m) to coral and seabed: coarse field, refined by exact vertex distances near the surface */
  const clear = (x, y, z) => {
    if (isInside(x, y, z)) return -0.05;
    const c0 = coarse(x, y, z);
    const sb = y - T.heightAt(x, z);
    if (c0 > 0.11) return Math.min(c0, sb);
    // the 3x3x3 block holds every vertex closer than one cell (5 cm): exact below that, field above
    const nv = nearest(x, y, z, 1);
    const d = nv.d <= CS ? nv.d : Math.min(nv.d, Math.max(c0, CS));
    return Math.min(d, sb);
  };
  return { clear, coarse, nearest, isInside, inside, total };
}

// ---------------------------------------------------------------------------------------------
// Arc-length paths for the roamers (built once): closed Catmull-Rom loop, resampled every 1 cm,
// with per-sample speed, cumulative time, tail-beat phase and optional pecking dwells.
// ---------------------------------------------------------------------------------------------
function catmullLoop(way, perSeg = 40) {
  const n = way.length, out = [];
  for (let i = 0; i < n; i++) {
    const p0 = way[(i - 1 + n) % n], p1 = way[i], p2 = way[(i + 1) % n], p3 = way[(i + 2) % n];
    for (let s = 0; s < perSeg; s++) {
      const t = s / perSeg, t2 = t * t, t3 = t2 * t;
      out.push(
        [0, 1, 2].map(
          (c) => 0.5 * (2 * p1[c] + (-p0[c] + p2[c]) * t + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2 + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t3)
        )
      );
    }
  }
  return out;
}
function resampleLoop(poly, ds = 0.01) {
  const n = poly.length;
  const cum = [0];
  for (let i = 1; i <= n; i++) {
    const a = poly[i - 1], b = poly[i % n];
    cum.push(cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
  }
  const L = cum[n];
  const m = Math.max(8, Math.floor(L / ds));
  const pts = new Float32Array(m * 3);
  let seg = 0;
  for (let k = 0; k < m; k++) {
    const s = (k / m) * L;
    while (cum[seg + 1] < s) seg++;
    const a = poly[seg], b = poly[(seg + 1) % n];
    const f = (s - cum[seg]) / Math.max(1e-9, cum[seg + 1] - cum[seg]);
    for (let c = 0; c < 3; c++) pts[k * 3 + c] = a[c] + (b[c] - a[c]) * f;
  }
  return { pts, n: m, L, ds: L / m };
}
/** speedFn(k) m/s per sample; dwell: Map sample -> seconds; tail: { f0, f1, fDwell } Hz = f0 + f1 * v */
function timePath(path, speedFn, dwell, tail) {
  const n = path.n;
  const T = new Float64Array(n + 1), PH = new Float64Array(n + 1), V = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const v = Math.max(0.02, speedFn(k));
    V[k] = v;
    const d = dwell.get(k) || 0;
    const dt = path.ds / v;
    T[k + 1] = T[k] + d + dt;
    PH[k + 1] = PH[k] + d * tail.fDwell + dt * (tail.f0 + tail.f1 * v);
  }
  // smoothed tangents
  const tan = new Float32Array(n * 3);
  const w = 6;
  for (let k = 0; k < n; k++) {
    const a = ((k + w) % n) * 3, b = ((k - w + n) % n) * 3;
    const dx = path.pts[a] - path.pts[b], dy = path.pts[a + 1] - path.pts[b + 1], dz = path.pts[a + 2] - path.pts[b + 2];
    const l = Math.hypot(dx, dy, dz) || 1;
    tan[k * 3] = dx / l;
    tan[k * 3 + 1] = dy / l;
    tan[k * 3 + 2] = dz / l;
  }
  return { ...path, T, PH, V, tan, period: T[n], dwell };
}
/** evaluate a timed path at absolute time t (+ offset): position, tangent, tail phase, speed, dwell state */
function evalPath(P, t, offsetS, arcOffset = 0, out = {}) {
  const tl = t + offsetS;
  const cyc = Math.floor(tl / P.period);
  const tau = tl - cyc * P.period;
  let lo = 0, hi = P.n;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (P.T[mid] <= tau) lo = mid;
    else hi = mid;
  }
  const k = lo;
  const d = P.dwell.get(k) || 0;
  const into = tau - P.T[k];
  let f = 0, dwellW = 0, dwellT = 0;
  if (into < d) {
    dwellT = into;
    dwellW = Math.min(1, into / 0.6, (d - into) / 0.6);
  } else f = (into - d) / Math.max(1e-6, P.T[k + 1] - P.T[k] - d);
  const ko = (((k + Math.round(arcOffset / P.ds)) % P.n) + P.n) % P.n;
  const k2 = (ko + 1) % P.n;
  for (let c = 0; c < 3; c++) {
    out[c] = P.pts[ko * 3 + c] + (P.pts[k2 * 3 + c] - P.pts[ko * 3 + c]) * f;
    out[3 + c] = P.tan[ko * 3 + c];
  }
  const phInto = P.PH[k] + (P.PH[k + 1] - P.PH[k]) * Math.min(1, into / Math.max(1e-6, P.T[k + 1] - P.T[k]));
  out.phase = ((((cyc * P.PH[P.n]) % 1) + phInto) % 1 + 1) % 1;
  out.speed = into < d ? 0 : P.V[k];
  out.dwellW = dwellW;
  out.dwellT = dwellT;
  out.k = k;
  return out;
}

// ---------------------------------------------------------------------------------------------
// The module
// ---------------------------------------------------------------------------------------------
export function createReefFish(env) {
  const group = new THREE.Group();
  group.name = "fish";
  env.groups.fish = group;
  env.scene.add(group);
  try {
    return buildFishSystem(env, group);
  } catch (e) {
    console.error("reef-fish: init failed", e);
    return { group, update() {}, warmTo() {} };
  }
}

function buildFishSystem(env, group) {
  const atlas = buildAtlas();
  const G = buildSharedGeometry();
  const M = buildFishMaterial(env, atlas, G);

  // ---- one instance buffer for every fish; roamers first (per-frame range upload), school last ----
  const ORDER = ["butterfly", "wrasse", "anthias", "chromis", "humbug", "clown", "fusilier"];
  const COUNTS = { butterfly: 4, wrasse: 3, anthias: 110, chromis: 70, humbug: 16, clown: 3, fusilier: 60 };
  const range = {};
  let NI = 0;
  for (const n of ORDER) {
    range[n] = { base: NI, count: COUNTS[n] };
    NI += COUNTS[n];
  }
  const ROAMERS = COUNTS.butterfly + COUNTS.wrasse;
  const inst = new Float32Array(NI * 16);
  const ibuf = new THREE.InstancedInterleavedBuffer(inst, 16);
  ibuf.setUsage(THREE.DynamicDrawUsage);
  const geo = G.geo;
  geo.setAttribute("iA", new THREE.InterleavedBufferAttribute(ibuf, 4, 0));
  geo.setAttribute("iB", new THREE.InterleavedBufferAttribute(ibuf, 4, 4));
  geo.setAttribute("iC", new THREE.InterleavedBufferAttribute(ibuf, 4, 8));
  geo.setAttribute("iD", new THREE.InterleavedBufferAttribute(ibuf, 4, 12));
  const mesh = new THREE.Mesh(geo, M.mat);
  mesh.name = "reef_fish";
  mesh.count = NI;
  mesh.frustumCulled = false;
  group.add(mesh);

  const search = typeof location !== "undefined" ? location.search : "";
  const dbg = /[?&]fishDebug=1/.test(search);
  const only = ((search.match(/[?&]fishOnly=([a-z,]+)/) || [])[1] || "").split(",").filter(Boolean);
  const shown = (name) => !only.length || only.includes(name);
  const lineup = /[?&]fishLineup=1/.test(search);

  const off = (name, i) => (range[name].base + i) * 16;
  /** write one fish: a = anchor+len, b = seed/slot/R/wander, c = hide/heading, d = [kSurge, variant] */
  const write = (name, i, a, b, c, d) => {
    const o = off(name, i);
    inst.set(a, o);
    if (!shown(name)) inst[o + 3] = 0;
    inst.set(b, o + 4);
    inst.set(c, o + 8);
    inst[o + 12] = d[0];
    inst[o + 13] = d[1];
    inst[o + 14] = G.speciesIndex[name];
    inst[o + 15] = 0;
  };
  const hideRest = (name, from) => {
    for (let i = from; i < range[name].count; i++) write(name, i, [0, -40, 0, 0], [0.5, 3, 0, 0], [0, 0, 0, 0], [0, 0]);
  };
  // parked (zero length, under the seabed) until placement; the pipeline still compiles in compileAsync
  for (const n of ORDER) hideRest(n, 0);
  let fullUpload = true;
  // warm the single fish pipeline while the Houdini GLB is still loading (off the critical path)
  if (env.renderer && env.renderer.compileAsync) {
    const tw = performance.now();
    env.renderer
      .compileAsync(group, env.camera, env.scene)
      .then(() => {
        if (dbg) console.log(`reef-fish: pipeline warmed in ${Math.round(performance.now() - tw)} ms`);
      })
      .catch((e) => console.error("reef-fish: pipeline warm-up failed", e));
  }

  const state = {
    ready: false,
    field: null,
    T: null,
    roam: { butterfly: [], wrasse: [] },
    sitesKey: "",
    anemoneFallback: true,
    schoolOn: false,
  };

  // -------------------------------------------------------------------------------------------
  // placement
  // -------------------------------------------------------------------------------------------
  const obstacleClear = (x, y, z) => {
    const obs = env.sites && env.sites.obstacles;
    if (!obs || !obs.length) return 9;
    let best = 9;
    for (const o of obs) {
      if (!o) continue;
      const ox = o.x ?? (o.position && o.position.x) ?? 0;
      const oz = o.z ?? (o.position && o.position.z) ?? 0;
      const y0 = o.y0 ?? 0, y1 = o.y1 ?? y0 + 0.3, r = o.r ?? o.radius ?? 0.2;
      const cy = Math.min(Math.max(y, y0), y1);
      const d = Math.hypot(x - ox, y - cy, z - oz) - r;
      if (d < best) best = d;
    }
    return best;
  };
  const clearAll = (x, y, z) => Math.min(state.field.clear(x, y, z), obstacleClear(x, y, z));
  const tooClose = (list, x, y, z, dmin) => {
    for (const p of list) if ((p[0] - x) ** 2 + (p[1] - y) ** 2 + (p[2] - z) ** 2 < dmin * dmin) return true;
    return false;
  };
  // hero-foreground guard: nothing between the hero camera and the bouquet face
  const inHeroForeground = (x, y, z) => z > 0.95 && Math.abs(x) < 1.25 && y < 1.6;

  /** hover anchors inside ellipsoid regions: [{c:[x,y,z], r:[rx,ry,rz], n}] */
  function placeHover(name, regions, opt, rng) {
    const out = [];
    const total = regions.reduce((a, r) => a + r.n, 0);
    for (const reg of regions) {
      let got = 0;
      for (let tries = 0; tries < reg.n * 400 && got < reg.n; tries++) {
        let dx, dy, dz;
        do {
          dx = rng() * 2 - 1;
          dy = rng() * 2 - 1;
          dz = rng() * 2 - 1;
        } while (dx * dx + dy * dy + dz * dz > 1);
        const x = reg.c[0] + dx * reg.r[0], y = reg.c[1] + dy * reg.r[1], z = reg.c[2] + dz * reg.r[2];
        if (inHeroForeground(x, y, z) && !opt.allowForeground) continue;
        const c = clearAll(x, y, z);
        if (c < opt.minClear || c > opt.maxClear) continue;
        if (tooClose(out, x, y, z, opt.spacing)) continue;
        out.push([x, y, z, c]);
        got++;
      }
    }
    const n = Math.min(out.length, range[name].count);
    for (let i = 0; i < n; i++) {
      const [x, y, z, c] = out[i];
      const len = opt.len[0] + (opt.len[1] - opt.len[0]) * rng();
      const room = Math.max(0, c - len * 0.6 - 0.02);
      const kS = Math.min(opt.kSurge, Math.max(0.15, (room - 0.02) / 0.16));
      const R = Math.min(opt.R, Math.max(0.0, room - kS * 0.12));
      const wA = Math.min(opt.wander, Math.max(0.004, room * 0.15));
      let hide = [0, 0, 0], hideP = 0;
      if (opt.hideP) {
        // tuck target: just outside the closest coral surface point (in between the branches)
        const nv = state.field.nearest(x, y, z, 3);
        if (Number.isFinite(nv.d) && nv.d < 0.4) {
          const k = Math.max(0, (nv.d - len * 0.45 - 0.012) / nv.d);
          hide = [(nv.x - x) * k, (nv.y - y) * k, (nv.z - z) * k];
          hideP = opt.hideP;
        }
      }
      const variant = opt.variantP && rng() < opt.variantP ? 1 : 0;
      const L = variant ? len * 1.12 : len;
      write(name, i, [x, y, z, L], [rng(), opt.slot[0] + (opt.slot[1] - opt.slot[0]) * rng(), R, wA], [...hide, hideP], [kS, variant]);
    }
    hideRest(name, n);
    if (dbg) console.log(`reef-fish: ${name} placed ${n}/${total}`);
    return out;
  }

  function placeClowns(rng) {
    const an = env.sites && Array.isArray(env.sites.anemones) && env.sites.anemones.length ? env.sites.anemones : null;
    let homes;
    if (an) {
      homes = [...an]
        .map((a) => ({ p: a.position || new THREE.Vector3(a.x || 0, a.y || 0, a.z || 0), r: a.radius || 0.15 }))
        .sort((a, b) => b.r - a.r);
      state.anemoneFallback = false;
    } else {
      const x = -0.75, z = 0.75;
      homes = [{ p: new THREE.Vector3(x, state.T.heightAt(x, z) + 0.06, z), r: 0.16 }];
      state.anemoneFallback = true;
    }
    for (let i = 0; i < range.clown.count; i++) {
      const h = homes[i < 2 ? 0 : Math.min(1, homes.length - 1)];
      const ang = rng() * TAU, rr = h.r * (0.15 + 0.35 * rng());
      let x = h.p.x + Math.cos(ang) * rr, z = h.p.z + Math.sin(ang) * rr;
      const y = h.p.y + 0.05 + h.r * 0.3 + 0.04 * rng();
      const len = i === 2 ? 0.055 : 0.075 + 0.012 * rng();
      if (state.field && state.field.clear(x, y, z) < len * 0.72) {
        const nv = state.field.nearest(x, y, z, 2);
        if (Number.isFinite(nv.d) && nv.d > 1e-4) {
          const ux = (x - nv.x) / nv.d, uz = (z - nv.z) / nv.d;
          x += ux * 0.06;
          z += uz * 0.06;
        }
      }
      const clr = state.field ? state.field.clear(x, y, z) : 0.2;
      const dartR = Math.max(0.02, Math.min(0.11, h.r * 0.5, clr - len * 0.55));
      write("clown", i, [x, y, z, len], [rng(), 1.6 + 1.4 * rng(), dartR, 0.008], [0, -0.05 - h.r * 0.15, 0, 0.35], [0.45, 0]);
    }
    if (dbg) {
      console.log("reef-fish: clownfish homed", an ? `to ${an.length} anemone(s)` : "at the fallback spot", JSON.stringify(homes.map((h) => [+h.p.x.toFixed(2), +h.p.y.toFixed(2), +h.p.z.toFixed(2), +h.r.toFixed(2)])));
      const ob = env.sites && env.sites.obstacles;
      console.log("reef-fish: obstacles", ob ? ob.length : 0);
    }
  }

  function placeSchool(rng) {
    const n = range.fusilier.count;
    const pts = [];
    for (let tries = 0; tries < 20000 && pts.length < n; tries++) {
      let dx, dy, dz;
      do {
        dx = rng() * 2 - 1;
        dy = rng() * 2 - 1;
        dz = rng() * 2 - 1;
      } while (dx * dx + dy * dy + dz * dz > 1);
      const p = [dx * 1.5, dy * 0.36, dz * 0.55];
      if (!tooClose(pts, p[0], p[1], p[2], 0.2)) pts.push(p);
    }
    for (let i = 0; i < n; i++) {
      const p = pts[i % pts.length];
      write("fusilier", i, [p[0], p[1], p[2], 0.15 + 0.05 * rng()], [rng(), 3, 0, 0.03], [0, 0, 0, 0], [0.45, 0]);
    }
  }

  /** butterflyfish: loop around the reef at mid height with pecking stops */
  function buildButterflyPaths(rng) {
    const F = state.field;
    const loops = [];
    for (let pi = 0; pi < 2; pi++) {
      const nW = 12;
      const way = [];
      const pecks = [];
      for (let w = 0; w < nW; w++) {
        const th = (pi ? -1 : 1) * ((TAU * w) / nW) + 0.4 + pi * 1.7 + (rng() - 0.5) * 0.3;
        const y = pi ? 0.85 + 0.45 * rng() : 0.45 + 0.4 * rng();
        const dx = Math.cos(th), dz = Math.sin(th);
        // march in from open water to the reef edge (coral clearance only: obstacles are routed around below)
        let r = 3.3;
        for (; r > 0.15; r -= 0.02) if (F.clear(dx * r, y, dz * r) < 0.22) break;
        r += 0.03;
        const x = dx * r;
        let z = dz * r;
        if (inHeroForeground(x, y, z)) z = Math.min(z, 1.15);
        way.push([x, y, z]);
        if (w % 4 === 1) {
          // peck point: toward the coral until ~3 cm clearance
          let pr = r;
          for (; pr > 0.1; pr -= 0.01) if (F.clear(dx * pr, y, dz * pr) < 0.035) break;
          const nv = F.nearest(dx * pr, y, dz * pr, 2);
          if (obstacleClear(dx * pr, y, dz * pr) > 0.06)
            pecks.push({ w, p: [dx * (pr + 0.015), y, dz * (pr + 0.015)], c: Number.isFinite(nv.d) ? [nv.x, nv.y, nv.z] : [0, y, 0] });
        }
      }
      const path = resampleLoop(catmullLoop(way), 0.01);
      // keep the dense path clear (push outward from the reef axis)
      for (let k = 0; k < path.n; k++) {
        let x = path.pts[k * 3], z = path.pts[k * 3 + 2];
        const y = path.pts[k * 3 + 1];
        for (let it = 0; it < 30 && clearAll(x, y, z) < 0.12; it++) {
          const l = Math.hypot(x, z) || 1;
          x += (x / l) * 0.02;
          z += (z / l) * 0.02;
        }
        path.pts[k * 3] = x;
        path.pts[k * 3 + 2] = z;
      }
      const dwell = new Map();
      const peckAt = new Map();
      for (const pk of pecks) {
        let best = 0, bd = Infinity;
        for (let k = 0; k < path.n; k++) {
          const d = (path.pts[k * 3] - way[pk.w][0]) ** 2 + (path.pts[k * 3 + 1] - way[pk.w][1]) ** 2 + (path.pts[k * 3 + 2] - way[pk.w][2]) ** 2;
          if (d < bd) {
            bd = d;
            best = k;
          }
        }
        dwell.set(best, 2.6 + 1.6 * rng());
        peckAt.set(best, pk);
      }
      const v0 = 0.1 + 0.02 * rng();
      const peckKs = [...dwell.keys()];
      const speed = (k) => {
        let g = 1 + 0.25 * Math.sin((k * path.ds * TAU) / 2.7 + pi);
        for (const pk of peckKs) {
          const dk = Math.min(Math.abs(k - pk), path.n - Math.abs(k - pk)) * path.ds;
          g *= 1 - 0.7 * Math.exp(-((dk / 0.18) ** 2));
        }
        return v0 * g;
      };
      const P = timePath(path, speed, dwell, { f0: 1.1, f1: 7.0, fDwell: 1.0 });
      P.peckAt = peckAt;
      loops.push(P);
    }
    return loops;
  }

  /** moon wrasse: weaving loops 0.1-0.3 m over the low and mid corals */
  function buildWrassePaths(rng) {
    const T = state.T;
    const loops = [];
    for (let wi = 0; wi < 3; wi++) {
      const pts = [];
      for (let tries = 0; tries < 6000 && pts.length < 9; tries++) {
        const x = -2.1 + 4.2 * rng(), z = -1.0 + 2.0 * rng();
        const top = T.coralTop(x, z);
        if (top < 0.3 || top > 1.62) continue;
        const y = top + 0.12 + 0.18 * rng();
        if (inHeroForeground(x, y, z) && y < 1.2) continue;
        if (clearAll(x, y, z) < 0.09) continue;
        if (tooClose(pts, x, y, z, 0.55)) continue;
        pts.push([x, y, z]);
      }
      if (pts.length < 4) continue;
      const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length, cz = pts.reduce((a, p) => a + p[2], 0) / pts.length;
      pts.sort((a, b) => Math.atan2(a[2] - cz, a[0] - cx) - Math.atan2(b[2] - cz, b[0] - cx));
      if (wi === 1) pts.reverse();
      const path = resampleLoop(catmullLoop(pts), 0.01);
      for (let k = 0; k < path.n; k++) {
        const x = path.pts[k * 3], z = path.pts[k * 3 + 2];
        let y = path.pts[k * 3 + 1];
        for (let it = 0; it < 40 && clearAll(x, y, z) < 0.08; it++) y += 0.015;
        path.pts[k * 3 + 1] = y;
      }
      // smooth the vertical corrections (never lowering a sample)
      const ys = new Float32Array(path.n);
      for (let pass = 0; pass < 3; pass++) {
        for (let k = 0; k < path.n; k++) {
          let s = 0;
          for (let o = -8; o <= 8; o++) s += path.pts[((((k + o) % path.n) + path.n) % path.n) * 3 + 1];
          ys[k] = Math.max(path.pts[k * 3 + 1], s / 17);
        }
        for (let k = 0; k < path.n; k++) path.pts[k * 3 + 1] = ys[k];
      }
      const v0 = 0.2 + 0.05 * rng();
      const bursts = [rng(), rng()].map((r) => Math.floor(r * path.n));
      const speed = (k) => {
        let g = 1 + 0.2 * Math.sin((k * path.ds * TAU) / 1.9 + wi);
        for (const b of bursts) {
          const dk = Math.min(Math.abs(k - b), path.n - Math.abs(k - b)) * path.ds;
          g += 1.1 * Math.exp(-((dk / 0.3) ** 2));
        }
        return v0 * g;
      };
      loops.push(timePath(path, speed, new Map(), { f0: 1.2, f1: 4.0, fDwell: 1.0 }));
    }
    return loops;
  }

  function placeAll() {
    const t0 = performance.now();
    // anthias: loose cloud in the open water above the bouquet, in front of the tall Acropora
    placeHover(
      "anthias",
      [
        { c: [0.0, 2.0, 0.0], r: [1.5, 0.6, 0.8], n: 72 },
        { c: [-0.1, 1.72, 0.5], r: [1.0, 0.18, 0.36], n: 28 },
        { c: [0.1, 2.75, -0.1], r: [1.2, 0.35, 0.7], n: 10 }, // a few higher up (silhouettes in the Look Up view)
      ],
      { minClear: 0.13, maxClear: 0.85, spacing: 0.1, len: [0.062, 0.088], R: 0.17, wander: 0.018, kSurge: 0.8, slot: [2.6, 6.0], variantP: 0.1 },
      env.makeRng(1101)
    );
    // chromis: low aggregations over the bouquet tips and the two side Acropora heads
    placeHover(
      "chromis",
      [
        { c: [-0.05, 1.7, 0.3], r: [0.7, 0.24, 0.42], n: 36 },
        { c: [-1.72, 2.12, -0.42], r: [0.38, 0.22, 0.32], n: 17 },
        { c: [1.74, 2.08, -0.45], r: [0.38, 0.22, 0.32], n: 17 },
      ],
      { minClear: 0.06, maxClear: 0.42, spacing: 0.075, len: [0.05, 0.07], R: 0.1, wander: 0.012, kSurge: 0.6, slot: [2.2, 5.0], hideP: 0.14 },
      env.makeRng(2203)
    );
    // humbug damsels: groups on three small coral heads
    placeHover(
      "humbug",
      [
        { c: [-1.56, 0.8, 0.66], r: [0.3, 0.3, 0.3], n: 6 },
        { c: [1.45, 0.76, 0.58], r: [0.3, 0.3, 0.3], n: 5 },
        { c: [-0.98, 0.95, 0.6], r: [0.28, 0.25, 0.28], n: 5 },
      ],
      { minClear: 0.035, maxClear: 0.16, spacing: 0.06, len: [0.04, 0.06], R: 0.12, wander: 0.01, kSurge: 0.35, slot: [1.6, 3.6], hideP: 0.42, allowForeground: true },
      env.makeRng(3307)
    );
    placeClowns(env.makeRng(4409));
    placeSchool(env.makeRng(5501));
    state.roam.butterfly = buildButterflyPaths(env.makeRng(6607));
    state.roam.wrasse = buildWrassePaths(env.makeRng(7703));
    if (dbg) {
      const desc = (P) => {
        const q = (k) => [0, 1, 2].map((c) => +P.pts[(Math.floor(k) % P.n) * 3 + c].toFixed(2));
        return `L=${P.L.toFixed(1)}m T=${P.period.toFixed(0)}s ${JSON.stringify([q(0), q(P.n / 4), q(P.n / 2), q((3 * P.n) / 4)])}`;
      };
      state.roam.butterfly.forEach((P, i) => console.log(`reef-fish: butterfly loop ${i}: ${desc(P)}`));
      state.roam.wrasse.forEach((P, i) => console.log(`reef-fish: wrasse loop ${i}: ${desc(P)}`));
    }
    // roamer constants (positions/headings are written every frame)
    for (let i = 0; i < range.butterfly.count; i++) {
      const ok = i < state.roam.butterfly.length * 2;
      write("butterfly", i, [0, -40, 0, ok ? (i % 2 ? 0.128 : 0.142) : 0], [(i * 0.618 + 0.11) % 1, 3, 0, 0.02], [1, 0, 0, 0], [0.22, 0]);
    }
    for (let i = 0; i < range.wrasse.count; i++) {
      const ok = i < state.roam.wrasse.length;
      write("wrasse", i, [0, -40, 0, ok ? [0.19, 0.165, 0.175][i % 3] : 0], [(i * 0.618 + 0.37) % 1, 3, 0, 0.015], [1, 0, 0, 0], [0.22, 0]);
    }
    fullUpload = true;
    if (lineup) applyLineup();
    state.placeMs = Math.round(performance.now() - t0);
    if (dbg) console.log(`reef-fish: placement ${state.placeMs} ms`);
  }

  /** dev: ?fishLineup=1 lines one fish of every species up in front of the hero camera (static, side-on) */
  function applyLineup() {
    for (const n of ORDER) hideRest(n, 0);
    const row = [["anthias", 0, 0.08], ["anthias", 1, 0.09], ["chromis", 0, 0.065], ["humbug", 0, 0.055], ["clown", 0, 0.08], ["butterfly", 0, 0.14], ["wrasse", 0, 0.18], ["fusilier", 0, 0.18]];
    let x = -0.62;
    row.forEach(([name, variant, len], k) => {
      x += len * 0.5 + 0.05;
      const p = [x, 1.2 + (k % 2) * 0.03, 1.9];
      const i = name === "anthias" ? variant : 0;
      if (SPECIES[name].mode === 1) write(name, i, [...p, len], [0.3, 3, 0, 0.02], [1, 0, 0, 0.25], [0, 0]);
      else if (SPECIES[name].mode === 2) {
        write(name, i, [0, 0, 0, len], [0.3, 3, 0, 0], [0, 0, 0, 0], [0, 0]);
        M.uSchoolPos.value.set(...p);
        M.uSchoolFwd.value.set(1, 0, 0);
      } else write(name, i, [...p, len], [0.3, 30, 0, 0], [0, 0, 0, 0], [0, variant]);
      x += len * 0.5;
    });
    M.uFlow.value.set(1, 0);
    fullUpload = true;
  }

  function sitesKey() {
    const s = env.sites || {};
    const an = Array.isArray(s.anemones) ? s.anemones.length : 0;
    const ob = Array.isArray(s.obstacles) ? s.obstacles.length : 0;
    return `${an}|${ob}`;
  }

  env.terrain
    .then((T) => {
      try {
        const t0 = performance.now();
        state.T = T;
        state.field = buildClearance(T);
        state.fieldMs = Math.round(performance.now() - t0);
        if (dbg) console.log(`reef-fish: clearance field ${state.fieldMs} ms, ${state.field.total} verts, ${state.field.inside} enclosed voxels`);
        state.sitesKey = sitesKey();
        placeAll();
        state.ready = true;
      } catch (e) {
        console.error("reef-fish: placement failed", e);
      }
    })
    .catch((e) => console.error("reef-fish: terrain unavailable", e));

  // -------------------------------------------------------------------------------------------
  // per-frame CPU work: 7 roamers + one school transform + site polling
  // -------------------------------------------------------------------------------------------
  const ev = {};
  const ev2 = {};
  const putRoamer = (name, i, x, y, z, hx, hy, hz, phase, amp) => {
    const o = off(name, i);
    inst[o] = x;
    inst[o + 1] = y;
    inst[o + 2] = z;
    inst[o + 7] = amp;
    const hl = Math.hypot(hx, hy, hz) || 1;
    inst[o + 8] = hx / hl;
    inst[o + 9] = hy / hl;
    inst[o + 10] = hz / hl;
    inst[o + 11] = phase;
  };
  function updateRoamers(t) {
    // butterflyfish pairs
    state.roam.butterfly.forEach((P, pi) => {
      for (let m = 0; m < 2; m++) {
        const i = pi * 2 + m;
        const e = evalPath(P, t, pi * 37.0, m ? -0.25 : 0, m ? ev2 : ev);
        let x = e[0], y = e[1], z = e[2];
        let hx = e[3], hy = e[4] * 0.6, hz = e[5];
        const lx = -hz, lz = hx;
        const ll = Math.hypot(lx, lz) || 1;
        if (m) {
          // partner: a little to the side and above
          x += (lx / ll) * 0.06;
          z += (lz / ll) * 0.06;
          y += 0.035;
        }
        const wv = Math.sin(t * 1.3 + i * 2.1) * 0.025;
        x += (lx / ll) * wv;
        z += (lz / ll) * wv;
        y += Math.sin(t * 0.9 + i) * 0.015;
        // pecking: lean in toward the coral and nod
        const pk = !m && P.peckAt.get(e.k);
        if (pk && e.dwellW > 0) {
          const w = e.dwellW;
          const cx = pk.c[0] - pk.p[0], cy = pk.c[1] - pk.p[1], cz = pk.c[2] - pk.p[2];
          const cl = Math.hypot(cx, cy, cz) || 1;
          const nod = Math.max(0, Math.sin(e.dwellT * TAU * 1.6)) * 0.008;
          x += (pk.p[0] - x) * w + (cx / cl) * nod * w;
          y += (pk.p[1] - y) * w + (cy / cl) * nod * w;
          z += (pk.p[2] - z) * w + (cz / cl) * nod * w;
          hx += (cx / cl - hx) * w;
          hy += (cy / cl - 0.2 - hy) * w * 0.7;
          hz += (cz / cl - hz) * w;
        }
        putRoamer("butterfly", i, x, y, z, hx, hy, hz, (e.phase + m * 0.37) % 1, 0.018 + Math.min(0.06, e.speed * 0.35));
      }
    });
    // wrasse
    state.roam.wrasse.forEach((P, i) => {
      const e = evalPath(P, t, i * 23.0, 0, ev);
      const lx = -e[5], lz = e[3];
      const ll = Math.hypot(lx, lz) || 1;
      const wv = Math.sin(t * 2.1 + i * 1.7) * 0.02;
      putRoamer("wrasse", i, e[0] + (lx / ll) * wv, e[1] + Math.sin(t * 1.1 + i) * 0.012, e[2] + (lz / ll) * wv, e[3], e[4] * 0.6, e[5], e.phase, 0.012 + Math.max(0, e.speed - 0.26) * 0.25);
    });
  }

  // fusilier school: one pass every 40 s on a chord 6-9 m out, fully fogged at both ends
  const _f = new THREE.Vector3();
  function updateSchool(t) {
    const PERIOD = 40, PASS = 26, OFFSET = 1.0;
    const tl = t + OFFSET;
    const cyc = Math.floor(tl / PERIOD);
    const tau = tl - cyc * PERIOD;
    state.schoolOn = tau < PASS && env.mood !== "night" && shown("fusilier");
    if (!state.schoolOn) return;
    const r = env.makeRng(9001 + cyc * 7919);
    let ang = r() * TAU;
    let d = 6.2 + 2.6 * r();
    let y = 1.5 + 1.1 * r();
    let sgn = r() < 0.5 ? -1 : 1;
    if (cyc === 0) {
      // first pass crosses the haze on the horizon band right of the reef in the wide framing
      ang = -0.8;
      d = 6.6;
      y = 2.3;
      sgn = 1;
    }
    const nx = Math.cos(ang), nz = Math.sin(ang);
    const cx = -nz * sgn, cz = nx * sgn;
    const half = Math.sqrt(Math.max(1, 12.8 * 12.8 - d * d));
    const v = (2 * half) / PASS;
    const along = -half + v * tau;
    const sw = Math.sin((TAU * tau) / 9.0) * 0.6, swd = Math.cos((TAU * tau) / 9.0) * 0.6 * (TAU / 9.0);
    M.uSchoolPos.value.set(nx * (d + sw) + cx * along, y + Math.sin((TAU * tau) / 7.0) * 0.25, nz * (d + sw) + cz * along);
    _f.set(cx * v + nx * swd, Math.cos((TAU * tau) / 7.0) * 0.25 * (TAU / 7.0), cz * v + nz * swd).normalize();
    M.uSchoolFwd.value.copy(_f);
  }

  function pollSites() {
    const key = sitesKey();
    if (key === state.sitesKey) return;
    state.sitesKey = key;
    try {
      placeAll();
    } catch (e) {
      console.error("reef-fish: re-placement failed", e);
    }
  }

  let dumped = !/[?&]fishDump=1/.test(search);
  function update(t) {
    M.uT.value = t % 3600;
    M.uNight.value = env.mood === "night" ? 1 : 0;
    if (!state.ready) return;
    if (!dumped && env.renderer.debug && env.renderer.debug.getShaderAsync) {
      dumped = true;
      env.renderer.debug
        .getShaderAsync(env.scene, env.camera, mesh)
        .then((s) => console.log(`reef-fish shader: vertex ${s.vertexShader.length} B, fragment ${s.fragmentShader.length} B\n=== VERTEX ===\n${s.vertexShader}\n=== FRAGMENT ===\n${s.fragmentShader}`))
        .catch((e) => console.error("reef-fish: shader dump failed", e));
    }
    pollSites();
    if (lineup) {
      state.schoolOn = true;
    } else {
      updateRoamers(t);
      updateSchool(t);
    }
    // fusiliers are the last instances: skip their vertex work while the school is out of sight
    mesh.count = state.schoolOn ? NI : NI - COUNTS.fusilier;
    if (fullUpload) {
      ibuf.clearUpdateRanges();
      fullUpload = false;
    } else {
      ibuf.addUpdateRange(0, ROAMERS * 16);
    }
    ibuf.needsUpdate = true;
  }

  return {
    group,
    mesh,
    material: M.mat,
    state,
    update,
    warmTo(t) {
      update(t);
    },
  };
}
