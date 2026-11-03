// -----------------------------------------------------------------------------
// reef-flora.js — surge-driven soft life (owned by the flora builder)
//
// One coherent water motion drives everything (expert P6 / art item 10): every organism bends with
// the shared surge field
//     disp(s) = env.surge(baseXZ, t, tau*s) * g * s^2        (s = 0 at the holdfast .. 1 at the tip)
// with y lowered so its length stays ~constant, plus a small per-instance flutter on top.
//
// Families (one material = one pipeline each; all GPU-animated, no per-frame CPU work):
//   tubes  sea whips (Junceella/Ellisella clusters) + bubble-tip anemone tentacles (Entacmaea):
//          instanced tubes with an analytic bend frame and a blunt-taper or bubble-tip profile
//          (2 draws of one material: 6x14 whip template, 5x9 bulb-dense tentacle template)
//   fans   gorgonian sea fans (Melithaea/Annella-like): cupped polar grids with a procedural
//          anastomosing net (Voronoi borders in arc/radius space) drawn with alpha-to-coverage,
//          oriented perpendicular to the surge and rocking a few degrees with a lag (1 draw)
//   grass  Thalassia/Cymodocea seagrass meadows in the middle ground, ~7k strap blades (1 draw)
//   leaf   merged, statically placed geometry that still sways per vertex: Halimeda and Padina
//          clumps at the reef base, the anemones' oral discs, Sarcophyton leather corals (1 draw)
// Shading: env.litProp (wrap key x baked sun shadow x caustics, depth light, sky/fill, rim, back-lit
// transmission) + env.fogTerms evaluated per vertex (like the coral shader) + env.output (grade,
// dither, encode) == env.finish with interpolated fog, so the flora lives in the same water as the
// Houdini corals. Thin geometry is double-sided with view-facing normals.
//
// Placement waits for env.terrain (seabed height, coral footprint distance). It publishes
// env.sites.anemones (clownfish homes; created synchronously and refined in place) and pushes coarse
// fish-avoidance capsules to env.sites.obstacles.
// Contract: see js/reef-env.js. createReefFlora(env) -> { group, update(t), warmTo(t) }.
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
  instanceIndex,
  hash,
  cameraPosition,
  cameraProjectionMatrix,
  cameraViewMatrix,
  sin,
  cos,
  mix,
  smoothstep,
  clamp,
  normalize,
  cross,
  dot,
  length,
  max,
  min,
  abs,
  fract,
  floor,
  step,
  sqrt,
  exp,
  pow,
  fwidth,
  uniform,
  texture,
} from "three/tsl";

const TAU = Math.PI * 2;
const rgb = (hex) => [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];

// ---- art direction: Indo-Pacific species, display-referred pastel albedos (like the coral attributes) ----
const PAL = {
  fans: [rgb(0xc2649c), rgb(0xd97e9a), rgb(0x9a6ccb), rgb(0xcc78a8), rgb(0xae62b0)], // dusty magenta / rose / violet
  whips: [
    [rgb(0xf6955f), rgb(0xfcc7a2)], // apricot
    [rgb(0xf1d77e), rgb(0xfaefc4)], // lemon
    [rgb(0xea8499), rgb(0xf7c0cc)], // rose
  ],
  anemone: { stalk: rgb(0xfbd3b6), tip: rgb(0xc59ee6), disc: rgb(0xeeb49c) },
  halimeda: rgb(0x9ad8ac),
  padina: rgb(0xeee2cf),
  leather: rgb(0xe6d2b8),
};

// Clownfish homes: two clonal bubble-tip anemones nestled at the front-left base of the bouquet
// (free sand, env.terrain.coralDist ~0.1-0.15 m). Published synchronously so the fish module can read
// them at any time; y (and the exact centre) are refined in place once the terrain is known.
const ANEMONES = [
  { x: -0.6, z: 0.8, y: 0.49, R: 0.135, n: 215, seed: 11, tip: rgb(0xc59ee6) },
  { x: -0.88, z: 0.965, y: 0.42, R: 0.155, n: 240, seed: 23, tip: rgb(0xd09ad8) },
];

// Sea fans (behind / beside the bouquet for silhouette against the water, never in the hero foreground)
const FANS = [
  { x: 2.9, z: -0.75, H: 0.78, aspect: 0.95, col: 0 }, // right flank (wide view)
  { x: 3.2, z: -1.85, H: 0.92, aspect: 1.05, col: 2 }, // its taller partner behind (violet)
  { x: -3.3, z: -0.1, H: 0.95, aspect: 1.05, col: 1 }, // left flank (orbit; clear of the barrel sponge)
  { x: 1.9, z: -2.3, H: 1.05, aspect: 1.0, col: 3 }, // back right (hero sliver, orbit)
  { x: -1.75, z: -1.95, H: 1.0, aspect: 0.95, col: 4 }, // back left (macro + orbit)
];

// Sea-whip clusters (sides and back of the reef; 5-9 whips of 0.6-1.4 m from one holdfast patch)
const WHIPS = [
  { x: -2.9, z: -0.1, n: 7, col: 1 },
  { x: -2.35, z: -1.45, n: 8, col: 0 },
  { x: -1.0, z: -1.6, n: 6, col: 2 },
  { x: 1.0, z: -1.85, n: 9, col: 0 },
  { x: 1.7, z: -1.6, n: 7, col: 1 },
  { x: 2.3, z: -1.15, n: 6, col: 2 },
  { x: 2.95, z: 0.45, n: 5, col: 0 },
  { x: 3.75, z: -0.85, n: 8, col: 1 },
  { x: -3.35, z: -0.85, n: 7, col: 0 },
];

// Seagrass meadows in the middle ground (3.5-8 m out), blob = union of soft circles
const MEADOWS = [
  { n: 2100, blobs: [[-3.4, -2.7, 1.1], [-2.6, -3.3, 0.9], [-4.1, -3.4, 0.9], [-3.3, -3.9, 0.7]] },
  { n: 2200, blobs: [[3.4, -3.0, 1.1], [4.2, -2.5, 0.85], [2.8, -3.8, 0.9], [4.0, -3.7, 0.8]] },
  { n: 1350, blobs: [[-0.6, -5.2, 1.0], [0.5, -5.5, 0.85], [-1.5, -5.6, 0.75], [0.0, -4.75, 0.55]] }, // behind the reef (orbit)
  { n: 650, blobs: [[5.0, 1.8, 0.7], [5.5, 1.3, 0.5], [4.6, 2.3, 0.45]] },
];

// Sarcophyton leather corals (optional life; slow "breathing" caps)
const LEATHERS = [
  { x: 2.38, z: 0.78, R: 0.15, h: 0.1 }, // right flank beside the brain pillow (wide view)
  { x: -2.55, z: 0.3, R: 0.14, h: 0.11 }, // left flank (wide view)
  { x: -1.85, z: -1.65, R: 0.2, h: 0.14 }, // behind the left pod (orbit)
];

export function createReefFlora(env) {
  const group = new THREE.Group();
  group.name = "flora";
  env.groups.flora = group;
  env.scene.add(group);

  const t = env.u.time;
  const uNight = uniform(0.0); // 1 in the Bio-Night mood (fluorescent anemone tips)
  env.sites.anemones = ANEMONES.map((a) => ({ position: new THREE.Vector3(a.x, a.y, a.z), radius: a.R + 0.05 }));
  const anemoneSites = env.sites.anemones;

  // ---------------------------------------------------------------------------
  // Shared TSL helpers
  // ---------------------------------------------------------------------------
  const clipPos = (posW) => cameraProjectionMatrix.mul(cameraViewMatrix.mul(vec4(posW, 1.0)));
  /** limit a vec2 node's length */
  const clampLen = (v, maxLen) => v.mul(min(float(1.0), maxLen.div(max(length(v), 1e-5))));
  /** small per-instance turbulence (0.7-1.5 Hz) riding on the coherent surge; seed in [0,1) */
  const flutter = (seed, s, amp) => {
    const f = float(0.7 * TAU).add(fract(seed.mul(7.31)).mul(0.8 * TAU));
    const ph = seed.mul(61.7);
    return vec2(sin(t.mul(f).add(ph).sub(s.mul(2.4))), cos(t.mul(f.mul(1.37)).add(ph.mul(1.9)).sub(s.mul(3.1)))).mul(amp);
  };
  /** unit normal turned toward the viewer (thin double-sided tissue) */
  const faceViewer = (n, posW) => {
    const nn = normalize(n);
    return nn.mul(step(0.0, dot(nn, cameraPosition.sub(posW))).mul(2.0).sub(1.0));
  };
  /** sin-free 2D hash (Dave Hoskins) */
  const hash22 = (p) => {
    const p3 = fract(vec3(p.x, p.y, p.x).mul(vec3(0.1031, 0.103, 0.0973))).toVar();
    const q = p3.add(dot(p3, vec3(p3.y, p3.z, p3.x).add(33.33)));
    return fract(vec2(q.x.add(q.y).mul(q.z), q.x.add(q.z).mul(q.y)));
  };
  // Voronoi net lookup: a tileable border-distance texture baked once on the CPU (2 fetches per fan pixel
  // instead of two unrolled 3x3 Voronoi searches). Returns vec2(distance to the nearest cell border in
  // cell units, random id of the cell pair) for p in cell units.
  const tNet0 = performance.now();
  const net = makeNetTexture(env.makeRng(8128));
  const netBakeMs = Math.round(performance.now() - tNet0);
  const netLookup = (p) => {
    const smp = texture(net.tex, p.div(net.cells));
    return vec2(smp.x.mul(net.eMax), smp.y);
  };
  /**
   * Fog per vertex, like the coral shader (the water column varies slowly over a blade, a tube ring or a
   * fan cell): env.fogTerms evaluated in the vertex stage and packed into two varyings.
   */
  const fogVary = (posW) => {
    const f = env.fogTerms(posW);
    const c = f.color.toVar();
    return { a: varying(vec4(f.T, c.x)), b: varying(vec2(c.y, c.z)) };
  };
  /** shaded -> fogged with the per-vertex terms -> grade + encode (== env.finish with interpolated fog) */
  const finishV = (shaded, fv, alpha = 1.0) => {
    const T = fv.a.xyz;
    const c = vec3(fv.a.w, fv.b.x, fv.b.y);
    return env.output(shaded.mul(T).add(c.mul(vec3(1.0).sub(T))), alpha);
  };
  const instAttr = (geo, name, cap) => {
    const a = new THREE.InstancedBufferAttribute(new Float32Array(cap * 4), 4);
    geo.setAttribute(name, a);
    return a;
  };
  const setV4 = (attr, i, a, b, c, d) => {
    const o = i * 4;
    const arr = attr.array;
    arr[o] = a;
    arr[o + 1] = b;
    arr[o + 2] = c;
    arr[o + 3] = d;
  };

  // ===========================================================================
  // 1) Tubes: sea whips + anemone tentacles (one material, two instanced draws)
  // ===========================================================================
  // two templates, one material/pipeline: whips 6 sides x 14 rings, tentacles 5 sides x 9 rings (bulb-dense)
  const WHIP_CAP = 96;
  const TENT_CAP = 560;
  const TUBE_RINGS = 14;
  const TUBE_S_LAST = 1 - Math.pow(1 / TUBE_RINGS, 1.45); // s of the last full whip ring (see makeTubeGeometry)
  const whipGeo = makeTubeGeometry(TUBE_RINGS, 6, 1.45);
  const tentGeo = makeTubeGeometry(9, 5, 1.6);
  const tubeAttrs = (geo, cap) => ({
    base: instAttr(geo, "tBase", cap), // xyz holdfast, length L
    axis: instAttr(geo, "tAxis", cap), // rest axis (unit), radius r0
    bend: instAttr(geo, "tBend", cap), // rest curve offset at the tip (x, z), surge gain g, lag tau
    colA: instAttr(geo, "tColA", cap), // base colour, bulb (0 whip .. 1 bubble tip)
    colB: instAttr(geo, "tColB", cap), // tip colour, holdfast AO
  });
  const whipA = tubeAttrs(whipGeo, WHIP_CAP);
  const tentA = tubeAttrs(tentGeo, TENT_CAP);
  const tubeMat = new THREE.MeshBasicNodeMaterial({ name: "flora_tube", side: THREE.FrontSide });
  {
    const iBase = attribute("tBase", "vec4");
    const iAxis = attribute("tAxis", "vec4");
    const iBend = attribute("tBend", "vec4");
    const iColA = attribute("tColA", "vec4");
    const iColB = attribute("tColB", "vec4");
    const base = iBase.xyz;
    const A = iAxis.xyz;
    const r0 = iAxis.w;
    const bulb = iColA.w;
    // both whips and bubble-tip tentacles use the template's tip-dense ring spacing directly
    void TUBE_S_LAST;
    const s = positionGeometry.x.toVar();
    const seed = hash(instanceIndex).toVar();
    // anemone tentacles "breathe" (slow length pulse); whips are stiff along their axis
    const L = iBase.w.mul(float(1.0).add(sin(t.mul(0.55).add(seed.mul(40.0))).mul(0.045).mul(bulb))).toVar();
    const W = env.surge(vec2(base.x, base.z), t, iBend.w.mul(s));
    const fl = flutter(seed, s, L.mul(mix(float(0.012), float(0.07), bulb)));
    const D = clampLen(W.mul(iBend.z).add(fl), L.mul(0.6)).toVar();
    const C = vec3(iBend.x, 0.0, iBend.y);
    const off = C.add(vec3(D.x, 0.0, D.y)).toVar();
    const s2 = s.mul(s);
    const oo = dot(off, off);
    const P = base.add(A.mul(L.mul(s))).add(off.mul(s2)).sub(vec3(0.0, oo.mul(s2).mul(s).div(L.mul(2.0).add(1e-4)), 0.0));
    const dP = A.mul(L).add(off.mul(s.mul(2.0))).sub(vec3(0.0, oo.mul(s2).mul(1.5).div(L.add(1e-4)), 0.0)).add(vec3(0.0, 1e-6, 0.0)).toVar();
    const T = normalize(dP).toVar();
    const Uref = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), step(0.9, abs(T.z)));
    const B1 = normalize(cross(T, Uref)).toVar();
    const B2 = cross(T, B1);
    const dir = B1.mul(positionGeometry.y).add(B2.mul(positionGeometry.z)).toVar();
    // radius profile: whips taper organically to a rounded paraboloid tip; tentacles have a thin stalk and a swollen bulb
    const taperK = float(0.40).mul(float(1.0).sub(bulb));
    const profile = (sv) => {
      const sc = clamp(sv, 0.0, 1.0);
      const bx = sc.sub(0.79).div(0.15);
      const shape = mix(float(1.0).sub(sc.mul(taperK)), float(0.5).add(exp(bx.mul(bx).negate()).mul(0.5)), bulb);
      const x = max(sc.sub(0.8), 0.0).div(0.2);
      const xW = max(sc.sub(0.94), 0.0).div(0.06);
      const cap = mix(sqrt(max(float(1.0).sub(xW.mul(xW)), 0.0)), sqrt(max(float(1.0).sub(x.mul(x)), 0.0)), bulb);
      return r0.mul(shape).mul(cap);
    };
    const r = profile(s);
    const dr = profile(s.add(0.012)).sub(profile(s.sub(0.012))).div(0.024);
    const posW = P.add(dir.mul(r)).toVar();
    const nW = normalize(dir.sub(T.mul(dr.div(length(dP)))));
    tubeMat.vertexNode = clipPos(posW);
    const vPos = varying(vec4(posW, s));
    const fogT = fogVary(posW);
    const vNrm = varying(vec4(nW, seed));
    const vColA = varying(iColA);
    const vColB = varying(iColB);
    tubeMat.fragmentNode = Fn(() => {
      const pW = vPos.xyz;
      const sv = vPos.w;
      const N = normalize(vNrm.xyz);
      const colA = vColA.xyz;
      const bl = vColA.w;
      const colB = vColB.xyz;
      // whips: velvet polyp coat, paler growing tip; tentacles: peach stalk, bulb, pale ring, lilac tip
      const whip = mix(colA, colB, smoothstep(0.68, 1.0, sv));
      const ring = smoothstep(0.84, 0.875, sv).mul(float(1.0).sub(smoothstep(0.9, 0.93, sv)));
      const tent = mix(mix(colA, colA.mul(vec3(1.0, 0.94, 0.94)), smoothstep(0.5, 0.78, sv)), colB, smoothstep(0.935, 0.975, sv));
      const alb = mix(whip, mix(tent, vec3(1.0, 0.97, 0.95), ring.mul(0.6)), bl);
      const aoW = mix(float(0.55), float(1.0), smoothstep(0.0, 0.3, sv));
      const aoT = mix(float(0.62), float(1.0), smoothstep(0.05, 0.6, sv));
      const ao = mix(aoW, aoT, bl).mul(vColB.w);
      const shaded = env.litProp({
        albedo: alb,
        N,
        posW: pW,
        ao,
        wrap: 0.55,
        trans: mix(float(0.25), float(0.55), bl),
        causticGain: 0.9,
        rim: 0.8,
        sheen: mix(float(0.4), float(0.3), bl),
      });
      // bubble-tip tentacles fluoresce faintly (keeps the cushion luminous in the canopy shade); in
      // Bio-Night the bulbs glow GFP-green with lilac tips under the blue moonlight
      const glow = mix(vec3(0.3, 0.95, 0.62), colB.mul(1.25), smoothstep(0.9, 0.97, sv)).mul(smoothstep(0.55, 0.85, sv)).mul(uNight.mul(0.45));
      return finishV(shaded.add(alb.mul(bl).mul(0.07)).add(glow.mul(bl)), fogT);
    })();
  }
  const whips = new THREE.Mesh(whipGeo, tubeMat);
  whips.name = "flora_whips";
  whipGeo.instanceCount = 0; // filled after placement (no instances drawn before)
  whips.frustumCulled = false;
  group.add(whips);
  const tents = new THREE.Mesh(tentGeo, tubeMat);
  tents.name = "flora_tentacles";
  tentGeo.instanceCount = 0;
  tents.frustumCulled = false;
  group.add(tents);

  // ===========================================================================
  // 2) Gorgonian sea fans (alpha-to-coverage lattice, one instanced draw)
  // ===========================================================================
  const FAN_CAP = 6;
  const THMAX = 1.05; // half-angle of the fan sector (rad)
  const fanGeo = makePolarGrid(28, 18);
  const fanA = {
    base: instAttr(fanGeo, "fBase", FAN_CAP), // holdfast xyz, height H
    frame: instAttr(fanGeo, "fFrame", FAN_CAP), // fan-plane right vector (x, z), aspect, cup
    col: instAttr(fanGeo, "fCol", FAN_CAP), // albedo, seed
    sway: instAttr(fanGeo, "fSway", FAN_CAP), // gain g, lag tau, lean, outline phase
  };
  const fanMat = new THREE.MeshBasicNodeMaterial({
    name: "flora_fan",
    side: THREE.DoubleSide,
    alphaToCoverage: true,
    transparent: false,
  });
  {
    const iBase = attribute("fBase", "vec4");
    const iFrame = attribute("fFrame", "vec4");
    const iCol = attribute("fCol", "vec4");
    const iSway = attribute("fSway", "vec4");
    const H = iBase.w;
    const seed = iCol.w;
    const Rv = vec3(iFrame.x, 0.0, iFrame.y);
    const Nf = vec3(iFrame.y, 0.0, iFrame.x.negate()); // == the surge direction (fan faces the water motion)
    const rn = positionGeometry.y;
    // the outer primary branches curve outward as they grow: narrow holdfast, broad rounded crown
    const th = positionGeometry.x.mul(THMAX).mul(float(0.55).add(smoothstep(0.0, 0.7, rn).mul(0.45))).toVar();
    const ph = iSway.w;
    const lobe = float(1.0).add(sin(th.mul(3.1).add(ph)).mul(0.07)).add(sin(th.mul(6.3).add(ph.mul(2.1))).mul(0.04));
    const rho = rn.mul(H).mul(pow(cos(th.mul(0.82)), 0.45)).mul(lobe).toVar();
    const uL = sin(th).mul(rho).mul(iFrame.z).toVar();
    const vL = cos(th).mul(rho).toVar();
    const sH = vL.div(H.add(1e-4)).toVar();
    const cup = iFrame.w;
    const wave = sin(uL.mul(9.0).div(H.add(1e-4)).add(seed.mul(6.0))).mul(0.016).mul(H).mul(sH);
    const nOff = cup.mul(uL).mul(uL).div(H.add(1e-4)).add(wave).add(iSway.z.mul(vL));
    // rocking: the whole fan sways with the water, the top lagging ~0.4 s behind the holdfast
    const W = env.surge(vec2(iBase.x, iBase.z), t, iSway.y.mul(sH));
    const D = clampLen(W.mul(iSway.x).add(flutter(seed, sH, H.mul(0.004))), H.mul(0.3)).toVar();
    const D3 = vec3(D.x, 0.0, D.y);
    const dd = dot(D, D);
    const posW = iBase.xyz
      .add(Rv.mul(uL))
      .add(vec3(0.0, vL, 0.0))
      .add(Nf.mul(nOff))
      .add(D3.mul(sH.mul(sH)))
      .sub(vec3(0.0, dd.mul(sH.mul(sH).mul(sH)).div(H.mul(2.0).add(1e-4)), 0.0))
      .toVar();
    const dPdu = Rv.add(Nf.mul(cup.mul(uL).mul(2.0).div(H.add(1e-4))));
    const dPdv = vec3(0.0, 1.0, 0.0).add(Nf.mul(iSway.z)).add(D3.mul(sH.mul(2.0).div(H.add(1e-4))));
    const nW = normalize(cross(dPdu, dPdv).add(vec3(1e-6, 0.0, 0.0)));
    fanMat.vertexNode = clipPos(posW);
    const vF0 = varying(vec4(uL, vL, rho, th));
    const vF1 = varying(vec4(posW, rn));
    const fogF = fogVary(posW);
    const vF2 = varying(vec4(nW, seed));
    const vF3 = varying(vec4(iCol.xyz, H));
    fanMat.fragmentNode = Fn(() => {
      const u = vF0.x;
      const v = vF0.y;
      const rh = vF0.z;
      const ang = vF0.w;
      const pW = vF1.xyz;
      const r = vF1.w;
      const sd = vF2.w;
      const col = vF3.xyz;
      const fanH = vF3.w;
      // (a) anastomosing net, two levels: Voronoi borders in (arc, radius) space, meshes elongated
      // along the growth direction. The fine net fades out once its meshes drop below ~6 px while the
      // coarser anastomoses keep >= ~1.4 px lines, so coverage stays binary (alpha-to-coverage never
      // has to dither a large half-covered area) and the lattice never shimmers.
      const pa = vec2(ang.mul(rh).div(0.02), rh.div(0.032)).add(sd.mul(37.0)).toVar();
      const pw = pa.add(vec2(sin(pa.y.mul(0.83).add(sd.mul(9.0))), sin(pa.x.mul(0.71).add(sd.mul(5.0)))).mul(0.32));
      const F = netLookup(pw).toVar();
      const e = F.x;
      const fe = fwidth(e).mul(0.75).add(1e-4);
      const hw = mix(float(0.15), float(0.105), smoothstep(0.15, 0.95, r)).mul(float(0.8).add(sin(u.mul(31.0).add(sd.mul(7.0))).mul(sin(v.mul(23.0).sub(sd.mul(3.0)))).mul(0.3)));
      const cellPx = max(fwidth(pa.x), fwidth(pa.y));
      // LOD by dissolve: whole strands drop out (per cell id) as the meshes shrink -> binary coverage
      const netFine = float(1.0).sub(smoothstep(hw.sub(fe), hw.add(fe), e)).mul(step(smoothstep(0.13, 0.26, cellPx), F.y));
      const pb = vec2(ang.mul(rh).div(0.055), rh.div(0.085)).add(sd.mul(91.0)).toVar();
      const pbw = pb.add(vec2(sin(pb.y.mul(1.1).add(sd.mul(4.0))), sin(pb.x.mul(0.9).add(sd.mul(8.0)))).mul(0.28));
      const G = netLookup(pbw).toVar();
      const e2 = G.x;
      const fe2 = fwidth(e2).mul(0.75).add(1e-4);
      const hwM = max(mix(float(0.062), float(0.048), r), max(fwidth(pb.x), fwidth(pb.y)).mul(0.7));
      const netMed = float(1.0).sub(smoothstep(hwM.sub(fe2), hwM.add(fe2), e2));
      const netA = max(netFine, netMed).mul(step(float(1.0).sub(smoothstep(0.07, 0.2, r)), G.y.mul(0.98).add(0.01)));
      // (b) primary branches radiating from the holdfast, forking once, and the short trunk
      const spread = smoothstep(0.0, 0.33, r).mul(THMAX * 0.97).add(0.03);
      const wig = sin(rh.mul(23.0).add(sd.mul(11.0))).mul(0.09);
      const branchDist = (n, w) => {
        const a = ang.div(spread).add(1.0).mul(0.5 * n).sub(0.5).add(w);
        const k = clamp(floor(a.add(0.5)), 0.0, n - 1);
        return abs(a.sub(k)).mul(spread.mul(2.0 / n)).mul(rh);
      };
      const fm = fwidth(rh).mul(0.8).add(1e-5); // metres per pixel (AA)
      const hb1 = max(mix(float(0.0062), float(0.003), r), fm.mul(0.8));
      const hb2 = max(mix(float(0.0038), float(0.0023), r), fm.mul(0.7));
      const br1 = float(1.0).sub(smoothstep(hb1.sub(fm), hb1.add(fm), branchDist(5, wig))).mul(smoothstep(0.05, 0.11, r));
      const br2 = float(1.0).sub(smoothstep(hb2.sub(fm), hb2.add(fm), branchDist(10, wig.mul(1.4)))).mul(smoothstep(0.38, 0.52, r));
      const ht = mix(float(0.011), float(0.0065), smoothstep(0.0, 0.16, r));
      const trunk = float(1.0).sub(smoothstep(ht.sub(fm), ht.add(fm), abs(u))).mul(float(1.0).sub(smoothstep(0.1, 0.2, r)));
      const ragged = sin(ang.mul(17.0).add(sd.mul(3.0))).mul(0.022).add(sin(ang.mul(41.0).add(sd)).mul(0.012));
      const fr = fwidth(r).add(1e-4);
      const cover = max(max(netA, br1), max(br2, trunk)).mul(float(1.0).sub(smoothstep(float(0.965).sub(fr), float(0.965).add(fr), r.add(ragged))));
      // colour: deeper at the holdfast, paler growing margins, pale polyp dots on the branches
      const q = vec2(u, v).div(0.0055).toVar();
      const qc = floor(q);
      const qf = fract(q).sub(0.5).sub(hash22(qc).sub(0.5).mul(0.5));
      const dots = float(1.0).sub(smoothstep(0.16, 0.3, length(qf))).mul(step(0.45, hash22(qc.add(17.0)).x));
      const dotsL = mix(dots, float(0.13), smoothstep(0.2, 0.5, max(fwidth(q.x), fwidth(q.y))));
      const base = col.mul(mix(float(0.7), float(1.0), smoothstep(0.0, 0.45, r)));
      const alb = mix(mix(base, vec3(1.0), smoothstep(0.75, 1.0, r).mul(0.12)), mix(col, vec3(1.0), 0.62), dotsL.mul(0.55));
      const N = faceViewer(vF2.xyz, pW);
      const ao = mix(float(0.55), float(1.0), smoothstep(0.0, 0.35, v.div(fanH)));
      // fans stand clear of the reef (only the back-left one can catch a cast shadow): one shadow tap
      // (env.sunVis1) dimming the lit result instead of litProp's 4-tap PCF, on a large lacy area
      const lit = env.litProp({ albedo: alb, N, posW: pW, ao, wrap: 0.65, trans: 0.5, causticGain: 0.85, shadow: false, rim: 0.6, sheen: 0.25 });
      const shaded = lit.mul(mix(float(0.55), float(1.0), env.sunVis1 ? env.sunVis1(pW) : float(1.0)));
      return finishV(shaded, fogF, cover);
    })();
  }
  const fans = new THREE.Mesh(fanGeo, fanMat);
  fans.name = "flora_fans";
  fans.count = FAN_CAP;
  fans.frustumCulled = false;
  fans.renderOrder = 2; // alpha-tested: after the opaque reef (keeps hidden-surface removal effective)
  group.add(fans);

  // ===========================================================================
  // 3) Seagrass meadows (one instanced draw of strap blades)
  // ===========================================================================
  const GRASS_CAP = 8192;
  const grassGeo = makeBladeGeometry(7);
  const grassA = {
    pos: instAttr(grassGeo, "gPos", GRASS_CAP), // root xyz, yaw
    shape: instAttr(grassGeo, "gShape", GRASS_CAP), // height, width, lean, seed
  };
  const grassMat = new THREE.MeshBasicNodeMaterial({ name: "flora_grass", side: THREE.DoubleSide });
  {
    const iPos = attribute("gPos", "vec4");
    const iShape = attribute("gShape", "vec4");
    const side = positionGeometry.x;
    const s = positionGeometry.y;
    const H = iShape.x;
    const seed = iShape.w;
    const yaw = iPos.w;
    const fwd = vec3(cos(yaw), 0.0, sin(yaw)).toVar();
    const fac = vec3(sin(yaw).negate(), 0.0, cos(yaw));
    const W = env.surge(vec2(iPos.x, iPos.z), t, s.mul(0.3));
    const D = clampLen(W.mul(0.95).add(flutter(seed, s, float(0.014))), H.mul(0.62)).toVar();
    const off = fac.mul(iShape.z.mul(H)).add(vec3(D.x, 0.0, D.y)).toVar();
    const s2 = s.mul(s);
    const oo = dot(off, off);
    const P = iPos.xyz.add(vec3(0.0, H.mul(s), 0.0)).add(off.mul(s2)).sub(vec3(0.0, oo.mul(s2).mul(s).div(H.mul(2.0).add(1e-4)), 0.0));
    const width = iShape.y.mul(float(1.0).sub(s2.mul(s).mul(0.78)));
    const posW = P.add(fwd.mul(side.mul(width))).toVar();
    const dP = vec3(0.0, H, 0.0).add(off.mul(s.mul(2.0))).sub(vec3(0.0, oo.mul(s2).mul(1.5).div(H.add(1e-4)), 0.0)).add(vec3(0.0, 1e-6, 0.0));
    const nW = cross(fwd, normalize(dP));
    grassMat.vertexNode = clipPos(posW);
    const vG0 = varying(vec4(posW, s));
    const fogG = fogVary(posW);
    const vG1 = varying(vec4(nW, seed));
    grassMat.fragmentNode = Fn(() => {
      const pW = vG0.xyz;
      const sv = vG0.w;
      const sd = vG1.w;
      const N = faceViewer(vG1.xyz, pW);
      // sage-mint blades, paler epiphyte-coated tips (a few older tan-tipped blades), AO at the base
      const baseCol = mix(vec3(0.5, 0.78, 0.6), vec3(0.62, 0.82, 0.56), fract(sd.mul(3.7)));
      const tipCol = mix(vec3(0.84, 0.9, 0.74), vec3(0.86, 0.8, 0.58), step(0.82, fract(sd.mul(13.7))));
      const alb = mix(baseCol, tipCol, smoothstep(0.5, 1.0, sv).mul(mix(float(0.45), float(1.0), fract(sd.mul(7.3)))));
      const ao = mix(float(0.36), float(1.0), smoothstep(0.0, 0.5, sv));
      // meadows lie 3.5-8 m out, beyond the reef's cast shadows: skip the shadow-map taps
      const shaded = env.litProp({ albedo: alb, N, posW: pW, ao, wrap: 0.55, trans: 0.6, causticGain: 0.8, shadow: false, rim: 0.5, sheen: 0.1 });
      return finishV(shaded, fogG);
    })();
  }
  const grass = new THREE.Mesh(grassGeo, grassMat);
  grass.name = "flora_grass";
  grass.count = GRASS_CAP;
  grass.frustumCulled = false;
  group.add(grass);

  // ===========================================================================
  // 4) Leaf family: merged world-space geometry (algae, anemone discs, leather corals), sways per vertex
  // ===========================================================================
  const LEAF_VCAP = 40000;
  const LEAF_ICAP = 90000;
  const leaf = new LeafBuilder(LEAF_VCAP, LEAF_ICAP);
  const leafGeo = leaf.geometry;
  const leafMat = new THREE.MeshBasicNodeMaterial({ name: "flora_leaf", side: THREE.DoubleSide });
  {
    const lCol = attribute("lCol", "vec4"); // albedo, transmission
    const lDat = attribute("lDat", "vec4"); // s (sway weight), ao, pattern (kind + param), lag tau
    const lBase = attribute("lBase", "vec4"); // holdfast x, z, gain, length
    const s = lDat.x;
    const L = lBase.w;
    const seed = fract(sin(lBase.x.mul(12.9898).add(lBase.y.mul(78.233))).mul(43758.5453)).toVar();
    const W = env.surge(vec2(lBase.x, lBase.y), t, lDat.w.mul(s));
    const D = clampLen(W.mul(lBase.z).add(flutter(seed, s, L.mul(0.02))), L.mul(0.5)).toVar();
    const kind = floor(lDat.z);
    // Sarcophyton caps breathe: slow radial swelling of the cap (kind 2)
    const breathe = step(1.5, kind).mul(step(kind, 2.5)).mul(sin(t.mul(0.62).add(seed.mul(20.0))).mul(0.035)).mul(s);
    const rel = vec2(positionGeometry.x.sub(lBase.x), positionGeometry.z.sub(lBase.y)).mul(breathe);
    const s2 = s.mul(s);
    const posW = positionGeometry
      .add(vec3(D.x.mul(s2).add(rel.x), dot(D, D).mul(s2).mul(s).div(L.mul(2.0).add(1e-4)).negate(), D.y.mul(s2).add(rel.y)))
      .toVar();
    leafMat.vertexNode = clipPos(posW);
    const vL0 = varying(vec4(posW, lDat.z));
    const fogL = fogVary(posW);
    const vL1 = varying(vec4(normalGeometry, lDat.y));
    const vL2 = varying(lCol);
    const vL3 = varying(vec3(positionGeometry.x, positionGeometry.z, s));
    leafMat.fragmentNode = Fn(() => {
      const pW = vL0.xyz;
      const pat = vL0.w;
      const k = floor(pat);
      const prm = fract(pat);
      const col = vL2.xyz;
      const isK = (n) => step(n - 0.5, k).mul(step(k, n + 0.5));
      // 1 Padina: concentric calcified bands; 2 leather cap: polyp stipple; 3 anemone disc: darker mouth;
      // 4 Halimeda: pale calcified segment rims
      const bands = smoothstep(0.32, 0.5, abs(fract(prm.mul(5.5)).sub(0.5))).mul(smoothstep(0.2, 0.5, prm));
      const q = vL3.xy.div(0.0045);
      const qf = fract(q).sub(0.5);
      const stip = float(1.0).sub(smoothstep(0.18, 0.34, length(qf))).mul(smoothstep(0.5, 0.9, prm));
      const stipL = mix(stip, float(0.15).mul(smoothstep(0.5, 0.9, prm)), smoothstep(0.25, 0.6, max(fwidth(q.x), fwidth(q.y))));
      const tint = vec3(1.0)
        .add(isK(1).mul(bands).mul(vec3(-0.2, -0.3, -0.38)))
        .add(isK(2).mul(stipL).mul(vec3(0.1, 0.1, 0.12)))
        .add(isK(3).mul(float(1.0).sub(smoothstep(0.05, 0.3, prm))).mul(vec3(-0.25, -0.38, -0.32)))
        .add(isK(4).mul(smoothstep(0.7, 0.98, prm)).mul(vec3(0.16, 0.1, 0.12)));
      const alb = col.mul(tint);
      const N = faceViewer(vL1.xyz, pW);
      const shaded = env.litProp({ albedo: alb, N, posW: pW, ao: vL1.w, wrap: 0.55, trans: vL2.w, causticGain: 0.85, rim: 0.6, sheen: 0.15 });
      return finishV(shaded, fogL);
    })();
  }
  const leafMesh = new THREE.Mesh(leafGeo, leafMat);
  leafMesh.name = "flora_leaf";
  leafMesh.frustumCulled = false;
  group.add(leafMesh);

  // dev A/B (?ab=flHideGrass,flHideFans,flHideTubes,flHideLeaf): hide single families to cost them
  if (env.AB.has("flHideGrass")) grass.visible = false;
  if (env.AB.has("flHideFans")) fans.visible = false;
  if (env.AB.has("flHideTubes")) whips.visible = tents.visible = false;
  if (env.AB.has("flHideLeaf")) leafMesh.visible = false;
  if (env.AB.has("flCompile")) {
    // dev: time the flora pipelines alone (node build + async pipeline compile), before the reef compiles
    setTimeout(async () => {
      const t0 = performance.now();
      try {
        const per = [];
        for (const m of [grass, leafMesh, whips, tents, fans]) {
          const t1 = performance.now();
          await env.renderer.compileAsync(m, env.camera, env.scene);
          per.push(`${m.name} ${Math.round(performance.now() - t1)}`);
        }
        console.log(`reef-flora compile: ${Math.round(performance.now() - t0)} ms (${per.join(", ")})`);
      } catch (e) {
        console.error("reef-flora compile probe failed:", e);
      }
    }, 0);
  }
  if (env.AB.has("flDump")) {
    // dev: generated WGSL sizes per family (watch for expression-tree blow-ups)
    setTimeout(async () => {
      for (const m of [whips, tents, fans, grass, leafMesh]) {
        try {
          const sh = await env.renderer.debug.getShaderAsync(env.scene, env.camera, m);
          console.log(`reef-flora shader ${m.name}: vertex ${sh.vertexShader.length} B, fragment ${sh.fragmentShader.length} B`);
          if (env.AB.has("flDumpFull")) console.log(`${m.name} VERTEX\n${sh.vertexShader}\n${m.name} FRAGMENT\n${sh.fragmentShader}`);
        } catch (e) {
          console.error("reef-flora dump failed:", e);
        }
      }
    }, 3000);
  }

  // ===========================================================================
  // Placement (once the Houdini GLB is in: seabed height + coral footprint)
  // ===========================================================================
  const stats = { whips: 0, tentacles: 0, fans: 0, blades: 0, halimeda: 0, padina: 0, leathers: 0 };
  env.terrain
    .then((T) => {
      const t0 = performance.now();
      const obstacles = env.sites.obstacles || (env.sites.obstacles = []);
      try {
        const nT = { whip: 0, tent: 0 };
        const pushTube = (b, L, axis, r0, curve, gain, tau, colA, bulb, colB, ao) => {
          const key = bulb > 0.5 ? "tent" : "whip";
          const A = bulb > 0.5 ? tentA : whipA;
          const i = nT[key];
          if (i >= (bulb > 0.5 ? TENT_CAP : WHIP_CAP)) return;
          setV4(A.base, i, b[0], b[1], b[2], L);
          setV4(A.axis, i, axis[0], axis[1], axis[2], r0);
          setV4(A.bend, i, curve[0], curve[1], gain, tau);
          setV4(A.colA, i, colA[0], colA[1], colA[2], bulb);
          setV4(A.colB, i, colB[0], colB[1], colB[2], ao);
          nT[key]++;
        };

        // ---- sea whips ----
        WHIPS.forEach((c, ci) => {
          const rng = env.makeRng(1000 + ci * 17);
          const [cx, cz] = findFree(T, c.x, c.z, 0.12, rng);
          const [ca, cb] = PAL.whips[c.col];
          let maxL = 0;
          for (let k = 0; k < c.n; k++) {
            const a = rng() * TAU;
            const rr = 0.015 + rng() * 0.07;
            const bx = cx + Math.cos(a) * rr;
            const bz = cz + Math.sin(a) * rr;
            const by = T.heightAt(bx, bz) - 0.02;
            const L = 0.55 + rng() * 0.65;
            maxL = Math.max(maxL, L);
            const lean = 0.06 + rng() * 0.26; // splay out of the common holdfast
            const ax = [Math.cos(a) * lean, 1, Math.sin(a) * lean];
            const al = Math.hypot(ax[0], ax[1], ax[2]);
            const ca2 = a + (rng() - 0.5) * 1.6; // arc outward from the holdfast like a loose fountain
            const curl = (0.16 + rng() * 0.26) * L;
            const jit = 0.92 + rng() * 0.16;
            pushTube(
              [bx, by, bz],
              L,
              [ax[0] / al, ax[1] / al, ax[2] / al],
              0.0068 + rng() * 0.0026,
              [Math.cos(ca2) * curl, Math.sin(ca2) * curl],
              0.8,
              0.6,
              [ca[0] * jit, ca[1] * jit, ca[2] * jit],
              0,
              cb,
              1.0
            );
            stats.whips++;
          }
          obstacles.push({ x: cx, z: cz, y0: T.heightAt(cx, cz), y1: T.heightAt(cx, cz) + maxL * 0.95, r: 0.22 });
        });

        // ---- bubble-tip anemones: oral disc (leaf) + phyllotaxis tentacles (tubes) ----
        ANEMONES.forEach((an, ai) => {
          const rng = env.makeRng(2000 + an.seed);
          const cy = T.heightAt(an.x, an.z);
          const n = T.normalAt(an.x, an.z);
          const up = new THREE.Vector3(n.x * 0.6, n.y, n.z * 0.6).normalize(); // follow the slope, but stay upright-ish
          const e1 = new THREE.Vector3(1, 0, 0).sub(up.clone().multiplyScalar(up.x)).normalize();
          const e2 = new THREE.Vector3().crossVectors(up, e1).normalize();
          const discH = 0.035;
          const centre = new THREE.Vector3(an.x, cy - 0.006, an.z);
          addAnemoneDisc(leaf, centre, up, e1, e2, an.R, discH, PAL.anemone.disc);
          const golden = Math.PI * (3 - Math.sqrt(5));
          for (let i = 0; i < an.n; i++) {
            const fr = Math.sqrt((i + 0.5) / an.n);
            const rr = an.R * 0.96 * fr;
            const ph = i * golden + rng() * 0.25;
            const ca = Math.cos(ph);
            const sa = Math.sin(ph);
            const radial = e1.clone().multiplyScalar(ca).add(e2.clone().multiplyScalar(sa));
            const hd = discH * (1 - fr * fr);
            const b = centre.clone().addScaledVector(radial, rr).addScaledVector(up, hd - 0.004);
            // trim tentacles that would grow into the coral the anemone is wedged against
            const L = 0.036 + rng() * 0.016;
            const splay = 0.08 + 0.6 * fr * fr + rng() * 0.12;
            const ax = up.clone().add(radial.clone().multiplyScalar(splay)).normalize();
            const tip = b.clone().addScaledVector(ax, L);
            if (T.coralDist(b.x, b.z) < 0.025 || T.coralDist(tip.x, tip.z) < 0.02) continue;
            const curl = radial.clone().multiplyScalar(L * (0.12 + rng() * 0.18));
            const shade = 0.82 + 0.18 * fr; // inner tentacles sit in the canopy's shade
            const jit = 0.95 + rng() * 0.08;
            const stalk = PAL.anemone.stalk;
            pushTube(
              [b.x, b.y, b.z],
              L,
              [ax.x, ax.y, ax.z],
              0.0082 + rng() * 0.0014,
              [curl.x, curl.z],
              0.11,
              0.25,
              [stalk[0] * jit, stalk[1] * jit, stalk[2] * jit],
              1,
              an.tip,
              shade
            );
            stats.tentacles++;
          }
          // refine the published clownfish home in place (centre of the tentacle disc)
          const site = anemoneSites[ai];
          site.position.set(an.x, cy + discH + 0.012, an.z);
          site.radius = an.R + 0.05;
        });
        whipGeo.instanceCount = nT.whip;
        tentGeo.instanceCount = nT.tent;
        for (const a of [...Object.values(whipA), ...Object.values(tentA)]) a.needsUpdate = true;

        // ---- sea fans: perpendicular to the surge (face the water motion), rocking with a lag ----
        const sd = env.surgeDir;
        let nFan = 0;
        FANS.forEach((f, fi) => {
          if (nFan >= FAN_CAP) return;
          const rng = env.makeRng(3000 + fi * 31);
          const [fx, fz] = findFree(T, f.x, f.z, 0.15, rng);
          const yawJ = (rng() - 0.5) * 0.3;
          const c = Math.cos(yawJ);
          const s = Math.sin(yawJ);
          // right vector = surge direction rotated +90 deg (+ jitter); normal = right x up = surge dir
          const rx0 = -sd.y;
          const rz0 = sd.x;
          const rx = rx0 * c - rz0 * s;
          const rz = rx0 * s + rz0 * c;
          const col = PAL.fans[f.col % PAL.fans.length];
          const by = T.heightAt(fx, fz) - 0.03;
          setV4(fanA.base, nFan, fx, by, fz, f.H);
          setV4(fanA.frame, nFan, rx, rz, f.aspect, 0.1 + rng() * 0.12);
          setV4(fanA.col, nFan, col[0], col[1], col[2], rng());
          // ~+-3.5 deg rocking at the max surge (0.16 m): tip displacement = 0.06 H -> g = 0.38 H
          setV4(fanA.sway, nFan, 0.38 * f.H, 0.4, (rng() - 0.5) * 0.1, rng() * TAU);
          obstacles.push({ x: fx, z: fz, y0: by, y1: by + f.H, r: 0.45 * f.H * f.aspect });
          nFan++;
          stats.fans++;
        });
        fans.count = Math.max(2, nFan);
        for (let i = nFan; i < FAN_CAP; i++) setV4(fanA.base, i, 0, -10, 0, 0);
        for (const a of Object.values(fanA)) a.needsUpdate = true;

        // ---- seagrass meadows (keep a bare grazing halo around the reef) ----
        let nB = 0;
        MEADOWS.forEach((m, mi) => {
          const rng = env.makeRng(4000 + mi * 101);
          let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
          for (const [bx, bz, br] of m.blobs) {
            minX = Math.min(minX, bx - br); maxX = Math.max(maxX, bx + br);
            minZ = Math.min(minZ, bz - br); maxZ = Math.max(maxZ, bz + br);
          }
          const density = (x, z) => {
            let d = 0;
            for (const [bx, bz, br] of m.blobs) {
              const q = Math.hypot(x - bx, z - bz) / br;
              d = Math.max(d, 1 - smooth01(0.55, 1.0, q));
            }
            // patchy: low-frequency modulation so the meadow edge is irregular
            d *= 0.75 + 0.25 * Math.sin(x * 2.3 + z * 1.7 + mi) * Math.sin(x * 1.1 - z * 2.9 + 2 * mi);
            return d;
          };
          let placed = 0;
          for (let tries = 0; placed < m.n && tries < m.n * 6 && nB < GRASS_CAP; tries++) {
            const x = minX + rng() * (maxX - minX);
            const z = minZ + rng() * (maxZ - minZ);
            const d = density(x, z);
            if (rng() > d) continue;
            if (T.coralDist(x, z) < 1.7 || Math.hypot(x, z) > 9.5) continue;
            const y = T.heightAt(x, z) - 0.01;
            const H = (0.14 + 0.22 * Math.sqrt(d) * (0.5 + 0.5 * rng())) * (0.88 + 0.24 * rng());
            setV4(grassA.pos, nB, x, y, z, rng() * TAU);
            setV4(grassA.shape, nB, H, 0.0065 + rng() * 0.004, (rng() - 0.25) * 0.5 * rng(), rng());
            nB++;
            placed++;
          }
          let cx = 0, cz = 0, rMax = 0;
          for (const [bx, bz] of m.blobs) { cx += bx / m.blobs.length; cz += bz / m.blobs.length; }
          for (const [bx, bz, br] of m.blobs) rMax = Math.max(rMax, Math.hypot(bx - cx, bz - cz) + br);
          obstacles.push({ x: cx, z: cz, y0: T.heightAt(cx, cz), y1: T.heightAt(cx, cz) + 0.32, r: rMax * 0.85 });
        });
        stats.blades = nB;
        grass.count = Math.max(2, nB);
        grassA.pos.needsUpdate = true;
        grassA.shape.needsUpdate = true;

        // ---- Halimeda / Padina clumps around the reef base ----
        {
          const rng = env.makeRng(5150);
          let nH = 0;
          for (let tries = 0; nH < 52 && tries < 4000; tries++) {
            const a = rng() * TAU;
            const rr = 0.4 + rng() * 2.6;
            const x = Math.cos(a) * rr * 1.25;
            const z = Math.sin(a) * rr * 0.85;
            const cd = T.coralDist(x, z);
            if (cd < 0.06 || cd > 0.55) continue;
            if (z > 0.55 && Math.abs(x) < 1.5 && rng() < 0.75) continue; // sparse in the hero foreground
            const y = T.heightAt(x, z) - 0.008;
            if (rng() < 0.62) {
              addHalimeda(leaf, x, y, z, 1.25 + rng() * 0.7, rng, PAL.halimeda);
              stats.halimeda++;
            } else {
              addPadina(leaf, x, y, z, 1.15 + rng() * 0.55, rng, PAL.padina);
              stats.padina++;
            }
            nH++;
          }
        }

        // ---- Sarcophyton leather corals ----
        LEATHERS.forEach((lc, li) => {
          const rng = env.makeRng(6000 + li * 7);
          const [x, z] = findFree(T, lc.x, lc.z, lc.R + 0.05, rng);
          const y = T.heightAt(x, z) - 0.02;
          addLeather(leaf, x, y, z, lc.R, lc.h, rng, PAL.leather);
          obstacles.push({ x, z, y0: y, y1: y + lc.h + 0.06, r: lc.R + 0.03 });
          stats.leathers++;
        });

        leaf.commit();
      } catch (e) {
        console.error("reef-flora placement failed:", e);
      }
      stats.ms = Math.round(performance.now() - t0);
      stats.netBakeMs = netBakeMs;
      env.floraStats = stats;
      console.log("reef-flora:", JSON.stringify(stats));
    })
    .catch((e) => console.error("reef-flora: terrain unavailable:", e));

  return {
    group,
    update() {
      // all motion is evaluated on the GPU from env.u.time (deterministic in t); only the mood switch here
      uNight.value = env.mood === "night" ? 1.0 : 0.0;
    },
    warmTo() {},
  };
}

// -----------------------------------------------------------------------------
// CPU helpers
// -----------------------------------------------------------------------------
function smooth01(a, b, x) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** nearest point to (x, z) with coralDist >= minCd (spiral search), deterministic */
function findFree(T, x, z, minCd, rng) {
  if (T.coralDist(x, z) >= minCd) return [x, z];
  for (let k = 1; k <= 60; k++) {
    const r = 0.05 * k;
    const a0 = rng() * TAU;
    for (let j = 0; j < 8; j++) {
      const a = a0 + (j / 8) * TAU;
      const px = x + Math.cos(a) * r;
      const pz = z + Math.sin(a) * r;
      if (T.coralDist(px, pz) >= minCd) return [px, pz];
    }
  }
  return [x, z];
}

/**
 * Tileable Voronoi border-distance texture for the sea-fan net (baked once, ~10 ms): R = (F2 - F1) / 2
 * in cell units (0..eMax), G = random id of the nearest cell (drives the strand dissolve LOD).
 * Linear filtering, no mips (the shader does its own LOD), repeat wrapping.
 */
function makeNetTexture(rng, cells = 24, res = 16) {
  const W = cells * res;
  const eMax = 0.25;
  const sx = new Float32Array(cells * cells);
  const sy = new Float32Array(cells * cells);
  const ids = new Float32Array(cells * cells);
  for (let i = 0; i < cells * cells; i++) {
    sx[i] = 0.07 + 0.86 * rng();
    sy[i] = 0.07 + 0.86 * rng();
    ids[i] = rng();
  }
  const data = new Uint8Array(W * W * 4);
  for (let y = 0; y < W; y++) {
    const py = (y + 0.5) / res;
    const cy = Math.floor(py);
    for (let x = 0; x < W; x++) {
      const px = (x + 0.5) / res;
      const cx = Math.floor(px);
      let f1 = 9, f2 = 9, id = 0;
      for (let j = -1; j <= 1; j++) {
        const gy = cy + j;
        const ky = ((gy % cells) + cells) % cells;
        for (let i = -1; i <= 1; i++) {
          const gx = cx + i;
          const k = ky * cells + (((gx % cells) + cells) % cells);
          const dx = gx + sx[k] - px;
          const dy = gy + sy[k] - py;
          const d = dx * dx + dy * dy;
          if (d < f1) {
            f2 = f1;
            f1 = d;
            id = ids[k];
          } else if (d < f2) f2 = d;
        }
      }
      const e = (Math.sqrt(f2) - Math.sqrt(f1)) * 0.5;
      const o = (y * W + x) * 4;
      data[o] = Math.min(255, Math.round((e / eMax) * 255));
      data[o + 1] = Math.round(id * 255);
      data[o + 2] = 0;
      data[o + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, W, W, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return { tex, cells, eMax };
}

/** tube template: rings x sides, position = (s, cos phi, sin phi); rings denser toward the tip */
function makeTubeGeometry(rings, sides, expo) {
  const pos = [];
  for (let r = 0; r <= rings; r++) {
    const s = 1 - Math.pow(1 - r / rings, expo);
    for (let k = 0; k < sides; k++) {
      const a = (k / sides) * TAU;
      pos.push(s, Math.cos(a), Math.sin(a));
    }
  }
  const idx = [];
  for (let r = 0; r < rings; r++)
    for (let k = 0; k < sides; k++) {
      const a = r * sides + k;
      const b = r * sides + ((k + 1) % sides);
      idx.push(a, b, b + sides, a, b + sides, a + sides); // CCW seen from outside
    }
  // InstancedBufferGeometry: the instance count lives on the geometry, so mesh.count stays 1 and the
  // whip and tentacle meshes share one material cache key (one node build / pipeline for both)
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.instanceCount = 0;
  return g;
}

/** sea-fan template: polar grid, position = (theta in -1..1, rho in 0..1, 0) */
function makePolarGrid(na, nr) {
  const pos = [];
  for (let j = 0; j <= nr; j++)
    for (let i = 0; i <= na; i++) pos.push((i / na) * 2 - 1, Math.pow(j / nr, 0.9), 0);
  const idx = [];
  for (let j = 0; j < nr; j++)
    for (let i = 0; i < na; i++) {
      const a = j * (na + 1) + i;
      idx.push(a, a + 1, a + na + 2, a, a + na + 2, a + na + 1);
    }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

/** strap-blade template: (rows+1) x 2 vertices, position = (side -0.5/0.5, s, 0) */
function makeBladeGeometry(rows) {
  const pos = [];
  for (let r = 0; r <= rows; r++) {
    const s = r / rows;
    pos.push(-0.5, s, 0, 0.5, s, 0);
  }
  const idx = [];
  for (let r = 0; r < rows; r++) {
    const a = r * 2;
    idx.push(a, a + 1, a + 3, a, a + 3, a + 2);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

/**
 * Fixed-capacity merged geometry (allocated up front so the pipeline compiles with the scene; filled
 * after placement). Per vertex: world position + normal, lCol (albedo, trans), lDat (s, ao, pattern, tau),
 * lBase (holdfast x, z, sway gain, organism length).
 */
class LeafBuilder {
  constructor(vcap, icap) {
    this.vcap = vcap;
    this.icap = icap;
    this.nv = 0;
    this.ni = 0;
    const g = new THREE.BufferGeometry();
    this.pos = new THREE.BufferAttribute(new Float32Array(vcap * 3), 3);
    this.nrm = new THREE.BufferAttribute(new Float32Array(vcap * 3), 3);
    this.col = new THREE.BufferAttribute(new Float32Array(vcap * 4), 4);
    this.dat = new THREE.BufferAttribute(new Float32Array(vcap * 4), 4);
    this.base = new THREE.BufferAttribute(new Float32Array(vcap * 4), 4);
    g.setAttribute("position", this.pos);
    g.setAttribute("normal", this.nrm);
    g.setAttribute("lCol", this.col);
    g.setAttribute("lDat", this.dat);
    g.setAttribute("lBase", this.base);
    this.index = new THREE.BufferAttribute(vcap > 65535 ? new Uint32Array(icap) : new Uint16Array(icap), 1);
    g.setIndex(this.index);
    g.setDrawRange(0, 0);
    this.geometry = g;
    this.cur = { col: [1, 1, 1, 0], base: [0, 0, 0, 0.1], tau: 0.2 };
  }
  /** set the per-organism attributes used by the following vertices */
  organism(col, trans, bx, bz, gain, L, tau) {
    this.cur = { col: [col[0], col[1], col[2], trans], base: [bx, bz, gain, L], tau };
  }
  vert(p, n, s, ao, pat, colOverride) {
    if (this.nv >= this.vcap) return -1;
    const i = this.nv++;
    const c = colOverride || this.cur.col;
    this.pos.array.set([p.x, p.y, p.z], i * 3);
    this.nrm.array.set([n.x, n.y, n.z], i * 3);
    this.col.array.set(colOverride ? [c[0], c[1], c[2], this.cur.col[3]] : c, i * 4);
    this.dat.array.set([s, ao, pat, this.cur.tau], i * 4);
    this.base.array.set(this.cur.base, i * 4);
    return i;
  }
  tri(a, b, c) {
    if (a < 0 || b < 0 || c < 0 || this.ni + 3 > this.icap) return;
    this.index.array[this.ni++] = a;
    this.index.array[this.ni++] = b;
    this.index.array[this.ni++] = c;
  }
  commit() {
    for (const a of [this.pos, this.nrm, this.col, this.dat, this.base, this.index]) a.needsUpdate = true;
    this.geometry.setDrawRange(0, this.ni);
  }
}

const _v = new THREE.Vector3();

/** Halimeda ("cactus alga"): branching chains of flat calcified kidney segments */
function addHalimeda(B, x, y, z, scale, rng, col) {
  const Hc = 0.1 * scale;
  B.organism(col, 0.35, x, z, 0.32, Hc, 0.25);
  const chains = 4 + Math.floor(rng() * 4);
  const seg = (c, up, faceN, a, b) => {
    const right = new THREE.Vector3().crossVectors(up, faceN).normalize();
    const n = new THREE.Vector3().crossVectors(right, up).normalize();
    const sOf = (p) => Math.min(1, Math.max(0, (p.y - y) / Hc));
    const aoOf = (p) => 0.5 + 0.5 * sOf(p);
    const ci = B.vert(c, n, sOf(c), aoOf(c), 4.0);
    const K = 9;
    const ids = [];
    for (let k = 0; k < K; k++) {
      const al = (k / K) * TAU;
      let dA = Math.abs(al - 1.5 * Math.PI); // kidney notch at the joint (bottom of the segment)
      dA = Math.min(dA, TAU - dA);
      const notch = 1 - 0.28 * Math.exp(-(dA * dA) / 0.25);
      const px = c.x + (right.x * Math.cos(al) * a + up.x * Math.sin(al) * b) * notch;
      const py = c.y + (right.y * Math.cos(al) * a + up.y * Math.sin(al) * b) * notch;
      const pz = c.z + (right.z * Math.cos(al) * a + up.z * Math.sin(al) * b) * notch;
      _v.set(px, py, pz);
      ids.push(B.vert(_v, n, sOf(_v), aoOf(_v), 4.95));
    }
    for (let k = 0; k < K; k++) B.tri(ci, ids[k], ids[(k + 1) % K]);
  };
  const grow = (p0, d0, nSeg, size, depth) => {
    const p = p0.clone();
    const d = d0.clone();
    const faceN = new THREE.Vector3(rng() - 0.5, 0, rng() - 0.5).normalize();
    for (let i = 0; i < nSeg; i++) {
      const a = size * (0.85 + rng() * 0.3);
      const b = a * 0.78;
      p.addScaledVector(d, b * 1.05);
      seg(p.clone(), d.clone(), faceN, a, b);
      p.addScaledVector(d, b * 0.75);
      d.x += (rng() - 0.5) * 0.35;
      d.z += (rng() - 0.5) * 0.35;
      d.y += 0.05;
      d.normalize();
      faceN.applyAxisAngle(d, (rng() - 0.5) * 0.5);
      if (depth < 1 && i > 0 && rng() < 0.3) {
        const bd = d.clone().add(new THREE.Vector3(rng() - 0.5, 0, rng() - 0.5).multiplyScalar(1.2)).normalize();
        grow(p, bd, 2 + Math.floor(rng() * 3), size * 0.85, depth + 1);
      }
    }
  };
  for (let c = 0; c < chains; c++) {
    const a = rng() * TAU;
    const p0 = new THREE.Vector3(x + Math.cos(a) * 0.012 * scale, y, z + Math.sin(a) * 0.012 * scale);
    const lean = 0.25 + rng() * 0.45;
    const d0 = new THREE.Vector3(Math.cos(a) * lean, 1, Math.sin(a) * lean).normalize();
    grow(p0, d0, 4 + Math.floor(rng() * 4), 0.0068 * scale, 0);
  }
}

/** Padina: a rosette of curled, fan-shaped calcified lobes with concentric bands */
function addPadina(B, x, y, z, scale, rng, col) {
  const R = 0.045 * scale;
  B.organism(col, 0.4, x, z, 0.2, R * 1.4, 0.2);
  const lobes = 3 + Math.floor(rng() * 3);
  const NA = 7;
  const NR = 4;
  for (let l = 0; l < lobes; l++) {
    const yaw = (l / lobes) * TAU + rng() * 0.6;
    const tilt = 0.25 + rng() * 0.35; // lobes stand up and open outward
    const half = 1.0 + rng() * 0.6;
    const Rl = R * (0.75 + rng() * 0.45);
    const fwd = new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw));
    const right = new THREE.Vector3(-Math.sin(yaw), 0, Math.cos(yaw));
    const ids = [];
    for (let j = 0; j <= NR; j++) {
      const rr = 0.12 + 0.88 * (j / NR);
      for (let i = 0; i <= NA; i++) {
        const ph = ((i / NA) * 2 - 1) * half;
        const lx = Math.sin(ph) * rr * Rl;
        const ly = Math.cos(ph) * rr * Rl * 0.95;
        const curl = rr * rr * Rl * 0.45 * (1 - Math.cos(ph) * 0.4); // rolled margin
        const out = ly * Math.sin(tilt) + curl;
        const p = new THREE.Vector3(x, y + ly * Math.cos(tilt) + 0.003, z).addScaledVector(right, lx).addScaledVector(fwd, out);
        const n = fwd.clone().multiplyScalar(Math.cos(tilt)).add(new THREE.Vector3(0, -Math.sin(tilt), 0)).addScaledVector(right, -Math.sin(ph) * 0.3).normalize();
        const s = Math.min(1, (p.y - y) / (R * 1.2));
        ids.push(B.vert(p, n, s, 0.55 + 0.45 * rr, 1.0 + Math.min(0.99, rr)));
      }
    }
    for (let j = 0; j < NR; j++)
      for (let i = 0; i < NA; i++) {
        const a = j * (NA + 1) + i;
        B.tri(ids[a], ids[a + 1], ids[a + NA + 2]);
        B.tri(ids[a], ids[a + NA + 2], ids[a + NA + 1]);
      }
  }
}

/** bubble-tip anemone oral disc: a low dome with a darker mouth, mostly hidden by the tentacles */
function addAnemoneDisc(B, c, up, e1, e2, R, h, col) {
  B.organism(col, 0.2, c.x, c.z, 0.04, 0.05, 0.2);
  const NR = 5;
  const NA = 20;
  const ci = B.vert(c.clone().addScaledVector(up, h), up, 0.2, 0.55, 3.0);
  const ids = [];
  for (let j = 1; j <= NR; j++) {
    const fr = j / NR;
    for (let i = 0; i < NA; i++) {
      const a = (i / NA) * TAU;
      const rad = e1.clone().multiplyScalar(Math.cos(a)).add(e2.clone().multiplyScalar(Math.sin(a)));
      const rr = R * 0.86 * fr * (1.0 + 0.04 * Math.sin(a * 5)); // the outer tentacles overhang the rim
      const hh = h * (1 - fr * fr) - (j === NR ? 0.02 : 0);
      const p = c.clone().addScaledVector(rad, rr).addScaledVector(up, hh);
      const n = up.clone().addScaledVector(rad, 2 * h * fr / R + (j === NR ? 0.8 : 0)).normalize();
      ids.push(B.vert(p, n, 0.2 * (1 - fr), 0.6 + 0.15 * (1 - fr), 3.0 + Math.min(0.99, fr)));
    }
  }
  for (let i = 0; i < NA; i++) B.tri(ci, ids[i], ids[(i + 1) % NA]);
  for (let j = 0; j < NR - 1; j++)
    for (let i = 0; i < NA; i++) {
      const a = j * NA + i;
      const b = j * NA + ((i + 1) % NA);
      B.tri(ids[a], ids[a + NA], ids[b + NA]);
      B.tri(ids[a], ids[b + NA], ids[b]);
    }
}

/** Sarcophyton toadstool leather coral: lathe stalk + cap with a wavy, folded rim */
function addLeather(B, x, y, z, R, h, rng, col) {
  B.organism(col, 0.25, x, z, 0.12, h + 0.05, 0.6);
  // profile (radius, height, s, ao, pattern) from the holdfast up the stalk, under the cap, rim, cap top
  const prof = [
    [0.55 * R, 0.0, 0.0, 0.45, 2.0],
    [0.42 * R, 0.35 * h, 0.2, 0.55, 2.0],
    [0.36 * R, 0.8 * h, 0.45, 0.6, 2.0],
    [0.48 * R, 1.0 * h, 0.6, 0.55, 2.05],
    [0.8 * R, 1.12 * h, 0.8, 0.62, 2.1],
    [1.0 * R, 1.2 * h + 0.012, 1.0, 0.8, 2.6],
    [0.97 * R, 1.2 * h + 0.026, 1.0, 0.95, 2.99],
    [0.7 * R, 1.2 * h + 0.034, 0.95, 1.0, 2.99],
    [0.35 * R, 1.2 * h + 0.04, 0.9, 1.0, 2.99],
    [0.0, 1.2 * h + 0.042, 0.88, 1.0, 2.99],
  ];
  const NA = 36;
  const folds = 5 + Math.floor(rng() * 3);
  const fph = rng() * TAU;
  const rows = [];
  for (let j = 0; j < prof.length; j++) {
    const [pr, py, s, ao, pat] = prof[j];
    const row = [];
    for (let i = 0; i < NA; i++) {
      const a = (i / NA) * TAU;
      const rimW = Math.max(0, (pr / R - 0.6) / 0.4);
      const fold = Math.sin(a * folds + fph) * 0.022 * rimW * rimW * (R / 0.17);
      const p = new THREE.Vector3(x + Math.cos(a) * pr, y + py + fold, z + Math.sin(a) * pr);
      // approximate normal from the profile slope
      const jn = Math.min(prof.length - 1, j + 1);
      const jp = Math.max(0, j - 1);
      const dr = prof[jn][0] - prof[jp][0];
      const dy = prof[jn][1] - prof[jp][1];
      const n = new THREE.Vector3(Math.cos(a) * dy, -dr, Math.sin(a) * dy).normalize();
      if (j >= prof.length - 4) n.set(Math.cos(a) * 0.25 * (pr / R), 1, Math.sin(a) * 0.25 * (pr / R)).normalize();
      row.push(B.vert(p, n, s, ao, pat));
    }
    rows.push(row);
  }
  for (let j = 0; j < rows.length - 1; j++)
    for (let i = 0; i < NA; i++) {
      const a = rows[j][i];
      const b = rows[j][(i + 1) % NA];
      const c = rows[j + 1][i];
      const d = rows[j + 1][(i + 1) % NA];
      B.tri(a, c, d);
      B.tri(a, d, b);
    }
}
