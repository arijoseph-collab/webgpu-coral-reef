// -----------------------------------------------------------------------------
// Underside of the water surface (the "Look Up" shot) — port of jeantimex/webgpu-water's
// surface-under.frag.wgsl (MIT; itself a port of Evan Wallace's WebGL Water, MIT) to three.js r172 TSL.
//
// A camera-following grid at the surface height is displaced by the simulated surface (wind sea + ripples,
// js/ocean-surface.js). Per pixel, as in surface-under: the view ray is refracted into the air
// (eta = 1.333) and reflected back down; Fresnel (Schlick-like, as in the original) mixes them, and
// `length(refract)` drops to zero past the critical angle, so outside Snell's window (48.6 deg) the surface is
// a total-internal-reflection mirror of the deep water. Adapted for the open sea: the pool walls/sphere are
// replaced by the shared water colour (js/reef-env.js) for reflected rays and a bright pastel sky + sun for
// refracted rays; the three colour channels refract with slightly different IORs, giving the faint rainbow
// rim that real Snell's windows show. Everything is fogged into the water column like the rest of the scene.
// -----------------------------------------------------------------------------
import * as THREE from "three/webgpu";
import {
  Fn,
  vec2,
  vec3,
  vec4,
  float,
  uniform,
  texture,
  positionGeometry,
  cameraPosition,
  cameraViewMatrix,
  cameraProjectionMatrix,
  varying,
  normalize,
  refract,
  reflect,
  dot,
  max,
  min,
  mix,
  pow,
  length,
  sqrt,
  smoothstep,
  clamp,
  exp,
  step,
} from "three/tsl";

export function createWaterSurface(env, { size = 36, segments = 144 } = {}) {
  const { ocean, u } = env;
  const group = new THREE.Group();
  group.name = "surface";
  env.groups.surface = group;
  env.scene.add(group);

  const uCenter = uniform(new THREE.Vector2()); // grid centre (world xz), snapped to surface texels
  const uSunAirW = uniform(new THREE.Vector3()); // sun direction above the water, world frame
  const L = ocean.tileSize;
  const rotC = Math.cos(ocean.tileRot), rotS = Math.sin(ocean.tileRot);
  const toTile = (xz) => vec2(xz.x.mul(rotC).sub(xz.y.mul(rotS)), xz.x.mul(rotS).add(xz.y.mul(rotC))).div(L);
  const surf = texture(ocean.surfaceRT.texture);

  const mat = new THREE.MeshBasicNodeMaterial({ name: "water_surface_under", side: THREE.DoubleSide });
  // vertex: camera-following grid displaced by the simulated height
  const local = positionGeometry; // PlaneGeometry in xy, rotated below
  const xz = vec2(local.x, local.y.negate()).add(uCenter);
  const hV = surf.sample(toTile(xz)).level(0).x;
  const posW = vec3(xz.x, u.surfaceY.add(hV), xz.y);
  const vPos = varying(posW);
  mat.vertexNode = cameraProjectionMatrix.mul(cameraViewMatrix).mul(vec4(posW, 1.0));

  mat.fragmentNode = Fn(() => {
    const p = vPos;
    const info = surf.sample(toTile(vec2(p.x, p.z)));
    // tile-frame normal -> world frame (undo the 23 deg tile rotation)
    const ntx = info.y, ntz = info.z;
    const nx = ntx.mul(rotC).add(ntz.mul(rotS));
    const nz = ntz.mul(rotC).sub(ntx.mul(rotS));
    const nUp = vec3(nx, sqrt(max(float(0.0), float(1.0).sub(nx.mul(nx)).sub(nz.mul(nz)))), nz);
    const n = nUp.negate(); // underwater: the surface faces down (surface-under: normal = -normal)
    const toP = p.sub(cameraPosition);
    const dist = length(toP);
    const incoming = toP.div(dist).toVar();
    const reflected = reflect(incoming, n);
    const refracted = refract(incoming, n, 1.333);
    // exact unpolarised dielectric Fresnel (water -> air): transmission falls smoothly to zero at the
    // 48.6 deg critical angle instead of switching on length(refract) (round-1 notes: no hard TIR islands,
    // no rainbow fringe)
    const cosI = clamp(dot(n, incoming.negate()), 0.0, 1.0);
    const sinT2 = float(1.333 * 1.333).mul(float(1.0).sub(cosI.mul(cosI)));
    const cosT = sqrt(max(float(1.0).sub(sinT2), 0.0));
    const rs = cosI.mul(1.333).sub(cosT).div(cosI.mul(1.333).add(cosT).add(1e-5));
    const rp = cosT.mul(1.333).sub(cosI).div(cosT.mul(1.333).add(cosI).add(1e-5));
    const trans = float(1.0).sub(rs.mul(rs).add(rp.mul(rp)).mul(0.5)).mul(step(sinT2, 1.0));
    // sky through the window: pastel cerulean/lavender zenith headroom, sun disc and glare
    const up = clamp(refracted.y, 0.0, 1.0);
    const sunCos = max(dot(normalize(refracted), uSunAirW), 0.0);
    const skyZenith = mix(u.bgShallow, u.skyHigh, 0.52).mul(0.76);
    const skyCol = mix(u.bgShallow.mul(0.66), skyZenith, pow(up, 0.8))
      .add(u.keyColor.mul(pow(sunCos, 1800.0).mul(3.2).add(pow(sunCos, 65.0).mul(0.20)).add(pow(sunCos, 8.0).mul(0.035))));
    // total internal reflection outside 48.6 deg: mirrors the darker deep water below so the window rim reads clearly
    const mirror = env.waterColor(normalize(vec3(reflected.x, min(reflected.y.mul(0.8).sub(0.12), float(-0.18)), reflected.z))).mul(0.76);
    const col = mix(mirror, skyCol, trans);
    // in-water sun beams converge on the window (folded into linear fogged colour before shoulder/encode so it never clips)
    const beams = env.beams(cameraPosition, incoming, dist, 10);
    const fogged = env.applyFog(col, p).add(u.shaftTint.mul(beams).mul(0.20));
    return env.output(fogged);
  })();

  const geo = new THREE.PlaneGeometry(size, size, segments, segments);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  mesh.renderOrder = 950;
  mesh.visible = false;
  group.add(mesh);

  const texel = L / ocean.gridSize;
  const fwd = new THREE.Vector3();
  return {
    group,
    mesh,
    update() {
      const cam = env.camera;
      // only draw when the view actually reaches the surface (top frustum edge > ~22 deg up)
      cam.getWorldDirection(fwd);
      const pitch = Math.asin(THREE.MathUtils.clamp(fwd.y, -1, 1));
      const top = pitch + THREE.MathUtils.degToRad(cam.fov * 0.5);
      mesh.visible = top > THREE.MathUtils.degToRad(22) && group.visible;
      if (!mesh.visible) return;
      uCenter.value.set(Math.round(cam.position.x / texel) * texel, Math.round(cam.position.z / texel) * texel);
      uSunAirW.value.copy(env.sunAir);
    },
    warmTo() {},
  };
}
