#!/usr/bin/env node
// Precomputes tileable 3D Voronoi honeycomb volumes for the coral TSL shader.
//
// Replaces the per-pixel 27-cell (+8-cell) Voronoi search with a single trilinear texture fetch.
// Output (RGBA8, N^3 voxels, x fastest, then y, then z):
//   honeycomb_field_<N>.bin : R = dRoundA / D_SCALE   (primary honeycomb, TILE_A cells per period)
//                             G = randA               (per-cell random of the nearest primary seed)
//                             B = dRoundB / D_SCALE   (secondary fine calice pores, TILE_B cells per period)
//                             A = f1A / F_SCALE       (distance to the nearest primary seed; the shader
//                                                      takes its surface gradient as the crater direction)
//
// dRound matches the original shader:  dRound = (f2 - f1) - k * max(0, f1 - f0)
// Usage: node tools/gen_honeycomb_volume.mjs [N=128]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const N = Number(process.argv[2] || 128);
const TILE_A = 8; // primary cells per texture period (VPC = N / 8)
const TILE_B = 15; // secondary cells per period (15 / 8 = 1.875 ~ original freq_b = 1.85 * freq_a)
const JITTER_A = 0.8;
const JITTER_B = 0.74;
const D_SCALE = 1.25;
const F_SCALE = 1.5;

const outDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "public", "textures");

// Deterministic integer hash -> 3 floats in [0, 1)
function hash3(ix, iy, iz, salt) {
  let h = Math.imul(ix, 0x8da6b343) ^ Math.imul(iy, 0xd8163841) ^ Math.imul(iz, 0xcb1ab31f) ^ Math.imul(salt, 0x165667b1);
  const out = new Float64Array(3);
  for (let k = 0; k < 3; k++) {
    h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
    h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
    h ^= h >>> 15;
    h = (h + 0x9e3779b9) | 0;
    out[k] = ((h >>> 8) & 0xffffff) / 16777216;
  }
  return out;
}

function buildSeeds(tile, jitter, salt) {
  const seeds = new Float64Array(tile * tile * tile * 4);
  for (let z = 0; z < tile; z++)
    for (let y = 0; y < tile; y++)
      for (let x = 0; x < tile; x++) {
        const h = hash3(x, y, z, salt);
        const o = (x + y * tile + z * tile * tile) * 4;
        seeds[o + 0] = 0.5 + (h[0] - 0.5) * jitter;
        seeds[o + 1] = 0.5 + (h[1] - 0.5) * jitter;
        seeds[o + 2] = 0.5 + (h[2] - 0.5) * jitter;
        seeds[o + 3] = hash3(x + 101, y + 211, z + 307, salt + 7)[0]; // per-cell random
      }
  return seeds;
}

// Exact 27-neighbour F1/F2 search in a periodic lattice of `tile` cells.
function voronoi(qx, qy, qz, tile, seeds, res) {
  const cx = Math.floor(qx), cy = Math.floor(qy), cz = Math.floor(qz);
  let f1 = 1e9, f2 = 1e9, vx = 0, vy = 0, vz = 0, rnd = 0.5;
  for (let dz = -1; dz <= 1; dz++)
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const nx = cx + dx, ny = cy + dy, nz = cz + dz;
        const wx = ((nx % tile) + tile) % tile, wy = ((ny % tile) + tile) % tile, wz = ((nz % tile) + tile) % tile;
        const o = (wx + wy * tile + wz * tile * tile) * 4;
        const ex = qx - (nx + seeds[o]), ey = qy - (ny + seeds[o + 1]), ez = qz - (nz + seeds[o + 2]);
        const d = Math.sqrt(ex * ex + ey * ey + ez * ez);
        if (d < f1) {
          f2 = f1;
          f1 = d;
          vx = ex; vy = ey; vz = ez;
          rnd = seeds[o + 3];
        } else if (d < f2) {
          f2 = d;
        }
      }
  res.f1 = f1; res.f2 = f2; res.vx = vx; res.vy = vy; res.vz = vz; res.rnd = rnd;
}

const seedsA = buildSeeds(TILE_A, JITTER_A, 17);
const seedsB = buildSeeds(TILE_B, JITTER_B, 91);
const field = new Uint8Array(N * N * N * 4);
const ra = {}, rb = {};
let maxDA = 0, maxDB = 0, maxF1 = 0;
const q8 = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
const t0 = Date.now();
for (let z = 0; z < N; z++) {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const u = (x + 0.5) / N, v = (y + 0.5) / N, w = (z + 0.5) / N;
      voronoi(u * TILE_A, v * TILE_A, w * TILE_A, TILE_A, seedsA, ra);
      voronoi(u * TILE_B, v * TILE_B, w * TILE_B, TILE_B, seedsB, rb);
      const dA = ra.f2 - ra.f1 - Math.max(0, ra.f1 - 0.22) * 0.2;
      const dB = rb.f2 - rb.f1 - Math.max(0, rb.f1 - 0.24) * 0.18;
      maxDA = Math.max(maxDA, dA);
      maxDB = Math.max(maxDB, dB);
      maxF1 = Math.max(maxF1, ra.f1);
      const o = (x + y * N + z * N * N) * 4;
      field[o + 0] = q8(dA / D_SCALE);
      field[o + 1] = q8(ra.rnd);
      field[o + 2] = q8(dB / D_SCALE);
      field[o + 3] = q8(ra.f1 / F_SCALE);
    }
  }
}
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, `honeycomb_field_${N}.bin`), field);
console.log(
  `N=${N} TILE_A=${TILE_A} TILE_B=${TILE_B} maxDRoundA=${maxDA.toFixed(3)} maxDRoundB=${maxDB.toFixed(3)} maxF1=${maxF1.toFixed(3)} ` +
    `(D_SCALE=${D_SCALE}, F_SCALE=${F_SCALE}) in ${Date.now() - t0} ms -> ${outDir}`
);
