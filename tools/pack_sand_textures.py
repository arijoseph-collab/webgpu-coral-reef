"""Pack the Houdini sand-lab bake (houdini/sand_outputs.npz, houdini/grain_tile.npz) into WebGPU textures.

public/textures/sand/
  sand_patch_nh.png   1024^2 RGBA8  R,G = normal x,z (0.5 + 0.5 n), B = cavity AO, A = height (normalised)
  sand_patch_mat.png  1024^2 RGBA8  R = albedo luminance (0.5 = x1), G = detritus, B = shell grit, A = coralline
  sand_grain.png       512^2 RGBA8  tileable 0.6 m grain tile: R,G = normal x,z, B = albedo (0.5 = x1), A = shell bits
The patch covers world x, z in [-4.5, 4.5] (texel row 0 = z = -4.5, column 0 = x = -4.5).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

root = sys.argv[1]
out = os.path.join(root, "public", "textures", "sand")
os.makedirs(out, exist_ok=True)
d = np.load(os.path.join(root, "houdini", "sand_outputs.npz"))


def to_rows_z_cols_x(a):
    """Volume layout [j][i] -> texture layout rows = +z, cols = +x (using the exported voxel positions)."""
    p00, p10, p01 = d["_p00"], d["_p10"], d["_p01"]
    i_is_x = abs(p10[0] - p00[0]) > abs(p10[2] - p00[2])
    if not i_is_x:
        a = a.T
    if (p10 - p00)[0 if i_is_x else 2] < 0:
        a = a[:, ::-1]
    if (p01 - p00)[2 if i_is_x else 0] < 0:
        a = a[::-1, :]
    return np.ascontiguousarray(a)


h = to_rows_z_cols_x(d["height"]).astype(np.float64)
crest = to_rows_z_cols_x(d["crest"])
dist = to_rows_z_cols_x(d["coral_dist"])
burrow = to_rows_z_cols_x(d["burrow"])
patch = to_rows_z_cols_x(d["patch"])
N = h.shape[0]
cell = 9.0 / N
h = h - np.median(h)


def blur(a, sigma_px):
    k = np.fft.fftfreq(a.shape[0])
    g = np.exp(-(k[:, None] ** 2 + k[None, :] ** 2) * (2 * np.pi * sigma_px) ** 2 / 2)
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g))


def normals(hh, texel):
    gx = (np.roll(hh, -1, 1) - np.roll(hh, 1, 1)) / (2 * texel)
    gz = (np.roll(hh, -1, 0) - np.roll(hh, 1, 0)) / (2 * texel)
    n = np.stack([-gx, np.ones_like(hh), -gz], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def u8(a):
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)


# ---- patch: normal + cavity + directional sun horizon shadow + metric height
# Amplify vortex-ripple & burrow/grit micro-relief so 3D vertex displacement + raking bump read crisply
h_disp = h * 1.45
n = normals(h_disp, cell)
cav = h_disp - blur(h_disp, 0.035 / cell)  # concave (< 0) -> occluded
cav_ao = np.clip(1.0 + cav * 26.0, 0.48, 1.0)

# 8-step horizon self-shadow march toward the refracted sun (SUN_DIR_WATER ~ (0.28, 0.92, 0.27), raking angle ~28 deg)
sun_horiz = np.array([0.28, 0.27], dtype=np.float64)
sun_horiz /= np.linalg.norm(sun_horiz)
tan_elev = 0.52  # effective raking elevation so lee slopes, burrow craters & grit chips cast crisp soft shadows
max_occ = np.zeros_like(h_disp)
yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
for s_step in (1.5, 3.0, 5.0, 7.5, 10.5, 14.0, 18.0, 23.0):
    sx = np.clip(xx + sun_horiz[0] * s_step, 0, N - 1.001)
    sz = np.clip(yy + sun_horiz[1] * s_step, 0, N - 1.001)
    x0 = np.floor(sx).astype(int); z0 = np.floor(sz).astype(int)
    tx = sx - x0; tz = sz - z0
    hs = (h_disp[z0, x0] * (1 - tx) * (1 - tz) + h_disp[z0, x0 + 1] * tx * (1 - tz) +
          h_disp[z0 + 1, x0] * (1 - tx) * tz + h_disp[z0 + 1, x0 + 1] * tx * tz)
    rise = (hs - h_disp) - (s_step * cell * tan_elev)
    max_occ = np.maximum(max_occ, rise / (s_step * cell * 0.45 + 1e-4))
sun_ao = np.clip(1.0 - np.clip(max_occ, 0.0, 1.0) * 0.42, 0.55, 1.0)
ao = np.clip(cav_ao * sun_ao, 0.42, 1.0)

# Fixed metric height encoding: 0.5 = 0 m, +-0.5 = +-0.035 m (+-3.5 cm) so JS mesh displacement is exact in metres!
HEIGHT_SCALE_M = 0.035
hn = np.clip(0.5 + h_disp / (2.0 * HEIGHT_SCALE_M), 0.0, 1.0)
nh = np.stack([n[..., 0] * 0.5 + 0.5, n[..., 2] * 0.5 + 0.5, ao, hn], -1)
Image.fromarray(u8(nh), "RGBA").save(os.path.join(out, "sand_patch_nh.png"), optimize=True)

# ---- patch: material masks
grit = np.clip((h - blur(h, 0.012 / cell)) * 260.0, 0, 1) * np.clip((dist - 0.05) / 0.3, 0, 1)
trough = np.clip(1.0 - crest * 1.6, 0, 1)  # crest: 0 trough .. 1 crest
sheltered = np.exp(-dist / 0.35)
detritus = np.clip(trough * 0.55 * (0.6 + 0.4 * np.clip(patch, -1, 1)) + sheltered * 0.25 + burrow * 0.35, 0, 1)
lum = 0.5 + 0.06 * crest - 0.05 * trough + 0.035 * np.clip(patch, -2, 2) + 0.04 * (burrow > 0.6) * (hn > 0.6)
rng = np.random.default_rng(5)
fleck_noise = blur(rng.standard_normal((N, N)), 1.2)
fleck_noise = (fleck_noise - fleck_noise.mean()) / fleck_noise.std()
coralline = np.clip((fleck_noise - 1.6) * 0.8, 0, 1) * np.exp(-np.maximum(dist - 0.1, 0) / 0.9)
mat = np.stack([np.clip(lum, 0, 1), detritus, grit, coralline], -1)
Image.fromarray(u8(mat), "RGBA").save(os.path.join(out, "sand_patch_mat.png"), optimize=True)

# ---- grain tile
g = np.load(os.path.join(root, "houdini", "grain_tile.npz"))
gh = g["height"].astype(np.float64)
gT = float(g["T"])
gcell = gT / gh.shape[0]
gn = normals(gh, gcell)
# grains are tiny: keep their normal variation moderate so the tile reads as texture, not gravel
gn = gn * np.array([0.55, 1.0, 0.55])
gn /= np.linalg.norm(gn, axis=-1, keepdims=True)
gtile = np.stack([gn[..., 0] * 0.5 + 0.5, gn[..., 2] * 0.5 + 0.5, np.clip(g["albedo"], 0, 1), np.clip(g["shell"], 0, 1)], -1)
Image.fromarray(u8(gtile), "RGBA").save(os.path.join(out, "sand_grain.png"), optimize=True)

stats = {
    "patch_texel_mm": round(cell * 1000, 2),
    "height_cm": [round(float(h.min()) * 100, 2), round(float(h.max()) * 100, 2)],
    "grit_fraction": round(float((grit > 0.3).mean()), 4),
    "detritus_mean": round(float(detritus.mean()), 3),
    "grain_texel_mm": round(gcell * 1000, 2),
    "files": sorted(os.listdir(out)),
}
print(json.dumps(stats))
