"""Pack high-relief WebGPU bump, normal, AO & height textures from the PolyHaven coral/sand scans.

Generates:
  public/textures/coral_bump_nh.png      1024^2 RGBA8 (tileable)
    R,G = tangent normal x,y (0.5 + 0.5 n) from coral_stone_wall + coral_fort_wall_01
    B   = micro-cavity AO
    A   = normalized height [0..1] for parallax relief

  public/textures/reef_substrate_nh.png  1024^2 RGBA8 (tileable)
    R,G = tangent normal x,y from coral_gravel + coast_sand_rocks_02
    B   = gravel/rock crevice AO
    A   = normalized height [0..1] for seabed Steep Parallax Occlusion Mapping (POM)
"""
import json
import os
import sys
import numpy as np
from PIL import Image

root = sys.argv[1] if len(sys.argv) > 1 else "/Users/arijoseph/Desktop/Houdini Install/webgpu-coral-reef"
ph = os.path.join(root, "public", "textures", "polyhaven")
out_dir = os.path.join(root, "public", "textures")
RES = 1024


def load_gray(name):
    im = Image.open(os.path.join(ph, name)).convert("L").resize((RES, RES), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    p1, p99 = np.percentile(a, 1.0), np.percentile(a, 99.0)
    return np.clip((a - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)


def load_nor(name):
    im = Image.open(os.path.join(ph, name)).convert("RGB").resize((RES, RES), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0 * 2.0 - 1.0
    norm = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-6
    return a / norm


def fft_blur(a, sigma_px):
    k = np.fft.fftfreq(a.shape[0])
    g = np.exp(-(k[:, None] ** 2 + k[None, :] ** 2) * (2.0 * np.pi * sigma_px) ** 2 * 0.5)
    return np.real(np.fft.ifft2(np.fft.fft2(a) * g)).astype(np.float32)


def height_to_normal(h, strength):
    gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5 * strength
    gy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5 * strength
    n = np.stack([-gx, -gy, np.ones_like(h)], axis=-1)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)


def blend_rnm(n1, n2, w2=1.0):
    n2s = np.stack([n2[..., 0] * w2, n2[..., 1] * w2, n2[..., 2]], axis=-1)
    n2s /= np.linalg.norm(n2s, axis=-1, keepdims=True) + 1e-6
    t = n1 + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    u = n2s * np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    r = t * np.sum(t * u, axis=-1, keepdims=True) / (t[..., 2:3] + 1e-6) - u
    return r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-6)


def u8(a):
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)


# 1) Coral calcified skeleton stony micro-relief (coral_stone_wall + coral_fort_wall_01)
cs_d = load_gray("coral_stone_wall_disp.png")
cs_n = load_nor("coral_stone_wall_nor.png")
cs_ao = load_gray("coral_stone_wall_ao.png")

cf_d = load_gray("coral_fort_wall_01_disp.png")
cf_n = load_nor("coral_fort_wall_01_nor.png")
cf_ao = load_gray("coral_fort_wall_01_ao.png")

# High-pass + meso-relief so it adds stony calcareous ridges, pits, and spinules without low-freq blobs
coral_h_raw = 0.62 * cs_d + 0.38 * cf_d
coral_hp = coral_h_raw - fft_blur(coral_h_raw, 28.0)
p2, p98 = np.percentile(coral_hp, 2.0), np.percentile(coral_hp, 98.0)
coral_h = np.clip((coral_hp - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

coral_hn = height_to_normal(coral_h, strength=14.0)
coral_scan_n = blend_rnm(cs_n, cf_n, 0.65)
# Note: flipY=false in WebGPU means +v is +row (down image), while PolyHaven OpenGL normals have +G = up (-row)
coral_scan_n = np.stack([coral_scan_n[..., 0], -coral_scan_n[..., 1], coral_scan_n[..., 2]], axis=-1)
coral_n = blend_rnm(coral_hn, coral_scan_n, 1.15)

cav = coral_h - fft_blur(coral_h, 6.0)
coral_ao = np.clip((0.65 * cs_ao + 0.35 * cf_ao) * np.clip(1.0 + cav * 1.4, 0.55, 1.08), 0.35, 1.0)

coral_pack = np.stack([coral_n[..., 0] * 0.5 + 0.5, coral_n[..., 1] * 0.5 + 0.5, coral_ao, coral_h], axis=-1)
Image.fromarray(u8(coral_pack), "RGBA").save(os.path.join(out_dir, "coral_bump_nh.png"), optimize=True)

# 2) Reef substrate: coarse coral gravel + coastal sand rocks (for seabed POM & reef-apron relief)
cg_d = load_gray("coral_gravel_disp.png")
cg_n = load_nor("coral_gravel_nor.png")
cg_ao = load_gray("coral_gravel_ao.png")

sr_d = load_gray("coast_sand_rocks_02_disp.png")
sr_n = load_nor("coast_sand_rocks_02_nor.png")
sr_ao = load_gray("coast_sand_rocks_02_ao.png")

sub_h_raw = 0.58 * cg_d + 0.42 * sr_d
sub_hp = sub_h_raw - fft_blur(sub_h_raw, 42.0)
sp2, sp98 = np.percentile(sub_hp, 2.0), np.percentile(sub_hp, 98.0)
sub_h = np.clip((sub_hp - sp2) / (sp98 - sp2 + 1e-6), 0.0, 1.0)

sub_hn = height_to_normal(sub_h, strength=16.0)
sub_scan_n = blend_rnm(cg_n, sr_n, 0.75)
sub_scan_n = np.stack([sub_scan_n[..., 0], -sub_scan_n[..., 1], sub_scan_n[..., 2]], axis=-1)
sub_n = blend_rnm(sub_hn, sub_scan_n, 1.10)

sub_cav = sub_h - fft_blur(sub_h, 7.0)
sub_ao = np.clip((0.55 * cg_ao + 0.45 * sr_ao) * np.clip(1.0 + sub_cav * 1.6, 0.48, 1.08), 0.30, 1.0)

sub_pack = np.stack([sub_n[..., 0] * 0.5 + 0.5, sub_n[..., 1] * 0.5 + 0.5, sub_ao, sub_h], axis=-1)
Image.fromarray(u8(sub_pack), "RGBA").save(os.path.join(out_dir, "reef_substrate_nh.png"), optimize=True)

print(json.dumps({
    "coral_bump_horiz_mean": round(float(np.hypot(coral_n[..., 0], coral_n[..., 1]).mean()), 3),
    "reef_sub_horiz_mean": round(float(np.hypot(sub_n[..., 0], sub_n[..., 1]).mean()), 3),
}))
