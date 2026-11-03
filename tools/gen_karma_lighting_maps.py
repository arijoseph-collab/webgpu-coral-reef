"""Generate physical HDR underwater dome maps & 2048^2 refracted wave-caustic gobo for Houdini Karma.

Outputs into webgpu-coral-reef/houdini/:
  karma_dome_pastel.hdr     2048x1024 RGBE HDR lat-long dome (Pastel Reference water column + Snell's window)
  karma_dome_tropical.hdr   2048x1024 RGBE HDR lat-long dome (Tropical Turquoise lagoon + Snell's window)
  karma_caustic_gobo.png    2048x2048 seamless refracted 40-wave Jacobian caustic projection texture
"""
import math
import os
import sys
import numpy as np
from PIL import Image

root = sys.argv[1] if len(sys.argv) > 1 else "/Users/arijoseph/Desktop/Houdini Install/webgpu-coral-reef"
out_dir = os.path.join(root, "houdini")
os.makedirs(out_dir, exist_ok=True)


def save_rgbe_hdr(path, rgb):
    """Write a standard Radiance .hdr (RGBE, uncompressed scanlines for universal USD/OpenImageIO/Karma support)."""
    h, w, _ = rgb.shape
    rgb = np.maximum(rgb, 0.0)
    max_c = np.max(rgb, axis=-1)
    rgbe = np.zeros((h, w, 4), dtype=np.uint8)
    mask = max_c > 1e-32
    mant, exp = np.frexp(max_c[mask])
    scale = mant * 256.0 / max_c[mask]
    rgbe[mask, 0] = np.clip(rgb[mask, 0] * scale, 0, 255).astype(np.uint8)
    rgbe[mask, 1] = np.clip(rgb[mask, 1] * scale, 0, 255).astype(np.uint8)
    rgbe[mask, 2] = np.clip(rgb[mask, 2] * scale, 0, 255).astype(np.uint8)
    rgbe[mask, 3] = np.clip(exp + 128, 0, 255).astype(np.uint8)
    header = (
        b"#?RADIANCE\n"
        b"FORMAT=32-bit_rle_rgbe\n\n"
        + f"-Y {h} +X {w}\n".encode("ascii")
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(rgbe.tobytes())


def make_dome(mode):
    W, H = 2048, 1024
    u = (np.arange(W, dtype=np.float32) + 0.5) / W
    v = (np.arange(H, dtype=np.float32) + 0.5) / H
    phi = (u[None, :] - 0.5) * (2.0 * math.pi)
    theta = v[:, None] * math.pi  # 0 = +Y zenith, pi = -Y nadir
    dy = np.cos(theta)
    sin_t = np.sin(theta)
    dx = sin_t * np.sin(phi)
    dz = sin_t * np.cos(phi)

    # Refracted sun direction in water: normalize((0.28, 0.92, 0.27))
    sun = np.array([0.28, 0.92, 0.27], dtype=np.float32)
    sun /= np.linalg.norm(sun)
    sun_dot = np.clip(dx * sun[0] + dy * sun[1] + dz * sun[2], -1.0, 1.0)

    if mode == "pastel":
        deep = np.array([0.16, 0.10, 0.34], dtype=np.float32)
        mid = np.array([0.44, 0.34, 0.70], dtype=np.float32)
        shallow = np.array([0.74, 0.70, 0.96], dtype=np.float32)
        window_sky = np.array([1.25, 1.20, 1.38], dtype=np.float32)
        sun_col = np.array([3.6, 3.3, 3.1], dtype=np.float32)
    else:
        deep = np.array([0.015, 0.09, 0.22], dtype=np.float32)
        mid = np.array([0.05, 0.36, 0.58], dtype=np.float32)
        shallow = np.array([0.34, 0.80, 0.94], dtype=np.float32)
        window_sky = np.array([0.95, 1.32, 1.45], dtype=np.float32)
        sun_col = np.array([3.8, 3.5, 3.0], dtype=np.float32)

    t_mid = np.clip((dy + 0.45) / 0.65, 0.0, 1.0)
    t_mid = t_mid * t_mid * (3.0 - 2.0 * t_mid)
    t_up = np.clip((dy - 0.02) / 0.68, 0.0, 1.0)
    t_up = t_up * t_up * (3.0 - 2.0 * t_up)

    col = deep[None, None, :] * (1.0 - t_mid[..., None]) + mid[None, None, :] * t_mid[..., None]
    col = col * (1.0 - 0.78 * t_up[..., None]) + shallow[None, None, :] * (0.78 * t_up[..., None])

    # Snell's window (critical angle 48.6 deg -> cos(48.6 deg) = 0.661)
    snell = np.clip((dy - 0.60) / 0.14, 0.0, 1.0)
    snell = snell * snell * (3.0 - 2.0 * snell)
    col = col * (1.0 - 0.65 * snell[..., None]) + window_sky[None, None, :] * (0.65 * snell[..., None])

    # Solar aureole inside Snell's window
    glow = (np.maximum(sun_dot, 0.0) ** 28.0) * 0.85 + (np.maximum(sun_dot, 0.0) ** 6.0) * 0.16
    col = col + sun_col[None, None, :] * glow[..., None]
    return col.astype(np.float32)


save_rgbe_hdr(os.path.join(out_dir, "karma_dome_pastel.hdr"), make_dome("pastel"))
save_rgbe_hdr(os.path.join(out_dir, "karma_dome_tropical.hdr"), make_dome("tropical"))

# Generate seamless 1024x1024 refracted 40-wave Jacobian caustic projection map (splatted onto 5 m periodic tile)
N = 1024
rng = np.random.default_rng(202611)
L = 5.0
yy, xx = np.mgrid[0:N, 0:N].astype(np.float32) / N * L
hx = np.zeros((N, N), dtype=np.float32)
hz = np.zeros((N, N), dtype=np.float32)
for i in range(40):
    f = i / 39.0
    lam = 0.12 * ((0.58 / 0.12) ** f)
    k = round(L / lam) * (2.0 * math.pi / L)
    ang = 0.6 + (rng.random() - 0.5) * 1.15
    kx = round(k * math.cos(ang) * L / (2.0 * math.pi)) * (2.0 * math.pi / L)
    kz = round(k * math.sin(ang) * L / (2.0 * math.pi)) * (2.0 * math.pi / L)
    km = math.hypot(kx, kz)
    if km < 1e-4:
        continue
    amp = 0.0022 * ((0.25 / lam) ** -0.35)
    ph = rng.random() * 2.0 * math.pi
    c = np.cos(kx * xx + kz * yy + ph)
    hx += amp * kx * c
    hz += amp * kz * c

# Refract vertical rays to 3.8 m depth and splat area-conserving irradiance onto periodic grid
depth = 3.6
eta = 1.0 / 1.334
shift_x = -hx * (1.0 - eta) * depth
shift_z = -hz * (1.0 - eta) * depth
px = ((xx + shift_x) / L * N) % N
pz = ((yy + shift_z) / L * N) % N
ix = np.floor(px).astype(np.int32) % N
iz = np.floor(pz).astype(np.int32) % N
tx = px - np.floor(px)
tz = pz - np.floor(pz)

irr = np.zeros((N, N), dtype=np.float32)
np.add.at(irr, (iz, ix), (1 - tx) * (1 - tz))
np.add.at(irr, (iz, (ix + 1) % N), tx * (1 - tz))
np.add.at(irr, ((iz + 1) % N, ix), (1 - tx) * tz)
np.add.at(irr, ((iz + 1) % N, (ix + 1) % N), tx * tz)

# Slight sun-disk penumbra blur so focal cusps are silky and anti-aliased
kf = np.fft.fftfreq(N)
gauss = np.exp(-(kf[:, None] ** 2 + kf[None, :] ** 2) * (2.0 * math.pi * 1.65) ** 2 * 0.5)
irr = np.real(np.fft.ifft2(np.fft.fft2(irr) * gauss)).astype(np.float32)
irr = np.clip(irr / (irr.mean() + 1e-6), 0.0, 4.0)
# Encode into [0..255] where mean ~ 0.32 -> 82/255 and bright cusps reach 1.0
gobo = np.clip((irr / 3.2) ** 0.85, 0.0, 1.0)
Image.fromarray(np.round(gobo * 255.0).astype(np.uint8), "L").save(os.path.join(out_dir, "karma_caustic_gobo.png"))
print("Generated Karma HDR domes and caustic gobo:", round(float(irr.min()), 3), round(float(irr.max()), 3))
