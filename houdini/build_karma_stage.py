"""
build_karma_stage.py — Houdini FX 22.0 Solaris + Karma Photorealistic Reef Benchmark Builder

Builds `/obj/karma_reef_scene` + `/stage` (100% pure MaterialX `mtlx*` + Karma path-tracing,
zero `sidefx/shps` dependency) for photorealistic ground-truth renders of the coral reef:
  1. High-resolution Houdini coral bouquet (`/obj/coral_reef/OUT_CORAL_REEF` merged from
     `coral_reef_biome.hipnc`, stray floating Acropora piece removed, Hero/Macro zone subdivided,
     and displaced in 3D by cellular calice craters, PolyHaven limestone scan height, Brain Coral
     meandroid septa teeth, and Foliose Fan radial costae ribs).
  2. 768x768 (1.18M-triangle) true 3D displaced Houdini Sand Lab floor (`houdini/sand_outputs.npz`
     Swift-Hohenberg vortex ripples, bifurcations, coral scour moats, burrows, grazing trails,
     shell grit + `reef_substrate_nh.png` 3D reef-apron gravel & carbonate clasts).
  3. Full 3D Indo-Pacific reef ecosystem: weathered coral rubble apron, pitted limestone boulders,
     ochre tube sponges, giant Xestospongia barrel sponge, clonal Entacmaea bubble-tip anemones,
     Amphiprion ocellaris clownfish, schooling Pseudanthias/Chromis/butterflyfish, lacy gorgonian
     sea fans, tapered sea whips, Thalassia seagrass meadows, Diadema urchins, blue Linckia star,
     iridescent Tridacna giant clam, distant bommies, and 3D marine snow motes.
  4. Solaris + MaterialX lookdev: Random-Walk Subsurface Scattering (`mtlxstandard_surface`),
     3D Worley `f2 - f1` trabecular pores, triplanar PolyHaven normal/bump detail, wet mucus
     clearcoat + thin-film interference, HDR Snell's-window underwater dome (`karma_dome_*.hdr`),
     collimated caustic projection (`karma_caustic_gobo.png`), volumetric `karmafogbox` god-rays,
     and physical thin-lens Depth-of-Field USD cameras (`cam_hero`, `cam_wide`, `cam_macro`).
"""

import math
import os
import time
import hou
import numpy as np
from PIL import Image

PROJ_ROOT = "/Users/arijoseph/Desktop/Houdini Install/webgpu-coral-reef"
BIOME_HIP = "/Users/arijoseph/Desktop/Houdini Install/coral_reef_biome.hipnc"
KARMA_HIP = os.path.join(PROJ_ROOT, "houdini", "coral_reef_karma.hipnc")
TEX_DIR = os.path.join(PROJ_ROOT, "public", "textures")
HOUDINI_DIR = os.path.join(PROJ_ROOT, "houdini")


# -----------------------------------------------------------------------------
# Fast vectorized 3D noise & sampling helpers (NumPy)
# -----------------------------------------------------------------------------
def hash33_np(ix, iy, iz):
    px = np.modf(ix * 0.1031)[0]
    py = np.modf(iy * 0.1030)[0]
    pz = np.modf(iz * 0.0973)[0]
    px = np.where(px < 0, px + 1.0, px)
    py = np.where(py < 0, py + 1.0, py)
    pz = np.where(pz < 0, pz + 1.0, pz)
    d = px * (py + 33.33) + py * (px + 33.33) + pz * (pz + 33.33)
    px += d
    py += d
    pz += d
    hx = np.modf((px + py) * pz)[0]
    hy = np.modf((px + px) * py)[0]
    hz = np.modf((py + px) * px)[0]
    return np.where(hx < 0, hx + 1.0, hx), np.where(hy < 0, hy + 1.0, hy), np.where(hz < 0, hz + 1.0, hz)


def worley_f2_f1_np(pts, freq=70.0, jitter=0.80):
    """Vectorized 27-neighbor 3D Worley f2 - f1 cellular honeycomb distance."""
    sp = pts * freq
    bx = np.floor(sp[:, 0])
    by = np.floor(sp[:, 1])
    bz = np.floor(sp[:, 2])
    fx = sp[:, 0] - bx
    fy = sp[:, 1] - by
    fz = sp[:, 2] - bz
    f1 = np.full(len(pts), 10.0, dtype=np.float32)
    f2 = np.full(len(pts), 10.0, dtype=np.float32)
    for dz in (-1.0, 0.0, 1.0):
        for dy in (-1.0, 0.0, 1.0):
            for dx in (-1.0, 0.0, 1.0):
                hx, hy, hz = hash33_np(bx + dx, by + dy, bz + dz)
                sx = dx + 0.5 + (hx - 0.5) * jitter - fx
                sy = dy + 0.5 + (hy - 0.5) * jitter - fy
                sz = dz + 0.5 + (hz - 0.5) * jitter - fz
                d = np.sqrt(sx * sx + sy * sy + sz * sz).astype(np.float32)
                m = d < f1
                f2 = np.where(m, f1, np.minimum(f2, d))
                f1 = np.where(m, d, f1)
    return f2 - f1, f1


def bilerp_img(img, u, v, wrap=True):
    """Vectorized bilinear sample of HxWxC float32 image in [0,1]. Row 0 = v=0."""
    H, W = img.shape[:2]
    if wrap:
        u = np.mod(u, 1.0)
        v = np.mod(v, 1.0)
    else:
        u = np.clip(u, 0.0, 0.999999)
        v = np.clip(v, 0.0, 0.999999)
    x = u * (W - 1)
    y = v * (H - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = (x0 + 1) % W if wrap else np.minimum(x0 + 1, W - 1)
    y1 = (y0 + 1) % H if wrap else np.minimum(y0 + 1, H - 1)
    tx = (x - x0)[..., None] if img.ndim == 3 else (x - x0)
    ty = (y - y0)[..., None] if img.ndim == 3 else (y - y0)
    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]
    return (c00 * (1.0 - tx) + c10 * tx) * (1.0 - ty) + (c01 * (1.0 - tx) + c11 * tx) * ty


def smoothstep_np(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_vertex_normals(pts, tris):
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    # Houdini clockwise outward winding: cross(v2 - v0, v1 - v0) points outward (+Y on flat grid)
    fn = np.cross(v2 - v0, v1 - v0)
    vn = np.zeros_like(pts, dtype=np.float32)
    np.add.at(vn, tris[:, 0], fn)
    np.add.at(vn, tris[:, 1], fn)
    np.add.at(vn, tris[:, 2], fn)
    l = np.linalg.norm(vn, axis=1, keepdims=True)
    return (vn / np.maximum(l, 1e-8)).astype(np.float32)


# -----------------------------------------------------------------------------
# Procedural mesh builders (ecosystem props, anemones, fish, fans, whips, grass)
# -----------------------------------------------------------------------------
class MeshAccumulator:
    def __init__(self):
        self.pts = []
        self.tris = []
        self.wall_cd = []
        self.cavity_cd = []
        self.ao = []
        self.tip_mask = []
        self.sea_blend = []
        self.names = []
        self.v_offset = 0

    def add(self, pts, tris, wall_cd, name, cavity_cd=None, ao=None, tip_mask=None, sea_blend=None, linearize=True):
        pts = np.asarray(pts, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.int32)
        n = len(pts)
        m = len(tris)
        if n == 0 or m == 0:
            return
        wcd = np.broadcast_to(np.asarray(wall_cd, dtype=np.float32), (n, 3)).copy()
        if linearize:
            wcd = np.clip(wcd, 0.005, 1.0) ** 2.05
        ccd = np.broadcast_to(np.asarray(cavity_cd if cavity_cd is not None else wcd * 0.42, dtype=np.float32), (n, 3)).copy()
        ao_a = np.broadcast_to(np.asarray(ao if ao is not None else 1.0, dtype=np.float32), (n,)).copy()
        tm_a = np.broadcast_to(np.asarray(tip_mask if tip_mask is not None else 0.0, dtype=np.float32), (n,)).copy()
        sb_a = np.broadcast_to(np.asarray(sea_blend if sea_blend is not None else 0.0, dtype=np.float32), (n,)).copy()

        self.pts.append(pts)
        self.tris.append(tris + self.v_offset)
        self.wall_cd.append(wcd)
        self.cavity_cd.append(ccd)
        self.ao.append(ao_a)
        self.tip_mask.append(tm_a)
        self.sea_blend.append(sb_a)
        self.names.extend([name] * m)
        self.v_offset += n

    def finalize(self):
        pts = np.vstack(self.pts)
        tris = np.vstack(self.tris)
        nrm = compute_vertex_normals(pts, tris)
        return {
            "P": pts,
            "tris": tris,
            "N": nrm,
            "wall_cd": np.vstack(self.wall_cd),
            "cavity_cd": np.vstack(self.cavity_cd),
            "ao": np.concatenate(self.ao),
            "tip_mask": np.concatenate(self.tip_mask),
            "sea_blend": np.concatenate(self.sea_blend),
            "names": self.names,
        }


def make_uv_sphere(nu=14, nv=10):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0.08, np.pi - 0.08, nv)
    uu, vv = np.meshgrid(u, v)
    x = (np.sin(vv) * np.cos(uu)).ravel()
    y = np.cos(vv).ravel()
    z = (np.sin(vv) * np.sin(uu)).ravel()
    pts = np.vstack([x, y, z, [0, 1, 0], [0, -1, 0]]).T.astype(np.float32) if False else None
    pts = np.column_stack([x, y, z]).astype(np.float32)
    top_i = len(pts)
    bot_i = len(pts) + 1
    pts = np.vstack([pts, [[0, 1, 0], [0, -1, 0]]]).astype(np.float32)
    tris = []
    for j in range(nv - 1):
        for i in range(nu):
            i1 = (i + 1) % nu
            a = j * nu + i
            b = j * nu + i1
            c = (j + 1) * nu + i
            d = (j + 1) * nu + i1
            tris.append([a, b, c])
            tris.append([b, d, c])
    for i in range(nu):
        i1 = (i + 1) % nu
        tris.append([top_i, i1, i])
        b0 = (nv - 1) * nu + i
        b1 = (nv - 1) * nu + i1
        tris.append([b0, b1, bot_i])
    return pts, np.array(tris, dtype=np.int32)


def make_tube_along_curve(spine_pts, radii, n_sides=8, cap_tip=True):
    """Builds a smooth swept 3D tube along spine_pts (M,3) with radii (M,)."""
    M = len(spine_pts)
    tangents = np.zeros_like(spine_pts)
    tangents[:-1] = spine_pts[1:] - spine_pts[:-1]
    tangents[-1] = tangents[-2]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(tangents[0], up)) > 0.9:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ang = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    verts = []
    s_vals = []
    nrm_ref = np.cross(tangents[0], up)
    nrm_ref /= max(np.linalg.norm(nrm_ref), 1e-6)
    for k in range(M):
        t = tangents[k]
        nrm_ref = nrm_ref - t * np.dot(nrm_ref, t)
        nl = np.linalg.norm(nrm_ref)
        if nl < 1e-5:
            nrm_ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            nrm_ref /= nl
        bn = np.cross(t, nrm_ref)
        ring = spine_pts[k][None, :] + radii[k] * (ca[:, None] * nrm_ref[None, :] + sa[:, None] * bn[None, :])
        verts.append(ring)
        s_vals.append(np.full(n_sides, k / max(M - 1, 1), dtype=np.float32))
    pts = np.vstack(verts)
    s_arr = np.concatenate(s_vals)
    tris = []
    for k in range(M - 1):
        for i in range(n_sides):
            i1 = (i + 1) % n_sides
            a = k * n_sides + i
            b = k * n_sides + i1
            c = (k + 1) * n_sides + i
            d = (k + 1) * n_sides + i1
            tris.append([a, c, b])
            tris.append([b, c, d])
    if cap_tip:
        tip_idx = len(pts)
        pts = np.vstack([pts, [spine_pts[-1] + tangents[-1] * (radii[-1] * 0.45)]])
        s_arr = np.append(s_arr, 1.0)
        base = (M - 1) * n_sides
        for i in range(n_sides):
            i1 = (i + 1) % n_sides
            tris.append([base + i, tip_idx, base + i1])
    return pts.astype(np.float32), np.array(tris, dtype=np.int32), s_arr.astype(np.float32)


def make_fish_mesh(length_m=0.075, height_m=0.036, width_m=0.016, is_clown=True):
    """Articulated fusiform reef fish with caudal + dorsal + pectoral fins and stripe pattern."""
    nu, nv = 16, 10
    pts, tris = make_uv_sphere(nu, nv)
    # Remap sphere y in [-1, 1] -> nose (+1) to peduncle (-1)
    s = (1.0 - pts[:, 1]) * 0.5  # 0 at nose, 1 at tail
    body_env = np.sin(np.clip(s, 0.02, 0.98) * np.pi) ** 0.68
    peduncle = np.where(s > 0.78, 0.32 + 0.68 * np.cos((s - 0.78) / 0.22 * np.pi * 0.5), 1.0)
    x = pts[:, 0] * width_m * body_env * peduncle
    z = pts[:, 2] * height_m * body_env * (0.85 + 0.25 * (1.0 - s))
    y_fwd = (0.5 - s) * length_m  # +y_fwd is nose along +X later
    f_pts = np.column_stack([y_fwd, z, x]).astype(np.float32)

    # Caudal fin fan
    tail_x0 = -0.46 * length_m
    tail_x1 = -0.74 * length_m
    fin_pts = [
        [tail_x0, 0.0, 0.0],
        [tail_x1, height_m * 0.72, 0.001],
        [tail_x1 * 1.05, height_m * 0.24, -0.001],
        [tail_x1 * 1.05, -height_m * 0.24, 0.001],
        [tail_x1, -height_m * 0.72, -0.001],
        # Dorsal ridge fin
        [0.12 * length_m, height_m * 0.72, 0.0],
        [-0.08 * length_m, height_m * 1.18, 0.001],
        [-0.32 * length_m, height_m * 0.92, -0.001],
        [-0.38 * length_m, height_m * 0.35, 0.0],
    ]
    b = len(f_pts)
    f_pts = np.vstack([f_pts, np.array(fin_pts, dtype=np.float32)])
    fin_tris = [
        [b, b + 1, b + 2], [b, b + 2, b + 3], [b, b + 3, b + 4],
        [b, b + 2, b + 1], [b, b + 3, b + 2], [b, b + 4, b + 3],
        [b + 5, b + 6, b + 7], [b + 5, b + 7, b + 8],
        [b + 5, b + 7, b + 6], [b + 5, b + 8, b + 7],
    ]
    tris = np.vstack([tris, np.array(fin_tris, dtype=np.int32)])

    u_head = (f_pts[:, 0] / length_m) + 0.5  # 1 at nose, 0 at tail
    if is_clown:
        cd = np.tile(np.array([0.98, 0.42, 0.11], dtype=np.float32), (len(f_pts), 1))
        # 3 white Amphiprion bars with thin dark rims
        for bar_c, bar_w in [(0.76, 0.065), (0.46, 0.075), (0.14, 0.055)]:
            d = np.abs(u_head - bar_c)
            rim = (d < bar_w * 1.38) & (d >= bar_w)
            wht = d < bar_w
            cd[rim] = [0.08, 0.06, 0.07]
            cd[wht] = [0.97, 0.96, 0.99]
        tip_black = f_pts[:, 0] < tail_x1 * 0.92
        cd[tip_black] = [0.10, 0.07, 0.08]
    else:
        cd = np.ones((len(f_pts), 3), dtype=np.float32)
    return f_pts, tris, cd


# -----------------------------------------------------------------------------
# Main Stage & SOP Builder
# -----------------------------------------------------------------------------
def build_karma_reef():
    t_start = time.time()
    hou.hipFile.clear(suppress_save_prompt=True)
    hou.hipFile.merge(BIOME_HIP)
    hou.hipFile.setName(KARMA_HIP)

    hou.ui.desktop("Solaris").setAsCurrent()
    viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
    if viewer:
        viewer.setHydraRenderer("Karma CPU")

    # Load textures & Houdini Sand Lab arrays
    sand_npz = np.load(os.path.join(HOUDINI_DIR, "sand_outputs.npz"))
    sand_h_grid = sand_npz["height"].astype(np.float32)  # (1024, 1024) in m
    coral_bump_img = np.asarray(Image.open(os.path.join(TEX_DIR, "coral_bump_nh.png")).convert("RGBA"), dtype=np.float32) / 255.0
    sub_nh_img = np.asarray(Image.open(os.path.join(TEX_DIR, "reef_substrate_nh.png")).convert("RGBA"), dtype=np.float32) / 255.0
    sand_nh_img = np.asarray(Image.open(os.path.join(TEX_DIR, "sand", "sand_patch_nh.png")).convert("RGBA"), dtype=np.float32) / 255.0
    sand_mat_img = np.asarray(Image.open(os.path.join(TEX_DIR, "sand", "sand_patch_mat.png")).convert("RGBA"), dtype=np.float32) / 255.0
    sand_grain_img = np.asarray(Image.open(os.path.join(TEX_DIR, "sand", "sand_grain.png")).convert("RGBA"), dtype=np.float32) / 255.0

    # 1. Extract Coral Reef & Original Seabed Profile from /obj/coral_reef/OUT_CORAL_REEF
    src_geo = hou.node("/obj/coral_reef/OUT_CORAL_REEF").geometry()
    hou.node("/obj/coral_reef").setDisplayFlag(False)
    for light_name in ["reef_cam", "env_light", "sun_light"]:
        if hou.node("/obj/" + light_name):
            hou.node("/obj/" + light_name).setDisplayFlag(False)

    npts = len(src_geo.points())
    P_all = np.array(src_geo.pointFloatAttribValues("P"), dtype=np.float32).reshape(npts, 3)
    N_all = np.array(src_geo.pointFloatAttribValues("N"), dtype=np.float32).reshape(npts, 3)
    wcd_all = np.array(src_geo.pointFloatAttribValues("wall_cd"), dtype=np.float32).reshape(npts, 3)
    ccd_all = np.array(src_geo.pointFloatAttribValues("cavity_cd"), dtype=np.float32).reshape(npts, 3)
    ao_all = np.array(src_geo.pointFloatAttribValues("ao"), dtype=np.float32)
    tm_all = np.array(src_geo.pointFloatAttribValues("tip_mask"), dtype=np.float32)
    pd_all = np.array(src_geo.pointFloatAttribValues("pore_depth"), dtype=np.float32)
    sb_all = np.array(src_geo.pointFloatAttribValues("sea_blend"), dtype=np.float32)

    mat_ids = np.array(src_geo.primIntAttribValues("mat_id"), dtype=np.int32)
    v0 = np.array(src_geo.primIntAttribValues("v0"), dtype=np.int32)
    v1 = np.array(src_geo.primIntAttribValues("v1"), dtype=np.int32)
    v2 = np.array(src_geo.primIntAttribValues("v2"), dtype=np.int32)
    poly_vtx = np.column_stack([v0, v1, v2])

    # Extract original 180x180 seabed height map so our 768x768 Sand Lab floor sits at the exact holdfast height!
    sea_tris = poly_vtx[mat_ids == 4]
    sea_v_idx = np.unique(sea_tris.ravel())
    sea_pts = P_all[sea_v_idx]
    N0 = int(round(math.sqrt(len(sea_pts))))
    sx0, sx1 = float(sea_pts[:, 0].min()), float(sea_pts[:, 0].max())
    sspan = sx1 - sx0
    base_y_grid = np.zeros((N0, N0), dtype=np.float32)
    base_wcd_grid = np.zeros((N0, N0, 3), dtype=np.float32)
    gi = np.clip(np.round((sea_pts[:, 0] - sx0) / sspan * (N0 - 1)).astype(np.int32), 0, N0 - 1)
    gj = np.clip(np.round((sea_pts[:, 2] - sx0) / sspan * (N0 - 1)).astype(np.int32), 0, N0 - 1)
    base_y_grid[gj, gi] = sea_pts[:, 1]
    base_wcd_grid[gj, gi] = wcd_all[sea_v_idx]

    def seabed_base_y(x_arr, z_arr):
        u = np.clip((x_arr - sx0) / sspan, 0.0, 1.0)
        v = np.clip((z_arr - sx0) / sspan, 0.0, 1.0)
        return bilerp_img(base_y_grid, u, v, wrap=False)

    # Compute 2D coral footprint distance grid (256x256 over [-4.5, 4.5]^2) so sand displacement & rubble clear coral bases!
    coral_mask_prims = mat_ids < 4
    coral_tris_raw = poly_vtx[coral_mask_prims]
    coral_mids_raw = mat_ids[coral_mask_prims]

    # Remove disconnected floating Acropora fragment in [0.640, 1.500, 0.215]..[0.775, 1.635, 0.320]
    tri_centers = (P_all[coral_tris_raw[:, 0]] + P_all[coral_tris_raw[:, 1]] + P_all[coral_tris_raw[:, 2]]) / 3.0
    in_float_box = (
        (coral_mids_raw == 3)
        & (tri_centers[:, 0] >= 0.640) & (tri_centers[:, 0] <= 0.775)
        & (tri_centers[:, 1] >= 1.500) & (tri_centers[:, 1] <= 1.635)
        & (tri_centers[:, 2] >= 0.215) & (tri_centers[:, 2] <= 0.320)
    )
    coral_tris_clean = coral_tris_raw[~in_float_box]
    coral_mids_clean = coral_mids_raw[~in_float_box]

    # Build coral footprint distance map (meters) over [-4.5, 4.5]^2
    FP_N = 256
    low_coral_pts = P_all[np.unique(coral_tris_clean.ravel())]
    low_coral_pts = low_coral_pts[(low_coral_pts[:, 1] < 0.82) & (np.abs(low_coral_pts[:, 0]) < 4.4) & (np.abs(low_coral_pts[:, 2]) < 4.4)]
    occ = np.zeros((FP_N, FP_N), dtype=bool)
    fpi = np.clip(((low_coral_pts[:, 0] + 4.5) / 9.0 * FP_N).astype(np.int32), 0, FP_N - 1)
    fpj = np.clip(((low_coral_pts[:, 2] + 4.5) / 9.0 * FP_N).astype(np.int32), 0, FP_N - 1)
    occ[fpj, fpi] = True
    # Fast Chamfer / jump-flood distance transform in meters
    dist_grid = np.where(occ, 0.0, 9.0).astype(np.float32)
    cell_m = 9.0 / FP_N
    for step_px in [64, 32, 16, 8, 4, 2, 1]:
        for dy in (-step_px, 0, step_px):
            for dx in (-step_px, 0, step_px):
                if dx == 0 and dy == 0:
                    continue
                sh = np.roll(np.roll(dist_grid, dy, axis=0), dx, axis=1)
                d_step = math.hypot(dx, dy) * cell_m
                dist_grid = np.minimum(dist_grid, sh + d_step)

    def coral_dist_at(x_arr, z_arr):
        u = np.clip((x_arr + 4.5) / 9.0, 0.0, 1.0)
        v = np.clip((z_arr + 4.5) / 9.0, 0.0, 1.0)
        return bilerp_img(dist_grid, u, v, wrap=False)

    acc = MeshAccumulator()

    # 2. Build 3D Displaced Coral Species Meshes (staghorn, brain, table, sponge)
    sp_specs = [
        (0, "staghorn", 76.0, 0.95, 0.0034),
        (1, "brain",    62.0, 0.38, 0.0028),
        (2, "table",    84.0, 0.84, 0.0026),
        (3, "sponge",   68.0, 0.54, 0.0028),
    ]
    for mid, sp_name, freq, pore_str, disp_amp in sp_specs:
        sp_tris_global = coral_tris_clean[coral_mids_clean == mid]
        u_verts, inv_tris = np.unique(sp_tris_global.ravel(), return_inverse=True)
        sp_tris = inv_tris.reshape(-1, 3).astype(np.int32)
        sp_P = P_all[u_verts].copy()
        sp_N = N_all[u_verts].copy()
        sp_wcd = wcd_all[u_verts].copy()
        sp_ccd = ccd_all[u_verts].copy()
        sp_ao = ao_all[u_verts].copy()
        sp_tm = tm_all[u_verts].copy()
        sp_pd = pd_all[u_verts].copy()
        sp_sb = sb_all[u_verts].copy()

        # Subdivide close-up Hero/Macro faces once so 3D calice craters & septa teeth have ~1.5 mm triangle density!
        tc = (sp_P[sp_tris[:, 0]] + sp_P[sp_tris[:, 1]] + sp_P[sp_tris[:, 2]]) / 3.0
        hero_mask = (np.abs(tc[:, 0]) < 0.72) & (tc[:, 1] > 0.55) & (tc[:, 1] < 1.75) & (tc[:, 2] > -0.35) & (tc[:, 2] < 1.15)
        if np.any(hero_mask):
            keep_t = sp_tris[~hero_mask]
            sub_t = sp_tris[hero_mask]
            a_i, b_i, c_i = sub_t[:, 0], sub_t[:, 1], sub_t[:, 2]
            n_cur = len(sp_P)
            # Midpoint vertices (ab, bc, ca)
            m_ab = n_cur + np.arange(len(sub_t), dtype=np.int32)
            m_bc = m_ab + len(sub_t)
            m_ca = m_bc + len(sub_t)

            def mid_arr(arr):
                return np.concatenate([
                    arr,
                    0.5 * (arr[a_i] + arr[b_i]),
                    0.5 * (arr[b_i] + arr[c_i]),
                    0.5 * (arr[c_i] + arr[a_i]),
                ], axis=0)

            sp_P = mid_arr(sp_P)
            sp_N = mid_arr(sp_N)
            sp_N /= np.maximum(np.linalg.norm(sp_N, axis=1, keepdims=True), 1e-6)
            sp_wcd = mid_arr(sp_wcd)
            sp_ccd = mid_arr(sp_ccd)
            sp_ao = mid_arr(sp_ao)
            sp_tm = mid_arr(sp_tm)
            sp_pd = mid_arr(sp_pd)
            sp_sb = mid_arr(sp_sb)
            new_t = np.vstack([
                np.column_stack([a_i, m_ab, m_ca]),
                np.column_stack([m_ab, b_i, m_bc]),
                np.column_stack([m_ca, m_bc, c_i]),
                np.column_stack([m_ab, m_bc, m_ca]),
            ]).astype(np.int32)
            sp_tris = np.vstack([keep_t, new_t])

        # Carve true 3D skeletal displacement along N!
        p_warp = sp_P + np.sin(sp_P * 11.5 + np.array([1.3, 4.7, 8.2], dtype=np.float32)) * 0.011
        df, _ = worley_f2_f1_np(p_warp, freq=freq, jitter=0.80)
        tip_supp = 1.0 - smoothstep_np(0.65, 0.98, sp_tm) * 0.80
        hold_supp = 1.0 - smoothstep_np(0.05, 0.90, sp_sb)
        wall_w = 0.17 + sp_tm * 0.22 + sp_sb * 0.16
        hole = smoothstep_np(0.0, 0.20, df - wall_w) * tip_supp * hold_supp * pore_str
        if mid == 3:
            hole = np.maximum(sp_pd * 0.75, hole * 0.45)

        # PolyHaven scan limestone height (triplanar sample of coral_bump_nh[..., 3])
        an = np.abs(sp_N) ** 4
        ws = an / np.maximum(an.sum(axis=1, keepdims=True), 1e-5)
        h_xy = bilerp_img(coral_bump_img[..., 3], sp_P[:, 0] * 2.6, sp_P[:, 1] * 2.6, wrap=True)
        h_xz = bilerp_img(coral_bump_img[..., 3], sp_P[:, 0] * 2.6, sp_P[:, 2] * 2.6, wrap=True)
        h_yz = bilerp_img(coral_bump_img[..., 3], sp_P[:, 2] * 2.6, sp_P[:, 1] * 2.6, wrap=True)
        scan_h = (h_xy * ws[:, 2] + h_xz * ws[:, 1] + h_yz * ws[:, 0]) - 0.5

        disp = (-hole * (disp_amp * 1.65) + scan_h * 0.0042 * (1.0 - 0.45 * sp_tm)) * hold_supp

        if mid == 1:
            # Brain coral comb-like meandroid septa teeth (~3.2 mm spacing across gyri slopes)
            slope_len = np.hypot(sp_N[:, 0], sp_N[:, 2])
            rx = sp_N[:, 0] / np.maximum(slope_len, 1e-3)
            rz = sp_N[:, 2] / np.maximum(slope_len, 1e-3)
            ax, az = -rz, rx
            septa_ph = (sp_P[:, 0] * ax + sp_P[:, 2] * az) * 195.0 + np.sin((sp_P[:, 0] * rx + sp_P[:, 2] * rz) * 42.0) * 1.6
            septa_h = np.sin(septa_ph) * smoothstep_np(0.12, 0.62, slope_len) * hold_supp * 0.0028
            disp += septa_h
        elif mid == 2:
            # Foliose fan radial vascular costae ribs
            rib_ang = np.arctan2(sp_P[:, 2], sp_P[:, 0])
            rib_ph = rib_ang * 46.0 + np.sin(sp_P[:, 1] * 16.0) * 1.3
            disp += np.sin(rib_ph) * (1.0 - 0.5 * sp_tm) * hold_supp * 0.0024

        sp_P = sp_P + sp_N * disp[:, None]
        # Convert display-referred sRGB vertex palettes to linear Rec.709 for Karma physically-based path tracing!
        sp_wcd_lin = np.clip(sp_wcd, 0.01, 1.0) ** 2.05
        sp_ccd_lin = (np.clip(sp_ccd, 0.01, 1.0) ** 2.15) * 0.72
        mix_k = np.clip((hole ** 0.85) * 0.86, 0.0, 0.95)[:, None]
        sp_wcd_baked = sp_wcd_lin * (1.0 + (1.0 - hole[:, None]) * 0.14 * hold_supp[:, None]) * (1.0 - mix_k) + sp_ccd_lin * mix_k
        sp_ao_baked = np.clip(sp_ao * (1.0 - hole * 0.52), 0.12, 1.0)

        acc.add(sp_P, sp_tris, sp_wcd_baked, sp_name, cavity_cd=sp_ccd_lin, ao=sp_ao_baked, tip_mask=sp_tm, sea_blend=sp_sb, linearize=False)

    # 3. Build 720x720 (1.03M-triangle) True 3D Displaced Houdini Sand Lab Ocean Floor!
    S_SAND = 720
    u_lin = np.linspace(-1.0, 1.0, S_SAND, dtype=np.float32)
    # Graded coordinates: ~1.1 cm resolution across the hero/wide reef floor, expanding smoothly to +-15.5 m!
    coords = 15.5 * u_lin * (0.24 + 0.76 * (u_lin * u_lin))
    gx, gz = np.meshgrid(coords, coords)
    fx = gx.ravel()
    fz = gz.ravel()

    by = seabed_base_y(fx, fz)
    puv_u = (fx + 4.5) / 9.0
    puv_v = (fz + 4.5) / 9.0
    edge = np.minimum(np.minimum(puv_u, puv_v), np.minimum(1.0 - puv_u, 1.0 - puv_v))
    w_patch = smoothstep_np(0.0, 0.07, edge)

    c_dist = coral_dist_at(fx, fz)
    coral_clear = smoothstep_np(0.03, 0.18, c_dist)
    apron_w = coral_clear * (1.0 - smoothstep_np(0.30, 1.55, c_dist))

    h_houdini = bilerp_img(sand_h_grid, puv_u, puv_v, wrap=False) * 2.05 * w_patch
    # Procedural outer sand ripples beyond the 9x9 m Houdini patch so the far floor has seamless ripples too
    surge_dx, surge_dz = math.sin(math.radians(35.0)), -math.cos(math.radians(35.0))
    along = fx * surge_dx + fz * surge_dz
    across = -fx * surge_dz + fz * surge_dx
    outer_rip = np.sin(along * (2 * math.pi / 0.105) + np.sin(across * 1.7) * 1.8) * 0.011 * (1.0 - w_patch)
    sub_rgba = bilerp_img(sub_nh_img, fx * 0.92, fz * 0.92, wrap=True)
    h_gravel = (sub_rgba[:, 3] - 0.46) * 0.036 * apron_w

    fy = by + (h_houdini + outer_rip + h_gravel) * coral_clear
    sand_pts = np.column_stack([fx, fy, fz]).astype(np.float32)

    # Rich aragonite sand albedo from Houdini Sand Lab material masks + grain tile + gravel apron
    pmat = bilerp_img(sand_mat_img, puv_u, puv_v, wrap=False)
    pnh = bilerp_img(sand_nh_img, puv_u, puv_v, wrap=False)
    gt = bilerp_img(sand_grain_img, fx / 0.6, fz / 0.6, wrap=True)
    sand_cd = np.tile(np.array([0.88, 0.82, 0.87], dtype=np.float32), (len(sand_pts), 1))
    h_lum = (1.0 - w_patch) + w_patch * (pmat[:, 0] * 2.0)
    sand_cd *= h_lum[:, None] * (1.0 + (gt[:, 2:3] - 0.5) * 0.26)
    sand_cd *= 1.0 - apron_w[:, None] * 0.12 + apron_w[:, None] * sub_rgba[:, 3:4] * 0.24
    # Detritus in troughs, white shell grit, pink coralline crust
    det_w = (pmat[:, 1] * w_patch * 0.64)[:, None]
    sand_cd = sand_cd * (1.0 - det_w) + (sand_cd * np.array([0.74, 0.70, 0.60], dtype=np.float32)) * det_w
    grit_w = np.maximum(pmat[:, 2] * w_patch * 0.62, gt[:, 3] * 0.38)[:, None]
    sand_cd = sand_cd * (1.0 - grit_w) + np.array([0.97, 0.95, 0.91], dtype=np.float32) * grit_w
    cca_w = (pmat[:, 3] * w_patch * 0.56)[:, None]
    sand_cd = sand_cd * (1.0 - cca_w) + np.array([0.93, 0.58, 0.70], dtype=np.float32) * cca_w
    sand_cd_lin = np.clip(sand_cd, 0.04, 1.0) ** 2.05
    # Beer-Lambert radial underwater distance attenuation on outer sand (r > 3.8 m melts smoothly into deep ocean cyclorama; zero white horizon shelf!)
    rad_xz = np.hypot(fx, fz)
    far_fade = smoothstep_np(3.8, 12.8, rad_xz)[:, None]
    deep_floor_col = np.array([0.24, 0.16, 0.46], dtype=np.float32) ** 2.05
    sand_cd_lin = sand_cd_lin * (1.0 - far_fade * 0.88) + deep_floor_col * (far_fade * 0.65)
    sand_ao = np.clip(((1.0 - w_patch) + w_patch * (pnh[:, 2] ** 1.35)) * (1.0 - apron_w * 0.45 * (1.0 - sub_rgba[:, 2])), 0.18, 1.0)

    ii, jj = np.meshgrid(np.arange(S_SAND - 1, dtype=np.int32), np.arange(S_SAND - 1, dtype=np.int32))
    a = (jj * S_SAND + ii).ravel()
    b = a + 1
    c = a + S_SAND
    d = c + 1
    sand_tris = np.vstack([np.column_stack([a, b, c]), np.column_stack([b, d, c])]).astype(np.int32)
    acc.add(sand_pts, sand_tris, sand_cd_lin, "sand_floor", ao=sand_ao, sea_blend=1.0, linearize=False)

    # 4. Reef Ecosystem Props (Boulders, Rubble Apron, Barrel/Tube Sponges, Anemones, Fish, Fans, Whips, Grass, Urchins, Star, Clam)
    rng = np.random.default_rng(20261102)
    sph_pts, sph_tris = make_uv_sphere(18, 12)

    def sand_y_exact(x, z):
        x_a = np.asarray([x], dtype=np.float32)
        z_a = np.asarray([z], dtype=np.float32)
        u = (x_a + 4.5) / 9.0
        v = (z_a + 4.5) / 9.0
        ed = np.minimum(np.minimum(u, v), np.minimum(1.0 - u, 1.0 - v))
        wp = smoothstep_np(0.0, 0.07, ed)
        cd = coral_dist_at(x_a, z_a)
        cc = smoothstep_np(0.03, 0.18, cd)
        hh = bilerp_img(sand_h_grid, u, v, wrap=False) * 1.55 * wp
        return float(seabed_base_y(x_a, z_a)[0] + hh[0] * cc[0])

    # 4a. Pitted Foreground Limestone Boulder + Encrusting Coral Head + Barrel Sponge + Tube Sponges
    boulder_specs = [
        (-1.46, 1.28, 0.44, 0.26, 0.38, [0.84, 0.78, 0.85]),
        (1.62, 0.92, 0.34, 0.20, 0.30, [0.82, 0.76, 0.84]),
        (-1.85, -0.45, 0.48, 0.30, 0.42, [0.76, 0.70, 0.80]),
        (1.95, -0.65, 0.42, 0.25, 0.36, [0.78, 0.73, 0.82]),
    ]
    b_sph_p, b_sph_t = make_uv_sphere(32, 22)
    for bx, bz, rx, ry, rz, bcol in boulder_specs:
        by0 = sand_y_exact(bx, bz)
        bp = b_sph_p * np.array([rx, ry, rz], dtype=np.float32)
        df_b, _ = worley_f2_f1_np(bp * 2.2 + np.array([bx, by0, bz]), freq=14.0, jitter=0.85)
        pit = smoothstep_np(0.12, 0.34, df_b)
        disp_b = 1.0 + 0.14 * np.sin(bp[:, 0] * 9.0 + bx) * np.cos(bp[:, 2] * 8.0 + bz) - 0.11 * pit
        bp = bp * disp_b[:, None] + np.array([bx, by0 + ry * 0.46, bz], dtype=np.float32)
        bcd = np.tile(np.array(bcol, dtype=np.float32), (len(bp), 1)) * (1.0 - 0.28 * pit[:, None])
        acc.add(bp, b_sph_t, bcd, "rubble", ao=1.0 - 0.35 * pit)

    # Giant Xestospongia barrel sponge (left mid-ground, matching WebGPU wide view)
    bar_x, bar_z = -2.42, -0.48
    bar_y = sand_y_exact(bar_x, bar_z)
    bar_spine = np.column_stack([np.full(14, bar_x), bar_y + np.linspace(0.0, 0.72, 14), np.full(14, bar_z)]).astype(np.float32)
    s_bar = np.linspace(0, 1, 14)
    bar_rad = 0.28 + 0.14 * np.sin(s_bar * math.pi * 0.82)
    bp, bt, bs = make_tube_along_curve(bar_spine, bar_rad, n_sides=28, cap_tip=False)
    bp += np.sin(bp * 22.0) * 0.014
    acc.add(bp, bt, [0.74, 0.52, 0.58], "rubble", ao=0.65 + 0.35 * bs)

    # Ochre tube sponge cluster (foreground left)
    for tx_off, tz_off, th, tr in [(-2.12, 1.56, 0.36, 0.055), (-2.00, 1.62, 0.30, 0.048), (-2.18, 1.66, 0.26, 0.044)]:
        ty0 = sand_y_exact(tx_off, tz_off)
        sp_c = np.column_stack([
            tx_off + np.linspace(0, 0.03, 10),
            ty0 + np.linspace(0, th, 10),
            tz_off + np.linspace(0, 0.02, 10),
        ]).astype(np.float32)
        tp_m, tt_m, ts_m = make_tube_along_curve(sp_c, tr * (0.82 + 0.32 * np.sin(np.linspace(0.2, 2.6, 10))), n_sides=16, cap_tip=False)
        acc.add(tp_m, tt_m, [0.94, 0.82, 0.48], "rubble", tip_mask=ts_m * 0.5)

    # 4b. 620 Weathered Coral Rubble Shards & Crustose-Coralline Cobbles around the Reef Apron
    rub_pal = np.array([
        [0.93, 0.89, 0.84],
        [0.84, 0.80, 0.90],
        [0.74, 0.69, 0.81],
        [0.91, 0.78, 0.81],
        [0.92, 0.60, 0.71],
    ], dtype=np.float32)
    placed_rub = 0
    attempts = 0
    while placed_rub < 620 and attempts < 5000:
        attempts += 1
        ang = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(0.45, 3.85)
        rx, rz = math.cos(ang) * rad, math.sin(ang) * rad
        cd = float(coral_dist_at(np.array([rx], dtype=np.float32), np.array([rz], dtype=np.float32))[0])
        if cd < 0.06 or cd > 1.85:
            continue
        if rng.random() > math.exp(-max(cd - 0.08, 0.0) / 0.52):
            continue
        ry0 = sand_y_exact(rx, rz)
        col = rub_pal[rng.integers(0, len(rub_pal))]
        if rng.random() < 0.42:
            # Broken staghorn branch shard lying on sand
            yaw = rng.uniform(0, 2 * math.pi)
            sh_len = rng.uniform(0.055, 0.145)
            sh_r = rng.uniform(0.010, 0.022)
            dx, dz = math.cos(yaw) * sh_len * 0.5, math.sin(yaw) * sh_len * 0.5
            spine = np.array([
                [rx - dx, ry0 + sh_r * 0.5, rz - dz],
                [rx, ry0 + sh_r * 0.85, rz],
                [rx + dx, ry0 + sh_r * 0.55, rz + dz],
            ], dtype=np.float32)
            rp, rt, _ = make_tube_along_curve(spine, np.array([sh_r * 0.85, sh_r, sh_r * 0.78], dtype=np.float32), n_sides=7)
        else:
            # Porous carbonate cobble
            sc = np.array([rng.uniform(0.018, 0.058), rng.uniform(0.010, 0.028), rng.uniform(0.016, 0.052)], dtype=np.float32)
            rp = (sph_pts * sc) + np.array([rx, ry0 + sc[1] * 0.45, rz], dtype=np.float32)
            rt = sph_tris
        acc.add(rp, rt, col, "rubble", ao=0.86)
        placed_rub += 1

    # 4c. Two Clonal Entacmaea quadricolor Bubble-Tip Anemones (front-left base)
    anemone_sites = [
        (-0.60, 0.80, 0.145, 195, [0.98, 0.82, 0.72], [0.80, 0.62, 0.94]),
        (-0.88, 0.965, 0.165, 220, [0.98, 0.84, 0.74], [0.85, 0.60, 0.88]),
    ]
    for ax, az, aR, n_tent, c_stalk, c_tip in anemone_sites:
        ay = sand_y_exact(ax, az) + 0.02
        c_stalk_np = np.array(c_stalk, dtype=np.float32)
        c_tip_np = np.array(c_tip, dtype=np.float32)
        for ti in range(n_tent):
            rr = aR * math.sqrt((ti + 0.5) / n_tent)
            th = ti * 2.39996323
            bx = ax + math.cos(th) * rr
            bz = az + math.sin(th) * rr
            t_len = rng.uniform(0.085, 0.145) * (1.0 - 0.28 * (rr / aR))
            s_t = np.linspace(0, 1, 9, dtype=np.float32)
            out_x = math.cos(th) * (rr / aR) * 0.055 + 0.022 * math.sin(ti * 0.7)
            out_z = math.sin(th) * (rr / aR) * 0.055 + 0.018 * math.cos(ti * 0.9)
            spine = np.column_stack([
                bx + out_x * (s_t ** 1.5),
                ay + t_len * s_t - 0.012 * (s_t ** 2),
                bz + out_z * (s_t ** 1.5),
            ]).astype(np.float32)
            # Signature Entacmaea pear-shaped sub-terminal bubble bulb + pointed nipple tip!
            bulb = np.exp(-((s_t - 0.76) / 0.14) ** 2)
            rad_t = 0.0055 * (1.0 - 0.55 * s_t) + 0.0068 * bulb
            tp_m, tt_m, ts_m = make_tube_along_curve(spine, rad_t, n_sides=8, cap_tip=True)
            t_cd = c_stalk_np[None, :] * (1.0 - ts_m[:, None]) + c_tip_np[None, :] * ts_m[:, None]
            acc.add(tp_m, tt_m, t_cd, "anemone", tip_mask=ts_m)

    # 4d. Symbiotic Clownfish (3) + Reef Canopy School (54 Pseudanthias, Chromis, & Butterflyfish)
    cf_pts, cf_tris, cf_cd = make_fish_mesh(0.072, 0.034, 0.015, is_clown=True)
    clown_poses = [
        (-0.63, sand_y_exact(-0.60, 0.80) + 0.16, 0.82, 0.45, 0.10),
        (-0.85, sand_y_exact(-0.88, 0.965) + 0.18, 0.95, -0.35, -0.08),
        (-0.74, sand_y_exact(-0.74, 0.88) + 0.22, 0.90, 2.40, 0.12),
    ]
    for fx0, fy0, fz0, yaw, pitch in clown_poses:
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp_p = math.cos(pitch), math.sin(pitch)
        R = np.array([[cy * cp, -sp_p, sy * cp], [sp_p, cp, 0.0], [-sy * cp, 0.0, cy * cp]], dtype=np.float32)
        p_rot = cf_pts @ R.T + np.array([fx0, fy0, fz0], dtype=np.float32)
        acc.add(p_rot, cf_tris, cf_cd, "fish")

    anth_pts, anth_tris, _ = make_fish_mesh(0.058, 0.024, 0.011, is_clown=False)
    for fi in range(54):
        ang = rng.uniform(-2.6, 2.6)
        rad = rng.uniform(0.35, 2.35)
        fx0 = math.sin(ang) * rad
        fz0 = math.cos(ang) * rad * 0.65 - 0.25
        fy0 = rng.uniform(1.15, 2.45)
        yaw = rng.uniform(-0.45, 0.45) + (0.0 if rng.random() < 0.7 else math.pi)
        sc_f = rng.uniform(0.78, 1.22)
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32) * sc_f
        p_rot = anth_pts @ R.T + np.array([fx0, fy0, fz0], dtype=np.float32)
        if fi % 3 == 0:
            fcol = np.array([0.52, 0.86, 0.95], dtype=np.float32)  # Blue-green Chromis
        else:
            fcol = np.array([0.98, 0.64, 0.42], dtype=np.float32)  # Peach-gold Pseudanthias
        acc.add(p_rot, anth_tris, fcol, "fish")

    # 2 Chaetodon lemon butterflyfish (hero/macro right accent)
    bf_pts, bf_tris, _ = make_fish_mesh(0.105, 0.072, 0.015, is_clown=False)
    for fx0, fy0, fz0, yaw in [(0.82, 0.78, 0.72, -2.3), (0.22, 0.96, 0.68, -1.9)]:
        cy, sy = math.cos(yaw), math.sin(yaw)
        R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        acc.add(bf_pts @ R.T + np.array([fx0, fy0, fz0], dtype=np.float32), bf_tris, [0.96, 0.87, 0.36], "fish")

    # 4e. Gorgonian Sea Fans, Tapered Sea Whips, Seagrass Meadows, Diadema Urchins, Blue Linckia Star, & Tridacna Clam
    fan_sites = [
        (2.85, -0.75, 0.82, [0.80, 0.42, 0.66]),
        (3.15, -1.80, 0.94, [0.64, 0.44, 0.82]),
        (-3.15, -0.15, 0.92, [0.86, 0.50, 0.64]),
        (1.85, -2.25, 1.02, [0.82, 0.48, 0.68]),
        (-1.75, -1.95, 0.98, [0.70, 0.40, 0.72]),
    ]
    for fx0, fz0, fH, fcol in fan_sites:
        fy0 = sand_y_exact(fx0, fz0)
        # Branching 3D gorgonian tree of 19 radiating ribs
        for bi in range(19):
            frac = (bi / 18.0) * 2.0 - 1.0
            ang_f = frac * 1.18
            s_f = np.linspace(0, 1, 10, dtype=np.float32)
            r_reach = fH * (0.65 + 0.35 * math.cos(frac * 1.1)) * s_f
            spine = np.column_stack([
                fx0 + np.sin(ang_f) * r_reach * 0.78,
                fy0 + np.cos(ang_f * 0.65) * r_reach,
                fz0 + np.sin(r_reach * 3.0 + bi) * 0.04 + (frac * 0.18) * r_reach,
            ]).astype(np.float32)
            rad_f = 0.0085 * (1.0 - 0.72 * s_f)
            fp_m, ft_m, fs_m = make_tube_along_curve(spine, rad_f, n_sides=6)
            acc.add(fp_m, ft_m, fcol, "flora", tip_mask=fs_m)

    whip_sites = [
        (-2.85, -0.10, 7, [0.96, 0.60, 0.40]),
        (-2.30, -1.45, 8, [0.95, 0.84, 0.50]),
        (1.00, -1.85, 8, [0.96, 0.60, 0.40]),
        (1.70, -1.60, 7, [0.92, 0.54, 0.62]),
        (2.35, -1.10, 6, [0.95, 0.84, 0.50]),
        (2.90, 0.45, 5, [0.96, 0.60, 0.40]),
    ]
    for wx, wz, wn_cnt, wcol in whip_sites:
        wy = sand_y_exact(wx, wz)
        for wi in range(wn_cnt):
            wh = rng.uniform(0.68, 1.32)
            s_w = np.linspace(0, 1, 11, dtype=np.float32)
            dx = rng.uniform(-0.18, 0.28)
            dz = rng.uniform(-0.22, 0.18)
            spine = np.column_stack([
                wx + rng.uniform(-0.06, 0.06) + dx * (s_w ** 1.7),
                wy + wh * s_w,
                wz + rng.uniform(-0.06, 0.06) + dz * (s_w ** 1.7),
            ]).astype(np.float32)
            # Organic paraboloid tip taper
            rad_w = 0.0095 * np.sqrt(np.clip(1.0 - 0.82 * (s_w ** 1.8), 0.08, 1.0))
            wp_m, wt_m, ws_m = make_tube_along_curve(spine, rad_w, n_sides=6)
            acc.add(wp_m, wt_m, wcol, "flora", tip_mask=ws_m)

    # Mid-ground Thalassia seagrass meadows (~950 curved 3D ribbons)
    meadow_centers = [(-3.2, -2.8, 1.2, 340), (3.3, -2.9, 1.2, 340), (-0.5, -4.9, 1.1, 180), (-2.35, 1.45, 0.35, 90)]
    for mx, mz, mR, mN in meadow_centers:
        for _ in range(mN):
            ga = rng.uniform(0, 2 * math.pi)
            gr = mR * math.sqrt(rng.random())
            gx0, gz0 = mx + math.cos(ga) * gr, mz + math.sin(ga) * gr
            gy0 = sand_y_exact(gx0, gz0)
            gh = rng.uniform(0.14, 0.30)
            gw = 0.0075
            yaw = rng.uniform(0, math.pi)
            wx_d, wz_d = math.cos(yaw) * gw, math.sin(yaw) * gw
            bend = rng.uniform(0.03, 0.09)
            b_pts = np.array([
                [gx0 - wx_d, gy0, gz0 - wz_d],
                [gx0 + wx_d, gy0, gz0 + wz_d],
                [gx0 - wx_d * 0.7 + bend * 0.4, gy0 + gh * 0.55, gz0 - wz_d * 0.7 - bend * 0.3],
                [gx0 + wx_d * 0.7 + bend * 0.4, gy0 + gh * 0.55, gz0 + wz_d * 0.7 - bend * 0.3],
                [gx0 + bend, gy0 + gh, gz0 - bend * 0.75],
            ], dtype=np.float32)
            b_tris = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [0, 2, 1], [1, 2, 3], [2, 4, 3]], dtype=np.int32)
            gcol = np.array([0.46, 0.72, 0.54], dtype=np.float32) * rng.uniform(0.85, 1.12)
            acc.add(b_pts, b_tris, gcol, "flora", tip_mask=np.array([0.0, 0.0, 0.5, 0.5, 1.0], dtype=np.float32))

    # Diadema long-spined black sea urchins (foreground accents)
    for ux, uz in [(-1.05, 1.38), (1.16, 1.34), (-1.62, 0.78), (0.68, 1.52)]:
        uy = sand_y_exact(ux, uz) + 0.035
        acc.add(sph_pts * 0.038 + np.array([ux, uy, uz], dtype=np.float32), sph_tris, [0.10, 0.08, 0.18], "rubble", ao=0.5)
        for si_u in range(56):
            th = si_u * 2.39996
            el = rng.uniform(0.08, 1.35)
            dx = math.cos(th) * math.sin(el)
            dy = math.cos(el)
            dz = math.sin(th) * math.sin(el)
            if dy < -0.15:
                continue
            slen = rng.uniform(0.11, 0.19)
            sp_u = np.array([[ux, uy, uz], [ux + dx * slen, uy + dy * slen, uz + dz * slen]], dtype=np.float32)
            up_m, ut_m, _ = make_tube_along_curve(sp_u, np.array([0.0022, 0.0004], dtype=np.float32), n_sides=4)
            acc.add(up_m, ut_m, [0.12, 0.09, 0.22], "rubble")

    # Blue Linckia laevigata sea star (front-left sand, matching WebGPU wide view!)
    lx, lz = -0.82, 1.56
    ly = sand_y_exact(lx, lz) + 0.012
    for arm_i in range(5):
        ang_l = arm_i * (2 * math.pi / 5.0) + 0.35
        s_l = np.linspace(0, 0.145, 7, dtype=np.float32)
        arm_sp = np.column_stack([
            lx + np.cos(ang_l) * s_l + np.sin(s_l * 16.0 + arm_i) * 0.008,
            ly + 0.004 * np.cos(s_l * 10.0),
            lz + np.sin(ang_l) * s_l,
        ]).astype(np.float32)
        rad_l = 0.016 * (1.0 - 0.42 * (s_l / 0.145))
        lp_m, lt_m, _ = make_tube_along_curve(arm_sp, rad_l, n_sides=8)
        acc.add(lp_m, lt_m, [0.38, 0.52, 0.94], "anemone")

    # Tridacna giant clam (front-right sand)
    cx0, cz0 = 1.48, 1.24
    cy0 = sand_y_exact(cx0, cz0) + 0.05
    for side_s in (-1.0, 1.0):
        sh_p = sph_pts * np.array([0.14, 0.09, 0.045], dtype=np.float32)
        sh_p[:, 2] += side_s * 0.045
        sh_p[:, 1] += 0.02 * np.sin(sh_p[:, 0] * 38.0)
        acc.add(sh_p + np.array([cx0, cy0, cz0], dtype=np.float32), sph_tris, [0.88, 0.85, 0.83], "rubble")
    man_p = sph_pts * np.array([0.125, 0.045, 0.052], dtype=np.float32)
    man_p[:, 2] += 0.016 * np.sin(man_p[:, 0] * 42.0)
    acc.add(man_p + np.array([cx0, cy0 + 0.035, cz0], dtype=np.float32), sph_tris, [0.24, 0.86, 0.84], "anemone", tip_mask=0.8)

    # Distant reef bommie silhouettes (6.0-8.5 m ring for multi-plane volumetric depth)
    for bi in range(12):
        ba = (bi / 12.0) * 2 * math.pi + 0.22
        br = rng.uniform(6.0, 8.4)
        bx0, bz0 = math.cos(ba) * br, math.sin(ba) * br
        if bz0 > 2.2:
            continue  # keep behind camera clean
        by0 = sand_y_exact(bx0, bz0)
        sc_b = np.array([rng.uniform(0.65, 1.25), rng.uniform(0.45, 0.95), rng.uniform(0.65, 1.15)], dtype=np.float32)
        bp = b_sph_p * sc_b + np.sin(b_sph_p * 5.0 + bi) * 0.09 + np.array([bx0, by0 + sc_b[1] * 0.35, bz0], dtype=np.float32)
        acc.add(bp, b_sph_t, [0.34, 0.29, 0.44], "rubble", ao=0.72)

    # 3D Translucent Marine Snow Motes (catch optical thin-lens bokeh!)
    snow_sph_p, snow_sph_t = make_uv_sphere(6, 5)
    for _ in range(280):
        sx_m = rng.uniform(-2.6, 2.6)
        sy_m = rng.uniform(0.25, 2.65)
        sz_m = rng.uniform(-1.5, 4.2)
        sr_m = rng.uniform(0.0016, 0.0038)
        acc.add(snow_sph_p * sr_m + np.array([sx_m, sy_m, sz_m], dtype=np.float32), snow_sph_t, [0.95, 0.94, 1.0], "snow", tip_mask=1.0)

    # 4f. Perforated Geometric Caustic Canopy (y = 4.95 m, invisible to primary camera rays, zero opacity shader!)
    # Refracted sunlight streams through open caustic filament apertures to cast crisp caustics & 3D volumetric god-rays!
    caus_img = np.asarray(Image.open(os.path.join(HOUDINI_DIR, "karma_caustic_gobo.png")).convert("L"), dtype=np.float32) / 255.0
    G_CAUS = 380
    cg_lin = np.linspace(-8.5, 8.5, G_CAUS, dtype=np.float32)
    cgx, cgz = np.meshgrid(cg_lin, cg_lin)
    cg_pts = np.column_stack([cgx.ravel(), np.full(G_CAUS * G_CAUS, 4.95, dtype=np.float32), cgz.ravel()])
    cii, cjj = np.meshgrid(np.arange(G_CAUS - 1, dtype=np.int32), np.arange(G_CAUS - 1, dtype=np.int32))
    ca0 = (cjj * G_CAUS + cii).ravel()
    cb0 = ca0 + 1
    cc0 = ca0 + G_CAUS
    cd0 = cc0 + 1
    cell_cx = 0.5 * (cg_pts[ca0, 0] + cg_pts[cd0, 0])
    cell_cz = 0.5 * (cg_pts[ca0, 2] + cg_pts[cd0, 2])
    c_val = bilerp_img(caus_img, cell_cx / 3.8, cell_cz / 3.8, wrap=True)
    # Keep dark wave-trough shade mask where c_val < 0.36; open sky apertures where c_val >= 0.36 (caustic filaments!)
    shade_mask = c_val < 0.36
    caus_tris_raw = np.vstack([
        np.column_stack([ca0[shade_mask], cb0[shade_mask], cc0[shade_mask]]),
        np.column_stack([cb0[shade_mask], cd0[shade_mask], cc0[shade_mask]]),
    ]).astype(np.int32)
    cu_v, cu_inv = np.unique(caus_tris_raw.ravel(), return_inverse=True)
    acc.add(cg_pts[cu_v], cu_inv.reshape(-1, 3).astype(np.int32), [0.0, 0.0, 0.0], "caustic_canopy", linearize=False)

    # 4g. Curved Underwater Depth-Graded Ocean Cyclorama (r = 16.0 m behind the reef)
    N_AZ, N_EL = 72, 28
    az = np.linspace(math.radians(-172.0), math.radians(-8.0), N_AZ, dtype=np.float32)  # behind -Z
    el_y = np.linspace(-0.4, 9.8, N_EL, dtype=np.float32)
    gaz, gy = np.meshgrid(az, el_y)
    R_CYC = 16.0
    cyc_pts = np.column_stack([(np.cos(gaz) * R_CYC).ravel(), gy.ravel(), (np.sin(gaz) * R_CYC).ravel()]).astype(np.float32)
    h_frac = np.clip((cyc_pts[:, 1] + 0.4) / 8.8, 0.0, 1.0)
    # Smooth underwater depth column gradient (deep indigo-violet -> luminous periwinkle -> bright Snell's-window lavender-cyan)
    c_deep = np.array([0.26, 0.17, 0.48], dtype=np.float32) ** 2.05
    c_mid  = np.array([0.52, 0.40, 0.76], dtype=np.float32) ** 2.05
    c_top  = np.array([0.78, 0.76, 0.98], dtype=np.float32) ** 2.05
    m_low = smoothstep_np(0.0, 0.45, h_frac)[:, None]
    m_hi  = smoothstep_np(0.35, 0.95, h_frac)[:, None]
    cyc_cd = (c_deep * (1.0 - m_low) + c_mid * m_low) * (1.0 - m_hi * 0.78) + c_top * (m_hi * 0.78)
    bii, bjj = np.meshgrid(np.arange(N_AZ - 1, dtype=np.int32), np.arange(N_EL - 1, dtype=np.int32))
    ba0 = (bjj * N_AZ + bii).ravel()
    bb0 = ba0 + 1
    bc0 = ba0 + N_AZ
    bd0 = bc0 + 1
    cyc_tris = np.vstack([np.column_stack([ba0, bc0, bb0]), np.column_stack([bb0, bc0, bd0])]).astype(np.int32)
    acc.add(cyc_pts, cyc_tris, cyc_cd, "water_backdrop", linearize=False)

    # 4h. Soft Volumetric Sun-Ray Shaft Ribbons behind the Coral Bouquet (aligned with refracted sun!)
    sun_d = np.array([0.36, 0.76, 0.42], dtype=np.float32)
    sun_d /= np.linalg.norm(sun_d)
    for bx_s, bz_s, bw_s, b_int in [(-2.4, -3.6, 0.75, 0.9), (-0.7, -3.2, 0.95, 1.0), (1.1, -3.4, 0.85, 0.95), (2.7, -3.8, 0.80, 0.85), (-0.1, -4.6, 1.25, 0.75)]:
        # Crossed soft-edged vertical beam ribbons tilted along sun_d
        for rot_a in (0.15, 1.15):
            rx_s, rz_s = math.cos(rot_a) * bw_s, math.sin(rot_a) * bw_s
            nu_b, nv_b = 7, 9
            uu_b = np.linspace(-1.0, 1.0, nu_b, dtype=np.float32)
            vv_b = np.linspace(0.0, 1.0, nv_b, dtype=np.float32)
            gu_b, gv_b = np.meshgrid(uu_b, vv_b)
            u_f, v_f = gu_b.ravel(), gv_b.ravel()
            # v_f = 0 near sand (y=0.3), v_f = 1 high in Snell's window (y=6.8)
            y_b = 0.3 + v_f * 6.5
            x_b = bx_s + u_f * rx_s * (1.25 - 0.35 * v_f) + sun_d[0] * (y_b - 1.5) * 0.68
            z_b = bz_s + u_f * rz_s * (1.25 - 0.35 * v_f) + sun_d[2] * (y_b - 1.5) * 0.68
            env_b = ((1.0 - u_f * u_f) ** 2) * smoothstep_np(0.0, 0.32, v_f) * (0.35 + 0.65 * v_f) * b_int
            sh_pts = np.column_stack([x_b, y_b, z_b]).astype(np.float32)
            sii, sjj = np.meshgrid(np.arange(nu_b - 1, dtype=np.int32), np.arange(nv_b - 1, dtype=np.int32))
            sa0 = (sjj * nu_b + sii).ravel()
            sb0 = sa0 + 1
            sc0 = sa0 + nu_b
            sd0 = sc0 + 1
            sh_tris = np.vstack([
                np.column_stack([sa0, sb0, sc0]), np.column_stack([sb0, sd0, sc0]),
                np.column_stack([sa0, sc0, sb0]), np.column_stack([sb0, sc0, sd0]),
            ]).astype(np.int32)
            sh_cd = np.outer(env_b, np.array([0.82, 0.90, 1.0], dtype=np.float32))
            acc.add(sh_pts, sh_tris, sh_cd, "god_rays", tip_mask=env_b, linearize=False)

    merged = acc.finalize()

    # 5. Populate /obj/karma_reef_scene via fast bulk hou.Geometry API
    obj = hou.node("/obj")
    if obj.node("karma_reef_scene"):
        obj.node("karma_reef_scene").destroy()
    kgeo_obj = obj.createNode("geo", "karma_reef_scene")
    kgeo_obj.setDisplayFlag(False)
    py_sop = kgeo_obj.createNode("python", "build_mesh")

    # Store arrays in hou.session so Python SOP cooks instantaneously without serialization overhead!
    hou.session.KARMA_MESH_DATA = merged
    py_sop.parm("python").set("""
import hou
geo = hou.pwd().geometry()
d = hou.session.KARMA_MESH_DATA
npts = len(d["P"])
geo.createPoints(d["P"].tolist())
geo.createPolygons(d["tris"].tolist())

for aname, size, defv in [
    ("N", 3, (0.0, 1.0, 0.0)),
    ("wall_cd", 3, (1.0, 1.0, 1.0)),
    ("cavity_cd", 3, (0.5, 0.2, 0.3)),
    ("ao", 1, 1.0),
    ("tip_mask", 1, 0.0),
    ("sea_blend", 1, 0.0),
]:
    geo.addAttrib(hou.attribType.Point, aname, defv)
    arr = d[aname]
    geo.setPointFloatAttribValuesFromString(aname, arr.astype("f4").tobytes())

geo.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
geo.setPointFloatAttribValuesFromString("Cd", d["wall_cd"].astype("f4").tobytes())

geo.addAttrib(hou.attribType.Prim, "name", "")
geo.setPrimStringAttribValues("name", d["names"])
""")
    out_null = kgeo_obj.createNode("null", "OUT_KARMA_REEF")
    out_null.setInput(0, py_sop)
    out_null.setDisplayFlag(True)
    out_null.setRenderFlag(True)
    out_null.geometry()  # force cook now
    py_sop.setHardLocked(True)  # persist cooked 3D reef & sand floor inside coral_reef_karma.hipnc!

    # -------------------------------------------------------------------------
    # 6. Build Solaris Stage (/stage) — Pure MaterialX + Karma Path-Tracing
    # -------------------------------------------------------------------------
    stage = hou.node("/stage")
    for c in list(stage.children()):
        c.destroy()

    si = stage.createNode("sopimport", "import_reef")
    si.parm("soppath").set(out_null.path())
    si.parm("primpath").set("/reef")
    si.parm("copycontents").set(1)
    si.parm("enable_partitionattribs").set(1)
    si.parm("partitionattribs").set("name")

    # Hide caustic_canopy from primary/indirect rays (casts sun & fogbox shadows ONLY!)
    rgs_canopy = stage.createNode("rendergeometrysettings", "rgs_canopy")
    rgs_canopy.setInput(0, si)
    rgs_canopy.parm("primpattern").set("/import_reef/caustic_canopy")
    for p in rgs_canopy.parms():
        pl = p.name().lower()
        if ("visibilitycamera" in pl or "visibilitydiffuse" in pl or "visibilityglossy" in pl or "visibilityreflect" in pl) and "control" in pl:
            p.set("set")
        elif ("visibilitycamera" in pl or "visibilitydiffuse" in pl or "visibilityglossy" in pl or "visibilityreflect" in pl):
            p.set(0)

    # Prevent water_backdrop and god_rays from casting shadows
    rgs_bg = stage.createNode("rendergeometrysettings", "rgs_backdrop")
    rgs_bg.setInput(0, rgs_canopy)
    rgs_bg.parm("primpattern").set("/import_reef/water_backdrop /import_reef/god_rays")
    for p in rgs_bg.parms():
        pl = p.name().lower()
        if ("visibilityshadow" in pl or "visibilitydiffuse" in pl) and "control" in pl:
            p.set("set")
        elif ("visibilityshadow" in pl or "visibilitydiffuse" in pl):
            p.set(0)

    ml = stage.createNode("materiallibrary", "reef_materials")
    ml.setInput(0, rgs_bg)
    ml.parm("matpathprefix").set("/materials/")

    def make_coral_mtlx(mat_name, freq, pore_low, pore_high, bump_scale, sss_amt, rough_base):
        col = ml.createNode("collect", mat_name)
        ss = ml.createNode("mtlxstandard_surface", mat_name + "_ss")
        ss.parm("base").set(0.88)
        ss.parm("diffuse_roughness").set(0.35)
        ss.parm("specular").set(0.55)
        ss.parm("specular_IOR").set(1.48)

        gp_wall = ml.createNode("mtlxgeompropvalue", mat_name + "_wall")
        gp_wall.parm("signature").set("color3")
        gp_wall.parm("geomprop").set("wall_cd")
        gp_wall.parm("default_color3r").set(0.92); gp_wall.parm("default_color3g").set(0.68); gp_wall.parm("default_color3b").set(0.72)

        gp_cav = ml.createNode("mtlxgeompropvalue", mat_name + "_cav")
        gp_cav.parm("signature").set("color3")
        gp_cav.parm("geomprop").set("cavity_cd")
        gp_cav.parm("default_color3r").set(0.55); gp_cav.parm("default_color3g").set(0.18); gp_cav.parm("default_color3b").set(0.28)

        gp_ao = ml.createNode("mtlxgeompropvalue", mat_name + "_ao")
        gp_ao.parm("geomprop").set("ao")
        gp_ao.parm("default").set(1.0)

        gp_tm = ml.createNode("mtlxgeompropvalue", mat_name + "_tm")
        gp_tm.parm("geomprop").set("tip_mask")
        gp_tm.parm("default").set(0.0)

        gp_sb = ml.createNode("mtlxgeompropvalue", mat_name + "_sb")
        gp_sb.parm("geomprop").set("sea_blend")
        gp_sb.parm("default").set(0.0)

        # Sub-millimeter render-time 3D Worley f2 - f1 pores + triplanar stony normal map!
        mp = ml.createNode("mtlxposition", mat_name + "_pos")
        mp.parm("space").set("world")
        mul_p = ml.createNode("mtlxmultiply", mat_name + "_mulp")
        mul_p.parm("signature").set("vector3")
        mul_p.setInput(0, mp, 0)
        mul_p.parm("in2_vector3x").set(freq); mul_p.parm("in2_vector3y").set(freq); mul_p.parm("in2_vector3z").set(freq)

        wn = ml.createNode("mtlxworleynoise3d", mat_name + "_worley")
        wn.parm("signature").set("vector2")
        wn.parm("jitter").set(0.82)
        wn.setInput(0, mul_p, 0)
        sep = ml.createNode("mtlxseparate2", mat_name + "_sep")
        sep.setInput(0, wn, 0)
        sub_w = ml.createNode("mtlxsubtract", mat_name + "_subw")
        sub_w.setInput(0, sep, 1)
        sub_w.setInput(1, sep, 0)

        sm_hole = ml.createNode("mtlxsmoothstep", mat_name + "_hole")
        sm_hole.setInput(0, sub_w, 0)
        sm_hole.parm("low").set(pore_low)
        sm_hole.parm("high").set(pore_high)

        # Suppress pores on growing tips & seabed holdfast
        inv_tm = ml.createNode("mtlxsubtract", mat_name + "_invtm")
        inv_tm.parm("in1").set(1.0)
        inv_tm.setInput(1, gp_tm, 0)
        inv_sb = ml.createNode("mtlxsubtract", mat_name + "_invsb")
        inv_sb.parm("in1").set(1.0)
        inv_sb.setInput(1, gp_sb, 0)
        mask_p = ml.createNode("mtlxmultiply", mat_name + "_maskp")
        mask_p.setInput(0, inv_tm, 0)
        mask_p.setInput(1, inv_sb, 0)
        eff_hole = ml.createNode("mtlxmultiply", mat_name + "_effhole")
        eff_hole.setInput(0, sm_hole, 0)
        eff_hole.setInput(1, mask_p, 0)

        # Mix wall_cd and cavity_cd strongly so OIDN's albedo guide preserves sub-pixel 3D honeycomb pores!
        mix_cd = ml.createNode("mtlxmix", mat_name + "_mixcd")
        mix_cd.parm("signature").set("color3")
        mix_cd.setInput(mix_cd.inputIndex("bg"), gp_wall, 0)
        mix_cd.setInput(mix_cd.inputIndex("fg"), gp_cav, 0)
        mul_hm = ml.createNode("mtlxmultiply", mat_name + "_hm")
        mul_hm.setInput(0, eff_hole, 0)
        mul_hm.parm("in2").set(0.84)
        mix_cd.setInput(mix_cd.inputIndex("mix"), mul_hm, 0)

        ra_ao = ml.createNode("mtlxrange", mat_name + "_raao")
        ra_ao.setInput(0, gp_ao, 0)
        ra_ao.parm("inlow").set(0.0); ra_ao.parm("inhigh").set(1.0)
        ra_ao.parm("outlow").set(0.38); ra_ao.parm("outhigh").set(1.04)
        conv_ao = ml.createNode("mtlxcombine3", mat_name + "_convao")
        conv_ao.parm("signature").set("color3")
        conv_ao.setInput(0, ra_ao, 0)
        conv_ao.setInput(1, ra_ao, 0)
        conv_ao.setInput(2, ra_ao, 0)
        cd_final = ml.createNode("mtlxmultiply", mat_name + "_cdfinal")
        cd_final.parm("signature").set("color3")
        cd_final.setInput(0, mix_cd, 0)
        cd_final.setInput(1, conv_ao, 0)
        ss.setInput(ss.inputIndex("base_color"), cd_final, 0)

        # Random-Walk Subsurface Scattering: luminous polyp tips & trabecular walls, suppressed inside dark pores!
        ra_sss = ml.createNode("mtlxrange", mat_name + "_rasss")
        ra_sss.setInput(0, eff_hole, 0)
        ra_sss.parm("inlow").set(0.0); ra_sss.parm("inhigh").set(1.0)
        ra_sss.parm("outlow").set(sss_amt); ra_sss.parm("outhigh").set(0.08)
        ss.setInput(ss.inputIndex("subsurface"), ra_sss, 0)
        ss.setInput(ss.inputIndex("subsurface_color"), gp_wall, 0)
        ss.parm("subsurface_radiusr").set(1.0); ss.parm("subsurface_radiusg").set(0.62); ss.parm("subsurface_radiusb").set(0.76)
        ss.parm("subsurface_scale").set(0.018)
        ss.parm("subsurface_anisotropy").set(0.35)

        # Roughness: porous carbonate walls (~0.68), smoother tips (~0.34), glossy wet mucus clearcoat!
        ra_rgh = ml.createNode("mtlxrange", mat_name + "_rargh")
        ra_rgh.setInput(0, gp_tm, 0)
        ra_rgh.parm("inlow").set(0.0); ra_rgh.parm("inhigh").set(1.0)
        ra_rgh.parm("outlow").set(rough_base); ra_rgh.parm("outhigh").set(0.32)
        ss.setInput(ss.inputIndex("specular_roughness"), ra_rgh, 0)

        ra_coat = ml.createNode("mtlxrange", mat_name + "_racoat")
        ra_coat.setInput(0, eff_hole, 0)
        ra_coat.parm("inlow").set(0.0); ra_coat.parm("inhigh").set(1.0)
        ra_coat.parm("outlow").set(0.80); ra_coat.parm("outhigh").set(0.08)
        ss.setInput(ss.inputIndex("coat"), ra_coat, 0)
        ss.parm("coat_roughness").set(0.068)
        ss.parm("coat_IOR").set(1.37)
        ss.parm("thin_film_thickness").set(380.0)
        ss.parm("thin_film_IOR").set(1.38)
        ss.parm("sheen").set(0.18)
        ss.setInput(ss.inputIndex("sheen_color"), gp_wall, 0)

        # Triplanar micro-normal perturbation from staghorn_normal.png (world/tangent detail on stony walls)
        tp_n = ml.createNode("mtlxtriplanarprojection", mat_name + "_tpn")
        tp_n.parm("signature").set("vector3")
        nrm_path = os.path.join(TEX_DIR, "staghorn_normal.png")
        for fk in ("filex", "filey", "filez"):
            tp_n.parm(fk).set(nrm_path)
        for ck in ("filexcolorspace", "fileycolorspace", "filezcolorspace"):
            if tp_n.parm(ck):
                tp_n.parm(ck).set("Raw")
        mul_tp = ml.createNode("mtlxmultiply", mat_name + "_multp")
        mul_tp.parm("signature").set("vector3")
        mul_tp.setInput(0, mp, 0)
        mul_tp.parm("in2_vector3x").set(3.4); mul_tp.parm("in2_vector3y").set(3.4); mul_tp.parm("in2_vector3z").set(3.4)
        tp_n.setInput(tp_n.inputIndex("position"), mul_tp, 0)
        mn = ml.createNode("mtlxnormal", mat_name + "_mn")
        mn.parm("space").set("world")
        tp_n.setInput(tp_n.inputIndex("normal"), mn, 0)

        # Perturb world normal slightly by (tp_n - 0.5) * bump_scale
        ra_n = ml.createNode("mtlxrange", mat_name + "_ran")
        ra_n.parm("signature").set("vector3")
        ra_n.setInput(0, tp_n, 0)
        for ch in ("x", "y", "z"):
            ra_n.parm("inlow_vector3" + ch).set(0.0)
            ra_n.parm("inhigh_vector3" + ch).set(1.0)
            ra_n.parm("outlow_vector3" + ch).set(-bump_scale)
            ra_n.parm("outhigh_vector3" + ch).set(bump_scale)
        add_n = ml.createNode("mtlxadd", mat_name + "_addn")
        add_n.parm("signature").set("vector3")
        add_n.setInput(0, mn, 0)
        add_n.setInput(1, ra_n, 0)
        nrm_out = ml.createNode("mtlxnormalize", mat_name + "_norm")
        nrm_out.setInput(0, add_n, 0)
        ss.setInput(ss.inputIndex("normal"), nrm_out, 0)

        col.setInput(0, ss, 0)
        return mat_name

    # Build Coral Species Materials
    make_coral_mtlx("mat_staghorn", 76.0, 0.15, 0.35, 0.50, 0.58, 0.62)
    make_coral_mtlx("mat_brain",    62.0, 0.19, 0.40, 0.54, 0.48, 0.60)
    make_coral_mtlx("mat_table",    84.0, 0.15, 0.35, 0.44, 0.66, 0.56)
    make_coral_mtlx("mat_sponge",   68.0, 0.17, 0.37, 0.48, 0.56, 0.60)

    # Build Aragonite Sand & Reef Apron Material (with micro-grain normal map!)
    col_sand = ml.createNode("collect", "mat_sand")
    ss_sand = ml.createNode("mtlxstandard_surface", "sand_ss")
    ss_sand.parm("base").set(0.92)
    ss_sand.parm("diffuse_roughness").set(0.45)
    ss_sand.parm("specular").set(0.38)
    ss_sand.parm("specular_roughness").set(0.54)
    ss_sand.parm("subsurface").set(0.18)
    ss_sand.parm("subsurface_scale").set(0.008)
    ss_sand.parm("coat").set(0.18)
    ss_sand.parm("coat_roughness").set(0.28)
    gp_sw = ml.createNode("mtlxgeompropvalue", "sand_wall"); gp_sw.parm("signature").set("color3"); gp_sw.parm("geomprop").set("wall_cd")
    gp_sao = ml.createNode("mtlxgeompropvalue", "sand_ao"); gp_sao.parm("geomprop").set("ao"); gp_sao.parm("default").set(1.0)
    ra_sao = ml.createNode("mtlxrange", "sand_raao"); ra_sao.setInput(0, gp_sao, 0)
    ra_sao.parm("inlow").set(0.0); ra_sao.parm("inhigh").set(1.0); ra_sao.parm("outlow").set(0.45); ra_sao.parm("outhigh").set(1.02)
    cv_sao = ml.createNode("mtlxcombine3", "sand_cvsao"); cv_sao.parm("signature").set("color3")
    cv_sao.setInput(0, ra_sao, 0); cv_sao.setInput(1, ra_sao, 0); cv_sao.setInput(2, ra_sao, 0)
    mul_scd = ml.createNode("mtlxmultiply", "sand_mulcd"); mul_scd.parm("signature").set("color3")
    mul_scd.setInput(0, gp_sw, 0); mul_scd.setInput(1, cv_sao, 0)
    ss_sand.setInput(ss_sand.inputIndex("base_color"), mul_scd, 0)
    ss_sand.setInput(ss_sand.inputIndex("subsurface_color"), gp_sw, 0)

    # Micro-grain normal perturbation from seabed_normal.png
    mp_s = ml.createNode("mtlxposition", "sand_pos"); mp_s.parm("space").set("world")
    tp_sn = ml.createNode("mtlxtriplanarprojection", "sand_tpn"); tp_sn.parm("signature").set("vector3")
    sn_path = os.path.join(TEX_DIR, "seabed_normal.png")
    for fk in ("filex", "filey", "filez"):
        tp_sn.parm(fk).set(sn_path)
    mul_sp = ml.createNode("mtlxmultiply", "sand_mulsp"); mul_sp.parm("signature").set("vector3"); mul_sp.setInput(0, mp_s, 0)
    mul_sp.parm("in2_vector3x").set(1.8); mul_sp.parm("in2_vector3y").set(1.8); mul_sp.parm("in2_vector3z").set(1.8)
    tp_sn.setInput(tp_sn.inputIndex("position"), mul_sp, 0)
    mn_s = ml.createNode("mtlxnormal", "sand_mn"); mn_s.parm("space").set("world")
    tp_sn.setInput(tp_sn.inputIndex("normal"), mn_s, 0)
    ra_sn = ml.createNode("mtlxrange", "sand_rasn"); ra_sn.parm("signature").set("vector3"); ra_sn.setInput(0, tp_sn, 0)
    for ch in ("x", "y", "z"):
        ra_sn.parm("inlow_vector3" + ch).set(0.0); ra_sn.parm("inhigh_vector3" + ch).set(1.0)
        ra_sn.parm("outlow_vector3" + ch).set(-0.28); ra_sn.parm("outhigh_vector3" + ch).set(0.28)
    add_sn = ml.createNode("mtlxadd", "sand_addsn"); add_sn.parm("signature").set("vector3")
    add_sn.setInput(0, mn_s, 0); add_sn.setInput(1, ra_sn, 0)
    nrm_sn = ml.createNode("mtlxnormalize", "sand_nrm"); nrm_sn.setInput(0, add_sn, 0)
    ss_sand.setInput(ss_sand.inputIndex("normal"), nrm_sn, 0)
    col_sand.setInput(0, ss_sand, 0)

    # Simple helper for Rubble, Anemone (high SSS!), Fish (iridescent scales), Flora, Snow
    def make_prop_mtlx(mat_name, sss, sss_scale, rough, coat, thin_walled=0):
        c = ml.createNode("collect", mat_name)
        s = ml.createNode("mtlxstandard_surface", mat_name + "_ss")
        s.parm("base").set(0.88)
        s.parm("specular_roughness").set(rough)
        s.parm("subsurface").set(sss)
        s.parm("subsurface_scale").set(sss_scale)
        s.parm("subsurface_anisotropy").set(0.4)
        s.parm("coat").set(coat)
        s.parm("coat_roughness").set(0.10)
        s.parm("coat_IOR").set(1.37)
        s.parm("thin_walled").set(thin_walled)
        gw = ml.createNode("mtlxgeompropvalue", mat_name + "_gw"); gw.parm("signature").set("color3"); gw.parm("geomprop").set("wall_cd")
        s.setInput(s.inputIndex("base_color"), gw, 0)
        s.setInput(s.inputIndex("subsurface_color"), gw, 0)
        c.setInput(0, s, 0)

    make_prop_mtlx("mat_rubble",  0.22, 0.010, 0.84, 0.04)
    rub_ss = ml.node("mat_rubble_ss")
    rub_ss.parm("specular").set(0.28)
    rub_ss.parm("diffuse_roughness").set(0.55)
    mp_r = ml.createNode("mtlxposition", "rub_pos"); mp_r.parm("space").set("world")
    tp_r = ml.createNode("mtlxtriplanarprojection", "rub_tpn"); tp_r.parm("signature").set("vector3")
    rub_nrm_p = os.path.join(TEX_DIR, "staghorn_normal.png")
    for fk in ("filex", "filey", "filez"):
        tp_r.parm(fk).set(rub_nrm_p)
    for ck in ("filexcolorspace", "fileycolorspace", "filezcolorspace"):
        if tp_r.parm(ck):
            tp_r.parm(ck).set("Raw")
    mul_r = ml.createNode("mtlxmultiply", "rub_mulp"); mul_r.parm("signature").set("vector3")
    mul_r.setInput(0, mp_r, 0)
    mul_r.parm("in2_vector3x").set(3.2); mul_r.parm("in2_vector3y").set(3.2); mul_r.parm("in2_vector3z").set(3.2)
    tp_r.setInput(tp_r.inputIndex("position"), mul_r, 0)
    mn_r = ml.createNode("mtlxnormal", "rub_mn"); mn_r.parm("space").set("world")
    tp_r.setInput(tp_r.inputIndex("normal"), mn_r, 0)
    ra_r = ml.createNode("mtlxrange", "rub_ran"); ra_r.parm("signature").set("vector3"); ra_r.setInput(0, tp_r, 0)
    for ch in ("x", "y", "z"):
        ra_r.parm("inlow_vector3" + ch).set(0.0); ra_r.parm("inhigh_vector3" + ch).set(1.0)
        ra_r.parm("outlow_vector3" + ch).set(-0.62); ra_r.parm("outhigh_vector3" + ch).set(0.62)
    add_r = ml.createNode("mtlxadd", "rub_addn"); add_r.parm("signature").set("vector3")
    add_r.setInput(0, mn_r, 0); add_r.setInput(1, ra_r, 0)
    nrm_r = ml.createNode("mtlxnormalize", "rub_norm"); nrm_r.setInput(0, add_r, 0)
    rub_ss.setInput(rub_ss.inputIndex("normal"), nrm_r, 0)

    make_prop_mtlx("mat_anemone", 0.82, 0.028, 0.22, 0.78)
    make_prop_mtlx("mat_fish",    0.32, 0.012, 0.18, 0.88)
    make_prop_mtlx("mat_flora",   0.58, 0.018, 0.44, 0.38, thin_walled=1)
    make_prop_mtlx("mat_snow",    0.85, 0.015, 0.25, 0.50)

    # Pure matte black shadow-blocker for caustic_canopy
    col_gobo = ml.createNode("collect", "mat_canopy")
    ss_gobo = ml.createNode("mtlxstandard_surface", "canopy_ss")
    ss_gobo.parm("base").set(0.0); ss_gobo.parm("specular").set(0.0)
    col_gobo.setInput(0, ss_gobo, 0)

    # Self-luminous ocean water-column cyclorama (tinted by mood_tint constant for Pastel vs Tropical!)
    col_bg = ml.createNode("collect", "mat_backdrop")
    ss_bg = ml.createNode("mtlxstandard_surface", "backdrop_ss")
    ss_bg.parm("base").set(0.0); ss_bg.parm("specular").set(0.0); ss_bg.parm("emission").set(1.05)
    gp_bg = ml.createNode("mtlxgeompropvalue", "backdrop_gw"); gp_bg.parm("signature").set("color3"); gp_bg.parm("geomprop").set("wall_cd")
    bg_tint = ml.createNode("mtlxconstant", "backdrop_tint"); bg_tint.parm("signature").set("color3")
    bg_tint.parm("value_color3r").set(1.0); bg_tint.parm("value_color3g").set(1.0); bg_tint.parm("value_color3b").set(1.0)
    bg_mul = ml.createNode("mtlxmultiply", "backdrop_mul"); bg_mul.parm("signature").set("color3")
    bg_mul.setInput(0, gp_bg, 0); bg_mul.setInput(1, bg_tint, 0)
    ss_bg.setInput(ss_bg.inputIndex("emission_color"), bg_mul, 0)
    col_bg.setInput(0, ss_bg, 0)

    # Translucent additive volumetric sun-ray ribbon material
    col_gr = ml.createNode("collect", "mat_godrays")
    ss_gr = ml.createNode("mtlxstandard_surface", "godrays_ss")
    ss_gr.parm("base").set(0.0); ss_gr.parm("specular").set(0.0); ss_gr.parm("thin_walled").set(1)
    ss_gr.parm("emission").set(1.25)
    gp_gr = ml.createNode("mtlxgeompropvalue", "godrays_gw"); gp_gr.parm("signature").set("color3"); gp_gr.parm("geomprop").set("wall_cd")
    gr_tint = ml.createNode("mtlxconstant", "godrays_tint"); gr_tint.parm("signature").set("color3")
    gr_tint.parm("value_color3r").set(1.0); gr_tint.parm("value_color3g").set(1.0); gr_tint.parm("value_color3b").set(1.0)
    gr_mul = ml.createNode("mtlxmultiply", "godrays_mul"); gr_mul.parm("signature").set("color3")
    gr_mul.setInput(0, gp_gr, 0); gr_mul.setInput(1, gr_tint, 0)
    ss_gr.setInput(ss_gr.inputIndex("emission_color"), gr_mul, 0)
    gp_gtm = ml.createNode("mtlxgeompropvalue", "godrays_tm"); gp_gtm.parm("geomprop").set("tip_mask")
    ra_gop = ml.createNode("mtlxrange", "godrays_op"); ra_gop.setInput(0, gp_gtm, 0)
    ra_gop.parm("inlow").set(0.0); ra_gop.parm("inhigh").set(1.0); ra_gop.parm("outlow").set(0.0); ra_gop.parm("outhigh").set(0.065)
    cb_gop = ml.createNode("mtlxcombine3", "godrays_op3"); cb_gop.parm("signature").set("color3")
    cb_gop.setInput(0, ra_gop, 0); cb_gop.setInput(1, ra_gop, 0); cb_gop.setInput(2, ra_gop, 0)
    ss_gr.setInput(ss_gr.inputIndex("opacity"), cb_gop, 0)
    col_gr.setInput(0, ss_gr, 0)

    assignments = [
        ("mat_staghorn", "/import_reef/staghorn"),
        ("mat_brain",    "/import_reef/brain"),
        ("mat_table",    "/import_reef/table"),
        ("mat_sponge",   "/import_reef/sponge"),
        ("mat_sand",     "/import_reef/sand_floor"),
        ("mat_rubble",   "/import_reef/rubble"),
        ("mat_anemone",  "/import_reef/anemone"),
        ("mat_fish",     "/import_reef/fish"),
        ("mat_flora",    "/import_reef/flora"),
        ("mat_snow",     "/import_reef/snow"),
        ("mat_canopy",   "/import_reef/caustic_canopy"),
        ("mat_backdrop", "/import_reef/water_backdrop"),
        ("mat_godrays",  "/import_reef/god_rays"),
    ]
    ml.parm("materials").set(len(assignments))
    for idx, (mnode, gpath) in enumerate(assignments, start=1):
        ml.parm(f"matnode{idx}").set(mnode)
        ml.parm(f"matpath{idx}").set(mnode)
        ml.parm(f"assign{idx}").set(1)
        ml.parm(f"geopath{idx}").set(gpath)

    # 7. Underwater Lighting: HDR Snell's-Window Dome + Caustic-Filtered Raking Sun + Fill + Rim + Volumetric Haze
    dome = stage.createNode("domelight::3.0", "water_dome")
    dome.setInput(0, ml)
    dome.parm("xn__inputstexturefile_r3ah").set(os.path.join(HOUDINI_DIR, "karma_dome_pastel.hdr"))
    dome.parm("xn__inputsintensity_i0a").set(0.42)

    # Primary refracted solar caustic beam (raking -41 deg so 3D sand ripples, septa teeth & pores cast rich shadows!)
    sun = stage.createNode("distantlight::2.0", "refracted_sun")
    sun.setInput(0, dome)
    sun.parm("rx").set(-41.5)
    sun.parm("ry").set(38.0)
    sun.parm("xn__inputsintensity_i0a").set(2.75)
    sun.parm("xn__inputsangle_zta").set(1.10)
    sun.parm("xn__inputscolor_ztar").set(1.04)
    sun.parm("xn__inputscolor_ztag").set(0.99)
    sun.parm("xn__inputscolor_ztab").set(0.96)

    # Gentle forward-scattered underwater sun halo
    sun_soft = stage.createNode("distantlight::2.0", "scattered_sun")
    sun_soft.setInput(0, sun)
    sun_soft.parm("rx").set(-41.5)
    sun_soft.parm("ry").set(38.0)
    sun_soft.parm("xn__inputsintensity_i0a").set(0.42)
    sun_soft.parm("xn__inputsangle_zta").set(16.0)
    sun_soft.parm("xn__inputscolor_ztar").set(0.94)
    sun_soft.parm("xn__inputscolor_ztag").set(0.95)
    sun_soft.parm("xn__inputscolor_ztab").set(1.00)

    # Soft front-left underwater fill light (matches WebGPU uFillDir = normalize(-0.65, 0.35, 0.65))
    fill = stage.createNode("distantlight::2.0", "water_fill")
    fill.setInput(0, sun_soft)
    fill.parm("rx").set(-22.0)
    fill.parm("ry").set(-45.0)
    fill.parm("xn__inputsintensity_i0a").set(0.45)
    fill.parm("xn__inputsangle_zta").set(22.0)
    fill.parm("xn__inputscolor_ztar").set(0.84)
    fill.parm("xn__inputscolor_ztag").set(0.90)
    fill.parm("xn__inputscolor_ztab").set(1.00)

    # Rim / back-scatter light (illuminates translucent coral tips, sea fans, & anemone bubble-tips from behind!)
    rim = stage.createNode("distantlight::2.0", "water_rim")
    rim.setInput(0, fill)
    rim.parm("rx").set(-30.0)
    rim.parm("ry").set(180.0)
    rim.parm("xn__inputsintensity_i0a").set(0.75)
    rim.parm("xn__inputsangle_zta").set(12.0)
    rim.parm("xn__inputscolor_ztar").set(0.92)
    rim.parm("xn__inputscolor_ztag").set(0.88)
    rim.parm("xn__inputscolor_ztab").set(1.00)

    # Volumetric Underwater Haze (shadow=0 for 6x faster full-frame OIDN convergence!)
    fog = stage.createNode("karmafogbox", "underwater_volume")
    fog.setInput(0, rim)
    fog.parm("primpath").set("/lights/underwater_volume")
    fog.parm("ty").set(2.2)
    fog.parm("sx").set(34.0); fog.parm("sy").set(6.5); fog.parm("sz").set(34.0)
    fog.parm("density").set(0.024)
    fog.parm("shadow").set(0)
    fog.parm("phase").set(0.58)
    fog.parm("colorr").set(0.52); fog.parm("colorg").set(0.44); fog.parm("colorb").set(0.78)

    # 8. Single Physical Thin-Lens DOF USD Camera (/cameras/render_cam — explicit *_control='set' for exact 16:9!)
    cam = stage.createNode("camera", "render_cam")
    cam.setInput(0, fog)
    cam.parm("primpath").set("/cameras/render_cam")
    cam.parm("tx").set(-0.20); cam.parm("ty").set(1.12); cam.parm("tz").set(2.95)
    cam.parm("lookatenable").set(1)
    cam.parm("lookatpositionx").set(0.05); cam.parm("lookatpositiony").set(1.16); cam.parm("lookatpositionz").set(0.02)
    vap = 20.955
    hap = vap * (16.0 / 9.0)
    for cparm in ("horizontalAperture_control", "verticalAperture_control", "focalLength_control", "focusDistance_control", "fStop_control", "clippingRange_control"):
        if cam.parm(cparm):
            cam.parm(cparm).set("set")
    cam.parm("horizontalAperture").set(hap)
    cam.parm("verticalAperture").set(vap)
    cam.parm("focalLength").set(vap / (2.0 * math.tan(math.radians(33.0 * 0.5))))
    cam.parm("focusDistance").set(2.94)
    cam.parm("fStop").set(9.0)
    cam.parm("clippingRange1").set(0.05)
    cam.parm("clippingRange2").set(50.0)

    krs = stage.createNode("karmarendersettings", "karma_settings")
    krs.setInput(0, cam)
    krs.parm("camera").set("/cameras/render_cam")
    krs.parm("engine").set("xpu")
    for rparm, rval in [("resolutionx", 1280), ("resolutiony", 720)]:
        try:
            krs.parm(rparm).set(rval)
        except hou.PermissionError:
            pass
    krs.parm("pathtracedsamples").set(32)
    krs.parm("diffuselimit").set(2)
    krs.parm("reflectionlimit").set(3)
    krs.parm("refractionlimit").set(3)
    krs.parm("volumelimit").set(1)
    krs.parm("ssslimit").set(2)
    krs.parm("colorlimit").set(3.8)
    krs.parm("truedisplace").set(0)
    krs.parm("denoiser").set("oidn")
    krs.parm("denoise_usealbedo").set(1)
    krs.parm("denoise_useN").set(1)
    krs.setDisplayFlag(True)
    stage.layoutChildren()

    hou.clearAllSelected()
    hou.hipFile.save(KARMA_HIP)
    return {
        "build_sec": round(time.time() - t_start, 2),
        "total_verts": int(len(merged["P"])),
        "total_tris": int(len(merged["tris"])),
        "hip_saved": KARMA_HIP,
    }
