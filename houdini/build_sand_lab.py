# -----------------------------------------------------------------------------
# WebGPU coral reef — Houdini "sand lab"
#
# Builds /obj/webgpu_sand_lab inside a running Houdini 22 session and bakes realistic reef-sand data for
# the WebGPU seabed shader (public/textures/sand/*.png via tools/pack_sand_textures.py):
#
#   heightfield (9 x 9 m around the reef, 1024^2, 8.8 mm voxels)
#     -> sand_ripples   (Python SOP) Swift-Hohenberg pattern formation with anisotropy along the wave
#                       surge (crests perpendicular to it) and the coral footprint as an obstacle, so ripple
#                       crests bend, fork and terminate around the coral heads like real vortex ripples;
#                       vortex-ripple profile (sharp crests, rounded troughs), patchy amplitude, scour moats
#                       around coral bases, bioturbation (burrow mounds + holes, worm casts) and grazing trails
#     -> heightfield_noise  (native) gentle large-scale undulation
#     -> heightfield_erode  (native, thermal only) relaxes over-steep slopes to a sand angle of repose
#     -> shell grit: heightfield_scatter -> copytopoints (flattened shell / rubble chips) -> heightfield_project
#     -> sand_export    (Python SOP) writes height + masks to houdini/sand_outputs.npz
#   grain tile (60 cm, 512^2 periodic) -> grain_tile (Python SOP) grains / shell fragments -> houdini/grain_tile.npz
#
# Run inside Houdini (Python shell or the MCP bridge):
#   exec(open('/path/to/webgpu-coral-reef/houdini/build_sand_lab.py').read()); build_sand_lab(ROOT)
# -----------------------------------------------------------------------------
import os
import hou

SAND_RIPPLES_CODE = r'''
import numpy as np, os
node = hou.pwd()
geo = node.geometry()
ROOT = node.parent().parm("root_dir").eval()
inp = np.load(os.path.join(ROOT, "houdini", "sand_inputs.npz"))
occ = inp["occ"].astype(np.float32)          # (N, N) rows = z, cols = x
N = occ.shape[0]
# close the holes between splatted coral vertices (3x3 dilation, then 3x3 erosion)
def _mx(a):
    b = a.copy()
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            b = np.maximum(b, np.roll(np.roll(a, dz, 0), dx, 1))
    return b
occ = 1.0 - _mx(1.0 - _mx(occ))
H = float(inp["H"])
cell = 2 * H / N                              # metres per voxel
rng = np.random.default_rng(int(node.parent().parm("seed").eval()))

# --- distance (m) to the coral footprint: 8-neighbour chamfer on a 4x max-pooled grid + bilinear upsample
P = 4
M = N // P
occ_c = occ.reshape(M, P, M, P).max(axis=(1, 3))
INF = 1e9
dc = [[0.0 if occ_c[j, i] > 0 else INF for i in range(M)] for j in range(M)]
R2 = 2 ** 0.5
for j in range(M):
    row = dc[j]; prev = dc[j - 1] if j > 0 else None
    for i in range(M):
        v = row[i]
        if i > 0 and row[i - 1] + 1 < v: v = row[i - 1] + 1
        if prev is not None:
            if prev[i] + 1 < v: v = prev[i] + 1
            if i > 0 and prev[i - 1] + R2 < v: v = prev[i - 1] + R2
            if i < M - 1 and prev[i + 1] + R2 < v: v = prev[i + 1] + R2
        row[i] = v
for j in range(M - 1, -1, -1):
    row = dc[j]; nxt = dc[j + 1] if j < M - 1 else None
    for i in range(M - 1, -1, -1):
        v = row[i]
        if i < M - 1 and row[i + 1] + 1 < v: v = row[i + 1] + 1
        if nxt is not None:
            if nxt[i] + 1 < v: v = nxt[i] + 1
            if i < M - 1 and nxt[i + 1] + R2 < v: v = nxt[i + 1] + R2
            if i > 0 and nxt[i - 1] + R2 < v: v = nxt[i - 1] + R2
        row[i] = v
dc = np.array(dc, dtype=np.float32) * (cell * P)
cz = (np.arange(N) + 0.5) / P - 0.5
c0 = np.clip(np.floor(cz).astype(int), 0, M - 2)
ct = np.clip(cz - c0, 0.0, 1.0).astype(np.float32)
rows_ = dc[c0, :] * (1 - ct)[:, None] + dc[c0 + 1, :] * ct[:, None]
dist = rows_[:, c0] * (1 - ct)[None, :] + rows_[:, c0 + 1] * ct[None, :]
dist = np.where(occ > 0, 0.0, np.maximum(dist - cell * P * 0.5, cell)).astype(np.float32)

# --- smooth low-frequency noise helper (FFT-filtered white noise, periodic)
kx = np.fft.fftfreq(N)[None, :]
kz = np.fft.fftfreq(N)[:, None]
def smooth_noise(scale_m, seed_shift=0):
    w = np.random.default_rng(int(node.parent().parm("seed").eval()) + seed_shift).standard_normal((N, N))
    f = np.fft.fft2(w)
    sig = cell / scale_m
    f *= np.exp(-(kx ** 2 + kz ** 2) / (2 * sig ** 2))
    n = np.real(np.fft.ifft2(f))
    return (n / (n.std() + 1e-9)).astype(np.float32)

# --- Swift-Hohenberg ripple pattern (semi-implicit spectral), anisotropic + obstacle damping
lam = float(node.parent().parm("ripple_wavelength").eval())     # metres
k0 = 2 * np.pi * cell / lam                                     # rad / voxel
surge = np.array([float(node.parent().parm("surge_x").eval()), float(node.parent().parm("surge_z").eval())])
surge /= np.linalg.norm(surge)
KX = 2 * np.pi * kx
KZ = 2 * np.pi * kz
k2 = KX ** 2 + KZ ** 2
k_along_crest = KX * (-surge[1]) + KZ * surge[0]               # component along the crest direction
r = 0.32
aniso = float(node.parent().parm("anisotropy").eval())
L = r - (k0 ** 2 - k2) ** 2 / (k0 ** 4) - aniso * (k_along_crest ** 2) / (k0 ** 2)
damp = np.clip(1.0 - dist / 0.14, 0.0, 1.0) ** 1.5             # no ripples on / right next to coral
u = (rng.standard_normal((N, N)) * 0.05).astype(np.float64) * (1 - damp)
dt = 0.6
steps = int(node.parent().parm("sim_steps").eval())
denom = 1.0 - dt * L
for it in range(steps):
    u = np.real(np.fft.ifft2(np.fft.fft2(u - dt * u ** 3) / denom))
    u *= (1.0 - damp * 0.35)
u = u / (np.percentile(np.abs(u), 99.5) + 1e-9)
u = np.clip(u, -1.0, 1.0)

# --- natural variation: warp the pattern with low-frequency noise (wavelength / orientation drift)
wx = smooth_noise(1.6, 11) * (0.035 / cell)
wz = smooth_noise(1.6, 12) * (0.035 / cell)
gz, gx = np.mgrid[0:N, 0:N].astype(np.float32)
sx = np.clip(gx + wx, 0, N - 1.001)
sz = np.clip(gz + wz, 0, N - 1.001)
x0 = np.floor(sx).astype(int); z0 = np.floor(sz).astype(int)
tx = sx - x0; tz = sz - z0
uw = (u[z0, x0] * (1 - tx) * (1 - tz) + u[z0, x0 + 1] * tx * (1 - tz) + u[z0 + 1, x0] * (1 - tx) * tz + u[z0 + 1, x0 + 1] * tx * tz)

# --- vortex-ripple profile: sharp crests, broad rounded troughs; patchy amplitude
s = (uw + 1.0) * 0.5                                            # 0 trough .. 1 crest
profile = s ** 1.7
amp_noise = np.clip(0.72 + 0.28 * smooth_noise(2.2, 21), 0.35, 1.0)
near = np.clip((dist - 0.1) / 0.5, 0.0, 1.0)                    # sheltered / scoured sand near coral: smooth
ripple_h = float(node.parent().parm("ripple_height").eval())
h = (profile - profile.mean()) * ripple_h * amp_noise * near

# --- scour moats around the coral bases (oscillatory flow scours a ring around obstacles)
h += -0.024 * np.exp(-dist / 0.12)

# --- bioturbation: burrow mounds with a central hole, worm casts, grazing trails
burrow = np.zeros((N, N), np.float32)
free = np.argwhere((dist > 0.35))
for i in range(int(node.parent().parm("burrows").eval())):
    zi, xi = free[rng.integers(len(free))]
    R = rng.uniform(0.018, 0.034) / cell
    hgt = rng.uniform(0.006, 0.012)
    rad = int(R * 2.2) + 2
    z_a, z_b, x_a, x_b = max(0, zi - rad), min(N, zi + rad), max(0, xi - rad), min(N, xi + rad)
    yy, xx = np.mgrid[z_a:z_b, x_a:x_b]
    rr = np.hypot(yy - zi, xx - xi) / R
    mound = hgt * np.exp(-rr ** 2 * 1.6)
    hole = -0.011 * np.exp(-(rr / 0.28) ** 2)
    h[z_a:z_b, x_a:x_b] += mound + hole
    burrow[z_a:z_b, x_a:x_b] = np.maximum(burrow[z_a:z_b, x_a:x_b], np.exp(-rr ** 2 * 1.2))
for i in range(int(node.parent().parm("trails").eval())):   # meandering sea-cucumber / snail trails
    z, x = free[rng.integers(len(free))].astype(np.float64)
    ang = rng.uniform(0, 2 * np.pi)
    for st in range(int(rng.uniform(80, 220))):
        ang += rng.normal(0, 0.12)
        z += np.sin(ang) * 0.006 / cell; x += np.cos(ang) * 0.006 / cell
        zi, xi = int(z), int(x)
        if zi < 3 or xi < 3 or zi > N - 4 or xi > N - 4 or occ[zi, xi] > 0: break
        h[zi - 1:zi + 2, xi - 1:xi + 2] -= 0.0011
        burrow[zi - 1:zi + 2, xi - 1:xi + 2] = np.maximum(burrow[zi - 1:zi + 2, xi - 1:xi + 2], 0.5)

# --- write into the heightfield ('height' layer, rows = z, cols = x) and add mask layers
height = None
for p in geo.prims():
    if p.attribValue("name") == "height":
        height = p
res = height.resolution()
p00 = height.indexToPos((0, 0, 0)); p10 = height.indexToPos((1, 0, 0)); p01 = height.indexToPos((0, 1, 0))
i_is_x = abs(p10[0] - p00[0]) > abs(p10[2] - p00[2])
def to_vox(a):
    # npz layout: rows = +z, cols = +x  ->  volume layout: [j][i]
    if not i_is_x:
        a = a.T
    if (p10 - p00)[0 if i_is_x else 2] < 0:
        a = a[:, ::-1]
    if (p01 - p00)[2 if i_is_x else 0] < 0:
        a = a[::-1, :]
    return np.ascontiguousarray(a, dtype=np.float32)
node.setUserData("vox_mapping", "i_is_x=%s p00=%s p10=%s p01=%s" % (i_is_x, tuple(p00), tuple(p10), tuple(p01)))
base = np.frombuffer(height.allVoxelsAsString(), dtype=np.float32).reshape(res[1], res[0])
def fit(a):
    # bilinear resample (rows = z, cols = x) to the heightfield resolution if it differs from N
    if a.shape == (res[1], res[0]):
        return a
    zz = (np.arange(res[1]) + 0.5) * a.shape[0] / res[1] - 0.5
    xx = (np.arange(res[0]) + 0.5) * a.shape[1] / res[0] - 0.5
    z0 = np.clip(np.floor(zz).astype(int), 0, a.shape[0] - 2); tz = np.clip(zz - z0, 0, 1)[:, None]
    x0 = np.clip(np.floor(xx).astype(int), 0, a.shape[1] - 2); tx = np.clip(xx - x0, 0, 1)[None, :]
    r0 = a[z0][:, x0] * (1 - tx) + a[z0][:, x0 + 1] * tx
    r1 = a[z0 + 1][:, x0] * (1 - tx) + a[z0 + 1][:, x0 + 1] * tx
    return (r0 * (1 - tz) + r1 * tz).astype(np.float32)
height.setAllVoxelsFromString((base + to_vox(fit(h))).tobytes())
_m4 = hou.Matrix4(height.transform())      # Volume.transform() is the 3x3 part; setTransform wants a 4x4
_c = height.vertex(0).point().position()   # the volume's centre carries the translation
_m4.setAt(3, 0, _c[0]); _m4.setAt(3, 1, _c[1]); _m4.setAt(3, 2, _c[2])
def add_layer(name, arr):
    v = geo.createVolume(res[0], res[1], 1, height.boundingBox())
    v.setTransform(_m4)
    v.setAttribValue("name", name)
    v.setAllVoxelsFromString(to_vox(fit(arr)).tobytes())
add_layer("crest", s.astype(np.float32))
add_layer("coral_dist", dist)
add_layer("burrow", burrow)
add_layer("patch", smooth_noise(0.9, 31))
'''

SAND_EXPORT_CODE = r'''
import numpy as np, os
node = hou.pwd()
geo = node.geometry()
ROOT = node.parent().parm("root_dir").eval()
layers = {}
for p in geo.prims():
    if isinstance(p, hou.Volume):
        res = p.resolution()
        layers[p.attribValue("name")] = np.frombuffer(p.allVoxelsAsString(), dtype=np.float32).reshape(res[1], res[0]).copy()
p0 = None
for p in geo.prims():
    if isinstance(p, hou.Volume) and p.attribValue("name") == "height":
        p0 = p
layers["_p00"] = np.array(p0.indexToPos((0, 0, 0))); layers["_p10"] = np.array(p0.indexToPos((1, 0, 0))); layers["_p01"] = np.array(p0.indexToPos((0, 1, 0)))
np.savez_compressed(os.path.join(ROOT, "houdini", "sand_outputs.npz"), **layers)
'''

GRAIN_TILE_CODE = r'''
import numpy as np, os
node = hou.pwd()
geo = node.geometry()
ROOT = node.parent().parm("root_dir").eval()
rng = np.random.default_rng(int(node.parent().parm("seed").eval()) + 7)
N = 512
T = 0.6                              # tile size (m), periodic
cell = T / N
height = np.zeros((N, N), np.float32)
albedo = np.full((N, N), 0.5, np.float32)
shell = np.zeros((N, N), np.float32)
dark = np.zeros((N, N), np.float32)
def stamp(z, x, rz, rx, ang, hgt, val, layer=None, lval=1.0):
    rad = int(max(rz, rx) * 2) + 2
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    c, s = np.cos(ang), np.sin(ang)
    u = (xx * c + yy * s) / rx; v = (-xx * s + yy * c) / rz
    rr = u * u + v * v
    bump = hgt * np.sqrt(np.clip(1 - rr, 0, 1))
    zi = (np.arange(-rad, rad + 1) + int(z)) % N
    xi = (np.arange(-rad, rad + 1) + int(x)) % N
    sub = np.ix_(zi, xi)
    m = bump > height[sub]
    height[sub] = np.where(m, bump, height[sub])
    albedo[sub] = np.where(m, val, albedo[sub])
    if layer is not None:
        layer[sub] = np.where(m, np.maximum(layer[sub], lval), layer[sub])
# fine quartz / carbonate grains (0.3 - 1.4 mm), densely packed
for i in range(int(node.parent().parm("grains").eval())):
    r = rng.uniform(0.00025, 0.0007) / cell
    stamp(rng.uniform(0, N), rng.uniform(0, N), r * rng.uniform(0.7, 1.0), r, rng.uniform(0, np.pi), r * cell * rng.uniform(0.6, 1.0),
          float(np.clip(rng.normal(0.5, 0.07), 0.25, 0.75)))
# dark mineral / organic grains
for i in range(int(node.parent().parm("grains").eval()) // 25):
    r = rng.uniform(0.0003, 0.0008) / cell
    stamp(rng.uniform(0, N), rng.uniform(0, N), r, r, 0.0, r * cell * 0.8, rng.uniform(0.18, 0.3), dark, 1.0)
# shell & coral fragments (2 - 7 mm), flat and bright
for i in range(int(node.parent().parm("shell_bits").eval())):
    rx = rng.uniform(0.001, 0.0035) / cell
    stamp(rng.uniform(0, N), rng.uniform(0, N), rx * rng.uniform(0.35, 0.8), rx, rng.uniform(0, np.pi), rx * cell * 0.18,
          rng.uniform(0.64, 0.84), shell, 1.0)
np.savez_compressed(os.path.join(ROOT, "houdini", "grain_tile.npz"), height=height, albedo=albedo, shell=shell, dark=dark, T=np.float32(T))
'''


def _parm(node, name, value):
    p = node.parm(name)
    if p is None:
        return False
    p.set(value)
    return True


def build_sand_lab(root_dir, steps=260, cook=True):
    obj = hou.node("/obj")
    old = obj.node("webgpu_sand_lab")
    if old:
        old.destroy()
    lab = obj.createNode("geo", "webgpu_sand_lab")
    for c in lab.children():
        c.destroy()
    # spare parms that drive the Python SOPs (tweakable from the parameter pane)
    ptg = lab.parmTemplateGroup()
    folder = hou.FolderParmTemplate("sandlab", "Sand Lab")
    folder.addParmTemplate(hou.StringParmTemplate("root_dir", "Project Root", 1, default_value=(root_dir,)))
    folder.addParmTemplate(hou.IntParmTemplate("seed", "Seed", 1, default_value=(20261102,)))
    folder.addParmTemplate(hou.FloatParmTemplate("ripple_wavelength", "Ripple Wavelength (m)", 1, default_value=(0.105,)))
    folder.addParmTemplate(hou.FloatParmTemplate("ripple_height", "Ripple Height (m)", 1, default_value=(0.018,)))
    folder.addParmTemplate(hou.FloatParmTemplate("anisotropy", "Crest Alignment", 1, default_value=(0.62,)))
    folder.addParmTemplate(hou.FloatParmTemplate("surge_x", "Surge Dir X", 1, default_value=(0.5736,)))
    folder.addParmTemplate(hou.FloatParmTemplate("surge_z", "Surge Dir Z", 1, default_value=(-0.8192,)))
    folder.addParmTemplate(hou.IntParmTemplate("sim_steps", "Pattern Steps", 1, default_value=(steps,)))
    folder.addParmTemplate(hou.IntParmTemplate("burrows", "Burrows", 1, default_value=(320,)))
    folder.addParmTemplate(hou.IntParmTemplate("trails", "Grazing Trails", 1, default_value=(26,)))
    folder.addParmTemplate(hou.IntParmTemplate("grains", "Grain Tile Grains", 1, default_value=(60000,)))
    folder.addParmTemplate(hou.IntParmTemplate("shell_bits", "Grain Tile Shell Bits", 1, default_value=(520,)))
    ptg.append(folder)
    lab.setParmTemplateGroup(ptg)

    hf = lab.createNode("heightfield", "sand_base")
    _parm(hf, "sizex", 9.0)
    _parm(hf, "sizey", 9.0)
    _parm(hf, "gridspacing", 9.0 / 1024.0)
    _parm(hf, "gridsamples", 1024)  # resolution comes from 'gridsamples' in the default division mode

    ripples = lab.createNode("python", "sand_ripples")
    ripples.parm("python").set(SAND_RIPPLES_CODE)
    ripples.setInput(0, hf)

    noise = lab.createNode("heightfield_noise", "sand_undulation")
    noise.setInput(0, ripples)
    _parm(noise, "amp", 0.012)
    _parm(noise, "elementsize", 1.6)

    erode = noise  # (thermal erosion can be inserted here: heightfield_erode, thermal only)

    # shell grit / rubble chips projected onto the sand (native scatter -> copy -> project)
    scatter = lab.createNode("heightfield_scatter", "grit_points")
    scatter.setInput(0, erode)
    for n, v in (("totalpointcount", 9000), ("seed", 7)):
        _parm(scatter, n, v)
    chip = lab.createNode("sphere", "grit_chip")
    for n, v in (("type", 2), ("radx", 0.006), ("rady", 0.0016), ("radz", 0.004), ("rows", 6), ("cols", 8)):
        _parm(chip, n, v)
    rand = lab.createNode("attribwrangle", "grit_variation")
    rand.setInput(0, scatter)
    rand.parm("snippet").set(
        "f@pscale = fit01(pow(rand(@ptnum * 1.37), 3.0), 0.5, 2.6);\n"
        "vector4 q = quaternion(radians(rand(@ptnum + 0.1) * 360.0), {0,1,0});\n"
        "p@orient = qmultiply(q, quaternion(radians(fit01(rand(@ptnum + 0.7), -12, 12)), {1,0,0}));\n"
        "@P.y += 0.001;\n"
    )
    copy = lab.createNode("copytopoints::2.0", "grit_copy")
    copy.setInput(0, chip)
    copy.setInput(1, rand)
    project = lab.createNode("heightfield_project", "grit_project")
    project.setInput(0, erode)
    project.setInput(1, copy)

    export = lab.createNode("python", "sand_export")
    export.parm("python").set(SAND_EXPORT_CODE)
    export.setInput(0, project)
    export.setDisplayFlag(True)
    export.setRenderFlag(True)

    grain_src = lab.createNode("null", "grain_tile_src")
    grain = lab.createNode("python", "grain_tile")
    grain.parm("python").set(GRAIN_TILE_CODE)
    grain.setInput(0, grain_src)

    lab.layoutChildren()
    info = {"nodes": [c.name() for c in lab.children()], "cook": {}}
    if cook:
        import time as _t
        # cook node by node so a failure is attributed to the right SOP (and never aborts the bake)
        for n in (hf, ripples, noise, scatter, chip, rand, copy, project, export, grain_src, grain):
            t0 = _t.time()
            try:
                n.cook(force=False)
                info["cook"][n.name()] = round(_t.time() - t0, 2)
            except Exception as exc:  # hou.OperationFailed: details are in node.errors()
                info["cook"][n.name()] = "FAILED: " + str(exc).splitlines()[0]
        info["errors"] = {c.name(): list(c.errors()) for c in lab.children() if c.errors()}
        info["warnings"] = {c.name(): list(c.warnings())[:2] for c in lab.children() if c.warnings()}
    return info
