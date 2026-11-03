"""Local dry run of the sand-lab Python SOP code with a mocked `hou` (numpy only) + preview PNGs."""
import sys, types, os, time
import numpy as np

ROOT = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 260
src = open(os.path.join(ROOT, 'houdini', 'build_sand_lab.py')).read()
ns = {}
code_ripples = src.split("SAND_RIPPLES_CODE = r'''")[1].split("'''")[0]
code_export = src.split("SAND_EXPORT_CODE = r'''")[1].split("'''")[0]
code_grain = src.split("GRAIN_TILE_CODE = r'''")[1].split("'''")[0]

PARMS = dict(root_dir=ROOT, seed=20261102, ripple_wavelength=0.105, ripple_height=0.018, anisotropy=0.62,
             surge_x=0.5736, surge_z=-0.8192, sim_steps=STEPS, burrows=320, trails=26, grains=60000, shell_bits=520)

class Parm:
    def __init__(self, v): self.v = v
    def eval(self): return self.v

class Volume:
    def __init__(self, res, name='height'):
        self.res = res; self.data = np.zeros(res[1] * res[0], np.float32); self.name = name
    def resolution(self): return self.res
    def attribValue(self, n): return self.name
    def setAttribValue(self, n, v): self.name = v
    def indexToPos(self, idx):
        i, j, _ = idx
        return np.array([-4.5 + (i + 0.5) * 9 / self.res[0], 0.0, -4.5 + (j + 0.5) * 9 / self.res[1]])
    def allVoxelsAsString(self): return self.data.tobytes()
    def setAllVoxelsFromString(self, b): self.data = np.frombuffer(b, np.float32).copy()
    def boundingBox(self): return None
    def transform(self): return None
    def setTransform(self, t): pass
    def vertex(self, i):
        return types.SimpleNamespace(point=lambda: types.SimpleNamespace(position=lambda: (0.0, 0.0, 0.0)))

class Geo:
    def __init__(self): self.vols = [Volume((1024, 1024, 1))]
    def prims(self): return self.vols
    def createVolume(self, x, y, z, bb):
        v = Volume((x, y, z), 'new'); self.vols.append(v); return v

class Parent:
    def parm(self, n): return Parm(PARMS[n])

class Node:
    def __init__(self): self.g = Geo(); self.ud = {}
    def geometry(self): return self.g
    def parent(self): return Parent()
    def setUserData(self, k, v): self.ud[k] = v

class Matrix4:
    def __init__(self, m=None): pass
    def setAt(self, r, c, v): pass
hou = types.SimpleNamespace(Volume=Volume, Matrix4=Matrix4)
node = Node()
hou.pwd = lambda: node
t0 = time.time()
exec(code_ripples, {'hou': hou, '__name__': 'sop'})
t1 = time.time()
print('ripples cook %.1fs' % (t1 - t0), node.ud)
exec(code_export, {'hou': hou, '__name__': 'sop'})
t2 = time.time()
exec(code_grain, {'hou': hou, '__name__': 'sop'})
print('export %.1fs grain %.1fs' % (t2 - t1, time.time() - t2))

from PIL import Image
d = np.load(os.path.join(ROOT, 'houdini', 'sand_outputs.npz'))
h = d['height']
print('layers', list(d.keys()), 'height range %.4f..%.4f' % (h.min(), h.max()))
# shaded preview: light from the refracted sun direction
sun = np.array([0.36, 0.8, 0.48]); sun /= np.linalg.norm(sun)
cell = 9 / 1024
gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * cell)
gz = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * cell)
n = np.stack([-gx, np.ones_like(h), -gz], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
shade = np.clip(n @ sun, 0, 1)
img = (np.clip(shade * 1.1, 0, 1) * 255).astype(np.uint8)
Image.fromarray(img).save('/tmp/sand_preview_full.png')
Image.fromarray(img[300:812, 300:812]).resize((1024, 1024), Image.NEAREST).save('/tmp/sand_preview_zoom.png')
g = np.load(os.path.join(ROOT, 'houdini', 'grain_tile.npz'))
gh = g['height']; ga = g['albedo']
ggx = (np.roll(gh, -1, 1) - np.roll(gh, 1, 1)) / (2 * 0.6 / 512)
ggz = (np.roll(gh, -1, 0) - np.roll(gh, 1, 0)) / (2 * 0.6 / 512)
gn = np.stack([-ggx, np.ones_like(gh), -ggz], -1); gn /= np.linalg.norm(gn, axis=-1, keepdims=True)
gs = np.clip(gn @ sun, 0, 1) * (0.5 + ga)
Image.fromarray((np.clip(gs, 0, 1) * 255).astype(np.uint8)).save('/tmp/grain_preview.png')
print('done')
