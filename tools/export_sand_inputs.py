"""Export the coral footprint + seabed heights from the Houdini GLB on the sand-lab grid (for Houdini).

Grid: N x N cells over world x, z in [-H, H] (row = z, col = x). Writes houdini/sand_inputs.npz with
  occ   (N,N) uint8   1 where any coral vertex projects into the cell
  top   (N,N) float32 highest coral point above the cell (-1 if none)
  base  (N,N) float32 lowest coral point above the cell (-1 if none)
  seabed(N,N) float32 seabed height (bilinear from the 180x180 Houdini seabed grid)
"""
import json, struct, sys
import numpy as np

glb, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
H = float(sys.argv[4]) if len(sys.argv) > 4 else 4.5

f = open(glb, 'rb').read()
clen, _ = struct.unpack_from('<II', f, 12)
j = json.loads(f[20:20 + clen])
boff = 20 + clen
blen, _ = struct.unpack_from('<II', f, boff)
bin_ = f[boff + 8: boff + 8 + blen]

def acc(i):
    a = j['accessors'][i]
    bv = j['bufferViews'][a['bufferView']]
    comp = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[a['componentType']]
    n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
    off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    arr = np.frombuffer(bin_, dtype=comp, count=a['count'] * n, offset=off)
    return arr.reshape(a['count'], n) if n > 1 else arr

occ = np.zeros((N, N), np.uint8)
top = np.full((N, N), -1.0, np.float32)
base = np.full((N, N), 99.0, np.float32)
seabed = None
cell = 2 * H / N
for m in j['meshes']:
    pos = acc(m['primitives'][0]['attributes']['POSITION']).astype(np.float64)
    if 'seabed' in m['name']:
        # regular grid -> bilinear resample on the lab grid
        n = int(round(np.sqrt(len(pos))))
        x0, x1 = pos[:, 0].min(), pos[:, 0].max()
        gi = np.round((pos[:, 0] - x0) / (x1 - x0) * (n - 1)).astype(int)
        gj = np.round((pos[:, 2] - x0) / (x1 - x0) * (n - 1)).astype(int)
        g = np.zeros((n, n))
        g[gj, gi] = pos[:, 1]
        c = (np.arange(N) + 0.5) * cell - H
        fx = (c - x0) / (x1 - x0) * (n - 1)
        i0 = np.clip(np.floor(fx).astype(int), 0, n - 2)
        t = fx - i0
        # rows (z) then cols (x)
        rows = g[i0, :] * (1 - t)[:, None] + g[i0 + 1, :] * t[:, None]
        seabed = (rows[:, i0] * (1 - t)[None, :] + rows[:, i0 + 1] * t[None, :]).astype(np.float32)
        continue
    ix = np.floor((pos[:, 0] + H) / cell).astype(int)
    iz = np.floor((pos[:, 2] + H) / cell).astype(int)
    ok = (ix >= 0) & (iz >= 0) & (ix < N) & (iz < N)
    ix, iz, y = ix[ok], iz[ok], pos[ok, 1].astype(np.float32)
    occ[iz, ix] = 1
    np.maximum.at(top, (iz, ix), y)
    np.minimum.at(base, (iz, ix), y)
base[base > 98] = -1
np.savez_compressed(out, occ=occ, top=top, base=base, seabed=seabed, H=np.float32(H), N=np.int32(N))
print('occupied cells', int(occ.sum()), 'of', N * N, '| seabed', float(seabed.min()), float(seabed.max()))
