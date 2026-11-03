# Houdini FX 22.0 Solaris/Karma → Three.js WebGPU + TSL Coral Reef Biome

A real-time **Three.js WebGPU + TSL (Three Shading Language)** tropical coral reef ecosystem calibrated directly against **Houdini FX 22.0 Solaris + Karma** path-traced ground-truth renders (`mtlxstandard_surface`, Random-Walk Subsurface Scattering, dual-lobe wet-mucus clearcoat, 2-bounce GI, and physical underwater extinction).

## Highlights

- **True 3D Geometric Vertex Displacement**:
  - Graded `360×360` carbonate sand seabed displaced on the GPU by Houdini-baked vortex ripples (`sand_patch_nh.png`) and PolyHaven reef-rubble aprons (`reef_substrate_nh.png`).
  - Calcified aragonite coral skeleton relief (`coral_bump_nh.png`) with automatic holdfast blending.
- **Steep Parallax Relief & 3D Volumetric Calice Pores**:
  - `128³` 3D Worley honeycomb calice volume (`honeycomb_field_128.bin`) with close-range view-ray crater parallax and finite-difference gradients.
  - 3-plane blended PolyHaven coral-limestone scan bump, `~3.2 mm` meandroid **brain-coral septa teeth**, and radial vascular **foliose fan costae ribs**.
- **Karma-Calibrated Lookdev & Underwater Light Transport**:
  - Random-Walk in-pore SSS transmission + smooth growing-tip pore taper (`tipSuppress`).
  - Dual-lobe specular pairing rough calcium carbonate with a tight wet-mucus clearcoat (`IOR 1.37` feel) and thin-film iridescence.
  - Real-time GPU water heightfield & projected-grid caustics (`js/ocean-surface.js`, adapted from `jeantimex/webgpu-water`), Snell-refracted sun shafts, depth-dependent Beer-Lambert spectral extinction, GPU-instanced reef fish schools, swaying sea fans/kelp, sea anemones, urchins, and bokeh marine snow.
- **High-Frame-Rate WebGPU Performance**: Runs at **90–140 FPS** at `2940×1660` (`4.88 MP`).

## Importing into Google AI Studio

1. Open [Google AI Studio](https://aistudio.google.com/) → **Build** → **Import from GitHub**.
2. Select `arijoseph-collab/webgpu-coral-reef`.
3. AI Studio will automatically detect `package.json`, `vite.config.js`, `metadata.json`, and `index.html`. Binary 3D/texture assets in `public/` also include an automatic CORS fallback to GitHub Raw if a cloud sandbox skips large binary files.

## Running Locally

### Option 1 — Vite Dev Server (Recommended)
```bash
npm install
npm run dev
```

### Option 2 — Zero-Dependency Node or Python Server
```bash
npm start
# or:
python3 serve.py
```
Then open `http://127.0.0.1:8085/`.

## URL Query Parameters

- `?mood=day|sunset|night` — Lighting & water-column preset (`day` pastel reef, `sunset` tropical lagoon, `night` bioluminescence).
- `?cam=wide|brain|sponge` — Camera framing preset (`wide` hero bouquet, `brain` wide reef & sand floor, `sponge` polyp & pore macro).
- `?bump=1.25` — Override 3D vertex displacement, POM & micro-bump amplitude (`0` disables all 3D relief for A/B comparison).
- `?hud=0` — Hide HUD overlays for clean full-bleed viewport capture.
