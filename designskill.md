---
name: designskill
description: >-
  MANDATORY DEFAULT UI DESIGN SYSTEM FOR ALL PROJECTS: Enforces the user's preferred Minimal Floating-Pill Interface (Google Cloud Kiosk × Arrival Configurator × Teenage Engineering OP-1 Field aesthetic). Activate and follow this skill whenever creating, styling, refactoring, or updating any web app, 3D/WebGPU viewer, configurator, studio tool, dashboard, or interactive HUD.
---

# Minimal Floating-Pill Interface Design System (`designskill.md`)

Use this minimal floating-pill interface architecture as the **default UI design language for all projects going forward** (3D/WebGPU viewers, configurators, creative tools, hardware workbenches, AR/CV experiences, and web apps).

---

## 1. Core Architectural Principles ("Stage First, Floating Pills Only")

1. **≥ 90% Unobstructed Viewport Stage**:
   - The 3D canvas, interactive stage, or primary visual medium occupies `100vw × 100vh` (`position: fixed; inset: 0`).
   - **Never** use persistent full-height sidebars, wide right-hand control drawers, multi-row top navbars, or stacked dashboard cards that crowd the viewport.
2. **Strictly 2 to 3 Floating Pill Clusters (`border-radius: 999px`)**:
   - **Pill 1 — Top-Left Brand & Mode Pill (`#topLeftPill.floating-pill`)**: Compact brand mark + mono sub-badge (`.brand-sub`) + `1px` vertical `.pill-divider` + compact `.icon-pill-btn` mode/camera chips + `HIDE UI` toggle.
   - **Pill 2 — Top-Right Telemetry & Primary CTA Pill (`#topRightPill.floating-pill`)**: Live pulsing `.status-dot` + concise tabular monospace telemetry readout (`.telemetry-pill`) + on-demand popover triggers (`🎛️ TUNE`, `🎯 REF`) or a single `.primary-pill-cta`.
   - **Pill 3 — Bottom-Center Floating Dock (`.bottom-dock`, when needed)**: Centered at `bottom: 24px`, housing a horizontal scrollable strip of `.mini-chip` / `.tap-chip` presets, color `.swatch-dot` circles, or a `.segmented-pill` bar.
3. **Progressive Disclosure via On-Demand Popover Cards (`.popover-card`, `border-radius: 28px`)**:
   - Sliders, advanced shader/rig parameters, fabrication specs, and reference image galleries **must not** sit permanently open on screen.
   - Place them inside compact floating popover cards (`width: 330px–372px`, `border-radius: 28px`, `padding: 18px–22px`) anchored to the top or bottom pill, toggled on click and dismissed via `Esc` or outside click.
4. **Bullet-Concise Micro-Copy**:
   - Pill labels are **1–2 words max** (`HERO`, `WIDE`, `MACRO`, `LAGOON`, `TUNE`, `EXPORT`).
   - Never place multi-sentence explanatory paragraphs inside floating HUD elements; use `title="..."` tooltips for secondary descriptions.
5. **Built-In Zero-Chrome Mode (`H` Shortcut + `?hud=0` + `.show-ui-pill`)**:
   - Every app must support toggling all UI overlays via a `HIDE UI` pill button, the `H` keyboard shortcut (`body.ui-hidden`), and the `?hud=0` URL query parameter (`body.no-hud`) for clean full-bleed captures.

---

## 2. Design Tokens (Light Cyclorama & Dark Translucent Glass)

Include Google Sans / Inter + Google Sans Mono / Roboto Mono / JetBrains Mono and the following CSS tokens:

```css
:root {
  --font-sans: "Google Sans", "Google Sans Text", "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: "Google Sans Mono", "Roboto Mono", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;

  /* Light Studio Cyclorama Theme (Default for Product / Hardware / CAD Configurators) */
  --page: #f2f2f0;
  --stage: #f7f7f5;
  --card: #ffffff;
  --pill-bg: rgba(255, 255, 255, 0.90);
  --ink: #141414;
  --muted: #6b6b70;
  --line: #e3e3e0;
  --accent: #3b6bff;
  --accent-soft: #eef2ff;
  --green: #2da44e;
  --amber: #f9c74f;
  --danger: #e24b3b;
  --shadow-card: 0 1px 2px rgba(20, 20, 20, 0.04), 0 12px 32px rgba(20, 20, 20, 0.07);
  --shadow-lift: 0 2px 4px rgba(20, 20, 20, 0.06), 0 18px 44px rgba(20, 20, 20, 0.12);
  --blur-md: 14px;
  --ease: cubic-bezier(0.2, 0.8, 0.2, 1);
}

/* Dark / Immersive 3D & WebGPU Scene Theme */
body[data-theme="dark"] {
  --page: #071019;
  --stage: #0b1826;
  --card: #111c2a;
  --pill-bg: rgba(12, 22, 34, 0.78);
  --ink: #f2f6fa;
  --muted: #8ea0b5;
  --line: rgba(255, 255, 255, 0.13);
  --accent: #38bdf8;
  --accent-soft: rgba(56, 189, 248, 0.15);
  --green: #2dd4bf;
  --shadow-card: 0 2px 6px rgba(0, 0, 0, 0.35), 0 16px 38px rgba(0, 0, 0, 0.48);
  --shadow-lift: 0 4px 12px rgba(0, 0, 0, 0.45), 0 24px 52px rgba(0, 0, 0, 0.62);
}
```

---

## 3. Canonical Component CSS

```css
*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  -webkit-font-smoothing: antialiased;
  user-select: none;
}

html, body {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  background: var(--page);
  color: var(--ink);
  font-family: var(--font-sans);
}

/* 1. Full-Viewport Stage */
#viewportContainer, #canvas-container {
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
}

/* 2. Shared Floating Pill Surface */
.floating-pill {
  background: var(--pill-bg);
  backdrop-filter: blur(var(--blur-md));
  -webkit-backdrop-filter: blur(var(--blur-md));
  border: 1.5px solid var(--line);
  box-shadow: var(--shadow-card);
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 11px;
  transition: transform 0.22s var(--ease), box-shadow 0.22s var(--ease), border-color 0.22s var(--ease);
}

.pill-divider {
  width: 1px;
  height: 20px;
  background: var(--line);
  margin: 0 3px;
  flex-shrink: 0;
}

/* 3. Top Bar Layout (Left Brand/Controls Pill + Right Telemetry/Action Pill) */
.top-bar {
  position: fixed;
  top: 18px;
  left: 20px;
  right: 20px;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  pointer-events: none;
  z-index: 20;
}

.top-bar > * {
  pointer-events: auto;
}

.brand-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 3px 8px 3px 6px;
  font-weight: 600;
  font-size: 13.5px;
  letter-spacing: -0.015em;
  color: var(--ink);
  white-space: nowrap;
}

.brand-sub {
  font-family: var(--font-mono);
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--accent);
  background: var(--accent-soft);
  border: 1px solid color-mix(in srgb, var(--accent) 32%, transparent);
  padding: 2px 8px;
  border-radius: 999px;
}

/* 4. Pill Action Buttons & Mode Chips */
.icon-pill-btn {
  border: 1.5px solid transparent;
  background: transparent;
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  height: 32px;
  padding: 0 12px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  cursor: pointer;
  white-space: nowrap;
  transition: transform 0.18s var(--ease), border-color 0.18s var(--ease), background-color 0.18s var(--ease), color 0.18s var(--ease);
}

.icon-pill-btn:hover {
  color: var(--ink);
  background: color-mix(in srgb, var(--ink) 6%, transparent);
}

.icon-pill-btn:active {
  transform: scale(0.97);
}

.icon-pill-btn.active {
  border-color: color-mix(in srgb, var(--accent) 55%, transparent);
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
}

/* 5. Telemetry Readout & Primary Pill CTA */
.telemetry-pill {
  height: 32px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 0 11px;
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--green);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--green) 24%, transparent);
}

.primary-pill-cta {
  border: none;
  background: var(--accent);
  color: #06131f;
  font-family: var(--font-sans);
  font-size: 12.5px;
  font-weight: 600;
  letter-spacing: -0.01em;
  height: 32px;
  padding: 0 15px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  transition: transform 0.18s var(--ease), opacity 0.18s var(--ease);
}

/* 6. On-Demand Floating Popover Card (28px Kiosk Sheet) */
.popover-card {
  margin-top: 10px;
  width: 348px;
  max-height: calc(100vh - 96px);
  overflow-y: auto;
  scrollbar-width: thin;
  background: var(--pill-bg);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  border: 1.5px solid var(--line);
  box-shadow: var(--shadow-lift);
  border-radius: 26px;
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.popover-card.hidden {
  display: none !important;
}

/* 7. Bottom-Center Floating Dock */
.bottom-dock {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  z-index: 20;
  max-width: calc(100vw - 32px);
  pointer-events: none;
}

.bottom-dock > * {
  pointer-events: auto;
}

/* 8. Segmented Pill Control */
.segmented-pill {
  display: flex;
  background: color-mix(in srgb, var(--ink) 6%, transparent);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 3px;
  gap: 3px;
}

.seg-btn {
  flex: 1;
  border: 1px solid transparent;
  background: transparent;
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  padding: 6px 11px;
  border-radius: 999px;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.18s var(--ease);
}

.seg-btn.active {
  background: var(--card);
  border-color: color-mix(in srgb, var(--accent) 50%, transparent);
  color: var(--accent);
  font-weight: 600;
}

/* 9. Zero-Chrome / Hide UI States */
body.no-hud .top-bar,
body.no-hud .bottom-dock,
body.no-hud .show-ui-pill,
body.ui-hidden .top-bar,
body.ui-hidden .bottom-dock {
  display: none !important;
}

.show-ui-pill {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 25;
  display: none;
}

body.ui-hidden .show-ui-pill {
  display: inline-flex;
}
```

---

## 4. Layout Rules & Anti-Patterns Checklist

- **DO**: Keep top-level controls to **≤ 2 floating pills** at the top (`#topLeftPill`, `#topRightPill`) and at most **1 compact floating pill dock** at the bottom center (`.bottom-dock`).
- **DO**: Group camera framing presets (`HERO`, `WIDE`, `MACRO`) and lighting/environment modes (`DAY`, `LAGOON`, `NIGHT`) into inline `.icon-pill-btn` chips separated by `.pill-divider`.
- **DO**: Hide parameter sliders, debug toggles, and reference/material inspectors inside a toggleable `.popover-card` (`🎛️ TUNE` / `🎯 REF`) so the 3D scene remains uncluttered by default.
- **DO**: Bind key `h` / `H` (and `Escape` for popovers) to toggle `document.body.classList.toggle("ui-hidden")` and support `?hud=0` (`document.body.classList.add("no-hud")`).
- **DON'T**: Never build fixed sidebars, multi-card vertical stacks on the right/left margin, permanent slider walls, or multi-sentence instructional banners over the canvas.
