#!/usr/bin/env node
// Headless-Chrome WebGPU benchmark runner for the coral reef viewer (no npm deps; Node >= 22).
//
// Usage:
//   node tools/bench.mjs [--out dir] [--w 1470] [--h 830] [--dpr 2] [--page /] [--timeout 150]
//                        label::query [label2::query2 ...]
// Example:
//   node tools/bench.mjs --out bench_results/base base_hero::mood=day&cam=wide
//
// Each run launches a fresh Chrome profile (cold shader cache), opens
//   http://127.0.0.1:8085<page>?bench=<label>&<query>
// waits for window.__BENCH_RESULT__, saves <label>.json, <label>.png (CDP screenshot) and
// <label>.log (console output), then prints a summary table.
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const BASE = process.env.CORAL_URL || "http://127.0.0.1:8085";
const DEBUG_PORT = Number(process.env.BENCH_PORT || 9333); // parallel runs: BENCH_PORT=9334 ...

const argv = process.argv.slice(2);
const opts = { out: "bench_results/latest", w: 1470, h: 830, dpr: 2, page: "/", timeout: 150, shot: 1 };
const runs = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a.startsWith("--")) {
    opts[a.slice(2)] = argv[++i];
  } else {
    const [label, query = ""] = a.split("::");
    runs.push({ label, query });
  }
}
for (const k of ["w", "h", "dpr", "timeout", "shot"]) opts[k] = Number(opts[k]);
fs.mkdirSync(opts.out, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.listeners = [];
    ws.onmessage = (ev) => {
      const msg = JSON.parse(typeof ev.data === "string" ? ev.data : ev.data.toString());
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      } else if (msg.method) {
        for (const fn of this.listeners) fn(msg);
      }
    };
    // A crashed/closed browser must not leave commands pending forever
    ws.onclose = () => {
      for (const { reject } of this.pending.values()) reject(new Error("DevTools connection closed"));
      this.pending.clear();
    };
  }
  send(method, params = {}, timeoutMs = 60000) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error(`CDP ${method} timed out`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (v) => (clearTimeout(timer), resolve(v)),
        reject: (e) => (clearTimeout(timer), reject(e)),
      });
    });
  }
  on(fn) {
    this.listeners.push(fn);
  }
}

async function runOne({ label, query }) {
  const profile = `/tmp/coral-bench-profile-${DEBUG_PORT}`;
  for (let i = 0; i < 5; i++) {
    try {
      fs.rmSync(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 });
      break;
    } catch (e) {
      await sleep(400); // Chrome from the previous run may still be shutting down (ENOTEMPTY race)
    }
  }
  const args = [
    "--headless=new",
    `--remote-debugging-port=${DEBUG_PORT}`,
    `--user-data-dir=${profile}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--disable-component-update",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--enable-unsafe-webgpu",
    "--ignore-gpu-blocklist",
    "--enable-dawn-features=allow_unsafe_apis",
    "--disable-dawn-features=timestamp_quantization",
    `--window-size=${opts.w},${opts.h}`,
    `--force-device-scale-factor=${opts.dpr}`,
    "about:blank",
  ];
  const proc = spawn(CHROME, args, { stdio: ["ignore", "ignore", "pipe"] });
  let stderr = "";
  proc.stderr.on("data", (d) => (stderr += d));
  const logs = [];
  let result = null;
  try {
    let page = null;
    for (let i = 0; i < 150 && !page; i++) {
      try {
        const list = await (await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`)).json();
        page = list.find((t) => t.type === "page");
      } catch {
        /* devtools not up yet */
      }
      if (!page) await sleep(100);
    }
    if (!page) throw new Error("DevTools page target not found");
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((res, rej) => {
      ws.onopen = res;
      ws.onerror = rej;
    });
    const cdp = new CDP(ws);
    cdp.on((msg) => {
      if (msg.method === "Runtime.consoleAPICalled") {
        logs.push(`[${msg.params.type}] ` + msg.params.args.map((a) => a.value ?? a.description ?? "").join(" "));
      } else if (msg.method === "Runtime.exceptionThrown") {
        const d = msg.params.exceptionDetails;
        logs.push("[exception] " + ((d.exception && d.exception.description) || d.text));
      } else if (msg.method === "Log.entryAdded") {
        logs.push(`[log:${msg.params.entry.level}] ${msg.params.entry.text}`);
      }
    });
    await cdp.send("Runtime.enable");
    await cdp.send("Log.enable");
    await cdp.send("Page.enable");
    await cdp.send("Emulation.setDeviceMetricsOverride", {
      width: opts.w,
      height: opts.h,
      deviceScaleFactor: opts.dpr,
      mobile: false,
    });
    // Live mode (query contains live=<seconds>): load the normal interactive page (no bench harness),
    // record requestAnimationFrame intervals + the on-page Performance HUD for N seconds.
    const liveSec = Number(new URLSearchParams(query).get("live") || 0);
    const q = liveSec ? query : `bench=${encodeURIComponent(label)}${query ? "&" + query : ""}`;
    const url = `${BASE}${opts.page}${opts.page.includes("?") ? "&" : "?"}${q}`;
    const t0 = Date.now();
    await cdp.send("Page.navigate", { url });
    const evaluate = async (expression) => (await cdp.send("Runtime.evaluate", { expression, returnByValue: true })).result.value;
    if (liveSec) {
      while (Date.now() - t0 < opts.timeout * 1000) {
        await sleep(300);
        if (await evaluate("!!document.querySelector('#loading-overlay.hidden')")) break;
      }
      const ready = Date.now();
      await evaluate(`(() => {
        window.__live = { frames: [], hud: [] };
        let last = performance.now();
        const t0 = last;
        const tick = (t) => { window.__live.frames.push(t - last); last = t; requestAnimationFrame(tick); };
        requestAnimationFrame(tick);
        setInterval(() => window.__live.hud.push(((performance.now() - t0) / 1000).toFixed(0) + 's ' + ['perf-fps', 'perf-gpu', 'perf-scale'].map((id) => document.getElementById(id).textContent).join(' | ')), 1000);
      })()`);
      // Optional scripted UI actions: actions=<sec>:click:<elementId>,<sec>:click:<elementId>...
      const actions = (new URLSearchParams(query).get("actions") || "")
        .split(",")
        .filter(Boolean)
        .map((a) => a.split(":"))
        .map(([sec, verb, id]) => ({ at: Number(sec) * 1000, verb, id }))
        .sort((a, b) => a.at - b.at);
      const tLive = Date.now();
      for (const act of actions) {
        const wait = act.at - (Date.now() - tLive);
        if (wait > 0) await sleep(wait);
        if (act.verb === "click") await evaluate(`document.getElementById(${JSON.stringify(act.id)}).click()`);
      }
      const remaining = liveSec * 1000 - (Date.now() - tLive);
      if (remaining > 0) await sleep(remaining);
      const live = JSON.parse(await evaluate("JSON.stringify(window.__live)"));
      const fr = live.frames.slice(5).sort((a, b) => a - b);
      const q = (p) => Math.round(fr[Math.min(fr.length - 1, Math.floor(p * (fr.length - 1)))] * 100) / 100;
      result = {
        label,
        query,
        live: true,
        loadToReadyMs: ready - t0,
        frames: fr.length,
        fps: Math.round((fr.length / (fr.reduce((a, b) => a + b, 0) / 1000)) * 10) / 10,
        frameMs: { p10: q(0.1), p50: q(0.5), p90: q(0.9), p99: q(0.99), max: fr[fr.length - 1] },
        longFrames: fr.filter((v) => v > 25).length,
        hud: live.hud,
      };
    }
    while (!liveSec && Date.now() - t0 < opts.timeout * 1000) {
      await sleep(400);
      const r = await cdp.send("Runtime.evaluate", {
        expression: "window.__BENCH_RESULT__ ? JSON.stringify(window.__BENCH_RESULT__) : null",
        returnByValue: true,
      });
      if (r.result && r.result.value) {
        result = JSON.parse(r.result.value);
        break;
      }
    }
    if (!result) result = { label, error: `timeout after ${opts.timeout}s` };
    result.wallSec = Math.round((Date.now() - t0) / 100) / 10;
    if (opts.shot) {
      await sleep(250);
      const s = await cdp.send("Page.captureScreenshot", { format: "png" }, 20000);
      fs.writeFileSync(path.join(opts.out, `${label}.png`), Buffer.from(s.data, "base64"));
    }
    try {
      await cdp.send("Browser.close", {}, 3000);
    } catch {
      /* ignore */
    }
  } catch (e) {
    result = { label, error: String(e && e.stack ? e.stack : e) };
  } finally {
    await sleep(300);
    try {
      proc.kill("SIGKILL");
    } catch {
      /* ignore */
    }
  }
  fs.writeFileSync(path.join(opts.out, `${label}.json`), JSON.stringify(result, null, 2));
  fs.writeFileSync(path.join(opts.out, `${label}.log`), logs.join("\n") + "\n--- chrome stderr (tail) ---\n" + stderr.split("\n").slice(-25).join("\n"));
  return result;
}

const rows = [];
const fmt = (v, w = 7) => String(v ?? "-").padStart(w);
for (const run of runs) {
  const r = await runOne(run);
  rows.push(r);
  if (r.live) {
    console.log(
      `${r.label} | LIVE ready ${r.loadToReadyMs} ms | ${r.frames} frames | fps ${r.fps} | frame p50 ${r.frameMs.p50} p90 ${r.frameMs.p90} p99 ${r.frameMs.p99} max ${Math.round(r.frameMs.max)} | >25ms: ${r.longFrames}\n   HUD: ${(opts.hudAll ? r.hud : r.hud.slice(-3)).join("\n        ")}`
    );
    continue;
  }
  console.log(
    `${r.label} | ${r.drawingBuffer ? r.drawingBuffer.join("x") : "-"} | load ${fmt(r.reefLoadMs, 5)} | ttff ${fmt(r.timeToFirstFrameMs, 5)} | wall ${r.wallSec}s` +
      (r.error ? ` | ERROR: ${r.error.slice(0, 300)}` : "")
  );
  const variants = r.variants || [];
  const base = variants[0];
  for (const v of variants) {
    const f = v.frameMs || {};
    const t = v.throughputMs || {};
    const rel = base && base.throughputMs && t.p50 ? (base.throughputMs.p50 / t.p50).toFixed(2) + "x" : "-";
    const relLat = base && base.frameMs && f.p50 ? (base.frameMs.p50 / f.p50).toFixed(2) + "x" : "-";
    console.log(
      `   ${v.spec.padEnd(34)} frame p10 ${fmt(f.p10)} p50 ${fmt(f.p50)} p90 ${fmt(f.p90)} | thru p50 ${fmt(t.p50)} min ${fmt(t.min)} | cpu ${fmt(v.cpuMs && v.cpuMs.p50, 6)} | speedup thru ${rel.padStart(6)} lat ${relLat.padStart(6)} | compile ${fmt(v.compileMs, 5)}`
    );
  }
}
fs.writeFileSync(path.join(opts.out, "_summary.json"), JSON.stringify(rows, null, 2));
// Chrome children / CDP sockets can keep the event loop alive after the summary is written
process.exit(0);
