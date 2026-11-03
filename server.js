import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || process.env.CORAL_PORT || 8085);
const OUT_DIR = path.join(__dirname, "bench_results");

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".mjs": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".glb": "model/gltf-binary",
  ".gltf": "model/gltf+json",
  ".bin": "application/octet-stream",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".hdr": "application/octet-stream",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function safeName(raw, fallback) {
  const cleaned = String(raw || "").replace(/[^A-Za-z0-9_.-]+/g, "_").slice(0, 120);
  return cleaned || fallback;
}

const server = http.createServer((req, res) => {
  const [rawPath, rawQuery = ""] = (req.url || "/").split("?");
  const urlPath = decodeURIComponent(rawPath || "/");

  if (req.method === "POST" && (urlPath === "/__bench" || urlPath === "/__shot")) {
    const params = new URLSearchParams(rawQuery);
    const ext = urlPath === "/__bench" ? ".json" : ".png";
    const name = safeName(params.get("name"), urlPath === "/__bench" ? "bench" : "shot");
    fs.mkdirSync(OUT_DIR, { recursive: true });
    const target = path.join(OUT_DIR, name + ext);
    const tmp = target + ".tmp";
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      fs.writeFileSync(tmp, Buffer.concat(chunks));
      fs.renameSync(tmp, target);
      res.writeHead(204, { "Access-Control-Allow-Origin": "*" });
      res.end();
    });
    return;
  }

  const relPath = urlPath === "/" ? "/index.html" : urlPath;
  const normalized = path.normalize(relPath).replace(/^(\.\.(\/|\\|$))+/, "");

  // Resolve against repository root first, then ./public
  let filePath = path.join(__dirname, normalized);
  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    filePath = path.join(__dirname, "public", normalized);
  }

  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("404 Not Found");
    return;
  }

  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || "application/octet-stream";
  const cacheControl =
    ext === ".html" || ext === ".js" || ext === ".css"
      ? "no-store"
      : "public, max-age=3600";

  res.writeHead(200, {
    "Content-Type": contentType,
    "Cache-Control": cacheControl,
  });
  fs.createReadStream(filePath).pipe(res);
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`WebGPU Coral Reef Server running at http://127.0.0.1:${PORT}/`);
});
