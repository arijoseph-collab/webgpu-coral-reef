#!/usr/bin/env python3
"""Robust static server for the WebGPU coral reef viewer.

- Serves repository root (index.html, app.js, style.css, js/*) and ./public/* on 127.0.0.1:8085.
- Accepts POST /__bench?name=<label> (JSON) and POST /__shot?name=<label> (PNG)
  so the in-page benchmark harness can hand results back to tools/bench.mjs.
"""
import http.server
import os
import re
import sys
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_ROOT = os.path.join(HERE, "public")
OUT_DIR = os.path.join(HERE, "bench_results")
LOG_PATH = os.path.join(HERE, "serve.log")
PORT = int(os.environ.get("CORAL_PORT", "8085"))

os.makedirs(OUT_DIR, exist_ok=True)


def _safe_name(raw, default):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw or "")[:120]
    return name or default


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".glb": "model/gltf-binary",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".css": "text/css",
        ".wasm": "application/wasm",
        ".bin": "application/octet-stream",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def translate_path(self, path):
        raw = urllib.parse.unquote(path.split("?", 1)[0].split("#", 1)[0])
        rel = raw.lstrip("/") or "index.html"
        rel = os.path.normpath(rel)
        if rel.startswith(".."):
            rel = "index.html"
        root_candidate = os.path.join(HERE, rel)
        if os.path.isfile(root_candidate):
            return root_candidate
        pub_candidate = os.path.join(PUBLIC_ROOT, rel)
        if os.path.isfile(pub_candidate):
            return pub_candidate
        return root_candidate

    def log_message(self, fmt, *args):
        try:
            with open(LOG_PATH, "a") as f:
                f.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))
        except Exception:
            pass

    def end_headers(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html") or path.endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        path, _, query = self.path.partition("?")
        params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        if path == "/__bench":
            name = _safe_name(params.get("name"), "bench")
            target = os.path.join(OUT_DIR, name + ".json")
        elif path == "/__shot":
            name = _safe_name(params.get("name"), "shot")
            target = os.path.join(OUT_DIR, name + ".png")
        else:
            self.send_response(404)
            self.end_headers()
            return
        tmp = target + ".tmp"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, target)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    print("Serving %s on http://127.0.0.1:%d/" % (HERE, PORT), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
