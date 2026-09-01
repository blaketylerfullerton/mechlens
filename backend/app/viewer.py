"""A static file server for `viewer/`, and nothing more.

This is deliberately *not* phase 6. There is no job queue, no model in the
process, no /trace or /steer — those belong in the FastAPI layer and are the
whole point of doing them later. All this does is put `backend/` behind
http:// so the viewer can `fetch` a trace, because browsers refuse
cross-origin reads on file:// and a static page otherwise cannot open one.

The viewer works without it: drag a trace onto the page and it renders. This
just saves the dragging and adds one convenience the drop zone cannot, a list
of what is on disk:

    GET /api/traces -> [{name, path, tokens, passes, size_mb}]

If this file ever grows a POST, it has become phase 6 and should move.
"""

from __future__ import annotations

import json
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .store import DEFAULT_TRACE_DIR, SIDECAR_SUFFIX

BACKEND_DIR = Path(__file__).resolve().parents[1]
VIEWER_DIR = BACKEND_DIR / "viewer"
DEFAULT_PORT = 8765


def trace_index(trace_dir: Path, root: Path) -> list[dict]:
    """Summarise every trace on disk, for the viewer's picker.

    Each file is parsed in full — a few MB times a handful of traces, once per
    page load. Cheap enough that a partial parse is not worth the fragility of
    reading a JSON document with a regex.
    """
    out = []
    for path in sorted(trace_dir.glob("*.json")):
        if path.name.endswith(SIDECAR_SUFFIX):
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # a half-written trace should not break the listing
        out.append(
            {
                "name": path.stem,
                # Served path, relative to the document root, so the browser can
                # fetch it directly.
                "path": "/" + path.resolve().relative_to(root).as_posix(),
                "tokens": len(doc.get("steps", [])),
                "passes": [p["name"] for p in doc.get("passes", [])],
                "size_mb": round(path.stat().st_size / 1e6, 2),
            }
        )
    # Richest traces first: a trace with the lens on it is the one worth opening.
    out.sort(key=lambda t: (-len(t["passes"]), t["name"]))
    return out


class Handler(SimpleHTTPRequestHandler):
    trace_dir: Path = DEFAULT_TRACE_DIR
    root: Path = BACKEND_DIR

    def do_GET(self) -> None:  # noqa: N802 — http.server's naming
        if self.path.split("?")[0] == "/api/traces":
            body = json.dumps(trace_index(self.trace_dir, self.root)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            # Traces change under the server as passes are re-run.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self) -> None:
        if self.path.endswith(".json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        return  # one line per asset is noise; failures still raise


def serve(
    port: int = DEFAULT_PORT,
    trace: str | None = None,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    open_browser: bool = True,
) -> None:
    if not VIEWER_DIR.is_dir():
        raise SystemExit(f"no viewer at {VIEWER_DIR}")

    Handler.trace_dir = Path(trace_dir).resolve()
    Handler.root = BACKEND_DIR
    handler = partial(Handler, directory=str(BACKEND_DIR))

    url = f"http://localhost:{port}/viewer/"
    if trace:
        path = Path(trace).resolve()
        if not path.is_file():
            raise SystemExit(f"no such trace: {trace}")
        try:
            url += "?trace=/" + path.relative_to(BACKEND_DIR).as_posix()
        except ValueError:
            raise SystemExit(
                f"{trace} is outside {BACKEND_DIR}; the server only serves that tree"
            ) from None

    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        found = trace_index(Handler.trace_dir, Handler.root)
        print(f"serving {BACKEND_DIR} at {url}")
        print(f"  {len(found)} trace(s): " + ", ".join(t["name"] for t in found))
        print("  ctrl-c to stop")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
