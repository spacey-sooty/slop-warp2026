"""Local server for the static site, plus the one thing a browser cannot do.

The site is client-side: it loads `data/event.json`, fetches scouting from the
deployed app, and fits and simulates in the browser. So this mostly just serves
files, and what you see locally is what a static host will serve.

The exception is refreshing from TBA. That needs an API key, and a key must never
ship to a browser, so the pull has to happen somewhere trusted. When this server
is running it exposes exactly one endpoint for it:

    POST /api/refresh-tba   re-pull TBA, rebuild data/event.json, return it

On a static host that endpoint is simply absent, and the page falls back to
re-fetching the published bundle -- which picks up whatever the last
`./sim.py export` deployed. The button is therefore honest in both modes; the UI
labels which one is in play.

There is deliberately still no simulate endpoint. The old one accepted up to
200,000 trials and pinned a core for ~14 seconds per request: an unauthenticated
CPU amplifier. The sampler lives in the browser now, so the endpoint and the
problem are both gone. Refreshing is bounded work -- one TBA pull and one fit --
and this binds to localhost by default.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
BUNDLE_PATH = WEB_DIR / "data" / "event.json"

# Some systems' mime databases still map .js to text/plain, which browsers refuse
# to execute as a module.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/json", ".json")


class RefreshService:
    """Re-pulls TBA and rewrites the bundle. Serialised: the button is cheap to
    spam, and two concurrent pulls would race on the same file."""

    def __init__(self, event_key, csv_path, ridge, scouting_url, out=BUNDLE_PATH):
        self.event_key = event_key
        self.csv_path = csv_path
        self.ridge = ridge
        self.scouting_url = scouting_url
        self.out = Path(out)
        self.lock = threading.Lock()

    def refresh(self) -> dict:
        from .export import bundle as build_bundle
        from .loader import load_event

        with self.lock:
            state = load_event(
                self.event_key, csv_path=self.csv_path, offline=False, refresh=True
            )
            payload = build_bundle(
                state,
                ridge=self.ridge,
                scouting_url=self.scouting_url,
                generated_at=time.time(),
            )
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.out.write_text(json.dumps(payload, separators=(",", ":")))
            return payload


class Handler(SimpleHTTPRequestHandler):
    service: RefreshService | None = None

    def log_message(self, fmt, *args):  # quieter console
        return

    def end_headers(self):
        # The bundle is rewritten while the server is up, and a cached copy would
        # silently show the previous pull.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/api/capabilities":
            # Lets the page label the TBA button accurately on load rather than
            # discovering the endpoint is missing only when it is pressed.
            self._send_json({"refreshTba": self.service is not None})
            return
        super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/refresh-tba":
            self.send_error(404)
            return
        if self.service is None:
            self._send_json({"error": "refresh is not available on this host"}, 404)
            return
        try:
            payload = self.service.refresh()
        except Exception as exc:  # a flaky venue network must not kill the server
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 502)
            return
        self._send_json(payload)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    directory: Path = WEB_DIR,
    service: RefreshService | None = None,
) -> None:
    bound = type("BoundHandler", (Handler,), {"service": service})
    httpd = ThreadingHTTPServer((host, port), partial(bound, directory=str(directory)))
    print(f"  serving {directory} at http://{host}:{port}/  (ctrl-c to stop)")
    if service is not None:
        print("  Refresh TBA will re-pull live; Refresh scouting is always live")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
