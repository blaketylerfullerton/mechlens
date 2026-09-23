"""Pair this GPU with a Mechlens Cloud workspace over an anonymous tunnel.

The GPU never authenticates to Cloudflare: `cloudflared tunnel --url` opens a
quick tunnel with no account and no credentials on this machine. The tunnel's
hostname is random and is never shown to a person — browsers talk only to the
cloud app, which proxies to it server-side — so a throwaway URL is sufficient.

Ownership is proved by the activation code: the cloud issues one, this process
prints it, and whoever is signed in to the workspace types it into the web UI.
"""
from __future__ import annotations

import io
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

TUNNEL_URL = re.compile(rb"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")
HEARTBEAT_SECONDS = 30
CLOUDFLARED_RELEASES = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
CLOUDFLARED_PATH = Path.home() / ".local/bin/cloudflared"
INSTALL_HINT = (
    "Could not download cloudflared automatically ({reason}).\n"
    "  Install it yourself: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/\n"
    "No Cloudflare account or login is needed.\n"
    "Alternatively pass --tunnel-url with a public HTTPS URL your GPU host already provides."
)


class CloudError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _post(url: str, payload: dict, token: str | None = None, timeout: int = 30) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + token} if token else {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", "replace")
        try:
            detail = json.loads(body).get("detail")
        except ValueError:
            detail = None
        raise CloudError(detail or f"{exc.code} from {url}", exc.code) from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise CloudError(f"cannot reach {url}: {exc}") from None


def find_or_install_cloudflared() -> str:
    """Return a cloudflared path, downloading the official binary to ~/.local/bin on first use."""
    found = shutil.which("cloudflared") or (str(CLOUDFLARED_PATH) if os.access(CLOUDFLARED_PATH, os.X_OK) else None)
    if found:
        return found
    system = platform.system().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    if system not in {"linux", "darwin"} or not arch:
        raise CloudError(INSTALL_HINT.format(reason=f"no prebuilt binary for {platform.system()} {platform.machine()}"))
    asset = f"cloudflared-{system}-{arch}" + (".tgz" if system == "darwin" else "")
    print(f"cloudflared not found; downloading {asset} to {CLOUDFLARED_PATH} (one time)...", flush=True)
    try:
        with urllib.request.urlopen(CLOUDFLARED_RELEASES + asset, timeout=120) as response:
            data = response.read()
        if system == "darwin":
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                data = archive.extractfile("cloudflared").read()
    except (urllib.error.URLError, TimeoutError, tarfile.TarError, KeyError, AttributeError) as exc:
        raise CloudError(INSTALL_HINT.format(reason=exc)) from None
    CLOUDFLARED_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = CLOUDFLARED_PATH.with_suffix(".partial")
    partial.write_bytes(data)
    partial.chmod(0o755)
    partial.replace(CLOUDFLARED_PATH)
    return str(CLOUDFLARED_PATH)


def normalize_cloud_url(value: str) -> str:
    url = urlsplit(value)
    if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
        raise ValueError("--cloud must be an HTTP(S) URL such as https://mechlens.example.com")
    return value.rstrip("/")


class CloudLink:
    """Owns the pairing handshake, the tunnel subprocess and the heartbeat."""

    def __init__(self, cloud_url: str, port: int, gpu_token: str, tunnel_url: str | None = None):
        self.cloud_url = cloud_url
        self.port = port
        self.gpu_token = gpu_token
        # An operator-supplied tunnel (ngrok, Tailscale Funnel, or the public
        # proxy URL rented GPU hosts already hand out) replaces cloudflared.
        self.managed_tunnel = tunnel_url is None
        self.session_token: str | None = None
        self.tunnel_url: str | None = tunnel_url
        self.code: str | None = None
        self._process: subprocess.Popen | None = None
        self._cloudflared: str | None = None
        self._found = threading.Event()
        self._stop = threading.Event()

    # -- startup, on the main thread so failures abort before the model loads --

    def start(self) -> None:
        if self.managed_tunnel:
            self._cloudflared = find_or_install_cloudflared()
        pairing = _post(f"{self.cloud_url}/api/pairing/start", {})
        self._poll_token = pairing["poll_token"]
        self.code = pairing["code"]
        minutes = int(pairing.get("expires_in", 900)) // 60
        print("", flush=True)
        print("Connect this GPU:", flush=True)
        print(f"  Open {pairing['activate_url']}", flush=True)
        print(f"  Enter code: {pairing['code']}", flush=True)
        print(f"  (the code expires in {minutes} minutes)", flush=True)
        print("", flush=True)
        self._start_tunnel()

    def _start_tunnel(self) -> None:
        if not self.managed_tunnel:
            print(f"Using the tunnel you supplied: {self.tunnel_url}", flush=True)
            self._found.set()
            return
        self._process = subprocess.Popen(
            [self._cloudflared, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        threading.Thread(target=self._read_tunnel, name="tunnel-reader", daemon=True).start()

    def _read_tunnel(self) -> None:
        assert self._process and self._process.stderr
        for line in self._process.stderr:
            if not self.tunnel_url:
                match = TUNNEL_URL.search(line)
                if match:
                    self.tunnel_url = match.group().decode()
                    self._found.set()
        # cloudflared exited; unblock anyone waiting rather than hang forever.
        self._found.set()

    # -- the rest runs on a daemon thread while uvicorn serves ---------------

    def run(self) -> None:
        try:
            workspace = self._await_activation()
            if not self._await_tunnel():
                return
            _post(f"{self.cloud_url}/api/gpu/register",
                  {"tunnel_url": self.tunnel_url, "gpu_token": self.gpu_token}, self.session_token)
        except CloudError as exc:
            print(f"Cloud pairing failed: {exc}", flush=True)
            return
        print("Connected.", flush=True)
        print(f"Your workspace: {self.cloud_url} ({workspace})", flush=True)
        print("Keep this process running while using your workspace.", flush=True)
        self._heartbeat()

    def _await_activation(self) -> str:
        while not self._stop.is_set():
            result = _post(f"{self.cloud_url}/api/pairing/poll", {"poll_token": self._poll_token})
            if result.get("status") == "connected":
                self.session_token = result["session_token"]
                return result["workspace"]["name"]
            self._stop.wait(2)
        raise CloudError("shutting down")

    def _await_tunnel(self) -> bool:
        self._found.wait(90)
        if not self.tunnel_url:
            print("Cloud pairing failed: cloudflared did not report a tunnel URL.", flush=True)
            return False
        # cloudflared prints the hostname before Cloudflare's edge routes to it.
        # Registering early would hand the cloud a URL that answers 502, so
        # probe until a request actually reaches this process.
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not self._stop.is_set():
            if self._probe():
                return True
            self._stop.wait(2)
        print(f"Cloud pairing failed: {self.tunnel_url} never answered /health with this run's "
              f"token, so the cloud was not told this GPU is ready.", flush=True)
        if self.managed_tunnel:
            print("  trycloudflare.com quick tunnels sometimes fail to route. Try again, or pass "
                  "--tunnel-url with your own tunnel or your GPU host's public URL.", flush=True)
        return False

    def _probe(self) -> bool:
        """True once /health answers through the tunnel with our own token.

        Nothing weaker works: while a quick tunnel is still being registered
        Cloudflare's edge returns 404, not a 5xx, so any "not a server error"
        test would pass before a single request reached this process.
        /health answers 200 whether or not the model has finished loading.
        """
        request = urllib.request.Request(
            f"{self.tunnel_url}/health", headers={"Authorization": "Bearer " + self.gpu_token})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError):
            return False

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                _post(f"{self.cloud_url}/api/gpu/heartbeat", {}, self.session_token)
            except CloudError as exc:
                # A restarted cloud or a dropped network should not kill the
                # inference server; keep retrying until the process is stopped.
                print(f"Cloud heartbeat failed ({exc}); retrying.", flush=True)

    def close(self) -> None:
        self._stop.set()
        if self.session_token:
            try:
                _post(f"{self.cloud_url}/api/gpu/disconnect", {}, self.session_token, timeout=5)
            except CloudError:
                pass
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
