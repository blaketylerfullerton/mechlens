"""Pair this GPU with a Mechlens Cloud workspace and serve it by dialing out.

The GPU never accepts a connection from the internet. It long-polls the cloud
("any requests for me?"), runs each request against this process's own API on
loopback, and posts the answer back. Outbound HTTPS is all it needs, so it
works on boxes where tunnels are blocked or get the machine killed (managed
JupyterHub, strict neoclouds, corporate networks). HTTPS_PROXY is honored.

Ownership is proved by the activation code: the cloud issues one, this process
prints it, and whoever is signed in to the workspace types it into the web UI.

`--tunnel-url` is the escape hatch for people who already have a public URL
for this API: the cloud then calls that URL directly and the relay is not used.
"""
from __future__ import annotations

import base64
import threading
from urllib.parse import urlsplit

import httpx

RELAY_WORKERS = 4  # so one large download does not hold up everything else
STATUS_SECONDS = 10
POLL_TIMEOUT = httpx.Timeout(30, read=45)  # the cloud holds a poll open for 25s
MAX_BACKOFF = 30
STATUS_LINES = {
    "loading": "Model loading... (the first run downloads the weights; this can take a few minutes)",
    "ready": "Model ready. Your workspace can run experiments now.",
    "error": "Model failed to load: {detail}",
}


class CloudError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Unpaired(Exception):
    """The cloud no longer knows this GPU's session."""


def normalize_cloud_url(value: str) -> str:
    url = urlsplit(value)
    if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment:
        raise ValueError("--cloud must be an HTTP(S) URL such as https://mechlens.example.com")
    return value.rstrip("/")


class CloudLink:
    """Owns the pairing handshake, the relay workers and the status reports."""

    def __init__(self, cloud_url: str, port: int, gpu_token: str, tunnel_url: str | None = None,
                 host: str = "127.0.0.1"):
        self.cloud_url = cloud_url
        self.gpu_token = gpu_token
        self.tunnel_url = tunnel_url
        self.local_url = f"http://{f'[{host}]' if ':' in host else host}:{port}"
        self.session_token: str | None = None
        self.code: str | None = None
        # trust_env: a box that only reaches the internet through HTTPS_PROXY
        # still works. The loopback client must never go through a proxy.
        self.cloud = httpx.Client(timeout=POLL_TIMEOUT, trust_env=True, follow_redirects=False)
        self.local = httpx.Client(timeout=httpx.Timeout(120, connect=5), trust_env=False)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._offline = False

    # -- startup, on the main thread so a failure aborts before the model loads --

    def start(self) -> None:
        pairing = self._call("/api/pairing/start", {})
        self._poll_token = pairing["poll_token"]
        self.code = pairing["code"]
        minutes = int(pairing.get("expires_in", 900)) // 60
        print("", flush=True)
        print("Connect this GPU:", flush=True)
        print(f"  Open {pairing['activate_url']}", flush=True)
        print(f"  Enter code: {pairing['code']}", flush=True)
        print(f"  (the code expires in {minutes} minutes)", flush=True)
        print("", flush=True)

    # -- the rest runs on a daemon thread while uvicorn serves ---------------

    def run(self) -> None:
        try:
            workspace = self._await_activation()
            print(f'Code accepted. Linked to workspace "{workspace}".', flush=True)
            registration = {"gpu_token": self.gpu_token}
            if self.tunnel_url:
                registration["tunnel_url"] = self.tunnel_url
            self._call("/api/gpu/register", registration, self.session_token)
        except CloudError as exc:
            print(f"Cloud pairing failed: {exc}", flush=True)
            return
        if self.tunnel_url:
            print(f"Connected. The cloud reaches this GPU at {self.tunnel_url}.", flush=True)
        else:
            for n in range(RELAY_WORKERS):
                threading.Thread(target=self._relay_worker, name=f"cloud-relay-{n}", daemon=True).start()
            print("Connected to the cloud. No tunnel needed; this GPU dials out.", flush=True)
        print("Keep this process running while you use your workspace.", flush=True)
        self._report_status()

    def _await_activation(self) -> str:
        while not self._stop.is_set():
            try:
                result = self._call("/api/pairing/poll", {"poll_token": self._poll_token})
            except CloudError as exc:
                # A network blip or a cloud redeploy must not throw away the
                # code the user may be typing in right now. Only an answer
                # from the cloud (the code expired) ends the wait.
                if exc.status is not None and exc.status < 500:
                    raise
                self._went_offline(exc)
                self._stop.wait(5)
                continue
            self._online()
            if result.get("status") == "connected":
                self.session_token = result["session_token"]
                return result["workspace"]["name"]
            self._stop.wait(2)
        raise CloudError("shutting down")

    # -- relay: take a request from the cloud, run it here, send the answer --

    def _relay_worker(self) -> None:
        backoff = 0
        while not self._stop.is_set():
            try:
                response = self.cloud.post(self.cloud_url + "/api/gpu/relay/next", headers=self._auth())
                if response.status_code == 401:
                    raise Unpaired()
                response.raise_for_status()
                self._online()
                backoff = 0
                if response.status_code == 200:
                    self._forward(response.json())
            except Unpaired:
                self._unpaired()
                return
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                backoff = min(MAX_BACKOFF, backoff * 2 or 1)
                self._went_offline(exc)
                self._stop.wait(backoff)

    def _forward(self, item: dict) -> None:
        """Run one relayed request against this process and upload the answer."""
        path = item["path"]
        respond = self.cloud_url + f"/api/gpu/relay/{item['id']}/response"
        # Only ever a path on this server, never a URL somewhere else.
        if not path.startswith("/") or path.startswith("//"):
            self._upload(respond, 400, "application/json", [b'{"detail": "Bad relay path"}'])
            return
        body = base64.b64decode(item["body_b64"])
        request = self.local.build_request(item["method"], self.local_url + path, content=body or None,
            headers={"Authorization": "Bearer " + self.gpu_token, "Content-Type": "application/json"})
        try:
            local = self.local.send(request, stream=True)
        except httpx.HTTPError as exc:
            self._upload(respond, 502, "application/json", [b'{"detail": "GPU server did not answer: %s"}' % type(exc).__name__.encode()])
            return
        try:
            self._upload(respond, local.status_code, local.headers.get("content-type", "application/json"),
                         local.iter_bytes(64 * 1024))
        finally:
            local.close()

    def _upload(self, url: str, status: int, content_type: str, chunks) -> None:
        response = self.cloud.post(url, content=chunks, timeout=httpx.Timeout(120, connect=30), headers={
            **self._auth(), "X-Relay-Status": str(status), "X-Relay-Content-Type": content_type,
            "Content-Type": "application/octet-stream"})
        if response.status_code == 401:
            raise Unpaired()
        # 404: the browser already gave up on this request. Nothing to do.

    # -- status: what the Compute page and this terminal show ----------------

    def _report_status(self) -> None:
        last = None
        while True:
            state, detail = self._model_status()
            if (state, detail) != last:
                print(STATUS_LINES[state].format(detail=detail), flush=True)
                last = (state, detail)
            try:
                self._call("/api/gpu/heartbeat", {"state": state, "detail": detail}, self.session_token)
                if self.tunnel_url:
                    self._online()  # with the relay, the workers track this
            except CloudError as exc:
                if exc.status == 401:
                    self._unpaired()
                    return
                if self.tunnel_url:
                    self._went_offline(exc)
            if self._stop.wait(STATUS_SECONDS):
                return

    def _model_status(self) -> tuple[str, str | None]:
        try:
            health = self.local.get(self.local_url + "/health", headers=self._auth(self.gpu_token), timeout=5).json()
            if health.get("status") in STATUS_LINES:
                return health["status"], health.get("detail")
        except (httpx.HTTPError, ValueError):
            pass
        return "loading", None  # the API is still binding its port

    # -- helpers -----------------------------------------------------------

    def _auth(self, token: str | None = None) -> dict:
        return {"Authorization": "Bearer " + (token or self.session_token or "")}

    def _call(self, path: str, payload: dict, token: str | None = None) -> dict:
        url = self.cloud_url + path
        try:
            response = self.cloud.post(url, json=payload, timeout=30,
                                       headers=self._auth(token) if token else {})
        except httpx.HTTPError as exc:
            raise CloudError(f"cannot reach {url}: {exc}") from None
        if not response.is_success:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            raise CloudError(detail if isinstance(detail, str) else f"{response.status_code} from {url}",
                             response.status_code)
        try:
            return response.json()
        except ValueError:
            raise CloudError(f"unexpected answer from {url}") from None

    def _went_offline(self, exc: Exception) -> None:
        with self._lock:
            if self._offline:
                return
            self._offline = True
        print(f"Lost connection to the cloud ({exc}); retrying...", flush=True)

    def _online(self) -> None:
        with self._lock:
            was_offline, self._offline = self._offline, False
        if was_offline:
            print("Reconnected to the cloud.", flush=True)

    def _unpaired(self) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            self._stop.set()
        print("The cloud disconnected this GPU (another GPU was paired to the workspace, or it was removed). "
              "Restart mechlens serve --cloud to pair again.", flush=True)

    def close(self) -> None:
        self._stop.set()
        if self.session_token:
            try:
                self.cloud.post(self.cloud_url + "/api/gpu/disconnect", json={}, headers=self._auth(), timeout=5)
            except httpx.HTTPError:
                pass
        self.cloud.close()
        self.local.close()
