"""The GPU half of the cloud relay, against a fake cloud and a fake local API."""
import base64
import json

import httpx

from app import cloud_link
from app.cloud_link import CloudLink


def make_link(cloud_handler, local_handler, tunnel_url=None):
    link = CloudLink("https://cloud.example", 8000, "gpu-secret", tunnel_url)
    link.session_token = "session"
    link.cloud = httpx.Client(transport=httpx.MockTransport(cloud_handler))
    link.local = httpx.Client(transport=httpx.MockTransport(local_handler))
    return link


def relayed(path, method="GET", body=b""):
    return {"id": "r1", "method": method, "path": path, "body_b64": base64.b64encode(body).decode()}


def test_forward_runs_the_request_locally_and_uploads_the_answer():
    local_seen, uploads = [], []

    def local(request):
        local_seen.append(request)
        return httpx.Response(202, json={"job_id": "j"})

    def cloud(request):
        uploads.append((request.url.path, dict(request.headers), request.read()))
        return httpx.Response(200, json={"ok": True})

    link = make_link(cloud, local)
    link._forward(relayed("/trace?x=1", "POST", b'{"prompt": "hi"}'))

    sent = local_seen[0]
    assert sent.url == "http://127.0.0.1:8000/trace?x=1"
    assert sent.method == "POST" and sent.read() == b'{"prompt": "hi"}'
    # The local API gets the GPU's own token, never whatever the cloud sent.
    assert sent.headers["authorization"] == "Bearer gpu-secret"
    path, headers, body = uploads[0]
    assert path == "/api/gpu/relay/r1/response"
    assert headers["authorization"] == "Bearer session"
    assert headers["x-relay-status"] == "202"
    assert json.loads(body) == {"job_id": "j"}


def test_forward_refuses_anything_but_a_local_path():
    uploads = []
    link = make_link(lambda r: uploads.append(r) or httpx.Response(200),
                     lambda r: (_ for _ in ()).throw(AssertionError("must not be called")))
    for path in ("https://evil.example/", "//evil.example/x", "health"):
        link._forward(relayed(path))
    assert [r.headers["x-relay-status"] for r in uploads] == ["400"] * 3


def test_forward_reports_a_dead_local_server_as_502():
    uploads = []

    def local(request):
        raise httpx.ConnectError("refused")

    link = make_link(lambda r: uploads.append(r) or httpx.Response(200), local)
    link._forward(relayed("/health"))
    assert uploads[0].headers["x-relay-status"] == "502"


def test_worker_serves_polls_then_stops_when_unpaired(capsys):
    polls = iter([
        httpx.Response(204),
        httpx.Response(200, json=relayed("/health")),
        httpx.Response(401),
    ])
    answered = []

    def cloud(request):
        if request.url.path.endswith("/next"):
            return next(polls)
        answered.append(request.read())
        return httpx.Response(200)

    link = make_link(cloud, lambda r: httpx.Response(200, json={"status": "ready"}))
    link._relay_worker()  # returns on the 401
    assert answered == [b'{"status":"ready"}']
    assert link._stop.is_set()
    assert "Restart mechlens serve --cloud" in capsys.readouterr().out


def test_worker_backs_off_and_announces_the_outage_once(capsys, monkeypatch):
    calls = []

    def cloud(request):
        calls.append(request)
        if len(calls) <= 3:
            raise httpx.ConnectError("network down")
        return httpx.Response(401)

    link = make_link(cloud, lambda r: httpx.Response(200))
    waits = []
    monkeypatch.setattr(link._stop, "wait", lambda seconds: waits.append(seconds) or False)
    link._relay_worker()
    assert waits == [1, 2, 4]
    out = capsys.readouterr().out
    assert out.count("Lost connection to the cloud") == 1


def test_status_is_reported_and_printed_when_it_changes(capsys, monkeypatch):
    health = iter([{"status": "loading"}, {"status": "loading"}, {"status": "ready"},
                   {"status": "error", "detail": "CUDA out of memory"}])
    beats = []

    def cloud(request):
        beats.append(json.loads(request.read()))
        return httpx.Response(200, json={"ok": True})

    link = make_link(cloud, lambda r: httpx.Response(200, json=next(health)))
    ticks = iter([False, False, False, True])
    monkeypatch.setattr(link._stop, "wait", lambda seconds: next(ticks))
    link._report_status()
    assert [b["state"] for b in beats] == ["loading", "loading", "ready", "error"]
    assert beats[-1]["detail"] == "CUDA out of memory"
    out = capsys.readouterr().out
    assert out.count("Model loading") == 1
    assert "Model ready" in out and "Model failed to load: CUDA out of memory" in out


def test_tunnel_mode_registers_the_url_and_starts_no_relay(monkeypatch):
    calls = []

    def cloud(request):
        calls.append((request.url.path, json.loads(request.read() or b"{}")))
        if request.url.path == "/api/pairing/poll":
            return httpx.Response(200, json={"status": "connected", "session_token": "s", "workspace": {"name": "W"}})
        return httpx.Response(200, json={"ok": True})

    link = make_link(cloud, lambda r: httpx.Response(200, json={"status": "ready"}), tunnel_url="https://mine.example")
    link._poll_token = "p"
    monkeypatch.setattr(link, "_report_status", lambda: None)
    started = []
    monkeypatch.setattr(cloud_link.threading, "Thread", lambda **kw: started.append(kw))
    link.run()
    assert ("/api/gpu/register", {"gpu_token": "gpu-secret", "tunnel_url": "https://mine.example"}) in calls
    assert started == []


def test_relay_mode_registers_without_a_url(monkeypatch):
    calls = []

    def cloud(request):
        calls.append((request.url.path, json.loads(request.read() or b"{}")))
        if request.url.path == "/api/pairing/poll":
            return httpx.Response(200, json={"status": "connected", "session_token": "s", "workspace": {"name": "W"}})
        return httpx.Response(200, json={"ok": True})

    link = make_link(cloud, lambda r: httpx.Response(200))
    link._poll_token = "p"
    monkeypatch.setattr(link, "_report_status", lambda: None)

    class FakeThread:
        started = 0
        def __init__(self, **kw): pass
        def start(self): FakeThread.started += 1
    monkeypatch.setattr(cloud_link.threading, "Thread", FakeThread)
    link.run()
    assert ("/api/gpu/register", {"gpu_token": "gpu-secret"}) in calls
    assert FakeThread.started == cloud_link.RELAY_WORKERS


def test_pairing_survives_a_network_blip_but_not_an_expired_code(monkeypatch):
    answers = iter([
        httpx.ConnectError("cloud redeploying"),
        httpx.Response(503, json={"detail": "starting"}),
        httpx.Response(200, json={"status": "pending"}),
        httpx.Response(200, json={"status": "connected", "session_token": "s", "workspace": {"name": "W"}}),
    ])

    def cloud(request):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    link = make_link(cloud, lambda r: httpx.Response(200))
    link._poll_token = "p"
    monkeypatch.setattr(link._stop, "wait", lambda seconds: False)
    assert link._await_activation() == "W"

    expired = make_link(lambda r: httpx.Response(404, json={"detail": "This pairing code expired"}), lambda r: httpx.Response(200))
    expired._poll_token = "p"
    try:
        expired._await_activation()
    except cloud_link.CloudError as exc:
        assert "expired" in str(exc)
    else:
        raise AssertionError("an expired code must end the wait")
