"""Bearer authentication for the complete inference ASGI application."""
from __future__ import annotations

import secrets
from starlette.responses import JSONResponse


class BearerAuthMiddleware:
    def __init__(self, app, token: str):
        self.app = app
        self.expected = ("Bearer " + token).encode("ascii")

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        values = [value for key, value in scope.get("headers", []) if key.lower() == b"authorization"]
        if len(values) != 1 or not secrets.compare_digest(values[0], self.expected):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                response = JSONResponse({"detail": "Invalid or missing bearer token"}, status_code=401,
                                        headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"})
                await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
