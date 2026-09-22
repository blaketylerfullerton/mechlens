"""Lightweight installed entry point; help never imports Torch or the model."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mechlens", description="Inspect and steer language models on your GPU.")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="start the inference API on this machine")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-dir", type=Path, help="labels, training and circuits directory (or MECHLENS_DATA_DIR)")
    serve.add_argument("--token-file", type=Path, help="read bearer token from a file; alternatively set MECHLENS_API_TOKEN")
    for command in ("trace", "enrich", "show", "experiment"):
        sub.add_parser(command, add_help=False, help=f"run the existing {command} CLI")
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"trace", "enrich", "show", "experiment"}:
        from .cli import build_parser as legacy_parser
        args = legacy_parser().parse_args(argv)
        args.func(args)
        return
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    token = os.environ.get("MECHLENS_API_TOKEN", "")
    if args.token_file:
        try:
            token = args.token_file.read_text().strip()
        except OSError as exc:
            parser.error(f"cannot read token file: {exc}")
        if not token:
            parser.error("token file is empty")
    if token and (not token.isascii() or any(c.isspace() for c in token)):
        parser.error("API token must contain ASCII characters without whitespace")
    if args.host not in {"127.0.0.1", "::1", "localhost"} and not token:
        parser.error("non-loopback binding requires --token-file or MECHLENS_API_TOKEN")
    if token:
        os.environ["MECHLENS_API_TOKEN"] = token
    # Preserve existing repo assets for editable installs; installed wheels use
    # a writable user directory instead of writing into site-packages.
    repo_data = Path(__file__).resolve().parents[1] / "data"
    default_data = repo_data if repo_data.is_dir() else Path.home() / ".local/share/mechlens"
    data_dir = (args.data_dir or Path(os.environ.get("MECHLENS_DATA_DIR", default_data))).expanduser().resolve()
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        parser.error(f"cannot create data directory: {exc}")
    os.environ["MECHLENS_DATA_DIR"] = str(data_dir)
    host = f"[{args.host}]" if ":" in args.host else args.host
    print(f"Mechlens inference API: http://{host}:{args.port}", flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    print(f"Authentication: {'bearer token required' if token else 'local access without a token'}", flush=True)
    print("Model readiness: GET /health. Stop with Ctrl+C. Cloud pairing is not available yet.", flush=True)
    import uvicorn
    uvicorn.run("app.service.app:app", host=args.host, port=args.port, workers=1, proxy_headers=False)


if __name__ == "__main__":
    main()
