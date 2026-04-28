"""CLI entrypoint for the Anon Tokyo web visualizer."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Sequence

import uvicorn


SRC_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Anon Tokyo web visualizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8766)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--reload", action="store_true")
    return parser


def _start_frontend(frontend_port: int, backend_port: int, host: str) -> subprocess.Popen | None:
    frontend_dir = SRC_ROOT / "visualize" / "web" / "frontend"
    env = os.environ.copy()
    env.pop("VITE_BACKEND_URL", None)
    env["VITE_BACKEND_PORT"] = str(backend_port)
    env["VITE_FRONTEND_PORT"] = str(frontend_port)
    try:
        return subprocess.Popen(["pnpm", "dev", "--host", "0.0.0.0", "--port", str(frontend_port)], cwd=frontend_dir, env=env)
    except FileNotFoundError:
        print("pnpm not found. Install pnpm or run the backend only with uvicorn.")
    return None


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    host = "0.0.0.0" if args.public else args.host

    frontend_proc = _start_frontend(args.frontend_port, args.backend_port, "127.0.0.1")
    print(
        {
            "backend": f"http://{host}:{args.backend_port}",
            "frontend": f"http://{host}:{args.frontend_port}",
        }
    )
    try:
        uvicorn.run(
            "anon_tokyo.visualize.web.backend.app:app",
            host=host,
            port=args.backend_port,
            reload=args.reload,
        )
    finally:
        if frontend_proc is not None:
            frontend_proc.terminate()


if __name__ == "__main__":
    main()
