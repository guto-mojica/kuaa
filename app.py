"""FastAPI entrypoint.

Default:
    python app.py

Demo:
    python app.py --config config/demo.yaml
"""
from __future__ import annotations

import argparse
import os

import uvicorn

from api.deps import CONFIG_ENV_VAR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Cinemateca web UI.")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a YAML config override. Also exported as "
            f"{CONFIG_ENV_VAR} for the uvicorn reload process."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", default=8501, type=int, help="Bind port.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.config:
        os.environ[CONFIG_ENV_VAR] = args.config

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
