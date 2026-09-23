# Copyright (c) 2026 Olus Ozbek Research Group
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import argparse

from .qt import run_qt_gui
from .server import run_gui


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Start the NPF GUI.")
    p.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    p.add_argument("--port", type=int, default=8765, help="Port to bind.")
    p.add_argument(
        "--backend",
        choices=["auto", "qt", "web"],
        default="auto",
        help="GUI backend to use. 'auto' prefers PySide6 and falls back to the web UI.",
    )
    p.add_argument("--no-browser", action="store_true", help="Do not automatically open a browser window when using the web backend.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.backend == "web":
        run_gui(host=args.host, port=args.port, open_browser=not args.no_browser)
        return
    if args.backend == "qt":
        run_qt_gui(host=args.host, port=args.port)
        return
    try:
        run_qt_gui(host=args.host, port=args.port)
    except RuntimeError as exc:
        if "PySide6" not in str(exc):
            raise
        run_gui(host=args.host, port=args.port, open_browser=not args.no_browser)
