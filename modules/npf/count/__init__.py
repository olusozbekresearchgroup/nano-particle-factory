"""CN and active-site analysis tools for Nano Particle Factory."""

__all__ = ["build_parser", "main"]


def build_parser():
    from .main import build_parser as _build_parser

    return _build_parser()


def main(argv=None):
    from .main import main as _main

    return _main(argv)

