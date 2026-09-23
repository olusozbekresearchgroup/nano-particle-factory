"""Compatibility wrapper for the nanoparticle generator."""

from .api import *  # noqa: F401,F403
from .cli import main


if __name__ == "__main__":
    main()
