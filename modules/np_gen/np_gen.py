"""Legacy compatibility entry point."""

from npf.api import *  # noqa: F401,F403
from npf.cli import main


if __name__ == "__main__":
    main()

