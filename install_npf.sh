#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PYTHON_BIN=${PYTHON:-python3}
VENV_DIR="$PROJECT_DIR/.venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Error: Python 3.10 or newer is required. Set PYTHON to its executable.\n' >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Error: Python 3.10 or newer is required.")
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    printf 'Creating virtual environment: %s\n' "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
printf 'Installing build tools...\n'
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
printf 'Installing Nano Particle Factory and dependencies...\n'
"$VENV_PYTHON" -m pip install --upgrade "$PROJECT_DIR"

"$VENV_PYTHON" - <<'PY'
import ase
import matplotlib
import numpy
import scipy
import spglib
import PySide6
import npf
print("NPF installation verified.")
print(f"  numpy {numpy.__version__}")
print(f"  ASE {ase.__version__}")
print(f"  scipy {scipy.__version__}")
PY

printf '\nInstallation complete.\n'
printf 'Activate with: source %s/bin/activate\n' "$VENV_DIR"
printf 'Start GUI with: %s/bin/npf-gui\n' "$VENV_DIR"
