#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_DEV=0
INSTALL_ALIGNMENT=0

for argument in "$@"; do
  case "$argument" in
    --dev) INSTALL_DEV=1 ;;
    --alignment) INSTALL_ALIGNMENT=1 ;;
    *)
      printf 'error: unknown option: %s\n' "$argument" >&2
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'error: Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -c 'import sys; raise SystemExit("Python 3.10 or newer is required") if sys.version_info < (3, 10) else None'
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

if [[ "$INSTALL_DEV" -eq 1 ]]; then
  "$VENV_DIR/bin/python" -m pip install -r requirements-dev.txt
fi

if [[ "$INSTALL_ALIGNMENT" -eq 1 ]]; then
  "$VENV_DIR/bin/python" -m pip install -r requirements-alignment.txt
fi

printf '\nEnvironment ready in %s.\n' "$VENV_DIR"
printf 'Run: %s/bin/python -m pytest -q\n' "$VENV_DIR"
