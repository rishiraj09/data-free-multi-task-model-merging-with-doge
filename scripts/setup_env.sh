#!/usr/bin/env bash
# Create or resume the project environment (local machine or cloud GPU).
# Usage: bash scripts/setup_env.sh [venv_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${1:-$ROOT/.venv}"
DOGE_COMMIT=ca0a0a10c45bc0965f52f0815d6a6581ee103f85

command -v uv >/dev/null || pip install uv

if [ ! -d "$ROOT/third_party/DOGE/.git" ]; then
  git clone https://github.com/WalkerWorldPeace/DOGE.git "$ROOT/third_party/DOGE"
fi
git -C "$ROOT/third_party/DOGE" fetch --depth 1 origin "$DOGE_COMMIT" 2>/dev/null || true
git -C "$ROOT/third_party/DOGE" checkout -q "$DOGE_COMMIT"

# Preserve installed packages when resuming an interrupted setup.
uv venv --allow-existing --python 3.10 "$VENV"
# requirements.lock is resolved on macOS arm64; on Linux/CUDA use the top-level pins.
if [ "$(uname -s)" = "Darwin" ]; then
  uv pip install --python "$VENV/bin/python" -r "$ROOT/requirements.lock"
else
  uv pip install --python "$VENV/bin/python" -r "$ROOT/requirements.txt"
fi
uv pip install --python "$VENV/bin/python" --no-deps -e "$ROOT/third_party/DOGE" -e "$ROOT"

if [ -f "$ROOT/tests/test_environment.py" ]; then
  "$VENV/bin/python" -m pytest -q -p no:warnings "$ROOT/tests/test_environment.py"
else
  echo "Local test suite is not included; skipping environment tests."
fi
"$VENV/bin/python" "$ROOT/scripts/record_env.py"
