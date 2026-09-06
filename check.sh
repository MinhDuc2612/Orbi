#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"

if [ ! -f .venv/pyvenv.cfg ] || [ ! -x .venv/bin/python ]; then
    printf '%s\n' 'ERROR: .venv is missing or incomplete.' >&2
    exit 1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python - <<'PY'
import platform
import shutil
import subprocess
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"ERROR: Python 3.12.x required; got {platform.python_version()}")
print(f"Python: {platform.python_version()}", flush=True)

if sys.prefix == sys.base_prefix:
    raise SystemExit("ERROR: .venv/bin/python is not running in a virtual environment.")
import mlx.core as mx

device = mx.default_device()
if device != mx.gpu:
    raise SystemExit(f"ERROR: MLX must use a GPU; got {device}")
print(f".venv: OK; MLX: {device}", flush=True)

subprocess.run(
    ["/usr/sbin/sysctl", "iogpu.wired_limit_mb"],
    check=True,
)
free = shutil.disk_usage("/System/Volumes/Data").free
print(f"Free disk (/System/Volumes/Data): {free / 1e9:.2f} GB ({free / 2**30:.2f} GiB)")
print("Recorded baseline: 19.48 tok/s (qwen3:8b, 2026-09-06; not re-measured)")
PY
