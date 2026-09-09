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
import os
from pathlib import Path
import plistlib
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

limit = int(subprocess.check_output(
    ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"], text=True).strip())
print(f"iogpu.wired_limit_mb: {limit}", flush=True)
failed = limit <= 0
if failed:
    print("WARNING: GPU wired limit has reset to 0. Run this yourself before benchmarking: "
          "sudo sysctl iogpu.wired_limit_mb=20480", file=sys.stderr)
free = shutil.disk_usage("/System/Volumes/Data").free
print(f"Free disk (/System/Volumes/Data): {free / 1e9:.2f} GB ({free / 2**30:.2f} GiB)")
print("Recorded baseline: 19.48 tok/s (qwen3:8b, 2026-09-06; not re-measured)")

root = Path.cwd()
registration = f"{root}/.venv/bin/orbi --schedule-backups"
service = f"gui/{os.getuid()}/local.orbi.backup"
job = subprocess.run(["/bin/launchctl", "print", service], text=True, capture_output=True)
try:
    path = root / ".session/local.orbi.backup.plist"
    schedule = plistlib.loads(path.read_bytes())
    expected = [str(root / ".venv/bin/python"), str(root / "orbi.py"), "--backup"]
    live = {line.strip() for line in job.stdout.splitlines()}
    required = {f"path = {path}", f"program = {expected[0]}", *expected,
                f"working directory = {root}", f"ORBI_CONFIG => {root / 'orbi.toml'}",
                '"Hour" => 3', '"Minute" => 0'}
    if not (schedule["ProgramArguments"] == expected
            and schedule["StartCalendarInterval"] == {"Hour": 3, "Minute": 0}
            and schedule["WorkingDirectory"] == str(root)
            and schedule["EnvironmentVariables"]["ORBI_CONFIG"] == str(root / "orbi.toml")
            and job.returncode == 0 and required <= live):
        raise ValueError("Backup job does not match the configured schedule")
except (OSError, ValueError, KeyError, TypeError):
    failed = True
    print(f"WARNING: Nightly backup registration is missing or mismatched. Run: {registration}", file=sys.stderr)
else:
    print(f"Nightly backup: verified 03:00 for this login; after login run: {registration}")
raise SystemExit(1 if failed else 0)
PY
