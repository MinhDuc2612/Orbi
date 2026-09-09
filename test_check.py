"""Check health failures without changing sysctl or launchd registration."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import plistlib
import subprocess
from unittest.mock import patch


def main():
    root = Path(__file__).resolve().parent
    body = (root / "check.sh").read_text().split(".venv/bin/python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    schedule = dict(ProgramArguments=[str(root / ".venv/bin/python"), str(root / "orbi.py"), "--backup"],
        StartCalendarInterval={"Hour": 3, "Minute": 0}, WorkingDirectory=str(root),
        EnvironmentVariables={"ORBI_CONFIG": str(root / "orbi.toml")})
    live = "\n".join([f"path = {root / '.session/local.orbi.backup.plist'}",
        f"program = {schedule['ProgramArguments'][0]}", *schedule["ProgramArguments"],
        f"working directory = {root}", f"ORBI_CONFIG => {root / 'orbi.toml'}",
        '"Hour" => 3', '"Minute" => 0'])

    def run(limit, registered=True, text=live, optimize=0):
        output, errors = io.StringIO(), io.StringIO()
        result = subprocess.CompletedProcess([], 0 if registered else 113, text, "")
        with patch("subprocess.check_output", return_value=str(limit)), \
                patch("subprocess.run", return_value=result), \
                patch.object(Path, "read_bytes", return_value=plistlib.dumps(schedule)), \
                patch.object(Path, "cwd", return_value=root), \
                redirect_stdout(output), redirect_stderr(errors):
            try:
                exec(compile(body, "check.sh", "exec", optimize=optimize), {})
            except SystemExit as exit:
                return exit.code, output.getvalue(), errors.getvalue()
        raise AssertionError("Health check did not set an exit status")

    for optimize in (0, 1):
        code, output, errors = run(20480, optimize=optimize)
        assert code == 0 and "Nightly backup: verified" in output and not errors
        code, output, errors = run(0, optimize=optimize)
        assert code != 0 and "wired limit has reset to 0" in errors
        assert "sudo sysctl iogpu.wired_limit_mb=20480" in errors
        code, output, errors = run(20480, registered=False, optimize=optimize)
        assert code != 0 and "Nightly backup: verified" not in output
        assert f"{root}/.venv/bin/orbi --schedule-backups" in errors
        for changed in (live.replace('"Hour" => 3', '"Hour" => 30'),
                        live.replace(str(root / "orbi.py"), "/unrelated/orbi.py")):
            code, output, errors = run(20480, text=changed, optimize=optimize)
            assert code != 0 and "Nightly backup: verified" not in output
    print("PASS: wired-limit reset and missing/mismatched backup jobs fail, including optimized Python; no system settings changed.")


if __name__ == "__main__":
    main()
