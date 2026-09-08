"""Real CLI/stream/signal checks; uses isolated data and the installed local models."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import pty
import select
import signal
import sqlite3
import subprocess
import time
import tomllib
import uuid

import orbi
from setup_orbi import verify


def main():
    root = Path(__file__).resolve().parent
    work = root / ".session" / ("cli-validation-" + uuid.uuid4().hex)
    work.mkdir(parents=True)
    project = work / "project"
    project.mkdir()
    other = work / "other"
    other.mkdir()
    database = work / "orbi.db"
    config = tomllib.loads((root / "orbi.toml").read_text())
    source = (root / "orbi.toml").read_text()
    paths = [(value, str((root / value).resolve())) for value in config["paths"].values()]
    paths += [(config["lanes"]["a"]["model"], str(root / config["lanes"]["a"]["model"])),
              (config["memory"]["embedding_model"], str(root / config["memory"]["embedding_model"])),
              (config["runtime"]["server"], str(root / config["runtime"]["server"]))]
    for old, new in paths:
        if old == config["paths"]["db_path"]:
            new = str(database)
        elif old == config["paths"]["backup_dir"]:
            new = str(work / "backups")
        source = source.replace(json.dumps(old), json.dumps(new))
    test_config = work / "orbi.toml"
    test_config.write_text(source)
    assert test_config.read_text() == source
    tomllib.loads(source)
    environment = dict(os.environ, ORBI_CONFIG=str(test_config),
        http_proxy="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9", no_proxy="", NO_PROXY="")
    executable = str(root / ".venv/bin/orbi")
    checks = []

    def run(*args, text="", cwd=project):
        return subprocess.run([executable, *args], input=text, text=True, capture_output=True,
                              cwd=cwd, env=environment, timeout=120)

    def passed(name, condition, evidence=""):
        assert condition, f"{name}: {evidence}"
        checks.append(name)
        print("PASS:", name, flush=True)

    result = run("Reply with exactly: AMBER_FOX")
    passed("prompt streams a real reply", result.returncode == 0 and "AMBER_FOX" in result.stdout, result)
    passed("piped output contains no orbs or ANSI", not result.stderr and
           not any(x in result.stdout for x in [*orbi.ORBS.values(), "\x1b"]), result)
    result = run("--continue", "Repeat exactly your previous answer.")
    passed("continue resumes stored conversation", result.returncode == 0 and "AMBER_FOX" in result.stdout, result)
    result = run(text="Reply with exactly: PIPED_OK")
    passed("piped input works", result.returncode == 0 and "PIPED_OK" in result.stdout, result)
    result = run("Remember this personal fact globally: My favorite tea is rooibos.")
    with sqlite3.connect(database) as db:
        saved = db.execute("SELECT scope,project FROM memories WHERE text LIKE '%rooibos%' AND tier<>'L0'").fetchall()
    passed("real remember tool stores a global fact", result.returncode == 0 and saved == [("global", None)], result)
    result = run("Use the recall tool with scope global to find my favorite tea, then name it.", cwd=other)
    with sqlite3.connect(database) as db:
        recalled = db.execute("SELECT m.payload FROM orbi_messages m JOIN orbi_sessions s ON m.session=s.id WHERE s.project=?",
                              (str(other),)).fetchall()
    passed("real recall tool reads global memory from another project", result.returncode == 0 and "rooibos" in result.stdout.lower()
           and any(c["function"]["name"] == "recall" for row in recalled
                   for c in json.loads(row[0]).get("tool_calls", [])), result)
    master, slave = pty.openpty()
    process = subprocess.Popen([executable], stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               cwd=project, env=environment)
    os.close(slave)
    try:
        os.write(master, b"Reply with exactly: INTERACTIVE_OK\n\x04")
        output, errors = process.communicate(timeout=90)
        passed("interactive prompts stay out of piped stdout", process.returncode == 0
               and b"INTERACTIVE_OK" in output and b"orbi>" not in output and b"orbi>" in errors,
               (process.returncode, output, errors))
    finally:
        os.close(master)
        if process.poll() is None:
            process.kill()
            process.wait()
    empty_project = work / "empty-project"
    empty_project.mkdir()
    result = run("--continue", "hello", cwd=empty_project)
    passed("continue cannot cross project scope", result.returncode != 0 and "No previous session" in result.stderr, result)
    result = run("word " * 6000)
    passed("oversized context fails nonzero", result.returncode != 0 and "exceeds" in result.stderr, result)

    def interrupted(sig):
        process = subprocess.Popen([executable, "Count from 1 to 1000, one integer per line. Do not skip numbers."],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=project, env=environment)
        try:
            ready, _, _ = select.select([process.stdout], [], [], 90)
            assert ready and process.poll() is None, "Generation did not start"
            first = process.stdout.read(1)
            assert first
            # Let periodic persistence run before testing an uncatchable crash.
            if sig == signal.SIGKILL:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    with sqlite3.connect(database) as db:
                        rows = db.execute("SELECT m.payload FROM orbi_messages m JOIN orbi_tasks t ON m.task=t.id WHERE t.pid=?",
                                          (process.pid,)).fetchall()
                    if any(json.loads(r[0]).get("role") == "assistant" and json.loads(r[0]).get("content") for r in rows):
                        break
                    time.sleep(.05)
                else:
                    raise AssertionError("Stream was not persisted before crash")
            process.send_signal(sig)
            output, errors = process.communicate(timeout=20)
            return process.pid, process.returncode, first + output, errors
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    pid, code, output, errors = interrupted(signal.SIGINT)
    with sqlite3.connect(database) as db:
        outcome = db.execute("SELECT outcome FROM orbi_tasks WHERE pid=?", (pid,)).fetchone()[0]
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    passed("Ctrl-C preserves state and exits130", code == 130 and outcome == "cancelled" and integrity == "ok",
           (code, outcome, errors))
    pid, code, output, errors = interrupted(signal.SIGKILL)
    passed("crash test killed only its CLI process", code == -signal.SIGKILL and bool(output), (code, errors))
    result = run("--continue", "Reply with exactly: RECOVERED")
    with sqlite3.connect(database) as db:
        row = db.execute("SELECT state,outcome FROM orbi_tasks WHERE pid=?", (pid,)).fetchone()
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    passed("crash state survives and resumes", result.returncode == 0 and "RECOVERED" in result.stdout and
           row == ("stalled", "crashed") and integrity == "ok", (row, result))

    class TTY(io.StringIO):
        def isatty(self):
            return True

    stdout, stderr = TTY(), TTY()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        task = orbi.Task(database, "orb-self-check")
        try:
            for state in orbi.ORBS:
                task.set(state)
                with sqlite3.connect(database) as db:
                    assert db.execute("SELECT state FROM orbi_tasks WHERE id=?", (task.id,)).fetchone()[0] == state
            task.set("thinking")
            task.last_activity -= 31
            deadline = time.monotonic() + 3
            while task.state != "stalled" and time.monotonic() < deadline:
                time.sleep(.05)
            assert task.state == "stalled"
        finally:
            task.finish("done")
    passed("all orb states persist and stall monitoring works", all(x in stderr.getvalue() for x in orbi.ORBS.values()))

    artifact = work / "artifact"
    artifact.write_bytes(b"verified")
    verify(artifact, 8, hashlib.sha256(b"verified").hexdigest())
    try:
        verify(artifact, 8, "0" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupt artifact accepted")
    passed("artifact corruption is rejected", True)
    result = dict(passed=True, checks=checks, database=str(database))
    target = root / ".session/cli-results.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    assert json.loads(target.read_text()) == result


if __name__ == "__main__":
    main()
