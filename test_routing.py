"""Routing persistence controls; --live measures the frozen 20 cases on installed Lane A."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import Mock, patch
import uuid

import orbi
import routing

FROZEN_SHA = "fdcf669576169038916ba421e097ba9fee3854aae01873c5b6e6f25287e3e86d"


def fixtures():
    raw = (orbi.ROOT / "bench_cases.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA
    cases = json.loads(raw)
    assert routing.ROUTING_POLICY == cases["routing_policy"]
    assert routing.SKILL_TAXONOMY == cases["skill_taxonomy"]
    return cases


def new_session(path, project):
    session = uuid.uuid4().hex
    with orbi.database(path) as db:
        db.execute("INSERT INTO orbi_sessions VALUES(?,?,?)", (session, project, time.time()))
    return session


def latest(path, session):
    with orbi.database(path) as db:
        row = db.execute("SELECT r.* FROM orbi_routes r JOIN orbi_tasks t ON t.id=r.task "
                         "WHERE t.session=? ORDER BY r.created DESC LIMIT 1", (session,)).fetchone()
    result = dict(row)
    result["decision"] = json.loads(result["decision"]) if result["decision"] else None
    return result


def controls():
    fixtures()
    from skill_catalog import BY_ID
    settings = dict(lanes=dict(a=dict(model=Path("installed-gemma.gguf"))))
    for category, leaf_id, expected_lane, expected_model in (
            ("embedding", "d07.s01.l01", "B", "microsoft/Harrier-oss-v1-27B"),
            ("model3d_generation", "d05.s03.l01", "B", "tencent/Hunyuan3D-2.1"),
            ("formal_reasoning", "d01.s01.l06", "B", "Goedel-Prover-V2-32B"),
            ("general_assistance", "d01.s01.l02", "A", "installed-gemma.gguf")):
        leaf = BY_ID[leaf_id]
        coarse = dict(skill=category, lane=routing.SKILL_TAXONOMY[category]["lane"])
        responses = [(coarse, {}), (dict(group=leaf["subdomain"]), {}),
                     (dict(skill=leaf_id, reason="Requested operation matches this leaf"), {})]
        for forced, lane, model in ((None, expected_lane, expected_model),
                                    ("A", "A", "installed-gemma.gguf"),
                                    ("C", "C", "moonshotai/Kimi-K3")):
            with patch.object(routing, "classify", side_effect=responses):
                decision = routing.decide(settings, "An independent control request", forced_lane=forced)
            assert decision["lane"] == lane and decision["model"] == model
            assert decision["preferred_model"] == leaf["model"]
    valid = dict(choices=[dict(finish_reason="stop", message=dict(content='{"group":"known"}'))])
    for response in (valid, dict(choices=[]),
                     dict(choices=[dict(finish_reason="length")]),
                     dict(choices=[dict(finish_reason="stop", message=dict(content='{"group":"unknown"}'))])):
        with patch.object(orbi, "json_request", side_effect=[dict(prompt="template"), dict(tokens=[1]), response]):
            try:
                routing.classify(dict(runtime=dict(port=8123, context_size=4096)), "policy", "request",
                                 {"group": {"type": "string", "enum": ["known"]}})
                assert response is valid
            except ValueError:
                assert response is not valid
    (orbi.ROOT / ".session").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=orbi.ROOT / ".session") as temporary:
        path = Path(temporary) / "routing.sqlite3"
        orbi.initialize(path)
        project = str(Path(temporary).resolve())
        config = dict(paths=dict(db_path=path))
        memory = Mock()
        memory.retrieve.return_value = dict(text="", items=[])
        for lane, mode, code, status, success in (
                ("A", "auto", 0, "done", 1), ("B", "auto", 3, "not_installed", None),
                ("C", "auto", 3, "not_installed", None), ("C", "job", 0, "deferred_not_installed", None)):
            session = new_session(path, project)
            choice = dict(skill="test-leaf", lane=lane, model="test-model", reason=["Test decision"])
            with patch.object(routing, "decide", return_value=choice), \
                    patch.object(orbi, "ensure_runtime") as runtime, \
                    patch.object(orbi, "fit_messages", return_value=[]), \
                    patch.object(orbi, "stream_reply", return_value=dict(role="assistant", content="Done")) as stream, \
                    redirect_stdout(io.StringIO()) as output:
                assert orbi.run_turn(config, memory, session, project, "test request", route_mode=mode) == code
            row = latest(path, session)
            assert row["prompt"] == "test request" and row["decision"] == choice
            assert row["status"] == status and row["succeeded"] == success
            assert stream.call_count == (1 if lane == "A" else 0)
            if lane != "A":
                assert f"(Lane {lane}) — not installed" in output.getvalue()
                assert runtime.call_count == 1  # Classifier only; no specialist runtime path.
        for failure, status in ((ValueError("invalid classification"), "error"), (KeyboardInterrupt(), "cancelled")):
            session = new_session(path, project)
            with patch.object(routing, "decide", side_effect=failure), patch.object(orbi, "ensure_runtime"):
                try:
                    orbi.run_turn(config, memory, session, project, "fail request", route_mode="auto")
                    raise AssertionError("Failure was swallowed")
                except type(failure):
                    pass
            row = latest(path, session)
            assert row["status"] == status and row["succeeded"] == 0 and row["error"]
        session = new_session(path, project)
        with patch.object(routing, "decide", return_value=dict(skill="test-leaf", lane="A", model="test-model")), \
                patch.object(orbi, "ensure_runtime"), patch.object(orbi, "fit_messages", return_value=[]), \
                patch.object(orbi, "stream_reply", side_effect=KeyboardInterrupt()):
            try:
                orbi.run_turn(config, memory, session, project, "cancel generation", route_mode="auto")
                raise AssertionError("Cancellation was swallowed")
            except KeyboardInterrupt:
                pass
        row = latest(path, session)
        assert row["lane"] == "A" and row["status"] == "cancelled" and row["succeeded"] == 0
        # Crash recovery finalizes in-flight decisions; deferred jobs remain pending.
        session = new_session(path, project)
        task = orbi.Task(path, session)
        task.finished.set()
        task.watcher.join()
        with orbi.database(path) as db:
            db.execute("UPDATE orbi_tasks SET owner_start='dead-owner' WHERE id=?", (task.id,))
            db.execute("INSERT INTO orbi_routes(task,prompt,kind,status,created,updated) VALUES(?,?,?,?,?,?)",
                       (task.id, "crash", "ask", "classifying", time.time(), time.time()))
        orbi.initialize(path)
        assert latest(path, session)["status"] == "crashed"
        assert latest(path, session)["succeeded"] == 0
        with patch.object(orbi, "settings", return_value=config), \
                patch.object(sys, "argv", ["orbi", "ask", "--decision", task.id]), \
                patch.object(Path, "cwd", return_value=Path(project)), redirect_stdout(io.StringIO()) as output:
            assert orbi.main() == 0
        assert json.loads(output.getvalue())["status"] == "crashed"
        with patch.object(orbi, "settings", return_value=config), \
                patch.object(sys, "argv", ["orbi", "ask", "--decision", task.id]):
            assert orbi.main() == 1  # Another project's decision is not exposed.
        backup_memory = orbi.Memory(path)
        snapshot = backup_memory.backup(Path(temporary) / "backups")
        with orbi.database(path) as db:
            before = [tuple(row) for row in db.execute("SELECT * FROM orbi_routes ORDER BY task")]
            db.execute("DELETE FROM orbi_routes")
            assert not db.execute("SELECT 1 FROM orbi_routes").fetchone()
        backup_memory.restore(snapshot)
        with orbi.database(path) as db:
            assert [tuple(row) for row in db.execute("SELECT * FROM orbi_routes ORDER BY task")] == before
    fixtures()
    print("PASS: routing persistence, unavailable lanes, deferred jobs, failures, cancellation, crash recovery and scoped inspection.")


def live():
    cases = fixtures()
    wired = subprocess.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"], text=True).strip()
    if wired != "20480":
        raise RuntimeError(f"Benchmark blocked: iogpu.wired_limit_mb={wired}; expected 20480")
    directory = orbi.ROOT / ".session" / ("routing-" + uuid.uuid4().hex)
    directory.mkdir()
    assert directory.is_dir()
    config = orbi.settings()
    config["paths"]["db_path"] = directory / "routing.sqlite3"
    orbi.initialize(config["paths"]["db_path"])
    project = str(directory)
    memory = orbi.Memory(config["paths"]["db_path"], orbi.url(config, True) + "/v1/embeddings", **{
        key: config["memory"][key] for key in ("max_items", "max_chars", "max_ms")})
    result = dict(fixture_sha256=FROZEN_SHA, wired_limit_mb=int(wired),
                  runtime_model=str(config["lanes"]["a"]["model"]), run_dir=str(directory), cases=[], score=0)
    try:
        for case in cases["routing"]:
            session = new_session(config["paths"]["db_path"], project)
            row = dict(id=case["id"], expected=case["expected"], correct=False)
            try:
                with redirect_stdout(io.StringIO()) as output:
                    row["exit_code"] = orbi.run_turn(config, memory, session, project, case["prompt"], route_mode="auto")
                row["output"] = output.getvalue()
                row["log"] = latest(config["paths"]["db_path"], session)
                row["lane_correct"] = row["log"]["lane"] == case["expected"]["lane"]
                row["category_correct"] = row["log"]["decision"]["category"] == case["expected"]["skill"]
                row["correct"] = row["lane_correct"] and row["category_correct"]
                assert row["exit_code"] == (0 if row["log"]["lane"] == "A" else 3)
                assert row["log"]["succeeded"] == (1 if row["log"]["lane"] == "A" else None)
            except Exception as error:
                row["error"] = repr(error)
                row["correct"] = False
            result["cases"].append(row)
            result["score"] = sum(r["correct"] for r in result["cases"])
            result["lane_score"] = sum(r.get("lane_correct", False) for r in result["cases"])
            result["category_score"] = sum(r.get("category_correct", False) for r in result["cases"])
            orbi.atomic_json(directory / "results.json", result)
            print(f"{case['id']}: lane={'PASS' if row['correct'] else 'FAIL'} {row.get('error', '')}", flush=True)
    finally:
        orbi.ensure_runtime(config, stop=True)
        fixtures()
        result["fixture_unchanged"] = True
        orbi.atomic_json(directory / "results.json", result)
        orbi.atomic_json(orbi.ROOT / ".session/routing-results.json", result)
    print(json.dumps(dict(score=result["score"], total=20, evidence=str(directory / "results.json"))))
    return 0 if result["score"] >= 18 else 1


if __name__ == "__main__":
    raise SystemExit(live() if sys.argv[1:] == ["--live"] else controls())
