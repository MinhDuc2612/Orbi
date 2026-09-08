"""Deterministic unit tests with synthetic vectors; never a model recall score."""

from contextlib import closing
import io
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np

import memory as m


def synthetic_vector(url, text, timeout):
    vector = np.zeros(1024, dtype="<f4")
    topics = [int(n) for n in re.findall(r"topic(\d+)", text)]
    if "lunar dinner preference" in text:
        topics = [4]
    for topic in topics or [0]:
        vector[topic] = 1
    return vector / np.linalg.norm(vector)


def rejects(action, kind=ValueError):
    try:
        action()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def checked_retrieve(memory, *args, **kwargs):
    started = time.monotonic()
    result = memory.retrieve(*args, **kwargs)
    elapsed = (time.monotonic() - started) * 1000
    assert elapsed <= 300, (elapsed, result)
    assert result["elapsed_ms"] <= 300
    assert len(result["items"]) <= 12 and len(result["text"]) <= 4000
    return result


def main():
    session = Path(__file__).resolve().parent / ".session"
    session.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="memory-unit-", dir=session) as temporary:
        root = Path(temporary)
        a, b = root / "project-a", root / "project-b"
        a.mkdir()
        b.mkdir()
        with patch.object(m, "_fetch_embedding", side_effect=synthetic_vector):
            memory = m.Memory(root / "unit.sqlite3", max_items=999, max_chars=99999, max_ms=999)
            assert (memory.max_items, memory.max_chars, memory.max_ms) == (12, 4000, 300)
            with closing(memory._connect()) as db:
                assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
                assert db.execute("PRAGMA busy_timeout").fetchone()[0] > 0
            global_id = memory.add("topic1 I prefer green tea.", scope="global", tier="L3")
            a_id = memory.add("topic2 Arcturus private billing context.", scope="project", project=a, tier="L2")
            b_id = memory.add("topic2 Betelgeuse private billing context.", scope="project", project=b, tier="L2")
            result = checked_retrieve(memory, "topic1 topic2", project=a)
            assert {r["id"] for r in result["items"]} == {global_id, a_id}
            assert "Betelgeuse" not in result["text"] and b_id not in {r["id"] for r in result["items"]}
            assert {r["id"] for r in checked_retrieve(memory, "topic1 topic2", scope="global", project=a)["items"]} == {global_id}
            assert {r["id"] for r in checked_retrieve(memory, "topic1 topic2", scope="project", project=a)["items"]} == {a_id}
            assert {r["id"] for r in checked_retrieve(memory, "topic1 topic2")["items"]} == {global_id}
            alias = root / "project-alias"
            alias.symlink_to(a, target_is_directory=True)
            assert {r["id"] for r in checked_retrieve(memory, "topic2", project=alias)["items"]} == {a_id}
            for options in (dict(scope="global", project=a), dict(scope="project"),
                            dict(scope="other"), dict(scope="global", tier="L4")):
                rejects(lambda options=options: memory.add("topic1 invalid", **options))
            rejects(lambda: memory.retrieve("topic1", scope="project"))
            rejects(lambda: m.Memory(root / "bad.sqlite3", embedding_url="http://example.com/v1/embeddings"))
            semantic_id = memory.add("topic4 Pasta is my favourite meal.", scope="global")
            result = checked_retrieve(memory, "lunar dinner preference")
            assert result["semantic_used"] and semantic_id in {r["id"] for r in result["items"]}
            tagged_id = memory.add("topic8 release", scope="global", tags=["launchpad"])
            assert tagged_id in {r["id"] for r in checked_retrieve(memory, "launchpad")["items"]}
            original = "topic9 " + "界" * 6000
            long_id = memory.add(original, scope="global")
            result = checked_retrieve(memory, "topic9")
            assert result["truncated"] and result["items"][0]["id"] == long_id
            assert result["items"][0]["embedding_truncated"] == 1
            with closing(memory._connect()) as db:
                assert db.execute("SELECT text FROM memories WHERE id=?", (long_id,)).fetchone()[0] == original
            for index in range(20):
                memory.add(f"topic10 entry {index}", scope="global")
            result = checked_retrieve(memory, "topic10")
            assert len(result["items"]) == 12 and result["truncated"]
            for tier in ("L0", "L1", "L2", "L3"):
                memory.add(f"topic11 {tier} context", scope="global", tier=tier)
            result = checked_retrieve(memory, "topic11")
            assert {r["tier"] for r in result["items"]} == {"L0", "L1", "L2", "L3"}
            assert {r["tier"] for r in result["items"][:2]} == {"L2", "L3"}
            with closing(sqlite3.connect(memory.db_path, isolation_level=None)) as locked:
                locked.execute("PRAGMA locking_mode=EXCLUSIVE")
                locked.execute("BEGIN EXCLUSIVE")
                result = checked_retrieve(memory, "topic1")
                assert result["truncated"] and "locked" in result["error"]
                locked.rollback()

            backups = root / "backups"
            snapshot = memory.backup(backups)
            assert snapshot.is_file() and original in snapshot.with_suffix(".md").read_text()
            namespace = re.match(r"orbi-memory-[0-9a-f]{16}-", snapshot.name)[0]
            old = backups / (namespace + "20000101T000000-" + "a" * 32 + ".sqlite3")
            old.write_bytes(snapshot.read_bytes())
            old.with_suffix(".md").write_text("old own snapshot")
            unrelated = backups / "unrelated.sqlite3"
            unrelated.write_text("keep me")
            memory.backup(backups)
            assert not old.exists() and not old.with_suffix(".md").exists()
            assert unrelated.read_text() == "keep me"
            later = memory.add("topic12 created after snapshot", scope="global")
            inode = memory.db_path.stat().st_ino
            with closing(sqlite3.connect(memory.db_path)) as peer:
                assert peer.execute("SELECT count(*) FROM memories WHERE id=?", (later,)).fetchone()[0] == 1
                memory.restore(snapshot)
                assert memory.db_path.stat().st_ino == inode
                assert peer.execute("SELECT count(*) FROM memories WHERE id=?", (later,)).fetchone()[0] == 0
            for suffix in ("", "-wal", "-shm"):
                Path(str(memory.db_path) + suffix).unlink(missing_ok=True)
            assert not memory.db_path.exists()
            memory.restore(snapshot)
            assert a_id in {r["id"] for r in checked_retrieve(memory, "topic2", project=a)["items"]}
            assert "Betelgeuse" not in checked_retrieve(memory, "topic2", project=a)["text"]

        release = threading.Event()

        def stalled(*args):
            release.wait()
            return synthetic_vector(*args)

        try:
            with patch.object(m, "_fetch_embedding", side_effect=stalled):
                result = checked_retrieve(memory, "topic1")
                assert not result["semantic_used"] and result["semantic_error"] == "Embedding deadline exceeded"
                assert global_id in {r["id"] for r in result["items"]}
                for _ in range(5):
                    result = checked_retrieve(memory, "topic1")
                    assert result["semantic_error"] == "Embedding worker is busy"
                assert sum(t.name == "orbi-embedding" for t in threading.enumerate()) == 1
        finally:
            release.set()
            assert m._EMBED_LOCK.acquire(timeout=1)
            m._EMBED_LOCK.release()
        with patch.object(m, "_fetch_embedding", side_effect=RuntimeError("embedding offline")):
            rejects(lambda: memory.add("unsaved", scope="global"), RuntimeError)
            result = checked_retrieve(memory, "topic1")
            assert "embedding offline" in result["semantic_error"] and global_id in {r["id"] for r in result["items"]}

        release = threading.Event()
        connect = memory._connect

        def stalled_sql(*args, **kwargs):
            release.wait()
            return connect(*args, **kwargs)

        try:
            with patch.object(memory, "_connect", side_effect=stalled_sql), \
                    patch.object(m, "_fetch_embedding", side_effect=synthetic_vector):
                result = checked_retrieve(memory, "topic1")
                assert result["truncated"] and result["error"] == "Retrieval deadline exceeded"
                assert checked_retrieve(memory, "topic1")["error"] == "Retrieval worker is busy"
                assert sum(t.name == "orbi-retrieval" for t in threading.enumerate()) == 1
        finally:
            release.set()
            assert m._RETRIEVE_LOCK.acquire(timeout=1)
            m._RETRIEVE_LOCK.release()

        for values in ([1] * 1023, [0] * 1024, [True] * 1024, [float("nan")] * 1024):
            payload = json.dumps({"data": [{"embedding": values}]}).encode()
            with patch.object(m._OPENER, "open", return_value=io.BytesIO(payload)):
                rejects(lambda: m._fetch_embedding(memory.embedding_url, "query", 1))
    print("PASS: memory unit checks, synthetic vectors only; no model recall score measured.")


if __name__ == "__main__":
    main()
