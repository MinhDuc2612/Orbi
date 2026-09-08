"""Scoped SQLite memory with bounded hybrid retrieval and coherent backups."""

from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
import uuid

import numpy as np


QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
_EMBED_LOCK = threading.Lock()
_RETRIEVE_LOCK = threading.Lock()
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _embedding_input(text, query=False):
    # ponytail: embed the first 1024 UTF-8 bytes; chunk long documents if recall needs it.
    prefix = text[:1024].encode("utf-8")[:1024].decode("utf-8", errors="ignore")
    return (QUERY_PREFIX if query else "") + prefix, len(prefix) < len(text)


def _fetch_embedding(url, text, timeout):
    request = urllib.request.Request(url, json.dumps({
        "input": text, "encoding_format": "float"}).encode(),
        {"Content-Type": "application/json"})
    with _OPENER.open(request, timeout=timeout) as response:
        raw = response.read(131073)
    if len(raw) > 131072:
        raise ValueError("Embedding response exceeds 128 KiB")
    data = json.loads(raw)
    rows = data.get("data", [])
    values = rows[0].get("embedding") if len(rows) == 1 else None
    if (not isinstance(values, list) or len(values) != 1024
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError("Expected one finite 1024-dimensional Harrier embedding")
    vector = np.asarray(values, dtype="<f4")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding has invalid norm")
    return vector / norm


def _start_embedding(url, text, timeout):
    if not _EMBED_LOCK.acquire(blocking=False):
        return None
    event, result = threading.Event(), {}

    def work():
        try:
            result["vector"] = _fetch_embedding(url, text, timeout)
        except Exception as error:
            result["error"] = repr(error)
        finally:
            _EMBED_LOCK.release()
            event.set()

    # ponytail: one in-flight request per process; a stuck peer disables semantic
    # retrieval until it returns. Daemon + nonblocking admission prevent thread leaks.
    try:
        threading.Thread(target=work, daemon=True, name="orbi-embedding").start()
    except Exception:
        _EMBED_LOCK.release()
        raise
    return event, result


def _project(value):
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Project is not a directory: {path}")
    return str(path)


class Memory:
    def __init__(self, db_path, embedding_url="http://127.0.0.1:8124/v1/embeddings",
                 max_items=12, max_chars=4000, max_ms=300):
        for name, value in (("max_items", max_items), ("max_chars", max_chars)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(max_ms) not in (int, float) or not math.isfinite(max_ms) or max_ms <= 0:
            raise ValueError("max_ms must be a positive finite number")
        url = urllib.parse.urlsplit(embedding_url)
        if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "::1", "localhost")
                or url.username or url.password or url.fragment):
            raise ValueError("Embeddings must use a local HTTP endpoint")
        self.db_path = Path(db_path).expanduser().resolve()
        self.embedding_url = embedding_url
        self.max_items, self.max_chars = min(max_items, 12), min(max_chars, 4000)
        self.max_ms = min(max_ms, 300)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            if db.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise RuntimeError("Memory requires SQLite WAL mode")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS memory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO memory_meta VALUES ('version', '1');
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY, text TEXT NOT NULL, scope TEXT NOT NULL,
                    project TEXT, tier TEXT NOT NULL CHECK(tier IN ('L0','L1','L2','L3')),
                    tags TEXT NOT NULL, created REAL NOT NULL, vector BLOB NOT NULL,
                    embedding_truncated INTEGER NOT NULL,
                    CHECK(length(vector)=4096),
                    CHECK((scope='global' AND project IS NULL) OR
                          (scope='project' AND project IS NOT NULL AND length(project)>0))
                );
                CREATE INDEX IF NOT EXISTS memory_scope ON memories(scope, project, tier);
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    text, tags, content='memories', content_rowid='id');
                CREATE TRIGGER IF NOT EXISTS memory_insert AFTER INSERT ON memories BEGIN
                    INSERT INTO memory_fts(rowid,text,tags) VALUES(new.id,new.text,new.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memory_delete AFTER DELETE ON memories BEGIN
                    INSERT INTO memory_fts(memory_fts,rowid,text,tags)
                    VALUES('delete',old.id,old.text,old.tags);
                END;
                CREATE TRIGGER IF NOT EXISTS memory_update AFTER UPDATE ON memories BEGIN
                    INSERT INTO memory_fts(memory_fts,rowid,text,tags)
                    VALUES('delete',old.id,old.text,old.tags);
                    INSERT INTO memory_fts(rowid,text,tags) VALUES(new.id,new.text,new.tags);
                END;
            """)
            if db.execute("SELECT value FROM memory_meta WHERE key='version'").fetchone()[0] != "1":
                raise ValueError("Unsupported memory schema version")
        if not self.db_path.is_file():
            raise OSError(f"Memory database was not created: {self.db_path}")

    def _connect(self, deadline=None, readonly=False):
        remaining = max(0, deadline - time.monotonic()) if deadline else 5
        # SQLite's busy sleeps can overshoot the requested timeout; fail reads fast.
        busy = min(remaining, .005) if readonly else remaining
        target = self.db_path.as_uri() + "?mode=ro" if readonly else str(self.db_path)
        db = sqlite3.connect(target, uri=readonly, timeout=busy, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute(f"PRAGMA busy_timeout={int(busy * 1000)}")
            if readonly:
                db.execute("PRAGMA query_only=ON")
            if deadline:
                db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 100)
        except Exception:
            db.close()
            raise
        return db

    def add(self, text, *, scope, project=None, tier="L1", tags=()):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Memory text must be a nonempty string")
        project = _project(project)
        if scope not in ("global", "project") or (scope == "global") != (project is None):
            raise ValueError("Global memory requires project=None; project memory requires a directory")
        if tier not in ("L0", "L1", "L2", "L3"):
            raise ValueError("Memory tier must be L0, L1, L2 or L3")
        if isinstance(tags, str):
            raise ValueError("tags must be a sequence of strings")
        tags = tuple(tags)
        if not all(isinstance(tag, str) for tag in tags):
            raise ValueError("tags must contain only strings")
        content, clipped = _embedding_input(text)
        pending = _start_embedding(self.embedding_url, content, 5)
        if pending is None:
            raise RuntimeError("Embedding worker is busy; memory was not saved")
        event, result = pending
        if not event.wait(5):
            raise TimeoutError("Embedding exceeded 5 seconds; memory was not saved")
        if "error" in result:
            raise RuntimeError(result["error"])
        with closing(self._connect()) as db:
            with db:
                cursor = db.execute("""INSERT INTO memories
                    (text,scope,project,tier,tags,created,vector,embedding_truncated)
                    VALUES(?,?,?,?,?,?,?,?)""", (text, scope, project, tier,
                    json.dumps(tags, ensure_ascii=False), time.time(),
                    result["vector"].astype("<f4").tobytes(), int(clipped)))
                row_id = cursor.lastrowid
            if db.execute("SELECT text FROM memories WHERE id=?", (row_id,)).fetchone()[0] != text:
                raise RuntimeError("Stored memory verification failed")
        return row_id

    def retrieve(self, query, *, project=None, scope="both"):
        started = time.monotonic()
        if not isinstance(query, str) or scope not in ("both", "global", "project"):
            raise ValueError("Expected a string query and scope both, global or project")

        def incomplete(reason):
            return dict(items=[], text="", elapsed_ms=(time.monotonic() - started) * 1000,
                        truncated=True, semantic_used=False, semantic_error=None, error=reason)

        if not _RETRIEVE_LOCK.acquire(blocking=False):
            return incomplete("Retrieval worker is busy")
        event, result = threading.Event(), {}

        def work():
            try:
                result["value"] = self._retrieve(query, project=project, scope=scope)
            except Exception as error:
                result["error"] = error
            finally:
                _RETRIEVE_LOCK.release()
                event.set()

        # ponytail: one retrieval per process bounds stalled SQLite/file I/O too;
        # use a fixed-size pool if parallel in-process requests become necessary.
        try:
            threading.Thread(target=work, daemon=True, name="orbi-retrieval").start()
        except Exception:
            _RETRIEVE_LOCK.release()
            raise
        if not event.wait(max(0, started + self.max_ms * .95 / 1000 - time.monotonic())):
            return incomplete("Retrieval deadline exceeded")
        if "error" in result:
            raise result["error"]
        result["value"]["elapsed_ms"] = (time.monotonic() - started) * 1000
        return result["value"]

    def _retrieve(self, query, *, project, scope):
        started = time.monotonic()
        # Reserve time for bounded rendering and connection cleanup, rather than overrun.
        deadline = started + self.max_ms * 0.85 / 1000
        project = _project(project)
        if scope == "project" and project is None:
            raise ValueError("Project retrieval requires a directory")
        if scope == "global" or project is None:
            where, parameters = "m.scope='global'", ()
        elif scope == "project":
            where, parameters = "m.scope='project' AND m.project=?", (project,)
        else:
            where, parameters = "(m.scope='global' OR (m.scope='project' AND m.project=?))", (project,)
        content, clipped = _embedding_input(query, query=True)
        has_query = bool(query[:4096].strip())
        pending = _start_embedding(self.embedding_url, content, max(.001, deadline - time.monotonic())) if has_query else None
        rows, lexical, semantic, truncated, error, semantic_error = {}, [], [], clipped, None, None
        semantic_used = False
        columns = "m.id,m.scope,m.project,m.tier,substr(m.text,1,?) AS text,length(m.text) AS original_chars,m.embedding_truncated"
        terms = re.findall(r"\w+", query[:4096], flags=re.UNICODE)[:32]
        match = " OR ".join('"' + term[:128] + '"' for term in terms)
        try:
            with closing(self._connect(deadline, readonly=True)) as db:
                if match:
                    found = db.execute(f"""SELECT {columns} FROM memory_fts
                        JOIN memories m ON m.id=memory_fts.rowid
                        WHERE ({where}) AND memory_fts MATCH ?
                        ORDER BY bm25(memory_fts),m.tier DESC,m.id DESC LIMIT 64""",
                        (self.max_chars, *parameters, match)).fetchall()
                else:
                    found = db.execute(f"""SELECT {columns} FROM memories m WHERE ({where})
                        AND m.tier IN ('L2','L3') ORDER BY m.tier DESC,m.id DESC LIMIT 64""",
                        (self.max_chars, *parameters)).fetchall()
                rows = {r["id"]: dict(r) for r in found}
                lexical = list(rows)
                truncated |= len(found) == 64
                if pending:
                    event, result = pending
                    event.wait(max(0, deadline - time.monotonic() - .015))
                    if not event.is_set():
                        semantic_error, truncated = "Embedding deadline exceeded", True
                    elif "error" in result:
                        semantic_error = result["error"]
                    else:
                        candidates = []
                        # ponytail: scoped linear scan capped at 4096 vectors/deadline;
                        # use an ANN index when this ceiling reduces measured recall.
                        cursor = db.execute(f"SELECT m.id,m.vector FROM memories m WHERE ({where}) LIMIT 4097", parameters)
                        semantic_used = True
                        scanned = 0
                        while time.monotonic() < deadline - .008:
                            batch = cursor.fetchmany(32)
                            if not batch:
                                break
                            for row in batch:
                                scanned += 1
                                if scanned > 4096:
                                    truncated = True
                                    break
                                vector = np.frombuffer(row["vector"], dtype="<f4")
                                score = float(vector @ result["vector"])
                                if math.isfinite(score) and score > 0:
                                    candidates.append((score, row["id"]))
                            if scanned > 4096:
                                break
                        else:
                            truncated = True
                        semantic = [i for _, i in sorted(candidates, reverse=True)[:64]]
                        missing = [i for i in semantic if i not in rows]
                        if missing and time.monotonic() < deadline:
                            placeholders = ",".join("?" for _ in missing)
                            found = db.execute(f"SELECT {columns} FROM memories m WHERE ({where}) AND m.id IN ({placeholders})",
                                               (self.max_chars, *parameters, *missing)).fetchall()
                            rows.update((r["id"], dict(r)) for r in found)
                elif has_query:
                    semantic_error = "Embedding worker is busy"
        except sqlite3.OperationalError as failure:
            error, truncated = repr(failure), True
        scores = {}
        for ranking in (lexical, semantic):
            for rank, row_id in enumerate(ranking, 1):
                if row_id in rows:
                    scores[row_id] = scores.get(row_id, 0) + 1 / (60 + rank)
        ordered = sorted(scores, key=lambda i: (-scores[i], -int(rows[i]["tier"][1]), -i))
        # Bootstrap from at most two relevant high-level memories, then retrieve details.
        bootstrap = [i for i in ordered if rows[i]["tier"] in ("L2", "L3")][:2]
        ordered = bootstrap + [i for i in ordered if i not in bootstrap]
        items, blocks, used = [], [], 0
        for row_id in ordered:
            if len(items) >= self.max_items:
                truncated = True
                break
            row = rows[row_id]
            label = f"[{row['scope']}:{row['project'] or 'all'} {row['tier']} #{row_id}]\n"
            remaining = self.max_chars - used - (2 if blocks else 0) - len(label)
            if remaining <= 0:
                truncated = True
                break
            item = dict(row, text=row["text"][:remaining])
            truncated |= len(item["text"]) < row["original_chars"]
            block = label + item["text"]
            used += len(block) + (2 if blocks else 0)
            blocks.append(block)
            items.append(item)
        return dict(items=items, text="\n\n".join(blocks),
                    elapsed_ms=(time.monotonic() - started) * 1000,
                    truncated=bool(truncated), semantic_used=semantic_used,
                    semantic_error=semantic_error, error=error)

    @staticmethod
    def _verify_snapshot(db):
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite snapshot failed integrity_check")
        if db.execute("SELECT value FROM memory_meta WHERE key='version'").fetchone()[0] != "1":
            raise ValueError("Unsupported memory snapshot")

    def backup(self, backup_dir):
        directory = Path(backup_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        namespace = "orbi-memory-" + hashlib.sha256(str(self.db_path).encode()).hexdigest()[:16] + "-"
        now = datetime.now(timezone.utc)
        stem = namespace + now.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex
        snapshot, mirror = directory / (stem + ".sqlite3"), directory / (stem + ".md")
        partial, markdown = directory / (stem + ".sqlite3.tmp"), directory / (stem + ".md.tmp")
        deadline = time.monotonic() + 30

        def progress(*_):
            if time.monotonic() >= deadline:
                raise TimeoutError("SQLite backup exceeded 30 seconds")

        try:
            with closing(self._connect(readonly=True)) as source, closing(sqlite3.connect(partial)) as target:
                source.backup(target, pages=128, progress=progress, sleep=.02)
                target.execute("PRAGMA journal_mode=DELETE")
                self._verify_snapshot(target)
                lines = ["# Orbi memory snapshot\n"]
                for row in target.execute("SELECT id,scope,project,tier,tags,text FROM memories ORDER BY id"):
                    row_id, scope, project, tier, tags, text = row
                    lines.append(f"\n## {row_id} · {scope} · {project or 'all'} · {tier}\n\nTags: {tags}\n\n{text}\n")
                payload = "".join(lines)
            markdown.write_text(payload, encoding="utf-8")
            if markdown.read_text(encoding="utf-8") != payload or not partial.is_file():
                raise OSError("Snapshot file verification failed")
            size = partial.stat().st_size
            markdown.replace(mirror)
            partial.replace(snapshot)  # Publishing the verified DB marks this pair complete.
            if not mirror.is_file() or snapshot.stat().st_size != size:
                raise OSError("Snapshot publication verification failed")
        finally:
            partial.unlink(missing_ok=True)
            markdown.unlink(missing_ok=True)
        pattern = re.compile(re.escape(namespace) + r"(\d{8}T\d{6})-[0-9a-f]{32}\.sqlite3")
        for old in directory.iterdir():
            match = pattern.fullmatch(old.name)
            if not match or old.is_symlink() or not old.is_file():
                continue
            stamp = datetime.strptime(match[1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            if stamp < now - timedelta(days=14):
                old.unlink()
                old.with_suffix(".md").unlink(missing_ok=True)
        return snapshot

    def restore(self, snapshot):
        snapshot = Path(snapshot).expanduser().resolve(strict=True)
        if snapshot == self.db_path:
            raise ValueError("Restore source must differ from the live database")
        deadline = time.monotonic() + 30

        def progress(*_):
            if time.monotonic() >= deadline:
                raise TimeoutError("SQLite restore exceeded 30 seconds")

        with closing(sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True)) as source:
            self._verify_snapshot(source)
            with closing(self._connect()) as target:
                source.backup(target, pages=128, progress=progress, sleep=.02)
                self._verify_snapshot(target)
                if target.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                    raise RuntimeError("Restored database did not enable WAL")
        if not self.db_path.is_file():
            raise OSError("Restored database is missing")
