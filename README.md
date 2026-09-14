# Orbi

Orbi is a local terminal assistant with streaming replies, persistent sessions,
and scoped memory. Phase 1 uses Gemma 4 26B-A4B UD-IQ3_S with Harrier embeddings.
Lane A measured 29.3336 tok/s; recall now scores 20/20. See [BENCHMARKS.md](BENCHMARKS.md).

The Lane A DWQ retest scored Gemma 18/20 and Granite 14/20 on exact tool arguments.
Gemma still needs retries for two first-pass errors. Lane A remains provisional;
Phase 2 has not started.
The CLI continues to use the Phase 1 IQ3_S checkpoint; full retest measurements and
startup memory warnings are recorded in [BENCHMARKS.md](BENCHMARKS.md).
The two Gemma failures change punctuation inside valid string arguments. A tool-schema
grammar can enforce structure; it does not guarantee the requested string is copied exactly.
The shared system prompt distinguishes exact text from request punctuation and
matches retrieved facts by meaning. The 2026-09-13 run scores recall 20/20;
all 20 separate checks with the answering fact removed correctly return `UNKNOWN`.
The 2026-09-14 run scores exact arguments 18/20 first-pass and 20/20 after one
retry per failed call. Both punctuation errors are corrected by the model on retry.
Regex feedback identifies the failed target and explains when a trailing wildcard
requires an extra character; emitted arguments are never repaired by code.
The top three semantic hits remain protected and greedy decoding is verified.

On this Mac, activate the existing environment and run:

```sh
source .venv/bin/activate
./check.sh
orbi "Hello"
orbi --continue
printf 'Summarize this text' | orbi
```

From a fresh checkout on Apple Silicon, use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python setup_orbi.py
```

Setup downloads and verifies the pinned runtime and two models (about 11.7 GB).
Servers start locally on demand; `orbi --stop` releases them. The measured context
limit is 4,096 tokens. Keep the server prompt cache disabled (`--cache-ram 0`).

Memory uses SQLite WAL, BM25 and real semantic vectors, with hard limits of
12 items, 4,000 rendered characters and 300 ms. Personal facts are global;
project facts and sessions use the current directory. Ask Orbi to remember a
fact, or start a new prompt in a project to retrieve its context. Phase 1 tools
are `remember` and `recall`; other lanes and general file/shell tools come later.
For explicitly delimited exact facts, `remember` checks the source before saving,
retries a mismatch once with a diff, then fails if the copied text still differs.
This check recognizes explicit source delimiters, not arbitrary natural-language wording.
Ctrl-C saves partial state and exits 130. Orbs appear only on a terminal.

`orbi --backup` writes a verified SQLite snapshot and markdown mirror to
`../backups`, with 14-day retention. `orbi --restore PATH` restores a snapshot
after all active turns finish; it refuses to overwrite an active turn.
`orbi --schedule-backups` registers a 03:00 macOS job for the current login;
run it again after logging in. Its plist stays inside `.session/`.

After login, register backups with this exact command on this machine:

```sh
/Users/minhduc/Orbi/code/.venv/bin/orbi --schedule-backups
```

`./check.sh` verifies the loaded 03:00 backup job against its plist and reports
this command. It exits non-zero if registration is missing or mismatched, or
if `iogpu.wired_limit_mb` is 0; it prints the manual sysctl command in that case.

Run `.venv/bin/python test_memory.py` for deterministic memory checks.
`test_cli.py` and `test_recall.py` use the installed models and isolated test data;
start the local services with one `orbi` prompt before running the recall test.
Model files, databases, backups and runtime logs stay out of Git.

Python is pinned to `>=3.12,<3.13`; `requirements.lock` pins the packages.
See [BASELINE.md](BASELINE.md) for the recorded measurements and
[../Orbiplan.md](../Orbiplan.md) for the local project plan (kept outside this repository).
