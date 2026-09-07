# Orbi

Orbi is a local-first terminal agent being built to route tasks to open models
and remember context. Phase 0 is complete. Phase 1 is measuring Lane A first;
the CLI, memory and orbs are still pending. See [BENCHMARKS.md](BENCHMARKS.md).

Run the foundation health check from this directory:

```sh
./check.sh
```

Python is pinned to `>=3.12,<3.13`; `requirements.lock` pins the installed packages.
See [BASELINE.md](BASELINE.md) for the recorded measurements and
[../Orbiplan.md](../Orbiplan.md) for the local project plan (kept outside this repository).
