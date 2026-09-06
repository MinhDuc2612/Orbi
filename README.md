# Orbi

Orbi is a planned local-first terminal agent that routes tasks to open models
and remembers context. Phase 0 provides Python 3.12 and MLX tooling only;
agent features and model downloads have not started.

Run the foundation health check from this directory:

```sh
./check.sh
```

Python is pinned to `>=3.12,<3.13`; `requirements.lock` pins the installed packages.
See [BASELINE.md](BASELINE.md) for the recorded measurements and
[../Orbiplan.md](../Orbiplan.md) for the local project plan (kept outside this repository).
