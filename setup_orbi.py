"""Install the three pinned local artifacts used by Orbi Phase 1."""

import hashlib
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
ARTIFACTS = (
    (".tools/llama-b10809.tar.gz", 11123196,
     "7d692df9e1e386e62f1c12b843903218041e6cd74c9415aa39a7ed3176f9eaa2",
     "https://github.com/ggml-org/llama.cpp/releases/download/b10809/llama-b10809-bin-macos-arm64.tar.gz"),
    ("models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ3_S.gguf", 11289671136,
     "878be93f9c238ea853b3fd1eb602637ce3cf1cddea56dc345d9a7bf2d6093e29",
     "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/"
     "c099eb48e663fd284577b04978a94ffccb261841/gemma-4-26B-A4B-it-UD-IQ3_S.gguf?download=true"),
    ("models/harrier/harrier-oss-v1-0.6B-Q4_K_M.gguf", 396705600,
     "90d684bf550ca2c50de9191f6d18c5d8b20d89dba78257478eb746e88c66ecb3",
     "https://huggingface.co/SuperPauly/harrier-oss-v1-0.6b-gguf/resolve/"
     "5283b8a315ebee0cdc65c04fd7646b3fce8e7c80/harrier-oss-v1-0.6B-Q4_K_M.gguf?download=true"),
)


def verify(path, size, expected):
    if path.stat().st_size != size:
        raise ValueError(f"Wrong size: {path}")
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")


def main():
    for name, size, digest, url in ARTIFACTS:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            partial = target.with_name(target.name + ".partial")
            subprocess.run(["curl", "--fail", "--location", "--retry", "5",
                            "--retry-all-errors", "--connect-timeout", "30",
                            "--speed-limit", "1024", "--speed-time", "120",
                            "--continue-at", "-", "--output", str(partial), url], check=True)
            verify(partial, size, digest)
            partial.replace(target)
            if not target.is_file() or partial.exists():
                raise OSError(f"Artifact publication failed: {target}")
        else:
            verify(target, size, digest)
        print(f"Verified {name}", flush=True)
    destination = ROOT / ".tools/llama-b10809"
    destination.mkdir(exist_ok=True)
    with tarfile.open(ROOT / ARTIFACTS[0][0]) as archive:
        archive.extractall(destination, filter="data")
    server = destination / "llama-b10809/llama-server"
    version = subprocess.check_output([str(server), "--version"], text=True,
                                      stderr=subprocess.STDOUT)
    if "10809" not in version:
        raise RuntimeError(f"Unexpected runtime version: {version}")
    print("Verified llama.cpp b10809; local artifacts ready.")


if __name__ == "__main__":
    main()
