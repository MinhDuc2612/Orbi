"""Pinned artifacts and an independent, reproducible Granite DWQ calibration."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import urllib.parse
import urllib.request

from setup_orbi import verify

ROOT = Path(__file__).resolve().parent
WORK = ROOT / ".session/retest-20260909"
MODELS = {
    "granite": ("ibm-granite/granite-4.1-8b", "1504002f650e656a0a3789d99574df12e3e94ed0", "granite-4.1-bf16"),
    "gemma": ("catalystsec/gemma-4-26B-A4B-it-4bit-DWQ", "c50241db43deef70c71a4bd0e1f32ff9229aeec0", "gemma-4-dwq-4bit"),
}
SEED, LENGTH = 123, 257


def tied_weights(weights):
    """Drop only a verified duplicate tied head; keep strict loading elsewhere."""
    import mlx.core as mx
    head = weights.get("lm_head.weight")
    if head is not None:
        embedding = weights["model.embed_tokens.weight"]
        if head.dtype != embedding.dtype or not mx.array_equal(head, embedding).item():
            raise ValueError("Granite tied lm_head differs from its embedding")
        weights = {key: value for key, value in weights.items() if key != "lm_head.weight"}
    return weights


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    if json.loads(temporary.read_text()) != data:
        raise OSError(f"Write verification failed: {temporary}")
    temporary.replace(path)
    if not path.is_file():
        raise OSError(f"Publication failed: {path}")


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(name):
    repo, revision, directory = MODELS[name]
    destination = ROOT / "models" / directory
    destination.mkdir(parents=True, exist_ok=True)
    manifest = fetch_json(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
    if manifest["sha"] != revision:
        raise ValueError("Artifact revision changed")
    write_json(WORK / f"{name}-dwq-download-source.json", manifest)
    def download_one(entry):
        filename = entry["rfilename"]
        target = destination / filename
        if not target.resolve().is_relative_to(destination) or Path(filename).name != filename:
            raise ValueError(f"Unexpected artifact path: {filename}")
        partial = target.with_name(target.name + ".partial")
        if not target.exists():
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true"
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "5",
                "--retry-all-errors", "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
                "--continue-at", "-", "--output", str(partial), url], check=True)
        source = target if target.exists() else partial
        if "lfs" in entry:
            digest = entry["lfs"]["sha256"]
            verify(source, entry["size"], digest)
        else:
            data = source.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blob != entry["blobId"]:
                raise ValueError(f"Git blob hash mismatch: {source}")
            digest = hashlib.sha256(data).hexdigest()
        if source == partial:
            partial.replace(target)
        if not target.is_file() or target.stat().st_size != entry["size"]:
            raise OSError(f"Artifact publication failed: {target}")
        print("Verified", filename, entry["size"], flush=True)
        return dict(file=filename, bytes=entry["size"], sha256=digest)

    entries = [entry for entry in manifest["siblings"]
               if entry["rfilename"].endswith((".safetensors", ".json", ".jinja", ".txt", ".md"))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(download_one, entries))
    write_json(destination / "orbi-artifact.json", dict(repo=repo, revision=revision, files=verified))


def calibration():
    target = WORK / "granite-calibration.json"
    if target.exists():
        raise FileExistsError(f"Calibration is already frozen: {target}")
    rows = []
    for offset in range(0, 800000, 100000):
        params = dict(dataset="allenai/tulu-3-sft-mixture", config="default", split="train", offset=offset, length=20)
        batch = fetch_json("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(params))
        if len(batch["rows"]) != 20 or any(row["truncated_cells"] for row in batch["rows"]):
            raise ValueError(f"Incomplete calibration rows at offset {offset}")
        rows.extend(row["row"] for row in batch["rows"])
    random.Random(SEED).shuffle(rows)
    if len(rows) != 160 or len({row["id"] for row in rows}) != 160:
        raise ValueError("Expected 160 distinct calibration examples")
    write_json(target, dict(dataset="allenai/tulu-3-sft-mixture", offsets=list(range(0, 800000, 100000)),
        seed=SEED, max_seq_length=LENGTH, train=rows[:128], valid=rows[128:]))
    print("Frozen calibration SHA256:", hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)


def train(stage):
    import mlx.core as mx
    import mlx.optimizers as optimizers
    from mlx.utils import tree_flatten
    from mlx_lm.quant.dwq import compute_dwq_targets, dwq_quantize
    from mlx_lm.models.granite import Model, ModelArgs
    from mlx_lm.utils import load_model, load_tokenizer, quantize_model, save

    if subprocess.check_output(["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"], text=True).strip() != "20480":
        raise RuntimeError("Restore wired limit to20480 before running this build")
    source = ROOT / "models" / MODELS["granite"][2]
    calibration_path = WORK / "granite-calibration.json"
    data = json.loads(calibration_path.read_text())
    tokenizer = load_tokenizer(source)

    def tokens(rows):
        return [(tokenizer.apply_chat_template(row["messages"], tokenize=True, return_dict=False)[:LENGTH], 0)
                for row in rows]

    train_data, valid_data = tokens(data["train"]), tokens(data["valid"])
    targets = WORK / "granite-targets"
    mx.random.seed(SEED)
    mx.set_cache_limit(256 * 2**20)
    mx.set_memory_limit(20 * 2**30)
    mx.set_wired_limit(18 * 2**30)
    class TiedGranite(Model):
        def sanitize(self, weights):
            if not self.args.tie_word_embeddings:
                raise ValueError("Expected the official tied Granite checkpoint")
            return tied_weights(weights)

    model, config = load_model(source, lazy=True,
                               get_model_classes=lambda config: (TiedGranite, ModelArgs))
    if stage == "targets":
        if targets.exists():
            raise FileExistsError(f"Targets already exist: {targets}")
        compute_dwq_targets(model, targets, train_data, valid_data, 1, LENGTH, SEED)
        expected = {"train": len(train_data), "valid": len(valid_data)}
        if any(len(list((targets / split).glob("*.safetensors"))) != n for split, n in expected.items()):
            raise RuntimeError("Teacher targets are incomplete")
        write_json(targets / "verified.json", dict(counts=expected,
            calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest()))
        print("Teacher targets verified", flush=True)
        return

    verified = json.loads((targets / "verified.json").read_text())
    if verified["calibration_sha256"] != hashlib.sha256(calibration_path.read_bytes()).hexdigest():
        raise ValueError("Calibration changed after computing teacher targets")
    destination = ROOT / "models/granite-4.1-dwq-4bit"
    if destination.exists():
        raise FileExistsError(f"DWQ output already exists: {destination}")
    model, config = quantize_model(model, config, group_size=64, bits=4)
    mx.eval(model.parameters())
    before = {name: hashlib.sha256(value.astype(mx.float32).__array__().tobytes()).hexdigest()
              for name, value in tree_flatten(model.parameters()) if name.endswith(("scales", "biases"))}

    def target_fn(_, index, split):
        saved = mx.load(targets / split / f"{index:010d}.safetensors")
        return saved["logits"], saved["indices"]

    dwq_quantize(model, target_fn, optimizers.Adam(learning_rate=1e-6, bias_correction=True),
                 train_data, valid_data, 1, LENGTH, SEED, gradient_checkpoint=True)
    after = {name: hashlib.sha256(value.astype(mx.float32).__array__().tobytes()).hexdigest()
             for name, value in tree_flatten(model.parameters()) if name in before}
    changed = [name for name in before if before[name] != after[name]]
    if not changed:
        raise RuntimeError("DWQ did not change any quantization scales or biases")
    save(destination, source, model, tokenizer, config)
    files = []
    for path in sorted(destination.glob("*.safetensors")):
        with path.open("rb") as stream:
            files.append(dict(file=path.name, bytes=path.stat().st_size,
                              sha256=hashlib.file_digest(stream, "sha256").hexdigest()))
    if not files or json.loads((destination / "config.json").read_text())["quantization"]["bits"] != 4:
        raise RuntimeError("DWQ output verification failed")
    write_json(destination / "orbi-dwq-build.json", dict(teacher=MODELS["granite"][0],
        revision=MODELS["granite"][1], calibration_sha256=verified["calibration_sha256"],
        train_examples=128, valid_examples=32, max_seq_length=LENGTH, seed=SEED,
        bits=4, group_size=64, learning_rate=1e-6, changed_quantization_arrays=len(changed), files=files))
    print("DWQ build verified; changed quantization arrays:", len(changed), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download-granite", "download-gemma", "calibration", "targets", "build", "self-check"])
    stage = parser.parse_args().stage
    os.chdir(ROOT)
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(ROOT / ".cache/huggingface"), XDG_CACHE_HOME=str(ROOT / ".cache"),
                      TMPDIR=str(ROOT / ".tmp"), TOKENIZERS_PARALLELISM="false")
    if stage == "self-check":
        import mlx.core as mx
        x = mx.array([[1., 2.]])
        assert list(tied_weights({"lm_head.weight": x, "model.embed_tokens.weight": x})) == ["model.embed_tokens.weight"]
        try:
            tied_weights({"lm_head.weight": x + 1, "model.embed_tokens.weight": x})
        except ValueError as error:
            assert str(error) == "Granite tied lm_head differs from its embedding"
        else:
            raise AssertionError("Mismatching tied weights accepted")
        print("PASS: tied-head exact duplicate and mismatch checks")
    elif stage.startswith("download-"):
        download(stage.removeprefix("download-"))
    elif stage == "calibration":
        calibration()
    else:
        train(stage)


if __name__ == "__main__":
    main()
