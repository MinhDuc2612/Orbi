# Lane A measurements

Phase 1 is in progress. No candidate has passed all four gates yet.

## Reference and gates

- Existing baseline: **qwen3:8b, 19.48 tok/s**, prompt evaluation 100 tok/s,
  load 4.1 s, measured 2026-09-06. It has not been re-measured.
- Gate (a): generation throughput **>15 tok/s**; also compare against 19.48 tok/s.
- Gate (b): routing **>=18/20**. Gate (c): callable tool JSON **20/20**.
- Gate (d): record peak RAM and system memory pressure during **ten minutes** of load.
- Host: Mac mini M4, 10 CPU / 10 GPU cores, 24 GiB RAM, wired limit 20480 MB.
  Free disk before download: 253.63 GB. Before the first run, system free-memory
  percentage was 66% and pre-existing swap usage was 2975.94 MiB.

## Candidate 1: Qwen3.8-Flash-Next

Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-IQ4_XS`.
Three GGUF shards total **93,682,584,224 bytes**; each size and SHA-256 was
verified against Hugging Face metadata. Download took approximately 56 minutes.
Weights are stored locally under `models/qwen3.8-flash-next/UD-IQ4_XS/`, excluded from Git.

The upstream license is **Qwen Community License 1.0**, correcting the planning
file's Apache-2.0 label. Sources: [official model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
[license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE),
[quantization](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-IQ4_XS).

Runtime: upstream llama.cpp **b10809**, commit
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`, the binary referenced by stable v0.4.0.
The installed Homebrew b10360 predates this model's architecture support.
The [arm64 runtime archive](https://github.com/ggml-org/llama.cpp/releases/download/b10809/llama-b10809-bin-macos-arm64.tar.gz)
was size- and SHA-256-verified: 11,123,196 bytes,
`7d692df9e1e386e62f1c12b843903218041e6cd74c9415aa39a7ed3176f9eaa2`.
Runtime lives under `.tools/llama-b10809/llama-b10809/`, excluded from Git.

Prompt: "Explain how a local command-line assistant can keep useful project memory
while limiting retrieved context. Give a clear practical answer."

Common flags:

```sh
-lm mmap --lazy-mode on --cpu-moe --no-repack --no-op-offload \
-c 4096 -t 8 -ctk q8_0 -ctv q8_0 -fa on --offline --no-warmup \
--perf -n 128 --seed 42 --temp 0 --simple-io --single-turn
```

| Run | GPU layers | Batch / ubatch | Result | Elapsed | Maximum RSS |
|---|---:|---:|---|---:|---:|
| 1 | all | 128 / 128 | Metal OOM before valid generation | 77.62 s | 7,937,196,032 bytes |
| 2 | 8 | 32 / 32 | **0.21 tok/s — FAIL gate (a)** | 703.24 s | 14,425,473,024 bytes |

Run 1 exact errors:

```text
Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
llama_decode: failed to decode, ret = -3
Compute error.
```

**Run 1 exited 0 despite the compute errors. It is a failed run, with no valid
tok/s measurement.** A successful process exit alone is insufficient for this runtime.
The 78-second failed run does not satisfy gate (d).
Raw stdout, stderr and exit statuses are retained in `.session/flash-next-throughput.*`
and `.session/flash-next-retry.*` locally.

The first runtime archive transfer also failed with
`http.client.IncompleteRead: IncompleteRead(8189970 bytes read, 2933226 more expected)`.
A curl retry recovered it and passed the full size/hash check before extraction.

Run 2 completed 128 generated tokens in 600,408.85 ms; llama.cpp reports **0.21 tok/s**.
Prompt evaluation: 76 tokens in 60,105.95 ms, **1.26 tok/s**. No compute error occurred.
This fails both the >15 tok/s gate and the 19.48 tok/s reference. Routing and tool gates
were skipped as instructed after gate (a) failed. Although the run lasted 703.24 seconds,
it was not a continuously sampled ten-minute memory-pressure test; gate (d) is unmeasured.
No Lane A winner or recall score is claimed.

## Candidate 2: Qwen3.8-27B + CMoE

**Unsupported as specified; not downloaded or benchmarked.** No verified converted
Qwen3.8-27B checkpoint or compatible Metal runtime was found. This is an availability
blocker, not a measured 0 tok/s result.

At official [CMoE revision 42dfc947](https://github.com/JarvisPei/CMoE/tree/42dfc94777a0de3620a67bdb5000d7fec56e5b6a):

- `run_cmoe.py` loads Llama/Llava classes, requires CUDA and assumes `model.model.layers`.
- `CMoE_utils.py` calls self-attention in every layer. Qwen3.8-27B's
  [configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
  uses `Qwen3_5ForConditionalGeneration` with hybrid linear/full attention.
- `CMoE_model.py` implements a custom two-projection router. Upstream b10809's
  Qwen35 MoE graph cannot reproduce it simply by renaming or exporting tensors.
- The repository provides no GGUF exporter or Metal implementation. Its documented
  zero-sample route also conflicts with an `inps[0]` access and unconditional
  fine-tuning call in the published runner; these are source findings, not executed errors.

A dense, unconverted Qwen or a differently trained MoE would not be this candidate.


## Candidate 3: Gemma 4 26B-A4B

**Downloaded and hash-verified; first speed/quality run completed.** Ordered fallback after candidate 1's
measured speed failure and candidate 2's implementation blocker.

- Official model: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it),
  revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`; Apache-2.0 in the current official card.
- Quantization: [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/tree/c099eb48e663fd284577b04978a94ffccb261841),
  revision `c099eb48e663fd284577b04978a94ffccb261841`.
- File: `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`, **13,597,177,568 bytes**.
  Expected SHA-256: `babd1e389d386352f71600765d37390f7dc993fbfad6725caccf996ffe34aecf`.
- Same b10809 runtime; intended flags: `-lm mmap -ngl 99 -fa on -ctk q8_0 -ctv q8_0
  -np 1 -c 4096 -t 8 --jinja --reasoning off --offline --perf`.
  Context is capped at 4,096 for these gates; larger contexts are not certified by this run.
- Quantization scope: this is the publisher’s GGUF importance quantization for the requested
  llama.cpp runtime. The plan’s MLX-only DWQ procedure does not apply to a GGUF file; no DWQ
  result or equivalent quality is claimed. Flash attention and Q8 KV must be confirmed in the load log.

### Frozen quality and RAM protocol

`bench_cases.json` contains 20 routing requests and 20 synthetic tool requests. Its SHA-256 is
`fdcf669576169038916ba421e097ba9fee3854aae01873c5b6e6f25287e3e86d`.
The referenced prompt set was absent from Task 2, so these cases were created and frozen before
seeing any candidate's quality output. They sample the plan's skill/lane policy; they do not
certify every skill in the research taxonomy. No synthetic function is executed.

`benchmark.py` evaluates the model plus llama.cpp's native tool parser and strict function
schemas through its local HTTP API. Valid tool JSON and exact intended arguments are reported
separately. Routing requires exact skill/lane JSON; requests use temperature 0 and seed 42.
The RAM gate maintains generation load for at least 600 seconds and samples process RSS,
system pressure and free-memory percentage every five seconds. Pass criterion set before the
run: pressure remains normal and system free-memory percentage stays at least 10%.
The sampled RSS maximum is labeled as such; `/usr/bin/time -l` additionally records the
server's process high-water RSS after shutdown.

```sh
.venv/bin/python benchmark.py speed --output .session/gemma-speed.json
.venv/bin/python benchmark.py quality --output .session/gemma-quality.json
.venv/bin/python benchmark.py soak --pid SERVER_PID --output .session/gemma-soak.json
```

2026-09-07 recovery: the initial Xet transfer exited 1 after exhausting retries:

```text
RuntimeError: Task error: File reconstruction error: CAS Client Error: Format error: I/O error: error decoding response body
```

No final Gemma weight file or verified hash was produced. Retrying the same pinned artifact
with resumable HTTP; no new candidate or quantization has been selected.


### First Gemma run (2026-09-07)

HTTP recovery completed with exit 0 in **863.10 seconds**, all 13,597,177,568 bytes and
SHA-256 verified before loading. Throughput: **25.2986 tok/s**, beating the 19.48 reference.
Routing's required flat JSON shape passed **1/20**. Post-hoc inspection found all **20/20
skill/lane choices correct**, but 19 replies used nested objects or Markdown fences. This
inspection does not replace the failed format gate. Tool calls: **20/20 schema-valid JSON**,
**19/20 exact arguments**; `t07` added punctuation/escaping to the requested regex.
Process high-water RSS: **4,276,174,848 bytes**. RAM soak skipped after the quality gate failed.

The original outputs are preserved under `.session/*-unconstrained.*`. Retesting the same frozen
cases with a standard strict JSON response schema for routing; enums allow every listed skill
and lane and do not encode expected answers. Tool schemas, prompts, seed and temperature stay
unchanged. This measures the usable constrained-output integration, not unconstrained formatting.


### Constrained routing retest (2026-09-07)

Same frozen cases and model, routing schema enforced through the server API:
**26.1991 tok/s**, **20/20 routing**, **20/20 schema-valid tool JSON**, **19/20 exact arguments**.
The original regex mismatch remains; no expected answer or prompt was edited to hide it.
The completed ten-minute memory-pressure run failed, as detailed below. The model-load log confirms 31/31
layers on GPU, flash attention enabled, a 12,952.19 MiB mapped model buffer and 175.31 MiB
of Q8 KV cache. These allocation figures must not be confused with CPU RSS.

Full-GPU gate (d) **FAILED** after **606.30 seconds**: minimum system free memory **13%**,
kernel warning pressure (level 2). Sampled peak RSS **9,851,535,360 bytes**; true process
high-water RSS **12,086,706,176 bytes**. This is a real ten-minute failure, despite passing
speed and quality. Preserved as `.session/*-full-gpu.*`. Retrying the same model with the
first four expert layers on CPU (`--n-cpu-moe 4 --no-repack --no-op-offload`) to reduce
GPU residency. No gate threshold or fixture is relaxed.


### CPU-expert retry (2026-09-07)

The same Gemma weights with `--n-cpu-moe 4 --no-repack --no-op-offload` reached
**23.2194 tok/s**, **20/20 routing**, **20/20 schema-valid tool JSON**, and **19/20 exact
arguments**. Warning memory pressure persisted; the GPU mapped model buffer remained
12,952.19 MiB. The operator stopped this retry after **363.82 seconds**. This is **not** a
second ten-minute measurement. The resulting `RemoteDisconnected('Remote end closed
connection without response')` and shutdown signal were operator-induced.
Sampled peak RSS **2,833,924,096 bytes**, process high-water RSS **9,793,257,472 bytes**,
minimum system free memory **13%**. Global GPU in-use memory peaked at **14,637,809,664
bytes**, including other applications. Evidence: `.session/*-cpu4.*`.
Gemma has not passed gate (d); its full-GPU 606.30-second failure remains recorded above.

## Candidate 4: Granite 4.1 8B — transfer in progress

The final candidate uses IBM's official Apache-2.0
[GGUF publication](https://huggingface.co/ibm-granite/granite-4.1-8b-GGUF/tree/865b82c2e7970d82e3731278c88c57ae7138359c).
Pinned revision: `865b82c2e7970d82e3731278c88c57ae7138359c`.
File: `granite-4.1-8b-Q4_K_M.gguf`, **5,347,914,400 bytes** (the actual artifact is larger
than the plan's approximately 4.3 GB estimate).
Expected SHA-256: `ed902ac9eb6adce5a90c6a08c8ea201b50e23fdc5976d1cd0362006afac5309e`.
The transfer must pass full size/hash verification before loading. All GPU layers, mmap,
Q8 KV, flash attention, 4,096 context, batch/ubatch 128, eight threads, and the same frozen
quality fixtures will be used. No throughput or gate result is inferred from qwen3:8b.
