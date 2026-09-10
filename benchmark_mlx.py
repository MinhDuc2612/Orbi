"""Local MLX Lane A gates using the unchanged benchmark.py fixtures and scoring.

8-bit KV uses MLX quantized attention, not Flash Attention. Gemma sliding
attention retains its explicit window mask while storing the full quantized
history because mlx-lm 0.31.3 cannot quantize RotatingKVCache. No JSON grammar,
answer repair, prompt-cache reuse, weight download, or tool execution is used.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import time

import benchmark as b


ROOT = Path(__file__).resolve().parent
CONTEXT = 32768
FIXTURE_SHA256 = "fdcf669576169038916ba421e097ba9fee3854aae01873c5b6e6f25287e3e86d"


def local_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path must be inside {ROOT}: {path}")
    return path


def save(path, value):
    """Atomically publish and verify this run's result, never unconditional success."""
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    if path.read_text() != encoded:
        raise OSError(f"Result verification failed: {path}")


def quantized_cache(model):
    from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache, make_prompt_cache

    args = model.args
    config = getattr(args, "text_config", None)
    shared = (config.get("num_kv_shared_layers", 0) if config is not None
              else getattr(args, "num_kv_shared_layers", 0))
    if shared:
        raise ValueError("Shared-KV Gemma variants require a separate quantized-attention adapter")
    original = make_prompt_cache(model)
    if not original or any(type(c) not in (KVCache, RotatingKVCache) for c in original):
        raise ValueError("Unsupported model cache; refusing a silent KV precision fallback")
    # The Gemma model passes window_size to make_mask, including single-token
    # decoding; a full cache therefore changes retention, not visible positions.
    if any(isinstance(c, RotatingKVCache) for c in original):
        if model.model_type not in ("gemma4", "gemma4_text"):
            raise ValueError("Full-cache replacement is verified only for Gemma4 window masks")
    return [QuantizedKVCache(group_size=64, bits=8) for _ in original]


def tool_message(raw, text, tokenizer, tools, finish):
    """Translate complete native tool frames through the installed model parser."""
    message = {"role": "assistant", "content": text}
    start, end = tokenizer.tool_call_start, tokenizer.tool_call_end
    if not tools or not tokenizer.has_tool_calling or start not in raw:
        return message, finish, None
    try:
        if not end or raw.count(start) != raw.count(end):
            raise ValueError("Incomplete native tool-call frame")
        calls = []
        remaining = raw
        while start in remaining:
            _, remaining = remaining.split(start, 1)
            payload, remaining = remaining.split(end, 1)
            parsed = tokenizer.tool_parser(payload, tools)
            for call in parsed if isinstance(parsed, list) else [parsed]:
                calls.append({"id": f"call_{len(calls)}", "type": "function",
                              "function": {"name": call["name"],
                                           "arguments": json.dumps(call["arguments"],
                                                                   allow_nan=False)}})
        message["tool_calls"] = calls
        # Reaching the length limit must not become a successful native stop.
        return message, "tool_calls" if calls and finish == "stop" else finish, None
    except Exception as error:
        return message, finish, repr(error)


class MLXChat:
    def __init__(self, path):
        import mlx.core as mx
        from mlx_lm import load

        path = local_path(path)
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"Local model directory/config missing: {path}")
        config = b.strict_json((path / "config.json").read_text())
        if config.get("model_type") not in ("gemma4", "gemma4_text", "granite"):
            raise ValueError("This harness supports only the named Gemma4/Granite candidates")
        quant = config.get("quantization", config.get("quantization_config", {}))
        if quant.get("bits") != 4:
            raise ValueError("Expected a verified 4-bit checkpoint; config does not declare 4 bits")
        if mx.default_device() != mx.gpu:
            raise RuntimeError(f"MLX is not using GPU: {mx.default_device()}")
        text_config = config.get("text_config", config)
        if text_config.get("max_position_embeddings", 0) < CONTEXT:
            raise ValueError("Model config does not support the required 32768-token context")
        # Local paths and offline mode prevent missing files from triggering downloads.
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[key] = "1"
        mx.set_cache_limit(256 * 2**20)
        mx.set_memory_limit(20 * 2**30)
        mx.set_wired_limit(18 * 2**30)
        started = time.perf_counter()
        self.model, self.tokenizer = load(str(path), tokenizer_config={"local_files_only": True})
        self.load_s = time.perf_counter() - started
        quantized_cache(self.model)  # Reject incompatible caches before any generation.
        self.metadata = dict(
            model_path=str(path), model_type=config["model_type"],
            config_sha256=hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
            quantization=quant, device=str(mx.default_device()), load_s=self.load_s,
            mlx=importlib.metadata.version("mlx"), mlx_lm=importlib.metadata.version("mlx-lm"),
            context_tokens=CONTEXT, kv_bits=8, kv_group_size=64, prefill_step_size=128,
            cache_limit_bytes=256 * 2**20, memory_limit_bytes=20 * 2**30,
            wired_limit_between_requests_bytes=18 * 2**30,
            wired_limit_during_generation_bytes=mx.device_info()["max_recommended_working_set_size"],
            wired_limit_note="mlx-lm stream_generate temporarily applies the device recommendation and restores the prior limit",
            cache_strategy="fresh full QuantizedKVCache per request; native window masks retained",
            attention_path="quantized_matmul + softmax + quantized_matmul; no Flash Attention",
            response_format_enforced=False, parallel_tool_calls_enforced=False,
            timing_policy="post-first-yield tokens / observed elapsed including final GPU drain; native TPS also retained",
            tool_parser=(self.tokenizer.tool_parser.__module__ if self.tokenizer.tool_parser else None),
            thinking_enabled=False, temperature=0, seed=42,
            weight_provenance="DWQ provenance and weight hashes must be verified separately",
            maximum_observed_prompt_tokens=0, maximum_observed_total_tokens=0,
            generation_errors=[],
            context_note="32768 is the admission limit; short fixtures do not test a filled 32K context",
        )

    def __call__(self, unused_url, messages, **options):
        import mlx.core as mx
        from mlx_lm import stream_generate

        unknown = options.keys() - {"max_tokens", "response_format", "tools", "tool_choice",
                                    "parallel_tool_calls"}
        if unknown:
            raise ValueError(f"Unsupported generation options: {sorted(unknown)}")
        max_tokens = options.get("max_tokens", 256)
        tools = options.get("tools")
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True,
            enable_thinking=False,
        )
        if type(max_tokens) is not int or max_tokens < 1 or len(prompt) + max_tokens > CONTEXT:
            raise ValueError(f"Prompt {len(prompt)} + reply {max_tokens} exceeds context {CONTEXT}")
        caches = quantized_cache(self.model)
        chunks, tokens = [], []
        mx.random.seed(42)
        started = time.perf_counter()
        generator = stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens,
            prompt_cache=caches, prefill_step_size=128,
            # Caches are quantized before prefill; no rotating-cache conversion.
            kv_bits=8, kv_group_size=64, quantized_kv_start=0,
        )
        last, first_at = None, None
        try:
            for last in generator:
                if first_at is None:
                    first_at = time.perf_counter()
                chunks.append(last.text)
                tokens.append(int(last.token))
            drained_at = time.perf_counter()
            if last is None or last.finish_reason not in ("stop", "length"):
                raise RuntimeError("MLX generation ended without a completed response")
            native_n, native_tps = last.generation_tokens, last.generation_tps
            if not math.isfinite(native_tps) or native_tps <= 0 or native_n < 2:
                raise ValueError("Insufficient or invalid MLX generation timing")
            # Exclude the already-delivered prefill token. Include the generator's
            # final GPU drain (it pipelines a lookahead token) in observed time,
            # rather than treating MLX's pre-drain timer as the full elapsed time.
            decode_s = drained_at - first_at
            decode_n = native_n - 1
            if decode_s <= 0 or native_n != len(tokens):
                raise ValueError("Invalid observed decode timing or MLX token count")
            text = "".join(chunks)
            raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
            message, finish, parse_error = tool_message(raw, text, self.tokenizer, tools,
                                                       last.finish_reason)
            self.metadata["maximum_observed_prompt_tokens"] = max(
                self.metadata["maximum_observed_prompt_tokens"], len(prompt))
            self.metadata["maximum_observed_total_tokens"] = max(
                self.metadata["maximum_observed_total_tokens"], len(prompt) + native_n)
            return dict(
                choices=[dict(message=message, finish_reason=finish)],
                timings=dict(predicted_n=decode_n, predicted_ms=decode_s * 1000,
                             predicted_per_second=decode_n / decode_s,
                             prompt_n=last.prompt_tokens, prompt_per_second=last.prompt_tps),
                native=dict(raw_text=raw, stream_text=text, token_ids=tokens,
                            finish_reason=last.finish_reason, generation_tokens=native_n,
                            generation_tps=native_tps, wall_s=time.perf_counter() - started,
                            mlx_peak_bytes=int(last.peak_memory * 1e9),
                            kv_cache_bytes=sum(c.nbytes for c in caches),
                            kv_cache_types=[type(c).__name__ for c in caches],
                            tool_parse_error=parse_error),
                request=dict(messages=messages, options=options, prompt_tokens=len(prompt),
                             response_format_enforced=False,
                             parallel_tool_calls_enforced=False),
            )
        except Exception as error:
            self.metadata["generation_errors"].append(dict(
                error=repr(error), messages=messages, options=options,
                raw_text=self.tokenizer.decode(tokens, skip_special_tokens=False),
                stream_text="".join(chunks), token_ids=tokens))
            raise
        finally:
            generator.close()
            caches.clear()
            mx.clear_cache()


def self_check():
    """Tiny synthetic GPU/cache checks only; no model weights or model score."""
    from types import SimpleNamespace
    from unittest.mock import patch
    import mlx.core as mx
    from mlx_lm.models import base, gemma4_text
    from mlx_lm.models.cache import QuantizedKVCache, RotatingKVCache
    from mlx_lm.tool_parsers import gemma4

    try:
        RotatingKVCache(3).to_quantized(bits=8)
    except NotImplementedError as error:
        assert str(error) == "RotatingKVCache Quantization NYI"
    else:
        raise AssertionError("Installed rotating-cache behavior changed; recheck the workaround")
    cache = QuantizedKVCache(group_size=64, bits=8)
    keys = mx.arange(7 * 64, dtype=mx.float32).reshape(1, 1, 7, 64) / 1000
    values = mx.sin(keys * 7)
    cache.update_and_fetch(keys[..., :5, :], values[..., :5, :])
    stub = SimpleNamespace(layers=[SimpleNamespace(layer_type="sliding_attention")], window_size=3)
    mask = gemma4_text.Gemma4TextModel._make_masks(stub, mx.zeros((1, 2, 8)), [cache])[0]
    assert mask.tolist() == [[False, False, False, True, True, True, False],
                             [False, False, False, False, True, True, True]]
    qkeys, qvalues = cache.update_and_fetch(keys[..., 5:, :], values[..., 5:, :])
    queries = mx.ones((1, 1, 2, 64)) / 8
    with patch.object(base, "quantized_scaled_dot_product_attention",
                      wraps=base.quantized_scaled_dot_product_attention) as quantized, \
            patch.object(mx.fast, "scaled_dot_product_attention",
                         side_effect=AssertionError("8-bit KV unexpectedly used fast SDPA")):
        actual = base.scaled_dot_product_attention(queries, qkeys, qvalues, cache, 1.0, mask)
        mx.eval(actual)
        assert quantized.call_count == 1
    dense_keys = mx.dequantize(*qkeys, group_size=64, bits=8)
    dense_values = mx.dequantize(*qvalues, group_size=64, bits=8)
    scores = queries @ dense_keys.swapaxes(-1, -2)
    expected = mx.softmax(mx.where(mask, scores, mx.finfo(scores.dtype).min), axis=-1) @ dense_values
    assert mx.allclose(actual, expected, atol=1e-5).item()
    single = gemma4_text.Gemma4TextModel._make_masks(stub, mx.zeros((1, 1, 8)), [cache])[0]
    assert single.tolist() == [[False, False, False, False, False, True, True, True]]
    tokenizer = SimpleNamespace(tool_call_start=gemma4.tool_call_start,
                               tool_call_end=gemma4.tool_call_end, has_tool_calling=True,
                               tool_parser=gemma4.parse_tool_call)
    raw = '<|tool_call>call:read_file{path:<|"|>a.py<|"|>,start_line:1,max_lines:2}<tool_call|>'
    msg, finish, error = tool_message(raw, raw, tokenizer, ["fixture"], "stop")
    assert not error and finish == "tool_calls" and len(msg["tool_calls"]) == 1
    assert tool_message(raw, raw, tokenizer, ["fixture"], "length")[1] == "length"
    assert tool_message(raw.replace('<tool_call|>', ''), raw, tokenizer, ["fixture"], "stop")[2]
    assert tool_message('```json\n{}\n```', '```json\n{}\n```', tokenizer, None, "stop")[0]["content"] == '```json\n{}\n```'
    # Control-flow/timing checks use synthetic completions, never loaded models.
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    adapter = MLXChat.__new__(MLXChat)
    adapter.model = SimpleNamespace(args=SimpleNamespace(), model_type="granite",
                                    make_cache=lambda: [KVCache()])
    adapter.metadata = dict(maximum_observed_prompt_tokens=0,
                            maximum_observed_total_tokens=0, generation_errors=[])
    adapter.tokenizer = SimpleNamespace(
        apply_chat_template=lambda *a, **k: [1] * CONTEXT,
        decode=lambda *a, **k: "raw", has_tool_calling=False,
        tool_call_start=None, tool_call_end=None,
    )
    with patch.object(mlx_lm, "stream_generate") as generate:
        try:
            adapter("", [{"role": "user", "content": "fixture"}], max_tokens=1)
        except ValueError as error:
            assert "exceeds context 32768" in str(error)
        else:
            raise AssertionError("Context cap accepted overflow")
        generate.assert_not_called()
    adapter.tokenizer.apply_chat_template = lambda *a, **k: [1, 2]
    def synthetic_generate(*args, **kwargs):
        assert all(c.bits == 8 for c in kwargs["prompt_cache"])
        for i, chunk in enumerate(['```json\n', '{}', '\n```'], 1):
            yield SimpleNamespace(text=chunk, token=i, generation_tokens=i,
                                  generation_tps=30, finish_reason="stop" if i == 3 else None,
                                  peak_memory=0, prompt_tokens=2, prompt_tps=10)
    # Empty synthetic caches have no nbytes state until populated.
    with patch.object(mlx_lm, "stream_generate", side_effect=synthetic_generate), \
            patch.object(QuantizedKVCache, "nbytes", new=property(lambda _: 0)), \
            patch.object(time, "perf_counter", side_effect=[0, 1, 1.1, 1.2]):
        response = adapter("", [], response_format={"type": "json_schema"})
    assert response["choices"][0]["message"]["content"] == '```json\n{}\n```'
    assert response["timings"]["predicted_n"] == 2
    assert math.isclose(response["timings"]["predicted_per_second"], 20)
    assert not response["request"]["response_format_enforced"]
    print("PASS: synthetic 8-bit KV, Gemma prefill/decode window masks, native tool frames; no model score.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["all", "speed", "quality", "soak"], default="all")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if not args.model or not args.output:
        parser.error("--model LOCAL_DIRECTORY and --output are required")
    output = local_path(args.output)
    modes = ["speed", "quality", "soak"] if args.mode == "all" else [args.mode]
    paths = {mode: output.with_name(output.stem + "-" + mode + ".json") for mode in modes}
    destinations = [output, *paths.values()]
    if any(p.exists() or p.with_suffix(p.suffix + ".tmp").exists() for p in destinations):
        parser.error("Output already exists; use a new path to preserve earlier measurements")
    if not output.parent.is_dir():
        parser.error("Output parent must already exist inside code/")
    result = dict(passed=False, qualified=False, mode=args.mode,
                  fixture_sha256=hashlib.sha256((ROOT / "bench_cases.json").read_bytes()).hexdigest(),
                  measured={}, started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    save(output, result)  # A killed process leaves an explicitly incomplete run.
    runner = None
    try:
        if args.duration < 600:
            raise ValueError("RAM gate requires at least 600 seconds")
        if result["fixture_sha256"] != FIXTURE_SHA256:
            raise ValueError("Frozen benchmark fixtures changed")
        result["wired_limit_mb"] = int(b.command("sysctl", "-n", "iogpu.wired_limit_mb"))
        if result["wired_limit_mb"] != 20480:
            raise RuntimeError(f"Expected iogpu.wired_limit_mb=20480; read {result['wired_limit_mb']}")
        result["before_load_memory"] = b.sample_memory(os.getpid(), time.monotonic())
        runner = MLXChat(args.model)
        result["runtime"] = runner.metadata
        result["initial_memory"] = b.sample_memory(os.getpid(), time.monotonic())
        for mode in modes:
            measured = (b.soak("mlx-local", os.getpid(), args.duration, chat_fn=runner)
                        if mode == "soak" else getattr(b, mode)("mlx-local", chat_fn=runner))
            result["measured"][mode] = measured
            save(paths[mode], measured)
            save(output, result)
        result["passed"] = len(result["measured"]) == len(modes) and all(
            m["passed"] for m in result["measured"].values())
        result["qualified"] = args.mode == "all" and result["passed"]
    except Exception as error:
        result["error"] = repr(error)
    except KeyboardInterrupt:
        result["error"] = "KeyboardInterrupt: operator cancelled; incomplete gates do not pass"
    finally:
        result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if runner is not None:
            import mlx.core as mx
            result["mlx_peak_bytes"] = mx.get_peak_memory()
        save(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("measured", "runtime")}, indent=2))
    return 0 if (result["qualified"] if args.mode == "all" else result["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
