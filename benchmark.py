"""Run real Lane A gates against a local llama.cpp server; never execute tools."""

import argparse
import json
import math
from pathlib import Path
import plistlib
import re
import subprocess
import threading
import time
import urllib.request


PROMPT = (
    "Explain how a local command-line assistant can keep useful project memory "
    "while limiting retrieved context. Give a clear practical answer."
)


def strict_json(text):
    def invalid(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")
    return json.loads(text, parse_constant=invalid)


def chat(url, messages, **options):
    body = dict(messages=messages, temperature=0, seed=42, max_tokens=256,
                cache_prompt=False, stream=False)
    body.update(options)
    request = urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = strict_json(response.read())
    if "error" in result or not result.get("choices"):
        raise RuntimeError(f"Invalid server completion: {result}")
    if result["choices"][0].get("finish_reason") not in ("stop", "length", "tool_calls"):
        raise RuntimeError(f"Failed or unfinished completion: {result}")
    timing = result.get("timings", {})
    for key in ("predicted_n", "predicted_ms", "predicted_per_second"):
        value = timing.get(key, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid generation timing: {timing}")
    return result


def speed(url):
    response = chat(url, [{"role": "user", "content": PROMPT}], max_tokens=128)
    timing = response["timings"]
    if timing["predicted_n"] < 32:
        raise ValueError("Fewer than 32 generated tokens; insufficient speed sample")
    rate = timing["predicted_per_second"]
    return dict(passed=rate > 15, beats_baseline=rate > 19.48,
                tok_s=rate, response=response)


def matches(value, schema):
    """Validate only the JSON Schema keywords used by the frozen fixtures."""
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool, "null": type(None)}
    allowed = schema["type"]
    allowed = allowed if isinstance(allowed, list) else [allowed]
    if not any(type(value) in (types[t] if isinstance(types[t], tuple) else (types[t],))
               for t in allowed):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        properties = schema["properties"]
        return (all(k in value for k in schema.get("required", []))
                and (schema.get("additionalProperties", True) or value.keys() <= properties.keys())
                and all(k in properties and matches(v, properties[k]) for k, v in value.items()))
    if isinstance(value, list):
        return (len(value) >= schema.get("minItems", 0)
                and all(matches(v, schema["items"]) for v in value))
    if type(value) in (int, float):
        return (math.isfinite(value)
                and schema.get("minimum", -float("inf")) <= value <= schema.get("maximum", float("inf")))
    return True


def quality(url):
    cases = json.loads(Path(__file__).with_name("bench_cases.json").read_text())
    system = "\n".join(cases["routing_policy"]) + "\n" + json.dumps(cases["skill_taxonomy"])
    route_format = {"type": "json_schema", "json_schema": {"name": "route", "strict": True,
        "schema": {"type": "object", "properties": {
            "skill": {"type": "string", "enum": list(cases["skill_taxonomy"])},
            "lane": {"type": "string", "enum": ["A", "B", "C"]}},
            "required": ["skill", "lane"], "additionalProperties": False}}}
    routing, calls = [], []
    for case in cases["routing"]:
        row = dict(id=case["id"], correct=False)
        try:
            row["response"] = chat(url, [{"role": "system", "content": system},
                                         {"role": "user", "content": case["prompt"]}],
                                   response_format=route_format)
            message = row["response"]["choices"][0]["message"]
            row["actual"] = strict_json(message["content"])
            row["correct"] = row["actual"] == case["expected"]
        except Exception as error:
            row["error"] = repr(error)
        routing.append(row)
        print(f"{row['id']}: routing {'PASS' if row['correct'] else 'FAIL'}", flush=True)
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in cases["tools"]}
    for case in cases["tool_calls"]:
        row = dict(id=case["id"], valid=False, correct=False)
        try:
            row["response"] = chat(url, [
                {"role": "system", "content": cases["tool_policy"]},
                {"role": "user", "content": case["prompt"]}],
                tools=cases["tools"], tool_choice="auto", parallel_tool_calls=False)
            emitted = row["response"]["choices"][0]["message"].get("tool_calls", [])
            if (row["response"]["choices"][0]["finish_reason"] != "tool_calls"
                    or len(emitted) != 1 or emitted[0]["type"] != "function"
                    or not isinstance(emitted[0].get("id"), str) or not emitted[0]["id"]):
                raise ValueError("Expected exactly one function call")
            call = emitted[0]["function"]
            args = strict_json(call["arguments"])
            row["valid"] = call["name"] in schemas and matches(args, schemas[call["name"]])
            if row["valid"] and "scope" in args:
                row["valid"] = (args["project"] is None if args["scope"] == "global"
                                else isinstance(args["project"], str) and bool(args["project"]))
            row["correct"] = row["valid"] and dict(name=call["name"], arguments=args) == case["expected"]
        except Exception as error:
            row["error"] = repr(error)
        calls.append(row)
        print(f"{row['id']}: callable={row['valid']}, exact={row['correct']}", flush=True)
    score = sum(r["correct"] for r in routing)
    valid = sum(r["valid"] for r in calls)
    return dict(passed=score >= 18 and valid == 20, routing_score=score,
                tool_validity=valid, tool_exact=sum(r["correct"] for r in calls),
                routing=routing, tool_calls=calls)


def command(*args):
    return subprocess.check_output(args, text=True, timeout=10).strip()


def sample_memory(pid, started):
    devices = plistlib.loads(command("ioreg", "-r", "-c", "AGXAccelerator", "-a").encode())
    gpu = next(d["PerformanceStatistics"] for d in devices if "PerformanceStatistics" in d)
    return dict(
        elapsed_s=time.monotonic() - started,
        rss_bytes=int(command("ps", "-p", str(pid), "-o", "rss=")) * 1024,
        pressure=int(command("sysctl", "-n", "kern.memorystatus_vm_pressure_level")),
        free_percent=int(re.search(r"free percentage: (\d+)%",
                                    command("memory_pressure", "-Q"))[1]),
        swap=command("sysctl", "-n", "vm.swapusage"),
        gpu_in_use_bytes=int(gpu["In use system memory"]),
        gpu_allocated_bytes=int(gpu["Alloc system memory"]),
    )


def soak(url, pid, duration):
    if duration < 600:
        raise ValueError("RAM gate requires at least 600 seconds")
    started = time.monotonic()
    samples, errors, rates = [], [], []
    done = threading.Event()

    def monitor():
        try:
            while not done.is_set():
                sample = sample_memory(pid, started)
                samples.append(sample)
                if sample["elapsed_s"] >= duration:
                    break
                done.wait(5)
        except Exception as error:
            errors.append(repr(error))
        finally:
            done.set()

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        while not done.is_set():
            response = chat(url, [{"role": "user", "content": PROMPT +
                                   f" Include practical example {len(rates) + 1}."}],
                            max_tokens=128)
            rates.append(response["timings"]["predicted_per_second"])
            print(f"load {time.monotonic() - started:.0f}s: "
                  f"{rates[-1]:.2f} tok/s", flush=True)
    except Exception as error:
        errors.append(repr(error))
        done.set()
    finally:
        worker.join(timeout=15)
    passed = (not errors and rates and samples and samples[-1]["elapsed_s"] >= duration
              and all(s["pressure"] == 1 and s["free_percent"] >= 10 for s in samples))
    return dict(passed=bool(passed), duration_s=time.monotonic() - started,
                peak_sampled_rss_bytes=max((s["rss_bytes"] for s in samples), default=0),
                peak_sampled_gpu_in_use_bytes=max((s["gpu_in_use_bytes"] for s in samples), default=0),
                peak_sampled_gpu_allocated_bytes=max((s["gpu_allocated_bytes"] for s in samples), default=0),
                sampling_interval_s=5,
                minimum_free_percent=min((s["free_percent"] for s in samples), default=0),
                samples=samples, rates=rates, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["speed", "quality", "soak"])
    parser.add_argument("--url", default="http://127.0.0.1:8123")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--duration", type=int, default=600)
    args = parser.parse_args()
    if args.mode == "soak" and not args.pid:
        parser.error("soak requires --pid of the llama-server process")
    try:
        if args.mode == "soak":
            result = soak(args.url, args.pid, args.duration)
        else:
            result = {"speed": speed, "quality": quality}[args.mode](args.url)
    except Exception as error:
        result = dict(passed=False, error=repr(error))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert json.loads(args.output.read_text()) == result
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("response", "samples", "rates", "routing", "tool_calls")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
