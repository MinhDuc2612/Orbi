"""Synthetic evaluator regression checks only; these are not model measurements."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from unittest.mock import patch

import benchmark as b


def rejects(action):
    try:
        action()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Invalid input was accepted")


def completion(content="ok", calls=None, finish="stop", rate=20):
    message = {"content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return {"choices": [{"message": message, "finish_reason": finish}],
            "timings": {"predicted_n": 128, "predicted_ms": 128000 / rate,
                        "predicted_per_second": rate}}


def run_http(payloads, action):
    replies = iter(payloads)
    with patch.object(b.urllib.request, "urlopen", side_effect=lambda *a, **k:
                      io.BytesIO(json.dumps(next(replies)).encode())), redirect_stdout(io.StringIO()):
        return action()


def main():
    cases = json.loads(Path(__file__).with_name("bench_cases.json").read_text())
    assert len(cases["routing"]) == len(cases["tool_calls"]) == 20
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in cases["tools"]}
    for case in cases["tool_calls"]:
        expected = case["expected"]
        schema, args = schemas[expected["name"]], expected["arguments"]
        assert b.matches(args, schema)
        assert not b.matches(dict(args, unexpected=True), schema)
        assert not b.matches(dict(list(args.items())[1:]), schema)
    for value in (True, float("inf"), -float("inf"), float("nan")):
        assert not b.matches({"operation": "sum", "values": [value]}, schemas["calculate"])
    assert not b.matches({"path": "a", "start_line": True, "max_lines": 1}, schemas["read_file"])
    assert not b.matches({"path": "a", "start_line": 0, "max_lines": 201}, schemas["read_file"])
    for text in ("NaN", "Infinity", "-Infinity", "{bad"):
        rejects(lambda text=text: b.strict_json(text))
    assert not b.matches(b.strict_json("1e400"), {"type": "number"})
    assert run_http([completion()], lambda: b.speed("http://unused"))["passed"]
    boundary = run_http([completion(rate=15)], lambda: b.speed("http://unused"))
    assert not boundary["passed"] and not boundary["beats_baseline"]
    for finish in ("error", None, ""):
        rejects(lambda finish=finish: run_http([completion(finish=finish)], lambda: b.chat("http://unused", [])))
    for key in ("predicted_n", "predicted_ms", "predicted_per_second"):
        for value in (0, -1, True, "20", float("inf"), None):
            bad = completion()
            bad["timings"][key] = value
            rejects(lambda bad=bad: run_http([bad], lambda: b.chat("http://unused", [])))
    bad = completion()
    bad["timings"]["predicted_n"] = 31
    rejects(lambda: run_http([bad], lambda: b.speed("http://unused")))
    routing = [completion(json.dumps(c["expected"])) for c in cases["routing"]]
    calls = [completion(calls=[{"type": "function", "id": c["id"], "function": {
        "name": c["expected"]["name"], "arguments": json.dumps(c["expected"]["arguments"])}}],
        finish="tool_calls") for c in cases["tool_calls"]]
    result = run_http(routing + calls, lambda: b.quality("http://unused"))
    assert result["passed"] and result["routing_score"] == result["tool_validity"] == result["tool_exact"] == 20
    bad = deepcopy(calls)
    emitted = [r["choices"][0]["message"]["tool_calls"][0] for r in bad]
    emitted[0]["function"]["arguments"] = "{"
    bad[1]["choices"][0]["finish_reason"] = "length"
    del emitted[2]["id"]
    emitted[3]["id"] = ""
    emitted[4]["function"]["name"] = "unknown_function"
    emitted[5]["function"]["arguments"] = "[]"
    bad[6]["choices"][0]["message"]["tool_calls"].append(deepcopy(emitted[6]))
    bad[7]["choices"][0]["finish_reason"] = None
    args = json.loads(emitted[8]["function"]["arguments"])
    args["project"] = "/wrong-global-scope"
    emitted[8]["function"]["arguments"] = json.dumps(args)
    emitted[16]["function"]["arguments"] = '{"operation":"sum","values":[Infinity]}'
    result = run_http(routing + bad, lambda: b.quality("http://unused"))
    assert not result["passed"] and result["routing_score"] == 20
    assert result["tool_validity"] == result["tool_exact"] == 10
    rejects(lambda: b.soak("http://unused", 1, 599))
    print("PASS: synthetic benchmark evaluator checks; no real model score measured.")


if __name__ == "__main__":
    main()
