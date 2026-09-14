"""Bounded Lane A classification and inspectable, decision-only specialist routing."""

import json
import time

# Explicit job defaults from Orbimodels.md. Leaf alternatives still take priority.
PLAN_DEFAULTS = {
    "document_ocr": "PaddlePaddle/PaddleOCR-VL-1.6",
    "speech_recognition": "Qwen/Qwen3-ASR-1.7B",
    "speech_synthesis": "Qwen/Qwen3-TTS-12Hz-1.7B",
    "embedding": "microsoft/Harrier-oss-v1-27B",
    "translation": "xiaomi-research/MiLMMT-46-12B",
    "image_generation": "Qwen/Qwen-Image-2512",
    "video_generation": "Wan 2.7",
    "model3d_generation": "tencent/Hunyuan3D-2.1",
    "visual_grounding": "allenai/Molmo2-8B",
    "reranking": "Qwen/Qwen3-Reranker-4B",
    "formal_reasoning": "zai-org/GLM-5.2",
    "algorithmic_coding": "deepseek-ai/DeepSeek-V4-Pro-0813",
    "architecture_debugging": "deepseek-ai/DeepSeek-V4-Pro-0813",
    "deep_research": "moonshotai/Kimi-K3",
}


def classify(config, system, prompt, properties):
    # Imported here so the CLI can also run as a script without an import cycle.
    from orbi import json_request, strict_json, system_messages, url

    messages = system_messages([dict(role="system", content=system), dict(role="user", content=prompt)])
    endpoint = url(config)
    rendered = json_request(endpoint + "/apply-template", dict(messages=messages, add_generation_prompt=True))["prompt"]
    count = len(json_request(endpoint + "/tokenize", dict(content=rendered, add_special=False))["tokens"])
    if count + 256 + 32 > config["runtime"]["context_size"]:
        raise ValueError(f"Routing prompt needs {count + 288} tokens; context limit is {config['runtime']['context_size']}")
    body = dict(messages=messages, temperature=0, top_p=1, samplers=["temperature"],
                seed=42, max_tokens=256, cache_prompt=False, stream=False,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "route", "strict": True, "schema": {
                        "type": "object", "properties": properties, "required": list(properties),
                        "additionalProperties": False}}})
    response = json_request(endpoint + "/v1/chat/completions", body, timeout=180)
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Classifier did not finish one complete decision")
    result = strict_json(choices[0]["message"]["content"])
    if not isinstance(result, dict) or set(result) != set(properties):
        raise ValueError("Classifier returned invalid decision fields")
    for key, rule in properties.items():
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"Classifier returned empty or invalid {key}")
        if "enum" in rule and result[key] not in rule["enum"]:
            raise ValueError(f"Classifier returned unknown {key}: {result[key]}")
    return result, dict(result=result, prompt_tokens=count, response_id=response.get("id"),
                        requested_sampling={k: body[k] for k in ("temperature", "top_p", "samplers", "seed")})


def decide(config, prompt, *, forced_lane=None):
    from skill_catalog import SKILLS, SUBDOMAINS, DOMAINS, BY_ID, SOURCE_SHA256

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Cannot route an empty request")
    if forced_lane not in (None, "A", "C"):
        raise ValueError("Only explicit Lane A or job Lane C overrides are supported")
    started = time.monotonic()
    policy = "\n".join(ROUTING_POLICY) + "\n" + json.dumps(SKILL_TAXONOMY)
    coarse, first = classify(config, policy, prompt, {
        "skill": {"type": "string", "enum": list(SKILL_TAXONOMY)},
        "lane": {"type": "string", "enum": ["A", "B", "C"]}})
    if coarse["lane"] != SKILL_TAXONOMY[coarse["skill"]]["lane"]:
        raise ValueError("Classifier skill and lane disagree")
    # ponytail: hierarchical menus bound the 302-leaf catalog to the existing 4K context;
    # measure a larger-context single pass before replacing this with a larger prompt.
    menu = "\n".join(f"{key}: {DOMAINS[value['domain']]} / {value['label']}" for key, value in SUBDOMAINS.items())
    select = ("Select the most specific skill group for the requested operation, not incidental topics. "
              "Do not execute the task. Return its group ID. Operational category: " + coarse["skill"])
    group, second = classify(config, select + "\n" + menu, prompt,
                             {"group": {"type": "string", "enum": list(SUBDOMAINS)}})
    leaves = [row for row in SKILLS if row["subdomain"] == group["group"]]
    leaf_menu = "\n".join(f"{row['id']}: {row['label']}" for row in leaves)
    leaf, third = classify(config,
        "Select the most specific matching leaf for the requested operation. Do not solve the task. "
        "Return its skill ID and a short reason identifying the requested output. Operational category: "
        + coarse["skill"] + "\n" + leaf_menu, prompt,
        {"skill": {"type": "string", "enum": [row["id"] for row in leaves]}, "reason": {"type": "string"}})
    chosen = BY_ID[leaf["skill"]]
    lane = "A" if coarse["lane"] == "A" else (chosen["lane"] or coarse["lane"])
    model = chosen["model"]
    reason = [SKILL_TAXONOMY[coarse["skill"]]["description"], leaf["reason"], chosen["lane_reason"]]
    model_source = "skill_catalog"
    if coarse["lane"] == "B" and chosen["selection"] != "min_viable":
        lane, model = "B", PLAN_DEFAULTS[coarse["skill"]]
        model_source = "Orbimodels.md specialist job default"
        reason.append("The request needs a specialist operation. Use the companion plan's explicit specialist "
                      "instead of the catalog's general model default; the original leaf pick stays recorded. "
                      "This specialist remains unverified for this leaf on this machine.")
    elif model is None and coarse["lane"] != "A":
        model = PLAN_DEFAULTS[coarse["skill"]]
        model_source = "Orbimodels.md default; leaf unpicked"
        reason.append("The leaf has no model pick; use the companion plan's explicit job default, unverified for this leaf.")
    selected_lane = lane
    if forced_lane:
        lane = forced_lane
        reason.append(f"Explicit user override to Lane {forced_lane}.")
    if lane == "A":
        model = config["lanes"]["a"]["model"].name
        model_source = "installed Lane A configuration"
        reason.append("Ordinary assistance uses the installed Lane A model under the plan's resident-first rule.")
    elif forced_lane == "C" and selected_lane != "C":
        # A forced asynchronous job needs a giant target, not a specialist relabelled as C.
        model = "moonshotai/Kimi-K3"
        model_source = "Orbimodels.md forced giant default"
        reason.append("Forced Lane C uses the plan's general giant default; the preferred leaf specialist remains recorded.")
    if lane != "A":
        reason.append("Specialist execution is unavailable in Phase 2; no model loading or fallback is attempted.")
    return dict(skill=chosen["id"], skill_label=chosen["label"], category=coarse["skill"],
                lane=lane, model=model, installed=lane == "A", forced_lane=forced_lane,
                preferred_model=chosen["model"], selection=chosen["selection"],
                model_source=model_source,
                source_status=chosen["source_status"], source_line=chosen["source_line"],
                catalog_sha256=SOURCE_SHA256, reason=reason, classifier=[first, second, third],
                elapsed_ms=round((time.monotonic() - started) * 1000, 3))


def describe(decision):
    return (f"{decision['skill']} ({decision['skill_label']}) → Lane {decision['lane']} → "
            f"{decision['model'] or 'unassigned model'}. " + " ".join(decision["reason"]) +
            f" Research status: {decision['source_status']}; selection: {decision['selection']}.")

ROUTING_POLICY = ['Return exactly one JSON object with skill and lane. Choose a skill from skill_taxonomy and the '
 'associated uppercase lane A, B or C. Do not solve the task.',
 'Lane A is the default for ordinary questions, supplied-text edits or summaries, small clear code '
 'changes and memory lookups.',
 'A requested specialist artifact or operation goes to B: OCR, audio transcription or synthesis, '
 'embedding computation, translation, image/video/3D generation, coordinate grounding or retrieval '
 'reranking.',
 'Classify by the requested output, not keyword mentions: editing an image prompt is text_edit; '
 'summarizing an existing transcript is text_summary; explaining a short C function is '
 'general_assistance.',
 'Use C only for substantial technical research synthesis, difficult proofs or algorithm design, or '
 'cross-component debugging requiring deep reasoning. Ordinary coding does not automatically need C.',
 'Classify the ideal lane by need even if the corresponding specialist is not installed yet. This is a '
 'routing benchmark, not permission to run or download a model.']

SKILL_TAXONOMY = {'general_assistance': {'lane': 'A',
                        'description': 'Ordinary conversation, familiar explanations and short code '
                                       'explanations that need no specialist.'},
 'text_edit': {'lane': 'A',
               'description': 'Rewrite or correct supplied text, including a prompt intended for a '
                              'future media job.'},
 'code_edit': {'lane': 'A',
               'description': 'Small, localized code fixes whose cause and intended behavior are '
                              'clear.'},
 'text_summary': {'lane': 'A',
                  'description': 'Summarize text already available; an existing transcript does not '
                                 'require speech recognition.'},
 'memory_lookup': {'lane': 'A',
                   'description': 'Look up an existing remembered fact; do not confuse using memory '
                                  'with explicitly computing embeddings.'},
 'document_ocr': {'lane': 'B',
                  'description': 'Extract text, tables or document structure from scanned pages or '
                                 'document images.'},
 'speech_recognition': {'lane': 'B',
                        'description': 'Convert an audio recording to written words, optionally with '
                                       'timestamps.'},
 'speech_synthesis': {'lane': 'B', 'description': 'Produce spoken audio from supplied text.'},
 'embedding': {'lane': 'B', 'description': 'Compute semantic vectors for supplied text or documents.'},
 'translation': {'lane': 'B',
                 'description': 'Translate content between languages, preserving meaning and required '
                                'terminology.'},
 'image_generation': {'lane': 'B',
                      'description': 'Render or edit an actual image artifact; writing only a textual '
                                     'image prompt stays in A.'},
 'video_generation': {'lane': 'B',
                      'description': 'Generate an actual moving-image clip as a queued media job.'},
 'model3d_generation': {'lane': 'B',
                        'description': 'Generate a three-dimensional mesh or textured 3D asset.'},
 'visual_grounding': {'lane': 'B',
                      'description': 'Locate or point to objects in an image using coordinates or '
                                     'bounding boxes.'},
 'reranking': {'lane': 'B',
               'description': 'Reorder supplied retrieval candidates by relevance to a query using the '
                              'retrieval specialist.'},
 'formal_reasoning': {'lane': 'C',
                      'description': 'Difficult mathematical proof or formal reasoning with several '
                                     'interacting conditions.'},
 'algorithmic_coding': {'lane': 'C',
                        'description': 'Substantial algorithm design with correctness proofs and '
                                       'demanding complexity constraints.'},
 'deep_research': {'lane': 'C',
                   'description': 'Synthesize a large body of technical research and resolve '
                                  'conflicting evidence; not a routine lookup.'},
 'architecture_debugging': {'lane': 'C',
                            'description': 'Difficult cross-component system reasoning, concurrency '
                                           'diagnosis and a substantial verified repair.'}}
