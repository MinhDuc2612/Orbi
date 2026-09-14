"""Static research-snapshot skill catalog, with no runtime dependency on the vault.

302 actual leaves in 8 domains / 39 subdomains; the plan's 341 is a stale count.
The source has 56 alternative markers, but only 13 explicitly name a usable
candidate (including 2 self-picks). Negative/unverified alternative text is not
silently promoted to a pick. Source verification tags are historical claims,
not current model availability, model qualification, or runtime measurements.
"""

SOURCE = 'Researchhub/sample1/OrbiLLM/Orbi OSSLLM research.md'
SOURCE_SHA256 = '4aba2f715b130adf0dfe5b62b7a9c24ac100908c71dabe7f94c3aff4b7965c5e'
SOURCE_COUNTS = {"leaves": 302, "domains": 8, "subdomains": 39, "verified": 46, "reasoned_default": 125, "not_applicable": 131, "alternative_markers": 56, "named_alternatives": 13}

DOMAINS = {
    'd01': 'Conversational & Reasoning',
    'd02': 'Agentic & Autonomous Systems',
    'd03': 'Agentic Coding / Software Engineering',
    'd04': 'Vision & Multimodal Understanding',
    'd05': 'Creative & Media Generation',
    'd06': 'Speech & Audio Processing',
    'd07': 'Data, Retrieval & Analytics',
    'd08': 'Infrastructure & Meta-AI',
}

SUBDOMAINS = {
    'd01.s01': {'domain': 'd01', 'label': 'Formal & Deductive Reasoning'},
    'd01.s02': {'domain': 'd01', 'label': 'Inferential & Abstract Reasoning'},
    'd01.s03': {'domain': 'd01', 'label': 'Genre & Register Writing Craft'},
    'd01.s04': {'domain': 'd01', 'label': 'Pragmatics & Social Language Understanding'},
    'd01.s05': {'domain': 'd01', 'label': 'Multi-Turn Dialogue Management'},
    'd02.s01': {'domain': 'd02', 'label': 'Tool-Use Execution Reliability'},
    'd02.s02': {'domain': 'd02', 'label': 'Planning & Execution Control'},
    'd02.s03': {'domain': 'd02', 'label': 'Memory & State Persistence'},
    'd02.s04': {'domain': 'd02', 'label': 'Multi-Agent Coordination'},
    'd02.s05': {'domain': 'd02', 'label': 'Environment & Interface Actuation'},
    'd03.s01': {'domain': 'd03', 'label': 'Frontend'},
    'd03.s02': {'domain': 'd03', 'label': 'Backend'},
    'd03.s03': {'domain': 'd03', 'label': 'Core Dev Practices'},
    'd03.s04': {'domain': 'd03', 'label': 'Specialized / Vertical Coding'},
    'd04.s01': {'domain': 'd04', 'label': 'Text & Structured Document Understanding'},
    'd04.s02': {'domain': 'd04', 'label': 'Chart, Diagram & Technical Graphic Interpretation'},
    'd04.s03': {'domain': 'd04', 'label': 'Spatial, Object & Scene Reasoning'},
    'd04.s04': {'domain': 'd04', 'label': 'Temporal & Video-Native Understanding'},
    'd04.s05': {'domain': 'd04', 'label': 'Applied & Domain-Specialized Visual Interpretation'},
    'd05.s01': {'domain': 'd05', 'label': 'Image Generation & Editing'},
    'd05.s02': {'domain': 'd05', 'label': 'Video Generation & Editing'},
    'd05.s03': {'domain': 'd05', 'label': '3D & Spatial Asset Generation'},
    'd05.s04': {'domain': 'd05', 'label': 'Audio & Voice Generation'},
    'd05.s05': {'domain': 'd05', 'label': 'Design & Brand Asset Generation'},
    'd06.s01': {'domain': 'd06', 'label': 'Transcription & Recognition'},
    'd06.s02': {'domain': 'd06', 'label': 'Speaker & Voice Analysis'},
    'd06.s03': {'domain': 'd06', 'label': 'Audio Signal Enhancement'},
    'd06.s04': {'domain': 'd06', 'label': 'Non-Speech & Environmental Audio'},
    'd06.s05': {'domain': 'd06', 'label': 'Real-Time Conversational Audio'},
    'd07.s01': {'domain': 'd07', 'label': 'Retrieval & Search Systems'},
    'd07.s02': {'domain': 'd07', 'label': 'Grounded Generation & Knowledge Structuring'},
    'd07.s03': {'domain': 'd07', 'label': 'Query & Database Interfacing'},
    'd07.s04': {'domain': 'd07', 'label': 'Quantitative & Temporal Analysis'},
    'd07.s05': {'domain': 'd07', 'label': 'Data Visualization & Reporting'},
    'd08.s01': {'domain': 'd08', 'label': 'Model Evaluation & Quality Assurance'},
    'd08.s02': {'domain': 'd08', 'label': 'Training & Post-Training Efficiency'},
    'd08.s03': {'domain': 'd08', 'label': 'Inference Optimization & Serving'},
    'd08.s04': {'domain': 'd08', 'label': 'Safety, Alignment & Adversarial Robustness'},
    'd08.s05': {'domain': 'd08', 'label': 'Prompt Engineering & Model Operations'},
}

# Total parameters, not active parameters: these are snapshot capacity estimates.
# GLM-5.2 and DeepSeek-V4-Pro-0813 follow Orbimodels.md's corrected totals.
# Music 3.0's 2B comes from the source's final verification paragraph (line 548).
# Unknown sizes stay unknown; none of these entries certifies a local artifact.
MODEL_PARAMS_B = {
    'Qwen3.5-27B': 27,
    'Qwen3-VL-235B-A22B-Thinking': 235,
    'GLM-5.2': 753,
    'Goedel-Prover-V2-32B': 32,
    'MiMo-V2.5-Pro': 1020,
    'DeepSeek-V4-Pro-0813': 1600,
    'Qwen3-32B': 32,
    'Qwen3-VL-32B': 32,
    'Qwen3.5-397B-A17B': 397,
    'Kimi K3': 2800,
    'Kimi K2.5 Thinking': 1000,
    'Qwen3.8-Max': None,
    'Qwen3-235B-A22B-Instruct-2507': 235,
    'Kimi K2': 1000,
    'Kyutai Moshi': 7,
    'Kimi K2.5': 1000,
    'Qwen3.5-122B-A10B': 122,
    'Kimi K2.6': 1000,
    'DeepSeek-V4-Pro': None,
    'Qwen3-Reranker-0.6B': 0.6,
    'GLM-5.1': None,
    'Skywork-Reward-V2-Llama-3.1-8B': 8,
    'MiniMax M2.1 Preview': 230,
    'Qwen3.6-27B': 27,
    'Codestral-22B-v0.1': 22,
    'Arctic-Text2SQL-R1-7B': 7,
    'GLM-OCR': 0.9,
    'Qwen3.8-27B': 27.78,
    'InternVL3.5-241B-A28B': 240.7,
    'Qwen-Image-2512': 20,
    'Wan 2.7': 27,
    'MiniMax Music 3.0': 2,
    'Step-Audio-EditX': 3,
    'Qwen3-Omni-30B-A3B-Instruct': 30,
    'Qwen3.6-35B-A3B': 35,
    'Qwen3-Coder-480B-A35B-Instruct': 480,
}

LANE_B_MAX_PARAMS_B = 32
# Explicit specialist assignments from Orbimodels.md's Lane B table. In
# particular, queued video remains B; queueing alone does not make a giant lane.
SPECIALIST_B = {"Qwen-Image-2512", "Wan 2.7", "Qwen3-Omni-30B-A3B-Instruct"}

# IDs are persisted identifiers for this snapshot. Preserve them when editing
# a label/model or adding a row; never regenerate IDs by reordering the source.
# id, source line, label, selected model, original pick, min-viable, source status, selection
_ROWS = (
    ('d01.s01.l01', 22, 'syllogistic/deductive validity checking', None, 'N/A — no clear open-weight leader found for this specific skill', None, 'N/A', 'unpicked'),
    ('d01.s01.l02', 23, 'propositional & first-order logic proof construction', 'Qwen3.5-27B', 'Qwen3.5-27B', 'Qwen3.5-27B', '✓ verified', 'min_viable'),
    ('d01.s01.l03', 24, 'constraint-satisfaction puzzle solving (logic grids, Sudoku-style)', 'Qwen3-VL-235B-A22B-Thinking', 'Qwen3-VL-235B-A22B-Thinking', None, '✓ verified', 'original'),
    ('d01.s01.l04', 25, 'probabilistic/Bayesian reasoning', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d01.s01.l05', 26, 'combinatorics & permutation-counting problems', 'GLM-5.2', 'GLM-5.2', None, 'N/A', 'original'),
    ('d01.s01.l06', 27, 'proof-by-contradiction construction', 'Goedel-Prover-V2-32B', 'DeepSeek-Prover-V2-671B', 'Goedel-Prover-V2-32B', '✓ verified', 'min_viable'),
    ('d01.s01.l07', 28, 'symbolic algebra manipulation', None, 'No clear open-weight leader on a dedicated symbolic-algebra benchmark', None, 'N/A', 'unpicked'),
    ('d01.s01.l08', 29, 'multi-step arithmetic word-problem decomposition', 'MiMo-V2.5-Pro', 'MiMo-V2.5-Pro', None, '✓ verified', 'original'),
    ('d01.s02.l01', 35, 'causal chain identification (cause vs correlation)', 'GLM-5.2', 'GLM-5.2', None, 'N/A', 'original'),
    ('d01.s02.l02', 36, "counterfactual scenario simulation ('what if X hadn't happened')", 'DeepSeek-V4-Pro-0813', 'DeepSeek-V4-Pro-0813', None, 'N/A', 'original'),
    ('d01.s02.l03', 37, 'analogical mapping across domains', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d01.s02.l04', 38, 'abductive inference / best-explanation selection', 'Qwen3-32B', 'No single verified open-weight leader — DeepSeek-V3.2, Qwen3-32B, and Llama3.3-70B are roughly co-competitive depending on the specific abductive-selection subtask', 'Qwen3-32B', 'N/A', 'min_viable'),
    ('d01.s02.l05', 39, 'temporal reasoning & event-sequence ordering', None, 'No clear open-weight leader on a dedicated temporal-reasoning/event-ordering benchmark', None, 'N/A', 'unpicked'),
    ('d01.s02.l06', 40, 'spatial relation reasoning from text description', 'Qwen3-VL-32B', 'No single verified open-weight leader', 'Qwen3-VL-32B', 'N/A', 'min_viable'),
    ('d01.s02.l07', 41, 'commonsense physical-plausibility judgment', 'Qwen3.5-397B-A17B', 'Qwen3.5-397B-A17B', None, '✓ verified', 'original'),
    ('d01.s02.l08', 42, 'hypothesis generation from partial evidence', None, 'N/A — no clear open-weight leader for this specific skill', None, 'N/A', 'unpicked'),
    ('d01.s03.l01', 48, 'screenplay/dialogue formatting & subtext writing', 'Kimi K3', 'Kimi K3', None, '✓ verified', 'original'),
    ('d01.s03.l02', 49, 'formal legal-document drafting (contracts, briefs)', 'Kimi K3', 'Kimi K3 (Moonshot AI)', None, 'N/A', 'original'),
    ('d01.s03.l03', 50, 'academic argument structuring & citation-style prose', None, 'N/A — no clear open-weight leader found', None, 'N/A', 'unpicked'),
    ('d01.s03.l04', 51, 'verse/poetry meter & rhyme construction', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d01.s03.l05', 52, 'persuasive rhetorical appeal construction (ethos/pathos/logos)', 'Kimi K2.5 Thinking', 'Kimi K2.5 Thinking', None, '✓ verified', 'original'),
    ('d01.s03.l06', 53, 'technical instruction writing (stepwise procedural clarity)', 'Qwen3.8-Max', 'Qwen3.8-Max', None, '✓ verified', 'original'),
    ('d01.s03.l07', 54, 'epistolary/correspondence tone matching', 'Kimi K3', 'Kimi K3', None, '✓ verified', 'original'),
    ('d01.s03.l08', 55, 'satirical/parody voice construction', 'Qwen3-235B-A22B-Instruct-2507', 'Qwen3-235B-A22B-Instruct-2507', None, '✓ verified', 'original'),
    ('d01.s04.l01', 61, 'sarcasm & irony detection from context', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d01.s04.l02', 62, 'implicature resolution (reading between the lines)', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d01.s04.l03', 63, 'lexical/syntactic ambiguity disambiguation', 'Kimi K3', 'Kimi K3', None, 'N/A', 'original'),
    ('d01.s04.l04', 64, 'register/formality-level adaptation mid-conversation', None, 'N/A — no clear leader', None, 'N/A', 'unpicked'),
    ('d01.s04.l05', 65, 'politeness-strategy generation (hedging, face-saving)', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d01.s04.l06', 66, 'humor/wordplay & pun generation', 'Kimi K2', 'Kimi K2', None, '✓ verified', 'original'),
    ('d01.s04.l07', 67, 'rhetorical-question vs literal-question classification', None, 'N/A (no dedicated benchmark found)', None, 'N/A', 'unpicked'),
    ('d01.s04.l08', 68, 'cultural-reference & idiom interpretation', 'Kimi K3', 'Kimi K3 (reasoned default — no dedicated leaderboard)', None, 'N/A', 'original'),
    ('d01.s05.l01', 74, 'persona consistency maintenance across turns', 'Qwen3.5-27B', 'No clear dedicated leader — best available evidence points to Qwen3.5-27B (open-weight) on general roleplay/persona-adjacent benchmarks', None, 'N/A', 'original'),
    ('d01.s05.l02', 75, 'topic-drift tracking & re-anchoring', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d01.s05.l03', 76, 'clarification-question generation on underspecified input', None, 'none', None, 'N/A', 'unpicked'),
    ('d01.s05.l04', 77, 'contradiction detection across conversation history', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d01.s05.l05', 78, 'turn-taking & interruption-style repair', 'Kyutai Moshi', 'Ultravox v0.7 (fixie-ai/ultravox-v0_7-glm-4_6)', 'Kyutai Moshi', 'N/A', 'min_viable'),
    ('d01.s05.l06', 79, 'user-goal inference from indirect requests', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d01.s05.l07', 80, 'graceful topic-shift & context-switch handling', 'Kimi K2.5', 'Kimi K2.5', None, '✓ verified', 'original'),
    ('d01.s05.l08', 81, 'grounding/confirmation-checking dialogue acts', None, 'N/A — no clear open-weight leader', None, 'N/A', 'unpicked'),
    ('d02.s01.l01', 91, 'single tool-call argument accuracy under ambiguous schemas', 'Qwen3.5-122B-A10B', 'Qwen3.5-397B-A17B', 'Qwen3.5-122B-A10B', '✓ verified', 'min_viable'),
    ('d02.s01.l02', 92, 'parallel/concurrent tool-call batching and result reconciliation', 'Qwen3.5-397B-A17B', 'Qwen3.5-397B-A17B', None, '✓ verified', 'original'),
    ('d02.s01.l03', 93, 'tool selection among near-duplicate tool definitions', 'Qwen3.5-397B-A17B', 'Qwen3.5-397B-A17B', None, 'N/A', 'original'),
    ('d02.s01.l04', 94, 'malformed tool-response recovery (partial JSON, truncated output)', 'Kimi K2.6', 'Kimi K2.6 (Moonshot AI)', None, 'N/A', 'original'),
    ('d02.s01.l05', 95, 'rate-limit/backoff-aware retry scheduling', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d02.s01.l06', 96, 'idempotent action design to avoid duplicate side effects on retry', 'GLM-5.2', 'GLM-5.2 (closest proxy pick — no benchmark targets idempotency specifically)', None, 'N/A', 'original'),
    ('d02.s01.l07', 97, 'tool-call cost/latency budgeting mid-task', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d02.s01.l08', 98, 'cross-tool state consistency (e.g. file written by one tool read by another)', 'Kimi K3', 'Kimi K3 (as a reasoned default, not a benchmarked leader for this specific skill)', None, 'N/A', 'original'),
    ('d02.s02.l01', 104, 'hierarchical task decomposition into executable subgoals', 'Qwen3.5-397B-A17B', 'Qwen3.5-397B-A17B', None, '✓ verified', 'original'),
    ('d02.s02.l02', 105, 'dynamic replanning on unexpected intermediate results', 'DeepSeek-V4-Pro', 'DeepSeek-V4-Pro', None, '✓ verified', 'original'),
    ('d02.s02.l03', 106, 'long-horizon task state tracking across hours/days', 'Kimi K3', 'Kimi K3', None, '✓ verified', 'original'),
    ('d02.s02.l04', 107, 'checkpoint/resume after interruption or context loss', None, 'N/A — no clear open-weight leader', None, 'N/A', 'unpicked'),
    ('d02.s02.l05', 108, 'self-critique and output verification before finalizing a step', None, 'N/A — no clear leader found', None, 'N/A', 'unpicked'),
    ('d02.s02.l06', 109, 'goal-drift detection and course-correction mid-task', 'GLM-5.2', 'GLM-5.2', None, 'N/A', 'original'),
    ('d02.s02.l07', 110, 'stopping-condition judgment (knowing when a task is actually done)', None, 'No clear open-weight leader identified', None, 'N/A', 'unpicked'),
    ('d02.s02.l08', 111, 'budget-aware trade-off between thoroughness and step count', None, 'no clear leader', None, 'N/A', 'unpicked'),
    ('d02.s03.l01', 117, 'cross-session fact write and later recall', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d02.s03.l02', 118, 'episodic task-history summarization for context compression', None, 'No single verifiable open-weight leader for this exact skill', None, 'N/A', 'unpicked'),
    ('d02.s03.l03', 119, 'conflicting-memory resolution (stale vs. updated fact)', 'Qwen3.5-27B', 'Qwen3.5-27B', None, '✓ verified', 'original'),
    ('d02.s03.l04', 120, 'structured memory schema design (what to store vs. discard)', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d02.s03.l05', 121, 'retrieval-triggered memory recall relevance ranking', 'Qwen3-Reranker-0.6B', 'Qwen3-Reranker-8B', 'Qwen3-Reranker-0.6B', '✓ verified', 'min_viable'),
    ('d02.s03.l06', 122, 'working-memory scratchpad management within a single long task', 'GLM-5.1', 'GLM-5.1', None, 'N/A', 'original'),
    ('d02.s04.l01', 128, 'sub-agent spawning with scoped task delegation', 'Kimi K3', 'Kimi K3', None, '✓ verified', 'original'),
    ('d02.s04.l02', 129, 'sub-agent output aggregation and conflict resolution', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d02.s04.l03', 130, 'role/persona assignment consistency across agent instances', 'Qwen3.5-27B', 'Qwen3.5-27B', None, '✓ verified', 'original'),
    ('d02.s04.l04', 131, 'inter-agent message-passing protocol adherence', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d02.s04.l05', 132, 'deadlock/livelock detection in agent-to-agent negotiation', None, 'none — no clear open-weight leader found', None, 'N/A', 'unpicked'),
    ('d02.s04.l06', 133, 'supervisor-agent quality gating of worker-agent output', 'Skywork-Reward-V2-Llama-3.1-8B', 'Skywork-Reward-V2-Llama-3.1-8B', 'Skywork-Reward-V2-Llama-3.1-8B', '✓ verified', 'min_viable'),
    ('d02.s04.l07', 134, 'cost-aware agent-count scaling for a given task', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d02.s05.l01', 140, 'browser DOM element targeting via selectors/accessibility tree', 'Qwen3.5-122B-A10B', 'Qwen3.5-122B-A10B', None, '✓ verified', 'original'),
    ('d02.s05.l02', 141, 'desktop GUI element targeting via screen coordinates/pixel vision', 'Qwen3.5-122B-A10B', 'Qwen3.5-122B-A10B', None, '✓ verified', 'original'),
    ('d02.s05.l03', 142, 'form-fill validation and error-message handling', None, 'none', None, 'N/A', 'unpicked'),
    ('d02.s05.l04', 143, 'API-chaining workflow construction (auth, pagination, schema mapping)', None, 'unknown', None, 'N/A', 'unpicked'),
    ('d02.s05.l05', 144, 'structured output/JSON-schema conformance under constrained decoding', None, 'unknown', None, 'N/A', 'unpicked'),
    ('d02.s05.l06', 145, 'sandboxed code execution as an intermediate reasoning step', None, 'unknown', None, 'N/A', 'unpicked'),
    ('d02.s05.l07', 146, 'file-system operation safety (scoped read/write/delete boundaries)', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d02.s05.l08', 147, 'human-in-the-loop confirmation gating for irreversible actions', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d03.s01.l01', 157, 'browser automation scripting (Playwright/Selenium-style)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s01.l02', 158, 'component framework code generation (React/Vue/Svelte)', 'MiniMax M2.1 Preview', 'Kimi K3', 'MiniMax M2.1 Preview', '✓ verified', 'min_viable'),
    ('d03.s01.l03', 159, 'UI/CSS styling generation', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s01.l04', 160, 'animation and motion code (CSS/JS transitions, gesture-driven)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s01.l05', 161, 'accessibility (a11y) code generation and ARIA auditing', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s01.l06', 162, 'design-to-code translation (Figma/screenshot-to-component)', 'GLM-5.2', 'Kimi K3', 'GLM-5.2', '✓ verified', 'min_viable'),
    ('d03.s01.l07', 163, 'state management code (Redux/Zustand/Pinia-style stores)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s01.l08', 164, 'cross-browser/responsive layout debugging', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s02.l01', 170, 'REST/GraphQL API design and endpoint implementation', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s02.l02', 171, 'database schema design and ORM code generation', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s02.l03', 172, 'distributed systems and microservices code', 'DeepSeek-V4-Pro-0813', 'DeepSeek-V4-Pro-0813', None, '⚠ reasoned default', 'original'),
    ('d03.s02.l04', 173, 'message queue/event-driven code (Kafka/RabbitMQ consumers)', 'DeepSeek-V4-Pro-0813', 'DeepSeek-V4-Pro-0813', None, '⚠ reasoned default', 'original'),
    ('d03.s02.l05', 174, 'authentication/authorization code (OAuth, JWT, RBAC flows)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s02.l06', 175, 'caching layer implementation (Redis, CDN edge logic)', 'DeepSeek-V4-Pro-0813', 'DeepSeek-V4-Pro-0813', None, '⚠ reasoned default', 'original'),
    ('d03.s02.l07', 176, 'server-side rate limiting and load-balancing logic', 'DeepSeek-V4-Pro-0813', 'DeepSeek-V4-Pro-0813', None, '⚠ reasoned default', 'original'),
    ('d03.s02.l08', 177, 'background job/worker scheduling code', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s03.l01', 183, 'debugging and root-cause analysis from stack traces/logs', 'Qwen3.6-27B', 'Kimi K3', 'Qwen3.6-27B', '✓ verified', 'min_viable'),
    ('d03.s03.l02', 184, 'code review comment generation', 'GLM-5.2', 'Kimi K3', 'GLM-5.2', '✓ verified', 'min_viable'),
    ('d03.s03.l03', 185, 'refactoring for readability/structure preservation', 'Kimi K3', 'Kimi K3', None, '✓ verified', 'original'),
    ('d03.s03.l04', 186, 'automated unit/integration test generation', 'Kimi K3', 'Kimi K3', None, '✓ verified', 'original'),
    ('d03.s03.l05', 187, 'documentation generation (docstrings, READMEs, API refs)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s03.l06', 188, 'git commit/PR authoring and changelog generation', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s03.l07', 189, 'CLI tool development and argument-parsing design', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s03.l08', 190, 'fill-in-the-middle autocomplete/inline suggestion', 'Codestral-22B-v0.1', 'Codestral-22B-v0.1', None, '✓ verified (identity only, score unconfirmed)', 'original'),
    ('d03.s03.l09', 191, 'dependency upgrade and breaking-change migration scripting', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s03.l10', 192, 'merge conflict resolution', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s04.l01', 198, 'SQL/database query generation and optimization', 'Arctic-Text2SQL-R1-7B', 'Qwen3-Coder-480B-A35B-Instruct', 'Arctic-Text2SQL-R1-7B', '✓ verified', 'min_viable'),
    ('d03.s04.l02', 199, 'SAST/security vulnerability scanning and patch suggestion', None, 'N/A — no clean open-weight leader', None, 'N/A', 'unpicked'),
    ('d03.s04.l03', 200, 'mobile app development (iOS/Android/React Native/Flutter)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s04.l04', 201, 'embedded/systems programming (C/Rust firmware)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s04.l05', 202, 'game development scripting (Unity/Unreal/gameplay logic)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s04.l06', 203, 'data pipeline/ETL code generation (Airflow/dbt/Spark)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s04.l07', 204, 'DevOps/infra-as-code generation (Docker/Terraform/CI-CD pipelines)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s04.l08', 205, 'performance profiling and hotspot optimization', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s04.l09', 206, 'compiler/interpreter/language-tooling development', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d03.s04.l10', 207, 'blockchain/smart contract code generation and audit', None, 'N/A — no benchmark or known leader found', None, 'N/A', 'unpicked'),
    ('d04.s01.l01', 215, 'printed-text OCR (multi-language, dense small fonts)', 'GLM-OCR', 'GLM-OCR', None, '✓ verified', 'original'),
    ('d04.s01.l02', 216, 'handwriting OCR (cursive, mixed print/cursive, forms)', 'GLM-OCR', 'GLM-OCR', None, '⚠ reasoned default', 'original'),
    ('d04.s01.l03', 217, 'document/PDF layout understanding (reading order, columns, headers/footers)', 'GLM-OCR', 'GLM-OCR', None, '✓ verified', 'original'),
    ('d04.s01.l04', 218, 'table structure extraction (merged cells, nested headers, rotated tables)', 'GLM-OCR', 'GLM-OCR', None, '✓ verified', 'original'),
    ('d04.s01.l05', 219, 'form field extraction (key-value pairs, checkboxes, structured templates)', 'GLM-OCR', 'GLM-OCR', None, '⚠ reasoned default', 'original'),
    ('d04.s01.l06', 220, 'receipt/invoice line-item parsing', 'GLM-OCR', 'GLM-OCR', None, '⚠ reasoned default', 'original'),
    ('d04.s01.l07', 221, 'signature/stamp/seal detection and verification', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s01.l08', 222, 'scene-text recognition in natural photos (street signs, product labels)', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s02.l01', 228, 'bar/line/pie chart value extraction', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s02.l02', 229, 'scatter-plot trend and outlier reading', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s02.l03', 230, 'flowchart/process-diagram logic tracing', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s02.l04', 231, 'circuit/schematic diagram interpretation', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s02.l05', 232, 'architecture/network topology diagram reading', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s02.l06', 233, 'mathematical notation and equation transcription', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s02.l07', 234, 'engineering blueprint/CAD drawing interpretation', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s02.l08', 235, 'infographic multi-panel layout comprehension', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s03.l01', 241, 'open-vocabulary object detection and localization', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d04.s03.l02', 242, 'bounding-box/segmentation mask grounding from text reference', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d04.s03.l03', 243, 'relative spatial reasoning (left-of/behind/stacked-on queries)', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d04.s03.l04', 244, 'object counting under occlusion and clutter', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d04.s03.l05', 245, 'pose/gesture estimation from a single image', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d04.s03.l06', 246, 'scene layout and depth-order reasoning', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d04.s03.l07', 247, 'fine-grained visual attribute comparison (same/different, defect spotting)', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d04.s03.l08', 248, 'referring-expression resolution (pick the one described)', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d04.s04.l01', 254, 'action/event recognition within a clip', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s04.l02', 255, 'temporal ordering and causality across scenes (what happened before/after)', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s04.l03', 256, 'long-video summarization with timestamp grounding', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s04.l04', 257, 'shot/scene-boundary segmentation', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s04.l05', 258, 'moving-object tracking and re-identification across frames', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s04.l06', 259, 'anomaly/change detection between frames or timepoints', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s04.l07', 260, 'video question answering requiring cross-frame memory', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '⚠ reasoned default', 'original'),
    ('d04.s05.l01', 266, 'radiology image reading (X-ray/CT/MRI finding localization)', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d04.s05.l02', 267, 'histopathology slide interpretation', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d04.s05.l03', 268, 'satellite/aerial imagery land-use and change classification', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d04.s05.l04', 269, 'GUI/screenshot element grounding for computer-use agents', 'Qwen3.8-27B', 'Qwen3.8-27B', None, '✓ verified', 'original'),
    ('d04.s05.l05', 270, "app/website state understanding (what screen, what's clickable)", 'Qwen3.8-27B', 'Qwen3.8-27B', None, '✓ verified', 'original'),
    ('d04.s05.l06', 271, 'industrial defect/quality-inspection visual detection', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d04.s05.l07', 272, 'retail planogram and shelf-compliance checking', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d04.s05.l08', 273, 'geolocation inference from image content', 'InternVL3.5-241B-A28B', 'InternVL3.5-241B-A28B', None, '⚠ reasoned default', 'original'),
    ('d05.s01.l01', 281, 'photorealistic image generation', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '✓ verified', 'original'),
    ('d05.s01.l02', 282, 'stylized/artistic image generation', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s01.l03', 283, 'text-to-image with precise prompt adherence (compositional binding)', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s01.l04', 284, 'inpainting/outpainting', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s01.l05', 285, 'image-to-image style transfer', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s01.l06', 286, 'background removal/replacement', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s01.l07', 287, 'super-resolution/upscaling', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s01.l08', 288, 'consistent character/subject generation across frames', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s02.l01', 294, 'text-to-video generation', 'Wan 2.7', 'Wan 2.7', None, '✓ verified', 'original'),
    ('d05.s02.l02', 295, 'image-to-video animation', 'Wan 2.7', 'Wan 2.7', None, '✓ verified', 'original'),
    ('d05.s02.l03', 296, 'video inpainting/object removal', 'Wan 2.7', 'Wan 2.7', None, '⚠ reasoned default', 'original'),
    ('d05.s02.l04', 297, 'video cutting/scene segmentation', 'Wan 2.7', 'Wan 2.7', None, '✓ verified', 'original'),
    ('d05.s02.l05', 298, 'automatic captioning/subtitling', 'Wan 2.7', 'Wan 2.7', None, '✓ verified', 'original'),
    ('d05.s02.l06', 299, 'camera motion/control synthesis', 'Wan 2.7', 'Wan 2.7', None, '⚠ reasoned default', 'original'),
    ('d05.s02.l07', 300, 'video frame interpolation', 'Wan 2.7', 'Wan 2.7', None, '⚠ reasoned default', 'original'),
    ('d05.s02.l08', 301, 'talking-head/lip-sync avatar generation', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s03.l01', 307, '3D mesh generation from text/image', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s03.l02', 308, 'texture/material synthesis', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s03.l03', 309, 'character rigging automation', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s03.l04', 310, 'scene layout/environment generation', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s03.l05', 311, 'point-cloud/NeRF reconstruction', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s03.l06', 312, 'procedural terrain generation', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s04.l01', 318, 'music composition (instrumental)', 'MiniMax Music 3.0', 'MiniMax Music 3.0', None, '✓ verified', 'original'),
    ('d05.s04.l02', 319, 'lyric-conditioned song generation', 'MiniMax Music 3.0', 'MiniMax Music 3.0', None, '✓ verified', 'original'),
    ('d05.s04.l03', 320, 'sound-effect/foley generation', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s04.l04', 321, 'singing voice synthesis', 'MiniMax Music 3.0', 'MiniMax Music 3.0', None, '⚠ reasoned default', 'original'),
    ('d05.s04.l05', 322, 'voice cloning (few-shot speaker adaptation)', 'Step-Audio-EditX', 'Step-Audio-EditX', None, '⚠ reasoned default', 'original'),
    ('d05.s04.l06', 323, 'natural conversational text-to-speech', 'Step-Audio-EditX', 'Step-Audio-EditX', None, '✓ verified', 'original'),
    ('d05.s04.l07', 324, 'emotion/prosody-controllable speech synthesis', 'Step-Audio-EditX', 'Step-Audio-EditX', None, '⚠ reasoned default', 'original'),
    ('d05.s04.l08', 325, 'audio stem separation/remixing', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s05.l01', 331, 'UI mockup/wireframe generation', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s05.l02', 332, 'logo/brand mark generation', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s05.l03', 333, 'icon set generation', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s05.l04', 334, 'marketing/social graphic layout generation', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s05.l05', 335, 'typography/font pairing generation', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d05.s05.l06', 336, 'presentation slide visual design generation', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d05.s05.l07', 337, 'packaging/print layout generation', 'Qwen-Image-2512', 'Qwen-Image-2512', None, '⚠ reasoned default', 'original'),
    ('d06.s01.l01', 345, 'streaming ASR with partial-hypothesis correction', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s01.l02', 346, 'code-switched / multilingual mixed-language transcription', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s01.l03', 347, 'domain-specific vocabulary adaptation (medical, legal, technical jargon)', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s01.l04', 348, 'verbatim vs. clean-read transcription mode control', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s01.l05', 349, 'rare-word and named-entity recall in transcripts', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s01.l06', 350, 'low-resource language ASR', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s01.l07', 351, "children's speech recognition", None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s01.l08', 352, 'far-field / distant-microphone recognition', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s02.l01', 358, 'speaker diarization', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s02.l02', 359, 'speaker identification/verification', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s02.l03', 360, 'voice biometric anti-spoofing / liveness detection', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s02.l04', 361, 'overlapping-speech separation', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s02.l05', 362, 'speaker counting in a recording', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s02.l06', 363, 'voice-based age/gender/demographic estimation', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s02.l07', 364, 'emotion detection from voice', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s02.l08', 365, 'vocal stress/deception cue detection', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s03.l01', 371, 'background noise removal/enhancement', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s03.l02', 372, 'echo and reverberation cancellation', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s03.l03', 373, 'audio source separation (music/vocal isolation, cocktail-party problem)', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s03.l04', 374, 'packet-loss concealment for VoIP audio', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s03.l05', 375, 'audio upsampling/bandwidth extension', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s03.l06', 376, 'clipping/distortion repair', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s03.l07', 377, 'wind/handling-noise suppression for field recordings', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s04.l01', 383, 'sound event detection/classification (glass break, alarm, gunshot)', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s04.l02', 384, 'acoustic scene classification (indoor/outdoor/vehicle context)', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s04.l03', 385, 'audio anomaly detection for industrial/machine monitoring', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s04.l04', 386, 'bioacoustic classification (bird/animal call ID)', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s04.l05', 387, 'audio fingerprinting/content matching', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s04.l06', 388, 'keyword spotting/wake-word detection', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s05.l01', 394, 'real-time low-latency transcription', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s05.l02', 395, 'real-time voice conversation (full-duplex)', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s05.l03', 396, 'accented-speech recognition', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s05.l04', 397, 'turn-taking/barge-in detection in live dialogue', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s05.l05', 398, 'voice activity detection under noisy conditions', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d06.s05.l06', 399, 'prosody/intonation-based intent disambiguation', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d06.s05.l07', 400, 'real-time language identification mid-utterance', 'Qwen3-Omni-30B-A3B-Instruct', 'Qwen3-Omni-30B-A3B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d07.s01.l01', 408, 'dense embedding retrieval quality', 'Qwen3.6-35B-A3B', 'Qwen3.6-35B-A3B', None, '⚠ reasoned default', 'original'),
    ('d07.s01.l02', 409, 'sparse/lexical retrieval (BM25-style) tuning', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d07.s01.l03', 410, 'hybrid retrieval fusion (dense + sparse score merging)', 'Qwen3.6-35B-A3B', 'Qwen3.6-35B-A3B', None, '⚠ reasoned default', 'original'),
    ('d07.s01.l04', 411, 'cross-encoder re-ranking of search results', 'Qwen3.6-35B-A3B', 'Qwen3.6-35B-A3B', None, '⚠ reasoned default', 'original'),
    ('d07.s01.l05', 412, 'query expansion and rewriting', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s01.l06', 413, 'multi-hop/iterative retrieval chaining', 'Qwen3.6-35B-A3B', 'Qwen3.6-35B-A3B', None, '⚠ reasoned default', 'original'),
    ('d07.s01.l07', 414, 'retrieval chunking strategy optimization', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d07.s01.l08', 415, 'vector index recall/latency tradeoff tuning', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d07.s02.l01', 421, 'RAG answer grounding and citation accuracy', 'Qwen3.6-35B-A3B', 'Qwen3.6-35B-A3B', None, '⚠ reasoned default', 'original'),
    ('d07.s02.l02', 422, 'hallucination detection against retrieved context', 'Qwen3.6-35B-A3B', 'Qwen3.6-35B-A3B', None, '⚠ reasoned default', 'original'),
    ('d07.s02.l03', 423, 'structured data extraction from unstructured text (schema-constrained)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s02.l04', 424, 'named entity recognition and entity linking', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s02.l05', 425, 'relation extraction for knowledge graph construction', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s02.l06', 426, 'knowledge graph schema/ontology design', 'DeepSeek-V4-Pro-0813', 'DeepSeek-V4-Pro-0813', None, '⚠ reasoned default', 'original'),
    ('d07.s02.l07', 427, 'coreference resolution across documents', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s02.l08', 428, 'table-to-structured-record extraction from documents', 'GLM-OCR', 'GLM-OCR', None, '⚠ reasoned default', 'original'),
    ('d07.s03.l01', 434, 'natural-language-to-SQL query generation', 'Qwen3-Coder-480B-A35B-Instruct', 'Qwen3-Coder-480B-A35B-Instruct', None, '✓ verified', 'original'),
    ('d07.s03.l02', 435, 'natural-language-to-SQL query self-repair', 'Qwen3-Coder-480B-A35B-Instruct', 'Qwen3-Coder-480B-A35B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d07.s03.l03', 436, 'NoSQL/document-store query generation', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s03.l04', 437, 'schema linking (mapping NL terms to table/column names)', 'Qwen3-Coder-480B-A35B-Instruct', 'Qwen3-Coder-480B-A35B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d07.s03.l05', 438, 'multi-table join reasoning from natural language', 'Qwen3-Coder-480B-A35B-Instruct', 'Qwen3-Coder-480B-A35B-Instruct', None, '⚠ reasoned default', 'original'),
    ('d07.s03.l06', 439, 'query plan/execution cost estimation from NL intent', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d07.s03.l07', 440, 'API/GraphQL query construction from natural language', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s04.l01', 446, 'time-series forecasting (trend/seasonality modeling)', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d07.s04.l02', 447, 'anomaly/outlier detection in streaming data', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d07.s04.l03', 448, 'tabular data statistical analysis (aggregation, grouping, pivoting)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s04.l04', 449, 'cohort/funnel analysis from event data', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s04.l05', 450, 'causal inference and A/B test result interpretation', 'DeepSeek-V4-Pro-0813', 'DeepSeek-V4-Pro-0813', None, '⚠ reasoned default', 'original'),
    ('d07.s04.l06', 451, 'data cleaning and schema normalization', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s04.l07', 452, 'missing-data imputation strategy selection', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d07.s05.l01', 458, 'chart-type selection for a given data shape', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s05.l02', 459, 'chart/plot code generation (matplotlib/plotly/d3-style)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s05.l03', 460, 'dashboard layout composition from multiple metrics', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s05.l04', 461, 'statistical result narration into plain-language insight', 'DeepSeek-V4-Pro-0813', 'DeepSeek-V4-Pro-0813', None, '⚠ reasoned default', 'original'),
    ('d07.s05.l05', 462, 'automated report generation from query results', 'DeepSeek-V4-Pro-0813', 'DeepSeek-V4-Pro-0813', None, '⚠ reasoned default', 'original'),
    ('d07.s05.l06', 463, 'interactive filter/drill-down spec generation', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d07.s05.l07', 464, 'color/encoding accessibility in generated visualizations', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d08.s01.l01', 474, 'LLM-as-judge calibration and bias correction', 'GLM-5.2', 'GLM-5.2', None, '⚠ reasoned default', 'original'),
    ('d08.s01.l02', 475, 'pairwise preference ranking (Elo/Bradley-Terry) for model comparison', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s01.l03', 476, 'hallucination detection and factuality scoring', 'GLM-5.2', 'GLM-5.2', None, '⚠ reasoned default', 'original'),
    ('d08.s01.l04', 477, 'benchmark contamination/data-leakage detection', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s01.l05', 478, 'cross-version regression testing for model updates', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s01.l06', 479, 'rubric-based automated grading pipelines', 'GLM-5.2', 'GLM-5.2', None, '⚠ reasoned default', 'original'),
    ('d08.s01.l07', 480, 'confidence calibration and uncertainty estimation', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s01.l08', 481, 'inter-rater reliability measurement between human and AI graders', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s02.l01', 487, 'LoRA/QLoRA fine-tuning configuration and rank selection', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s02.l02', 488, 'RLHF/RLAIF reward model design', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s02.l03', 489, 'DPO/preference-optimization pipeline construction', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s02.l04', 490, 'synthetic training-data generation and filtering', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d08.s02.l05', 491, 'knowledge distillation from larger to smaller models', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s02.l06', 492, 'pretraining data-mixture and curriculum design', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s02.l07', 493, 'catastrophic forgetting mitigation during fine-tuning', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s02.l08', 494, 'sparse upcycling (dense-to-MoE conversion)', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s03.l01', 500, 'post-training quantization (GGUF/AWQ/GPTQ) for hardware minimization', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s03.l02', 501, 'KV-cache management and paged attention tuning', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s03.l03', 502, 'speculative decoding draft-model pairing', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s03.l04', 503, 'model routing (task-to-model selection under cost/latency constraints)', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s03.l05', 504, 'continuous batching and throughput tuning', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s03.l06', 505, 'MoE expert-streaming/offload scheduling', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s03.l07', 506, 'prompt/context caching for repeated-prefix workloads', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s03.l08', 507, 'serving-framework configuration (vLLM/SGLang/TensorRT-LLM)', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s04.l01', 513, 'jailbreak-resistance and safety-alignment tuning', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d08.s04.l02', 514, 'red-teaming and adversarial prompt discovery', 'GLM-5.2', 'GLM-5.2', None, '⚠ reasoned default', 'original'),
    ('d08.s04.l03', 515, 'prompt-injection defense for tool-using agents', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s04.l04', 516, 'guardrail and content-filter policy design', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s04.l05', 517, 'refusal-behavior/abliteration detection and auditing', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d08.s04.l06', 518, 'AI-content watermarking and provenance detection', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d08.s04.l07', 519, 'bias and fairness auditing across demographic slices', None, 'N/A', None, 'N/A', 'unpicked'),
    ('d08.s04.l08', 520, 'differential-privacy enforcement in model training', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s05.l01', 526, 'automated prompt optimization/auto-prompting', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d08.s05.l02', 527, 'system-prompt versioning and A/B testing', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s05.l03', 528, 'few-shot exemplar selection and ordering', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s05.l04', 529, 'structured-output/schema-constrained generation reliability', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
    ('d08.s05.l05', 530, 'model/version canary rollout and rollback', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s05.l06', 531, 'cost-per-token budget monitoring and alerting', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s05.l07', 532, 'context-window packing and truncation strategy', None, 'N/A — technique/tool category, not a model-choice skill', None, 'N/A', 'unpicked'),
    ('d08.s05.l08', 533, 'agent-loop failure-mode diagnosis (infinite loops, tool-call drift)', 'Kimi K3', 'Kimi K3', None, '⚠ reasoned default', 'original'),
)

def _skill(row):
    identity, line, label, model, original, alternative, source_status, selection = row
    subdomain = identity.rsplit(".", 1)[0]
    domain = SUBDOMAINS[subdomain]["domain"]
    size = MODEL_PARAMS_B.get(model)
    if model is None:
        lane, reason = None, "No model selected in the source; leave unassigned."
    elif model in SPECIALIST_B:
        lane, reason = "B", "Explicit specialist in Orbimodels.md Lane B; local availability unverified."
    elif size is not None and size <= LANE_B_MAX_PARAMS_B:
        # ponytail: reasoned <=32B/roughly16GB4bit capacity default; unload A and qualify before serving.
        lane, reason = "B", "Reasoned <=32B capacity default; fit unmeasured, requires unloading Lane A."
    else:
        lane, reason = "C", "Reasoned giant/unknown-size default; local capacity and availability unverified."
    return dict(id=identity, domain=domain, domain_label=DOMAINS[domain],
                subdomain=subdomain, subdomain_label=SUBDOMAINS[subdomain]["label"],
                label=label, model=model, original_pick=original, min_viable=alternative,
                source_status=source_status, status="research_snapshot" if model else "unpicked",
                selection=selection, source=SOURCE, source_line=line,
                params_b=size, lane=lane, lane_reason=reason)


SKILLS = tuple(_skill(row) for row in _ROWS)
BY_ID = {skill["id"]: skill for skill in SKILLS}


def self_check(source=None):
    """Check the complete catalog, optionally against the read-only source snapshot."""
    from collections import Counter
    assert len(SKILLS) == len(BY_ID) == 302 and len(DOMAINS) == 8 and len(SUBDOMAINS) == 39
    assert len({skill["source_line"] for skill in SKILLS}) == 302
    assert list(Counter(skill["domain"] for skill in SKILLS).values()) == [40, 37, 36, 39, 37, 36, 37, 40]
    assert max(Counter(skill["subdomain"] for skill in SKILLS).values()) == 10
    assert sum(skill["source_status"].startswith("✓ verified") for skill in SKILLS) == 46
    assert sum(skill["source_status"] == "⚠ reasoned default" for skill in SKILLS) == 125
    assert sum(skill["source_status"] == "N/A" for skill in SKILLS) == 131
    assert sum(skill["selection"] == "min_viable" for skill in SKILLS) == 13
    for skill in SKILLS:
        assert skill["domain"] in DOMAINS and skill["subdomain"] in SUBDOMAINS
        assert skill["source"] == SOURCE and 1 <= skill["source_line"] <= 559
        assert skill["label"] and skill["original_pick"]
        if skill["min_viable"] is not None:
            assert skill["model"] == skill["min_viable"]
        if skill["model"] is None:
            assert skill["lane"] is None and skill["status"] == "unpicked"
        else:
            assert skill["model"] in MODEL_PARAMS_B and skill["status"] == "research_snapshot"
            assert skill["lane"] in ("B", "C")
            assert "unverified" in skill["lane_reason"] or "unmeasured" in skill["lane_reason"]
    assert BY_ID["d03.s04.l01"]["model"] == "Arctic-Text2SQL-R1-7B"
    assert BY_ID["d03.s04.l01"]["lane"] == "B"
    assert BY_ID["d03.s01.l02"]["model"] == "MiniMax M2.1 Preview"
    assert BY_ID["d03.s01.l02"]["lane"] == "C"
    assert BY_ID["d01.s02.l04"]["source_status"] == "N/A"  # A named alt does not upgrade evidence.
    assert BY_ID["d02.s05.l01"]["model"] == "Qwen3.5-122B-A10B"  # Unverified Fara mention is not an alt.
    if source is not None:
        import hashlib
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
        lines = raw.decode().splitlines()
        leaves = [line for line in lines if line.startswith("| ") and not line.startswith("| Leaf skill")]
        assert len(leaves) == len(SKILLS)
        assert raw.count(b"**Min-viable alt:**") == 56
        for skill in SKILLS:
            cells = [cell.strip() for cell in lines[skill["source_line"] - 1].strip("|").split("|")]
            assert (cells[0], cells[1], cells[4]) == (skill["label"], skill["original_pick"], skill["source_status"])
            if skill["min_viable"]:
                assert "**Min-viable alt:** " + skill["min_viable"] in cells[3]


if __name__ == "__main__":
    from pathlib import Path
    source = Path(__file__).resolve().parent.parent / SOURCE
    self_check(source if source.is_file() else None)
    print("PASS: 302 leaves, 8 domains, 39 subdomains, 13 explicit alternatives; "
          + ("all source rows and SHA verified." if source.is_file() else "vault absent; structural check only."))
