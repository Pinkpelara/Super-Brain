# chatgbt
# Super-Brain Strict RAG Upgrade (Patch Layer)

This repository now contains:

- `chatbot.html` (full single-file chatbot HTML with inline strict RAG enhancer)
- `super_brain_rag_patch.js` (optional external drop-in patch file)

`chatbot.html` is now fully self-contained; it does not require external patch loading.

---

## What this patch improves

### 1) Full-corpus activation every query
- Forces broader retrieval behavior each turn.
- Adds per-document anchor sampling to avoid narrow top-k keyword bias.
- Ensures stronger cross-document coverage in retrieved evidence.

### 2) Meaning-level linking (not just keyword overlap)
- Builds an additional concept graph from chunk co-occurrence.
- Expands retrieval via implicit concept bridges.
- Keeps semantic reranking in the loop when embeddings are available.

### 3) Explicit internal mental model before answering
- Builds a structured model:
  - entities
  - concepts
  - relationships
  - constraints
  - assumptions
  - evidence gaps
- Injects this model into the system context before final answer generation.

### 4) Strict evidence-grounded RAG behavior
- Enforces source-only factual grounding.
- Preserves continuity-safe session context for reference resolution.
- Prevents prior assistant turns from being treated as factual evidence.

### 5) Stronger claim/citation repair
- Adds an extra repair trigger if citation density is low, uncertainty handling is missing, or claim diagnostics indicate weak grounding.
- Reinforces required sectioning:
  - Mental Model
  - Evidence-Based Answer
  - Uncertainties & Missing Information

### 6) Iterative retrieval + evidence sufficiency guard
- Runs iterative retrieval query expansions until retrieval coverage stabilizes.
- Enforces broader cross-source coverage for broad/comparative/decision questions.
- If evidence is still insufficient, explicitly requests missing documents/data instead of guessing.

### 7) Continuity-safe session handling
- Keeps conversational continuity for follow-up references.
- Prevents prior assistant turns from becoming factual evidence in fresh reasoning passes.

### 8) Hard citation/structure quality gate
- Adds a stricter claim-level gate (citation coverage, support ratio, citation precision, and required section checks).
- Attempts automatic repair when gate checks fail.
- If gate failures persist, the assistant refuses to over-claim and asks for specific missing evidence/documents.

### 9) Benchmark + reliability evaluation harness
- Adds expert-style benchmark tasks across cognitive categories (cross-source decisions, conflict resolution, constraint planning, gap detection).
- Adds recall/coverage validation over large corpora (distinct-doc spread, coverage %, dominant-source concentration).
- Adds decision-quality scoring and comparison against configurable human baseline metrics.
- Tracks failure rates directly:
  - unsupported claims,
  - uncited claims,
  - citation errors,
  - source-dominance risk,
  - missing trade-off/decision reasoning,
  - hallucination-risk indicator.

### 10) Hard anti-summarizer cognitive-depth gate
- Adds a synthesis-time gate (not only post-hoc eval) that rejects summary-style responses.
- Enforces mandatory expert reasoning structure:
  - Mental Model
  - Human Cognitive Processing Loop (evidence intake, attention filtering, perception/interpretation, dual-process reasoning, decision/action, feedback)
  - Evidence-Based Expert Analysis
  - (for decision queries) Options & Trade-offs, Recommendation, Risks
  - Uncertainties & Missing Information
- Requires explicit heuristic/bias checks and mitigation notes to reduce shallow one-shot conclusions.
- Runs iterative rewrite/repair passes when reasoning depth is weak, sections are missing, or cross-source decision reasoning is shallow.

### 11) Adversarial + structured analytical reasoning
- Adds adversarial retrieval ("devil's advocate") passes that actively search for contradictory/limiting evidence.
- Forces inclusion of an explicit contradiction line in final reasoning:
  - `While the primary evidence suggests X, the corpus also contains evidence that contradicts/modifies this conclusion: [Citation].`
- Requires structured analytical frameworks and logic skeleton sections:
  - Analytical Framework
  - Explicitly Stated Facts
  - Inferred Implications
  - Perspective Analysis (when stakeholder conflicts are relevant)
- Adds perspective-taking evidence maps to simulate multi-stakeholder viewpoints in policy/decision questions.

---

## Wiring guarantees (current build)

The current implementation hard-wires previously fragile features:

- Core `synthesize()` now always runs a strict cognitive post-processing gate unless an outer super-brain wrapper intentionally takes over that step.
- Completion reliability features are connected in runtime:
  - response caching (TTL + bounded size),
  - fallback model chain on transient API failures,
  - telemetry for cache hits and fallback usage.
- Chat restore/render path now escapes user/system content before HTML injection points to prevent stored HTML/script injection.
- Stop-word token sets are reused instead of recreated per call in hot paths.

---

## Human-cognition alignment update

The cognitive layer now emphasizes **latent processing behavior** rather than forcing visible keyword headers.

What changed:

- Reasoning/depth gates now score:
  - direct answer clarity,
  - evidence integration,
  - counterevidence handling,
  - uncertainty calibration,
  - decision trade-off quality,
  instead of requiring fixed section titles.
- Repair rewrites now request **natural expert communication** (answer first, rationale, limits), not mandatory scaffold blocks.
- Prompt policy now asks the model to run an internal cognitive loop:
  - recognize the decision/problem,
  - gather and filter relevant evidence,
  - evaluate options and conflicts,
  - execute recommendation logic,
  - reflect with uncertainty + feedback.

This is closer to the decision/reasoning principles described in the linked human-cognition/problem-solving references and avoids turning cognition into a header template.

---

## Nature paper inspiration

This patch’s cognition prior is explicitly aligned with:

**A foundation model to predict and capture human cognition**  
https://www.nature.com/articles/s41586-025-09215-4

In practical terms, the patch emphasizes:
- latent task abstraction over surface wording,
- cross-context generalization through relation-level structure,
- evidence-constrained synthesis loops.

---

## Integration

Open `chatbot.html` directly in the browser and configure your API key in the Admin Panel.

If you want to patch another existing HTML implementation, use `super_brain_rag_patch.js` as an optional runtime layer.

---

## Notes

- This is an **improvement layer**, not a rewrite.
- It expects your base code structure and method names to remain close to your provided implementation.
