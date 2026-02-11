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

### 4) Strict fresh-pass RAG behavior
- Enforces source-only answering.
- Excludes prior chat memory from reasoning pass (fresh build per question).
- Adds stronger prompt constraints to avoid hidden-knowledge style responses.

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
