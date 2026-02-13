/*
 * Super-Brain RAG Patch
 * ---------------------------------
 * Drop-in upgrade for the existing chatbot HTML script.
 *
 * This file intentionally DOES NOT replace your current implementation.
 * It monkey-patches the existing classes to improve:
 * - full-corpus activation on each query,
 * - semantic/concept bridge retrieval,
 * - explicit mental-model construction before answer generation,
 * - strict source grounding with continuity-safe context handling,
 * - stronger claim-level citation and uncertainty repair.
 *
 * Nature inspiration reference:
 * https://www.nature.com/articles/s41586-025-09215-4
 */
(function superBrainRagPatch() {
    'use strict';

    const PATCH_VERSION = '2026.02.11-superbrain-v6';
    const FULL_SWEEP_MIN_ANCHORS_PER_DOC = 2;
    const DEFAULT_TARGET_K_SMALL = 22;
    const DEFAULT_TARGET_K_MEDIUM = 30;
    const DEFAULT_TARGET_K_LARGE = 42;
    const MAX_MERGED_MULTIPLIER = 3;
    const MAX_LINK_EXPANSION = 28;
    const MAX_MENTAL_MODEL_SOURCES = 36;
    const MAX_EVIDENCE_DIGEST_CHARS = 14000;
    const MIN_CITATIONS_FOR_RICH_CORPUS = 4;
    const MAX_RETRIEVAL_REFINEMENT_PASSES = 4;
    const RETRIEVAL_STABLE_CHUNK_DELTA = 3;
    const MAX_MODEL_REFINEMENT_PASSES = 3;
    const MODEL_GAP_THRESHOLD = 6;
    const MAX_GAP_REFINEMENT_QUERIES = 4;
    const MAX_ANALYSIS_REWRITE_ATTEMPTS = 2;
    const MIN_DECISION_REASONING_SIGNALS = 4;
    const MIN_ANALYSIS_REASONING_SIGNALS = 2;
    const MIN_HUMAN_COGNITION_STAGE_SIGNALS = 4;
    const MIN_HUMAN_COGNITION_STAGE_SIGNALS_DECISION = 5;
    const MAX_ADVERSARIAL_RETRIEVAL_PASSES = 3;
    const MAX_ADVERSARIAL_CHUNKS = 16;
    const MAX_PREDICTIVE_HYPOTHESES = 5;
    const MIN_CROSS_SOURCE_DOCS = 2;

    const COGNITION_PRIOR = [
        'Abstract latent task structure from surface wording.',
        'Generalize across cover stories via shared concept relations.',
        'Prioritize relation-level semantic links over lexical overlap.',
        'Verify final claims against retrieved source evidence only.'
    ];

    function uniqBy(items, keyFn) {
        const seen = new Set();
        const out = [];
        for (const item of items || []) {
            const key = keyFn(item);
            if (key === undefined || key === null) continue;
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(item);
        }
        return out;
    }

    function clamp(v, lo, hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    function safeLower(text) {
        return String(text || '').toLowerCase();
    }

    function normalizeConcept(text) {
        return String(text || '')
            .trim()
            .toLowerCase()
            .replace(/[^\w\s-]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }

    function summarizeActivation(report) {
        if (!report) return '';
        const concepts = (report.bridgeConcepts || []).slice(0, 8)
            .map(item => `${item.concept} (${item.docs} docs)`)
            .join(', ');
        return [
            `## FULL-CORPUS ACTIVATION REPORT`,
            `- Activated documents in retrieval: ${report.activatedDocuments}/${report.totalDocuments}`,
            `- Activated chunk count: ${report.activatedChunks}/${report.totalChunks}`,
            `- Activation coverage: ${report.activationCoveragePct}%`,
            `- Retrieval passes: ${report.retrievalPasses || 1} (stabilized: ${report.retrievalStabilized ? 'yes' : 'no'})`,
            `- Cross-document bridge concepts: ${concepts || 'N/A'}`
        ].join('\n');
    }

    function patchMethod(klass, methodName, wrapFactory) {
        if (!klass || !klass.prototype) return false;
        const original = klass.prototype[methodName];
        if (typeof original !== 'function') return false;
        if (original.__superBrainPatchVersion === PATCH_VERSION) return true;

        const wrapped = wrapFactory(original);
        if (typeof wrapped !== 'function') return false;
        wrapped.__superBrainPatchVersion = PATCH_VERSION;
        klass.prototype[methodName] = wrapped;
        return true;
    }

    function installRetrieverMethods(CognitiveRetriever) {
        if (!CognitiveRetriever || !CognitiveRetriever.prototype) return;

        if (typeof CognitiveRetriever.prototype.extractRichConcepts !== 'function') {
            CognitiveRetriever.prototype.extractRichConcepts = function extractRichConcepts(text) {
                const source = String(text || '');
                if (!source) return [];

                const concepts = new Set();

                const named = source.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b/g) || [];
                for (const phrase of named) {
                    const v = normalizeConcept(phrase);
                    if (v.length >= 3) concepts.add(v);
                }

                const quoted = source.match(/"([^"]{3,90})"/g) || [];
                for (const q of quoted) {
                    const v = normalizeConcept(q.replace(/"/g, ''));
                    if (v.length >= 3) concepts.add(v);
                }

                const tech = source.match(/\b[a-z]+(?:[-_][a-z0-9]+)+\b/gi) || [];
                for (const t of tech) {
                    const v = normalizeConcept(t);
                    if (v.length >= 3) concepts.add(v);
                }

                const tokens = typeof this.tokenize === 'function'
                    ? this.tokenize(source)
                    : normalizeConcept(source).split(/\s+/).filter(Boolean);
                const freq = new Map();
                for (const token of tokens) {
                    if (token.length < 4) continue;
                    freq.set(token, (freq.get(token) || 0) + 1);
                }
                const topTokens = Array.from(freq.entries())
                    .filter(([, count]) => count >= 2)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 18)
                    .map(([token]) => token);
                for (const token of topTokens) concepts.add(token);

                return Array.from(concepts).slice(0, 28);
            };
        }

        if (typeof CognitiveRetriever.prototype.buildSuperConceptGraph !== 'function') {
            CognitiveRetriever.prototype.buildSuperConceptGraph = function buildSuperConceptGraph() {
                this.superConceptGraph = new Map();
                this.conceptToChunkIds = new Map();
                this.chunkConceptsCache = new Map();
                this.chunkToDocMap = new Map();

                const addEdge = (a, b, weight) => {
                    if (!a || !b || a === b) return;
                    if (!this.superConceptGraph.has(a)) this.superConceptGraph.set(a, new Map());
                    const neighbors = this.superConceptGraph.get(a);
                    neighbors.set(b, (neighbors.get(b) || 0) + weight);
                };

                for (const chunk of this.allChunks || []) {
                    const conceptList = this.extractRichConcepts(chunk.text).slice(0, 18);
                    this.chunkConceptsCache.set(chunk.id, conceptList);
                    this.chunkToDocMap.set(chunk.id, chunk.filename);

                    for (const concept of conceptList) {
                        if (!this.conceptToChunkIds.has(concept)) {
                            this.conceptToChunkIds.set(concept, new Set());
                        }
                        this.conceptToChunkIds.get(concept).add(chunk.id);
                    }

                    for (let i = 0; i < conceptList.length; i++) {
                        for (let j = i + 1; j < conceptList.length; j++) {
                            const distance = Math.abs(i - j) + 1;
                            const w = 1 / distance;
                            addEdge(conceptList[i], conceptList[j], w);
                            addEdge(conceptList[j], conceptList[i], w);
                        }
                    }
                }
            };
        }

        if (typeof CognitiveRetriever.prototype.getDocumentAnchors !== 'function') {
            CognitiveRetriever.prototype.getDocumentAnchors = function getDocumentAnchors(query, intent = {}) {
                const docs = this.documents || [];
                if (!docs.length) return [];

                const tokens = typeof this.tokenize === 'function'
                    ? this.tokenize(query)
                    : normalizeConcept(query).split(/\s+/).filter(Boolean);
                const entities = typeof this.extractEntities === 'function' ? this.extractEntities(query) : [];
                const concepts = typeof this.extractConcepts === 'function' ? this.extractConcepts(query) : [];
                const anchors = [];

                for (const doc of docs) {
                    const chunks = (doc.chunks || [])
                        .map(chunk => this.chunkLookup?.get(chunk.id) || chunk)
                        .filter(Boolean);
                    if (!chunks.length) continue;

                    const scored = chunks.map(chunk => {
                        const score = typeof this.calculateRelevanceScore === 'function'
                            ? this.calculateRelevanceScore(chunk, query, tokens, entities, concepts)
                            : 0;
                        return { ...chunk, score };
                    }).sort((a, b) => b.score - a.score);

                    const topRelevantCount = intent.broadCoverage ? 2 : 1;
                    anchors.push(...scored.slice(0, topRelevantCount));

                    const ordered = chunks.slice().sort((a, b) => (a.index || 0) - (b.index || 0));
                    const coverageCount = Math.max(
                        FULL_SWEEP_MIN_ANCHORS_PER_DOC,
                        intent.broadCoverage ? 3 : 2
                    );
                    const distributed = typeof this.pickEvenlyDistributed === 'function'
                        ? this.pickEvenlyDistributed(ordered, coverageCount)
                        : ordered.slice(0, coverageCount);
                    anchors.push(...distributed);
                }

                return uniqBy(anchors, item => item.id);
            };
        }

        if (typeof CognitiveRetriever.prototype.expandImplicitConceptLinks !== 'function') {
            CognitiveRetriever.prototype.expandImplicitConceptLinks = function expandImplicitConceptLinks(seedChunks, query, limit = MAX_LINK_EXPANSION) {
                const graph = this.superConceptGraph || new Map();
                const c2c = this.conceptToChunkIds || new Map();
                if (!graph.size || !c2c.size || !seedChunks?.length) return [];

                const queryTokens = typeof this.tokenize === 'function'
                    ? this.tokenize(query)
                    : normalizeConcept(query).split(/\s+/).filter(Boolean);
                const queryEntities = typeof this.extractEntities === 'function' ? this.extractEntities(query) : [];
                const queryConcepts = typeof this.extractConcepts === 'function' ? this.extractConcepts(query) : [];

                const seedIds = new Set(seedChunks.map(chunk => chunk.id));
                const seedConceptWeight = new Map();
                for (const chunk of seedChunks) {
                    const concepts = this.chunkConceptsCache?.get(chunk.id)
                        || this.extractRichConcepts(chunk.text);
                    const base = (chunk.score || 0.2) + 0.25;
                    for (const concept of concepts) {
                        seedConceptWeight.set(concept, (seedConceptWeight.get(concept) || 0) + base);
                    }
                }

                const topSeedConcepts = Array.from(seedConceptWeight.entries())
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 14);

                const candidateConcepts = new Map();
                for (const [concept, weight] of topSeedConcepts) {
                    candidateConcepts.set(concept, Math.max(candidateConcepts.get(concept) || 0, weight));
                    const neighbors = graph.get(concept);
                    if (!neighbors) continue;
                    const topNeighbors = Array.from(neighbors.entries())
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 5);
                    for (const [neighbor, edgeWeight] of topNeighbors) {
                        const combined = (weight * 0.65) + (edgeWeight * 0.35);
                        candidateConcepts.set(neighbor, Math.max(candidateConcepts.get(neighbor) || 0, combined));
                    }
                }

                const candidates = [];
                for (const [concept, bridgeWeight] of candidateConcepts.entries()) {
                    const chunkIds = c2c.get(concept);
                    if (!chunkIds) continue;
                    for (const chunkId of chunkIds) {
                        if (seedIds.has(chunkId)) continue;
                        const chunk = this.chunkLookup?.get(chunkId);
                        if (!chunk) continue;
                        const lexical = typeof this.calculateRelevanceScore === 'function'
                            ? this.calculateRelevanceScore(chunk, query, queryTokens, queryEntities, queryConcepts)
                            : 0;
                        const score = (lexical * 0.72) + (Math.min(bridgeWeight, 5) * 0.08);
                        candidates.push({ ...chunk, score, bridgeConcept: concept });
                    }
                }

                const uniqueCandidates = uniqBy(
                    candidates.sort((a, b) => b.score - a.score),
                    item => item.id
                );
                return uniqueCandidates.slice(0, limit);
            };
        }

        if (typeof CognitiveRetriever.prototype.enforceDocumentCoverage !== 'function') {
            CognitiveRetriever.prototype.enforceDocumentCoverage = function enforceDocumentCoverage(results, anchors, topK) {
                const out = uniqBy(results || [], item => item.id);
                const seenDocs = new Set(out.map(item => item.filename));
                const targetDocs = Math.min(
                    this.documents?.length || 0,
                    Math.max(1, Math.ceil((topK || out.length || 1) * 0.7))
                );

                for (const anchor of anchors || []) {
                    if (out.length >= (topK || out.length)) break;
                    if (seenDocs.has(anchor.filename)) continue;
                    out.push(anchor);
                    seenDocs.add(anchor.filename);
                    if (seenDocs.size >= targetDocs) break;
                }

                for (const anchor of anchors || []) {
                    if (out.length >= (topK || out.length)) break;
                    if (out.some(item => item.id === anchor.id)) continue;
                    out.push(anchor);
                }

                return out.slice(0, topK || out.length);
            };
        }

        if (typeof CognitiveRetriever.prototype.buildCorpusActivationReport !== 'function') {
            CognitiveRetriever.prototype.buildCorpusActivationReport = function buildCorpusActivationReport(query, selectedChunks) {
                const totalDocs = this.documents?.length || 0;
                const totalChunks = this.allChunks?.length || 0;
                const activatedDocs = new Set((selectedChunks || []).map(chunk => chunk.filename));
                const activatedChunks = selectedChunks?.length || 0;
                const coverage = totalDocs ? Math.round((activatedDocs.size / totalDocs) * 100) : 0;

                const bridgeConcepts = [];
                const conceptToDocs = new Map();
                for (const [concept, chunkIds] of (this.conceptToChunkIds || new Map()).entries()) {
                    const docs = new Set();
                    for (const chunkId of chunkIds) {
                        const doc = this.chunkToDocMap?.get(chunkId);
                        if (doc) docs.add(doc);
                    }
                    if (docs.size > 1) {
                        conceptToDocs.set(concept, docs.size);
                    }
                }

                for (const [concept, docs] of Array.from(conceptToDocs.entries())
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 10)) {
                    bridgeConcepts.push({ concept, docs });
                }

                return {
                    query,
                    totalDocuments: totalDocs,
                    totalChunks,
                    activatedDocuments: activatedDocs.size,
                    activatedChunks,
                    activationCoveragePct: coverage,
                    bridgeConcepts
                };
            };
        }

        if (typeof CognitiveRetriever.prototype.superBrainResolveQueryFromSession !== 'function') {
            CognitiveRetriever.prototype.superBrainResolveQueryFromSession = function superBrainResolveQueryFromSession(query) {
                const raw = String(query || '').trim();
                if (!raw) return raw;
                const hasReference = /\b(it|this|that|those|these|they|them|previous|same one|compare it|what about)\b/i.test(raw);
                if (!hasReference) return raw;
                const history = (typeof window !== 'undefined' && window.app?.synthesizer?.conversationHistory)
                    ? window.app.synthesizer.conversationHistory
                    : [];
                if (!Array.isArray(history) || !history.length) return raw;
                const recentUser = history.slice().reverse().find(item => item?.role === 'user' && String(item?.content || '').trim());
                if (!recentUser) return raw;
                const hint = String(recentUser.content || '').replace(/\s+/g, ' ').trim().slice(0, 240);
                if (!hint) return raw;
                return `${raw}\n\n[Continuity hint from prior user turn: ${hint}]`;
            };
        }

        if (typeof CognitiveRetriever.prototype.superBrainSanitizeRetrievalQuery !== 'function') {
            CognitiveRetriever.prototype.superBrainSanitizeRetrievalQuery = function superBrainSanitizeRetrievalQuery(query) {
                const raw = String(query || '').replace(/\s+/g, ' ').trim();
                if (!raw) return raw;
                const diagnosticMarkers = /\b(example output|for troubleshooting|troubleshoot|debug|not to tailor|what is going on)\b/i.test(raw);
                if (!diagnosticMarkers) return raw;
                const blockStart = String(query || '').search(/\n##\s+[A-Za-z]/);
                if (blockStart > 0) {
                    const head = String(query || '').slice(0, blockStart).replace(/\s+/g, ' ').trim();
                    if (head) return head;
                }
                return raw;
            };
        }

        if (typeof CognitiveRetriever.prototype.superBrainBuildIntentGraph !== 'function') {
            CognitiveRetriever.prototype.superBrainBuildIntentGraph = function superBrainBuildIntentGraph(query) {
                const normalized = String(query || '').replace(/\s+/g, ' ').trim();
                const entities = typeof this.extractEntities === 'function' ? this.extractEntities(normalized).slice(0, 20) : [];
                const concepts = typeof this.extractConcepts === 'function' ? this.extractConcepts(normalized).slice(0, 24) : [];
                const constraints = (normalized.match(/\b(must|cannot|only|required|unless|except|without|at least|at most)\b[^.?!;]{0,80}/gi) || []).slice(0, 10);
                const relations = (normalized.match(/\b(cause|impact|depends on|related to|trade[- ]?off|compare|versus|vs)\b[^.?!;]{0,80}/gi) || []).slice(0, 10);
                const unknowns = (normalized.match(/\b(uncertain|unknown|missing|need|evidence|prove|validate)\b[^.?!;]{0,80}/gi) || []).slice(0, 10);
                return { raw: normalized, entities, concepts, constraints, relations, unknowns };
            };
        }

        if (typeof CognitiveRetriever.prototype.superBrainBuildPredictiveModel !== 'function') {
            CognitiveRetriever.prototype.superBrainBuildPredictiveModel = function superBrainBuildPredictiveModel(query, intentGraph = null, intent = {}) {
                const graph = intentGraph || this.superBrainBuildIntentGraph(query);
                const raw = String(query || '').replace(/\s+/g, ' ').trim();
                const hypotheses = [];
                const pushHypothesis = (type, statement, confidence = 0.5) => {
                    const cleaned = String(statement || '').replace(/\s+/g, ' ').trim();
                    if (!cleaned) return;
                    hypotheses.push({ type, statement: cleaned, confidence, nonFactualPrior: true });
                };

                pushHypothesis('baseline_rule', `Test hypothesis: identify the governing baseline rule(s) for "${raw}".`, 0.58);
                if (graph.constraints?.length) {
                    pushHypothesis('constraint', `Test hypothesis: constraints materially narrow feasible options (${graph.constraints.slice(0, 2).join('; ')}).`, 0.63);
                }
                if (graph.relations?.length || intent.comparative) {
                    pushHypothesis('tradeoff', 'Test hypothesis: competing factors require explicit trade-off analysis instead of one-rule resolution.', 0.66);
                }
                if (/\b(late|deadline|exception|override|appeal|special case)\b/i.test(raw)) {
                    pushHypothesis('exception', 'Test hypothesis: exception/override clauses may exist and must be verified directly in source text.', 0.72);
                }
                if (/\b(policy|manual|guideline|rule|regulation|procedure)\b/i.test(raw)) {
                    pushHypothesis('hierarchy', 'Test hypothesis: precedence may be resolved by explicit hierarchy language (specificity/authority/recency) if citations confirm it.', 0.76);
                }

                const weightingPriors = {
                    authority: 'Prefer official, policy-grade, and source-of-record documents over summaries.',
                    recency: 'Prefer newer versions where temporal conflict exists.',
                    contextualRelevance: 'Prefer clauses that match the exact scenario and constraints from the query.'
                };
                const anticipatedConclusion = hypotheses.length
                    ? `Provisional non-factual prior for retrieval planning: ${hypotheses.slice(0, 2).map(item => item.statement).join(' ')}`
                    : `Provisional non-factual prior for retrieval planning only; no conclusion until evidence is cited for: ${raw}`;

                return {
                    query: raw,
                    anticipatedConclusion,
                    hypotheses: uniqBy(hypotheses, item => `${item.type}:${item.statement}`).slice(0, MAX_PREDICTIVE_HYPOTHESES),
                    expectedEvidenceNeeds: [
                        'baseline rule text',
                        'exception/override clauses',
                        'conflict resolution precedence statements',
                        'implementation/action constraints'
                    ],
                    weightingPriors
                };
            };
        }

        if (typeof CognitiveRetriever.prototype.superBrainBuildProbeQueries !== 'function') {
            CognitiveRetriever.prototype.superBrainBuildProbeQueries = function superBrainBuildProbeQueries(query, intentGraph = null, intent = {}, predictiveModel = null) {
                const graph = intentGraph || this.superBrainBuildIntentGraph(query);
                const predictive = predictiveModel || this.superBrainBuildPredictiveModel(query, graph, intent);
                const probes = [];
                const push = (value) => {
                    const cleaned = String(value || '').replace(/\s+/g, ' ').trim();
                    if (cleaned) probes.push(cleaned);
                };
                push(`${query} semantic evidence relationships constraints`);
                if (graph.entities.length) push(`${query} ${graph.entities.slice(0, 4).join(' ')} evidence across documents`);
                if (graph.concepts.length) push(`${query} ${graph.concepts.slice(0, 5).join(' ')} implicit links`);
                if (graph.constraints.length) push(`${query} constraints exceptions conflicts`);
                if (graph.relations.length || intent.comparative) push(`${query} compare trade-offs and dependencies`);
                if (intent.timeline) push(`${query} chronology sequence and milestones`);
                if (graph.unknowns.length) push(`${query} missing evidence validation`);
                if (predictive?.hypotheses?.length) {
                    for (const hypothesis of predictive.hypotheses.slice(0, 3)) {
                        push(`${query} validate hypothesis: ${hypothesis.statement}`);
                    }
                }
                push(`${query} specific clause that overrides general rule`);
                return uniqBy(probes, item => normalizeConcept(item)).slice(0, MAX_RETRIEVAL_REFINEMENT_PASSES * 2);
            };
        }

        if (typeof CognitiveRetriever.prototype.superBrainDeepScanAugment !== 'function') {
            CognitiveRetriever.prototype.superBrainDeepScanAugment = function superBrainDeepScanAugment(query, selectedChunks = [], targetTopK = DEFAULT_TARGET_K_SMALL) {
                const out = Array.isArray(selectedChunks) ? selectedChunks.slice() : [];
                const seen = new Set(out.map(item => item.id));
                const docNames = uniqBy(out.map(item => item.filename).filter(Boolean), item => item);
                if (!docNames.length) return out;
                const qTokens = typeof this.tokenize === 'function' ? this.tokenize(query) : normalizeConcept(query).split(/\s+/).filter(Boolean);
                const qEntities = typeof this.extractEntities === 'function' ? this.extractEntities(query) : [];
                const qConcepts = typeof this.extractConcepts === 'function' ? this.extractConcepts(query) : [];

                for (const filename of docNames) {
                    const docChunks = (this.allChunks || [])
                        .filter(item => item.filename === filename)
                        .sort((a, b) => (a.index || 0) - (b.index || 0));
                    if (docChunks.length < 4) continue;

                    const midIdx = Math.floor(docChunks.length / 2);
                    const lateIdx = Math.floor(docChunks.length * 0.8);
                    const candidates = [docChunks[midIdx], docChunks[lateIdx], docChunks[docChunks.length - 1]]
                        .filter(Boolean)
                        .map(chunk => ({
                            ...chunk,
                            score: typeof this.calculateRelevanceScore === 'function'
                                ? this.calculateRelevanceScore(chunk, query, qTokens, qEntities, qConcepts)
                                : 0
                        }))
                        .sort((a, b) => (b.score || 0) - (a.score || 0));

                    for (const candidate of candidates) {
                        if (out.length >= targetTopK) break;
                        if (!candidate?.id || seen.has(candidate.id)) continue;
                        out.push(candidate);
                        seen.add(candidate.id);
                    }
                    if (out.length >= targetTopK) break;
                }

                return out.slice(0, targetTopK);
            };
        }

        if (typeof CognitiveRetriever.prototype.superBrainRunRetrievalRefinement !== 'function') {
            CognitiveRetriever.prototype.superBrainRunRetrievalRefinement = async function superBrainRunRetrievalRefinement(query, seedChunks = [], topK = DEFAULT_TARGET_K_SMALL, options = {}, intent = {}, originalRetrieve = null) {
                if (typeof originalRetrieve !== 'function') {
                    return {
                        chunks: seedChunks || [],
                        passes: 1,
                        stabilized: true,
                        intentGraph: this.superBrainBuildIntentGraph(query),
                        predictiveModel: this.superBrainBuildPredictiveModel(query, this.superBrainBuildIntentGraph(query), intent)
                    };
                }

                const intentGraph = this.superBrainBuildIntentGraph(query);
                const predictiveModel = this.superBrainBuildPredictiveModel(query, intentGraph, intent);
                const probes = this.superBrainBuildProbeQueries(query, intentGraph, intent, predictiveModel);
                let merged = Array.isArray(seedChunks) ? seedChunks.slice() : [];
                let passes = 1;
                let stableHits = 0;
                let prevChunkCount = merged.length;
                let prevDocCount = new Set(merged.map(item => item.filename)).size;

                for (const probe of probes) {
                    if (passes >= MAX_RETRIEVAL_REFINEMENT_PASSES) break;
                    let passBase = [];
                    try {
                        passBase = await originalRetrieve.call(
                            this,
                            probe,
                            Math.max(18, Math.ceil(topK * 0.8)),
                            { ...options, intent: { ...intent, broadCoverage: true } }
                        );
                    } catch {
                        passBase = [];
                    }
                    const anchors = typeof this.getDocumentAnchors === 'function' ? this.getDocumentAnchors(probe, { ...intent, broadCoverage: true }) : [];
                    const links = typeof this.expandImplicitConceptLinks === 'function'
                        ? this.expandImplicitConceptLinks(passBase || [], probe, Math.max(8, Math.floor(MAX_LINK_EXPANSION * 0.8)))
                        : [];
                    const mergedProbe = typeof this.mergeUniqueChunks === 'function'
                        ? this.mergeUniqueChunks(
                            this.mergeUniqueChunks(passBase || [], anchors || [], Math.max(topK * 2, 90)),
                            links || [],
                            Math.max(topK * 3, 120)
                        )
                        : uniqBy([...(passBase || []), ...(anchors || []), ...(links || [])], item => item.id);
                    merged = typeof this.mergeUniqueChunks === 'function'
                        ? this.mergeUniqueChunks(merged, mergedProbe, Math.max(topK * 4, 170))
                        : uniqBy([...merged, ...mergedProbe], item => item.id);

                    const chunkCount = merged.length;
                    const docCount = new Set(merged.map(item => item.filename)).size;
                    const chunkDelta = chunkCount - prevChunkCount;
                    const docDelta = docCount - prevDocCount;
                    if (docDelta <= 0 && chunkDelta <= RETRIEVAL_STABLE_CHUNK_DELTA) stableHits += 1;
                    else stableHits = 0;
                    prevChunkCount = chunkCount;
                    prevDocCount = docCount;
                    passes += 1;
                    if (stableHits >= 1) break;
                }

                return {
                    chunks: merged,
                    passes,
                    stabilized: stableHits >= 1,
                    intentGraph,
                    predictiveModel
                };
            };
        }

        if (typeof CognitiveRetriever.prototype.superBrainContradictionSignalScore !== 'function') {
            CognitiveRetriever.prototype.superBrainContradictionSignalScore = function superBrainContradictionSignalScore(text) {
                const value = safeLower(text);
                if (!value) return 0;
                const patterns = [
                    /\bhowever\b/g,
                    /\bbut\b/g,
                    /\bexcept\b/g,
                    /\bunless\b/g,
                    /\balthough\b/g,
                    /\bdespite\b/g,
                    /\blimit(?:ation|ed|s)?\b/g,
                    /\bconstraint(?:s)?\b/g,
                    /\brisk(?:s)?\b/g,
                    /\bcounter(?:example|evidence)?\b/g,
                    /\bcontradict(?:s|ion|ory)?\b/g,
                    /\bfail(?:ure|s|ed|ing)?\b/g
                ];
                let score = 0;
                for (const pattern of patterns) {
                    const matches = value.match(pattern);
                    if (matches?.length) score += Math.min(2, matches.length) * 0.5;
                }
                return Math.min(6, score);
            };
        }

        if (typeof CognitiveRetriever.prototype.superBrainBuildAdversarialQueries !== 'function') {
            CognitiveRetriever.prototype.superBrainBuildAdversarialQueries = function superBrainBuildAdversarialQueries(query, seedChunks = [], intentGraph = null, predictiveModel = null) {
                const graph = intentGraph || this.superBrainBuildIntentGraph(query);
                const predictive = predictiveModel || this.superBrainBuildPredictiveModel(query, graph, {});
                const probes = [];
                const push = (value) => {
                    const cleaned = String(value || '').replace(/\s+/g, ' ').trim();
                    if (!cleaned) return;
                    probes.push(cleaned);
                };
                push(`${query} contradictory evidence limitations exceptions`);
                push(`${query} counterexamples boundary conditions`);
                push(`${query} risks unintended consequences failure modes`);
                if (graph.constraints?.length) push(`${query} constraints violated edge cases`);
                if (graph.relations?.length) push(`${query} alternative causal explanation contradictory evidence`);
                const seedConcepts = uniqBy(
                    (seedChunks || []).flatMap(chunk => typeof this.extractRichConcepts === 'function'
                        ? this.extractRichConcepts(chunk?.text || '')
                        : []),
                    item => normalizeConcept(item)
                ).slice(0, 6);
                if (seedConcepts.length) {
                    push(`${query} ${seedConcepts.join(' ')} contradictions exceptions`);
                }
                if (predictive?.anticipatedConclusion) {
                    push(`${query} evidence that challenges this expectation: ${predictive.anticipatedConclusion}`);
                }
                return uniqBy(probes, item => normalizeConcept(item)).slice(0, MAX_ADVERSARIAL_RETRIEVAL_PASSES + 2);
            };
        }

        if (typeof CognitiveRetriever.prototype.superBrainRunAdversarialRetrieval !== 'function') {
            CognitiveRetriever.prototype.superBrainRunAdversarialRetrieval = async function superBrainRunAdversarialRetrieval(
                query,
                seedChunks = [],
                options = {},
                intent = {},
                originalRetrieve = null
            ) {
                if (typeof originalRetrieve !== 'function') {
                    return { chunks: [], passes: 0, probes: [] };
                }
                const intentGraph = this.superBrainBuildIntentGraph(query);
                const predictiveModel = options?.predictiveModel || this.superBrainBuildPredictiveModel(query, intentGraph, intent);
                const probes = this.superBrainBuildAdversarialQueries(query, seedChunks, intentGraph, predictiveModel);
                let passes = 0;
                let aggregated = [];
                const qTokens = typeof this.tokenize === 'function' ? this.tokenize(query) : normalizeConcept(query).split(/\s+/).filter(Boolean);
                const qEntities = typeof this.extractEntities === 'function' ? this.extractEntities(query) : [];
                const qConcepts = typeof this.extractConcepts === 'function' ? this.extractConcepts(query) : [];

                for (const probe of probes) {
                    if (passes >= MAX_ADVERSARIAL_RETRIEVAL_PASSES) break;
                    passes += 1;
                    let retrieved = [];
                    try {
                        retrieved = await originalRetrieve.call(
                            this,
                            probe,
                            Math.max(DEFAULT_TARGET_K_SMALL, 18),
                            { ...options, intent: { ...intent, broadCoverage: true, comparative: true } }
                        );
                    } catch {
                        retrieved = [];
                    }
                    const anchors = typeof this.getDocumentAnchors === 'function'
                        ? this.getDocumentAnchors(probe, { ...intent, broadCoverage: true, comparative: true })
                        : [];
                    const mergedProbe = typeof this.mergeUniqueChunks === 'function'
                        ? this.mergeUniqueChunks(retrieved || [], anchors || [], 96)
                        : uniqBy([...(retrieved || []), ...(anchors || [])], item => item.id);
                    const scoredProbe = (mergedProbe || []).map(chunk => {
                        const lexical = chunk?.score || (typeof this.calculateRelevanceScore === 'function'
                            ? this.calculateRelevanceScore(chunk, query, qTokens, qEntities, qConcepts)
                            : 0);
                        const contradictionSignal = this.superBrainContradictionSignalScore(chunk?.text || '');
                        const score = lexical * 0.72 + Math.min(5, contradictionSignal) * 0.09;
                        return { ...chunk, score, adversarialSignal: contradictionSignal, adversarialProbe: probe };
                    }).sort((a, b) => (b.score || 0) - (a.score || 0));

                    aggregated = typeof this.mergeUniqueChunks === 'function'
                        ? this.mergeUniqueChunks(aggregated, scoredProbe, 220)
                        : uniqBy([...(aggregated || []), ...(scoredProbe || [])], item => item.id);
                }

                const prioritized = uniqBy(
                    aggregated
                        .filter(item => (item?.adversarialSignal || 0) > 0)
                        .sort((a, b) => ((b.adversarialSignal || 0) - (a.adversarialSignal || 0)) || ((b.score || 0) - (a.score || 0))),
                    item => item?.id
                );
                const selected = prioritized.length
                    ? prioritized.slice(0, MAX_ADVERSARIAL_CHUNKS)
                    : uniqBy((aggregated || []).sort((a, b) => (b.score || 0) - (a.score || 0)), item => item?.id).slice(0, Math.min(MAX_ADVERSARIAL_CHUNKS, 8));
                return {
                    chunks: selected,
                    passes,
                    probes
                };
            };
        }
    }

    function installSynthesizerMethods(CognitiveSynthesizer) {
        if (!CognitiveSynthesizer || !CognitiveSynthesizer.prototype) return;

        if (typeof CognitiveSynthesizer.prototype.superBrainFlattenSources !== 'function') {
            CognitiveSynthesizer.prototype.superBrainFlattenSources = function superBrainFlattenSources(relevantChunks, attachmentContext = null) {
                const corpusSources = (relevantChunks || []).map(chunk => ({
                    ...chunk,
                    filename: chunk.filename || 'unknown',
                    scope: 'corpus'
                }));
                const attachmentSources = (attachmentContext?.chunks || []).map(chunk => ({
                    ...chunk,
                    filename: attachmentContext.name || chunk.filename || 'attachment',
                    scope: 'attachment'
                }));
                return uniqBy([...attachmentSources, ...corpusSources], item => item.id || `${item.filename}:${item.index}`);
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainTokenize !== 'function') {
            CognitiveSynthesizer.prototype.superBrainTokenize = function superBrainTokenize(text) {
                const stop = new Set([
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                    'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'it',
                    'this', 'that', 'these', 'those', 'can', 'could', 'should', 'would', 'may',
                    'might', 'will', 'also', 'than', 'then', 'into', 'over', 'under'
                ]);
                return safeLower(text)
                    .replace(/[^\w\s-]/g, ' ')
                    .split(/\s+/)
                    .filter(token => token.length > 2 && !stop.has(token));
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildEvidenceDigest !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildEvidenceDigest = function superBrainBuildEvidenceDigest(sources) {
                const selected = (sources || []).slice(0, MAX_MENTAL_MODEL_SOURCES);
                const lines = [];
                let chars = 0;
                for (const source of selected) {
                    const page = source.page ? `page ${source.page}` : 'page N/A';
                    const excerpt = String(source.text || '').replace(/\s+/g, ' ').trim().slice(0, 320);
                    const line = `- [${source.filename}, ${page}] ${excerpt}`;
                    if (chars + line.length > MAX_EVIDENCE_DIGEST_CHARS) break;
                    lines.push(line);
                    chars += line.length;
                }
                return lines.join('\n');
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainHeuristicModel !== 'function') {
            CognitiveSynthesizer.prototype.superBrainHeuristicModel = function superBrainHeuristicModel(query, sources) {
                const text = (sources || []).map(s => s.text || '').join('\n');
                const entities = Array.from(new Set(
                    (text.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b/g) || [])
                        .map(item => normalizeConcept(item))
                        .filter(Boolean)
                )).slice(0, 14);

                const tokens = this.superBrainTokenize(text);
                const freq = new Map();
                for (const token of tokens) {
                    freq.set(token, (freq.get(token) || 0) + 1);
                }
                const concepts = Array.from(freq.entries())
                    .filter(([, count]) => count >= 3)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 18)
                    .map(([token]) => token);

                const relationSentences = String(text).split(/(?<=[.!?])\s+/)
                    .filter(sentence => /\b(causes?|leads to|results? in|depends on|requires?|blocks?|enables?|increases?|decreases?)\b/i.test(sentence))
                    .slice(0, 8)
                    .map(sentence => sentence.trim().slice(0, 220));

                const constraints = String(text).split(/(?<=[.!?])\s+/)
                    .filter(sentence => /\b(must|required|cannot|only|at least|at most|limit|constraint)\b/i.test(sentence))
                    .slice(0, 8)
                    .map(sentence => sentence.trim().slice(0, 220));

                const assumptions = String(text).split(/(?<=[.!?])\s+/)
                    .filter(sentence => /\b(assume|assuming|likely|may|might|unclear)\b/i.test(sentence))
                    .slice(0, 6)
                    .map(sentence => sentence.trim().slice(0, 220));

                const queryTokens = this.superBrainTokenize(query);
                const evidenceTokenSet = new Set(tokens);
                const gaps = queryTokens
                    .filter(token => !evidenceTokenSet.has(token))
                    .slice(0, 10)
                    .map(token => `Insufficient direct evidence for concept: "${token}"`);

                return {
                    entities,
                    concepts,
                    relationships: relationSentences.map(sentence => ({ statement: sentence })),
                    constraints,
                    assumptions,
                    gaps
                };
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainNormalizeModel !== 'function') {
            CognitiveSynthesizer.prototype.superBrainNormalizeModel = function superBrainNormalizeModel(parsed, fallback) {
                const base = fallback || {
                    entities: [],
                    concepts: [],
                    relationships: [],
                    constraints: [],
                    assumptions: [],
                    gaps: []
                };
                if (!parsed || typeof parsed !== 'object') return base;
                return {
                    entities: Array.isArray(parsed.entities) ? parsed.entities.slice(0, 16) : base.entities,
                    concepts: Array.isArray(parsed.concepts) ? parsed.concepts.slice(0, 20) : base.concepts,
                    relationships: Array.isArray(parsed.relationships) ? parsed.relationships.slice(0, 12) : base.relationships,
                    constraints: Array.isArray(parsed.constraints) ? parsed.constraints.slice(0, 10) : base.constraints,
                    assumptions: Array.isArray(parsed.assumptions) ? parsed.assumptions.slice(0, 10) : base.assumptions,
                    gaps: Array.isArray(parsed.gaps) ? parsed.gaps.slice(0, 10) : base.gaps
                };
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainFormatModelBlock !== 'function') {
            CognitiveSynthesizer.prototype.superBrainFormatModelBlock = function superBrainFormatModelBlock(model) {
                const relLines = (model.relationships || []).map(rel => {
                    if (typeof rel === 'string') return `- ${rel}`;
                    const s = rel.subject ? String(rel.subject) : '';
                    const p = rel.predicate ? String(rel.predicate) : '';
                    const o = rel.object ? String(rel.object) : '';
                    const stmt = rel.statement ? String(rel.statement) : '';
                    if (stmt) return `- ${stmt}`;
                    if (s || p || o) return `- ${s} ${p} ${o}`.trim();
                    return null;
                }).filter(Boolean);

                return [
                    '## PRE-BUILT MENTAL MODEL (STRICT SOURCE-GROUNDED)',
                    `- Entities: ${(model.entities || []).join(', ') || 'N/A'}`,
                    `- Concepts: ${(model.concepts || []).join(', ') || 'N/A'}`,
                    '- Relationships:',
                    ...(relLines.length ? relLines : ['- N/A']),
                    '- Constraints:',
                    ...((model.constraints || []).length ? model.constraints.map(x => `- ${x}`) : ['- N/A']),
                    '- Assumptions:',
                    ...((model.assumptions || []).length ? model.assumptions.map(x => `- ${x}`) : ['- N/A']),
                    '- Evidence gaps:',
                    ...((model.gaps || []).length ? model.gaps.map(x => `- ${x}`) : ['- N/A'])
                ].join('\n');
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildModel !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildModel = async function superBrainBuildModel(query, relevantChunks, attachmentContext = null, corpusStats = {}) {
                const sources = this.superBrainFlattenSources(relevantChunks, attachmentContext);
                const heuristic = this.superBrainHeuristicModel(query, sources);
                if (!this.apiKey || !sources.length) {
                    return heuristic;
                }

                const evidenceDigest = this.superBrainBuildEvidenceDigest(sources);
                const activationReport = corpusStats.activationReport ? summarizeActivation(corpusStats.activationReport) : '';

                const messages = [
                    {
                        role: 'system',
                        content: `You are an internal cognition module for strict RAG.
Return ONLY valid JSON with keys:
- entities: string[]
- concepts: string[]
- relationships: array (strings OR objects {subject,predicate,object,evidence})
- constraints: string[]
- assumptions: string[]
- gaps: string[]
Rules:
- Use only provided evidence.
- Do not include outside knowledge.
- Keep items concise and non-redundant.`
                    },
                    {
                        role: 'user',
                        content: `Question:
${query}

${activationReport}

Evidence:
${evidenceDigest}`
                    }
                ];

                try {
                    const raw = await this.callCompletion(messages, {
                        temperature: 0.1,
                        max_tokens: 900,
                        top_p: 0.9
                    });
                    const parsed = this.safeJsonParse(raw);
                    return this.superBrainNormalizeModel(parsed, heuristic);
                } catch {
                    return heuristic;
                }
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBestSentence !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBestSentence = function superBrainBestSentence(text, queryTokens = []) {
                const sentences = String(text || '')
                    .split(/(?<=[.!?])\s+/)
                    .map(s => s.trim())
                    .filter(Boolean);
                if (!sentences.length) return String(text || '').slice(0, 220);
                if (!queryTokens.length) return sentences[0].slice(0, 220);
                let best = sentences[0];
                let bestScore = -1;
                for (const sentence of sentences) {
                    const lower = safeLower(sentence);
                    const score = queryTokens.reduce((acc, token) => acc + (lower.includes(token) ? 1 : 0), 0);
                    if (score > bestScore) {
                        best = sentence;
                        bestScore = score;
                    }
                }
                return best.slice(0, 240);
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildAdversarialEvidenceBlock !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildAdversarialEvidenceBlock = function superBrainBuildAdversarialEvidenceBlock(query, adversarialState = null, maxItems = 8) {
                const adversarialChunks = Array.isArray(adversarialState?.chunks)
                    ? adversarialState.chunks
                    : (Array.isArray(adversarialState) ? adversarialState : []);
                const passes = Number(adversarialState?.passes || 0);
                if (!adversarialChunks.length || passes <= 0) return '';
                const queryTokens = this.superBrainTokenize(query);
                const lines = [
                    '## ADVERSARIAL RETRIEVAL (DEVIL\'S ADVOCATE)',
                    '- Secondary retrieval pass explicitly searched for disconfirming evidence, exceptions, and limitations.',
                    `- Adversarial retrieval passes: ${passes}`
                ];
                let added = 0;
                for (const chunk of adversarialChunks.slice(0, Math.max(3, maxItems))) {
                    const sentence = this.superBrainBestSentence(chunk.text || '', queryTokens).replace(/\s+/g, ' ').trim();
                    if (!sentence) continue;
                    const quote = sentence.slice(0, 110).replace(/"/g, '\'');
                    lines.push(`- ${sentence.slice(0, 220)} [Source: ${chunk.filename || 'Unknown'}, page ${chunk.page || 'N/A'}, "${quote || 'relevant quote'}"]`);
                    added += 1;
                    if (added >= maxItems) break;
                }
                lines.push('- Include contradiction sentence only if disconfirming evidence is explicitly cited.');
                return lines.join('\n');
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildPerspectiveEvidenceBlock !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildPerspectiveEvidenceBlock = function superBrainBuildPerspectiveEvidenceBlock(
                query,
                sources = [],
                instructionProfile = {}
            ) {
                const perspectiveMode = !!instructionProfile?.perspectiveMode || /\b(stakeholder|policy|should we|implement|impact|conflict|trade-?off|goals?)\b/i.test(String(query || ''));
                if (!perspectiveMode) return '';

                const candidates = [
                    { name: 'Operations', regex: /\b(operation|process|workflow|timeline|resource|capacity|implementation)\b/i },
                    { name: 'Risk/Compliance', regex: /\b(risk|compliance|regulation|legal|safety|policy|governance)\b/i },
                    { name: 'Financial', regex: /\b(cost|budget|expense|roi|price|funding|financial)\b/i },
                    { name: 'User/Stakeholder', regex: /\b(user|customer|student|patient|employee|stakeholder|experience)\b/i }
                ];
                const queryTokens = this.superBrainTokenize(query);
                const lines = ['## PERSPECTIVE EVIDENCE MAP'];
                const usedSources = new Set();
                let perspectiveCount = 0;

                for (const candidate of candidates) {
                    let chosen = null;
                    for (const source of (sources || [])) {
                        if (!source?.text || usedSources.has(source.id)) continue;
                        if (!candidate.regex.test(String(source.text))) continue;
                        chosen = source;
                        break;
                    }
                    if (!chosen) continue;
                    usedSources.add(chosen.id);
                    const sentence = this.superBrainBestSentence(chosen.text || '', queryTokens).replace(/\s+/g, ' ').trim();
                    if (!sentence) continue;
                    const quote = sentence.slice(0, 110).replace(/"/g, '\'');
                    lines.push(`- From Perspective ${candidate.name}, the corpus highlights: ${sentence.slice(0, 220)} [Source: ${chosen.filename || 'Unknown'}, page ${chosen.page || 'N/A'}, "${quote || 'relevant quote'}"]`);
                    perspectiveCount += 1;
                    if (perspectiveCount >= 3) break;
                }

                if (perspectiveCount < 2) {
                    const fallback = uniqBy(
                        (sources || []).filter(item => item?.text).slice(0, 6),
                        item => item?.id || `${item?.filename || 'unknown'}:${item?.index || 0}`
                    );
                    for (let i = 0; i < fallback.length && perspectiveCount < 2; i++) {
                        const source = fallback[i];
                        const label = perspectiveCount === 0 ? 'A' : 'B';
                        const sentence = this.superBrainBestSentence(source.text || '', queryTokens).replace(/\s+/g, ' ').trim();
                        if (!sentence) continue;
                        const quote = sentence.slice(0, 110).replace(/"/g, '\'');
                        lines.push(`- From Perspective ${label}, the corpus highlights: ${sentence.slice(0, 220)} [Source: ${source.filename || 'Unknown'}, page ${source.page || 'N/A'}, "${quote || 'relevant quote'}"]`);
                        perspectiveCount += 1;
                    }
                }

                if (perspectiveCount < 2) return '';
                lines.push('- Use these perspectives to analyze conflicts in goals before final recommendation.');
                return lines.join('\n');
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainEvidenceHierarchyWeight !== 'function') {
            CognitiveSynthesizer.prototype.superBrainEvidenceHierarchyWeight = function superBrainEvidenceHierarchyWeight(source, queryTokens = []) {
                const filename = safeLower(source?.filename || '');
                const text = String(source?.text || '');
                const textLower = safeLower(text);
                let authority = 0.45;
                if (/\b(policy|manual|regulation|official|handbook|governance|guideline|o?fr)\b/.test(filename)) authority = 0.9;
                else if (/\b(procedure|protocol|standard|framework)\b/.test(filename)) authority = 0.78;
                else if (/\b(faq|summary|notes|announcement|memo|blog)\b/.test(filename)) authority = 0.56;

                const yearMatches = filename.match(/\b(19|20)\d{2}\b/g) || text.match(/\b(19|20)\d{2}\b/g) || [];
                const years = yearMatches.map(y => parseInt(y, 10)).filter(Number.isFinite);
                const latestYear = years.length ? Math.max(...years) : null;
                const nowYear = new Date().getFullYear();
                let recency = 0.58;
                if (latestYear) {
                    const delta = Math.max(0, nowYear - latestYear);
                    recency = Math.max(0.35, 1 - (delta / 14));
                }

                let relevance = 0.4;
                if (queryTokens.length) {
                    const overlap = queryTokens.filter(token => textLower.includes(token)).length;
                    relevance = Math.min(1, (overlap / Math.max(1, queryTokens.length)) + 0.18);
                }
                const weighted = (authority * 0.42) + (recency * 0.24) + (relevance * 0.34);
                return { authority, recency, relevance, weighted, latestYear };
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildPredictiveResolutionBlock !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildPredictiveResolutionBlock = function superBrainBuildPredictiveResolutionBlock(
                query,
                predictiveModel = null,
                sources = []
            ) {
                if (!predictiveModel) return '';
                const queryTokens = this.superBrainTokenize(query);
                const hypotheses = Array.isArray(predictiveModel.hypotheses) ? predictiveModel.hypotheses : [];
                const selectedSources = (sources || []).slice(0, 26);
                const support = [];
                const contradiction = [];
                for (const source of selectedSources) {
                    const sentence = this.superBrainBestSentence(source?.text || '', queryTokens).replace(/\s+/g, ' ').trim();
                    if (!sentence) continue;
                    const lower = safeLower(sentence);
                    const contradictionSignal = /\b(however|but|except|unless|despite|contradict|override|supersede|limit)\b/.test(lower);
                    const quote = sentence.slice(0, 110).replace(/"/g, '\'');
                    const line = `${sentence.slice(0, 220)} [Source: ${source.filename || 'Unknown'}, page ${source.page || 'N/A'}, "${quote || 'relevant quote'}"]`;
                    if (contradictionSignal) contradiction.push(line);
                    else support.push(line);
                }

                const lines = [
                    '## Predictive Model (Pre-Evidence Expectation)',
                    '- Predictive hypothesis status: provisional non-factual prior used only to guide retrieval.',
                    `- Anticipated hypothesis set (not factual claims): ${predictiveModel.anticipatedConclusion || 'N/A'}`,
                    '- Hypotheses:'
                ];
                if (hypotheses.length) {
                    for (const item of hypotheses.slice(0, MAX_PREDICTIVE_HYPOTHESES)) {
                        lines.push(`- (${item.type}) ${item.statement}`);
                    }
                } else {
                    lines.push('- No explicit predictive hypotheses were generated.');
                }
                lines.push('', '## Prediction Errors, Tension Detection & Resolution');
                if (support.length) {
                    lines.push('- Evidence that supports the predictive model:');
                    lines.push(...support.slice(0, 3).map(item => `- ${item}`));
                }
                if (contradiction.length) {
                    lines.push('- Prediction errors / disconfirming evidence:');
                    lines.push(...contradiction.slice(0, 3).map(item => `- ${item}`));
                } else {
                    lines.push('- No major prediction errors detected in current evidence sample.');
                }
                lines.push('- Tension resolution rule: when baseline and exception conflict, only resolve with directly cited precedence/exception language from sources.');
                return lines.join('\n');
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildContextualHierarchyBlock !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildContextualHierarchyBlock = function superBrainBuildContextualHierarchyBlock(query, sources = []) {
                const selected = uniqBy(
                    (sources || []).filter(item => item?.text),
                    item => item?.id || `${item?.filename || 'unknown'}:${item?.index || 0}:${item?.page || 'na'}`
                ).slice(0, 24);
                if (!selected.length) return '';
                const queryTokens = this.superBrainTokenize(query);
                const ranked = selected.map(source => {
                    const weight = this.superBrainEvidenceHierarchyWeight(source, queryTokens);
                    return { source, ...weight };
                }).sort((a, b) => b.weighted - a.weighted);
                const winner = ranked[0];
                const challenger = ranked[1];
                const precedenceRegex = /\b(takes precedence|shall prevail|prevails over|overrides?|supersedes?|in case of conflict|except as provided|subject to)\b/i;
                const precedenceEvidence = ranked
                    .map(item => {
                        const sentence = this.superBrainBestSentence(item.source?.text || '', queryTokens).replace(/\s+/g, ' ').trim();
                        if (!sentence || !precedenceRegex.test(sentence)) return null;
                        return { item, sentence };
                    })
                    .filter(Boolean)
                    .slice(0, 2);

                const lines = ['## Contextual Hierarchy (Evidence Weighting)'];
                lines.push('- Weighting dimensions: Authority (42%), Recency (24%), Contextual Relevance (34%).');
                lines.push('- Important: weighting is retrieval triage only; final precedence claims require explicit cited precedence language.');
                lines.push('- Top weighted evidence:');
                for (const item of ranked.slice(0, 4)) {
                    lines.push(`- ${item.source.filename || 'Unknown'} -> score ${item.weighted.toFixed(2)} (authority ${item.authority.toFixed(2)}, recency ${item.recency.toFixed(2)}, relevance ${item.relevance.toFixed(2)})`);
                }
                if (winner && precedenceEvidence.length) {
                    const sentence = this.superBrainBestSentence(winner.source.text || '', queryTokens).replace(/\s+/g, ' ').trim();
                    const quote = sentence.slice(0, 110).replace(/"/g, '\'');
                    lines.push(`- Winning context source: ${winner.source.filename || 'Unknown'} [Source: ${winner.source.filename || 'Unknown'}, page ${winner.source.page || 'N/A'}, "${quote || 'relevant quote'}"]`);
                }
                if (precedenceEvidence.length) {
                    for (const entry of precedenceEvidence) {
                        const quote = entry.sentence.slice(0, 110).replace(/"/g, '\'');
                        lines.push(`- Explicit precedence clause: ${entry.sentence.slice(0, 220)} [Source: ${entry.item.source.filename || 'Unknown'}, page ${entry.item.source.page || 'N/A'}, "${quote || 'relevant quote'}"]`);
                    }
                } else {
                    lines.push('- No explicit precedence clause found in current evidence. Do NOT claim a definitive policy winner; request additional governing documents if needed.');
                }
                if (winner && challenger && precedenceEvidence.length) {
                    const why = winner.weighted >= challenger.weighted
                        ? `Evidence from ${winner.source.filename || 'top source'} is preferred because explicit precedence language is present and context match is stronger.`
                        : 'Evidence tension remains unresolved; additional authoritative material is required.';
                    lines.push(`- Conflict resolution: ${why}`);
                }
                return lines.join('\n');
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainNormalizeCitationSyntax !== 'function') {
            CognitiveSynthesizer.prototype.superBrainNormalizeCitationSyntax = function superBrainNormalizeCitationSyntax(text) {
                let out = String(text || '');
                if (!out) return out;
                out = out.replace(/\[(source)\s*:/gi, '[Source:');
                out = out.replace(/\(Source:\s*([^)]+)\)/gi, '[Source: $1]');
                out = out.replace(/\[Source:\s*([^,\]\n]+)\s*,\s*"([^"]+)"\]/gi, '[Source: $1, page N/A, "$2"]');
                out = out.replace(/\[Source:\s*([^,\]\n]+)\s*,\s*(?:page\s*)?([^,\]\n]+)\s*\]/gi, '[Source: $1, page $2, "citation excerpt unavailable"]');
                return out;
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainHasMalformedCitationPatterns !== 'function') {
            CognitiveSynthesizer.prototype.superBrainHasMalformedCitationPatterns = function superBrainHasMalformedCitationPatterns(text) {
                const value = String(text || '');
                if (!value) return false;
                if (/\bCitations?:\s*\n/i.test(value) && !/\[Source:\s*[^,\]]+,\s*(?:page\s*)?[^,\]]+,\s*"[^"]+"\]/i.test(value)) {
                    return true;
                }
                if (/\b[A-Za-z0-9._ -]+\.(?:pdf|docx|md)\s*\.\.\./i.test(value)) return true;
                if (/\[Source:\s*[^\]]*$/m.test(value)) return true;
                return false;
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainExtractDistinctCitedDocs !== 'function') {
            CognitiveSynthesizer.prototype.superBrainExtractDistinctCitedDocs = function superBrainExtractDistinctCitedDocs(text) {
                const value = String(text || '');
                if (!value) return [];
                let citations = [];
                if (typeof this.extractCitations === 'function') {
                    citations = this.extractCitations(value) || [];
                } else {
                    const regex = /\[Source:\s*([^,\]]+),\s*(?:page\s*)?([^,\]]+),\s*"([^"]+)"\]/gi;
                    let match;
                    while ((match = regex.exec(value)) !== null) {
                        citations.push({ filename: String(match[1] || '').trim() });
                    }
                }
                return uniqBy(
                    citations.map(item => String(item?.filename || '').trim()).filter(Boolean),
                    item => normalizeConcept(item)
                );
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainEvaluateCognitiveDepth !== 'function') {
            CognitiveSynthesizer.prototype.superBrainEvaluateCognitiveDepth = function superBrainEvaluateCognitiveDepth(
                query,
                responseText,
                instructionProfile = {},
                diagnostics = null,
                sources = [],
                activationReport = {}
            ) {
                if (instructionProfile?.diagnosticMode) {
                    return {
                        passed: true,
                        reasons: [],
                        diagnosticMode: true,
                        adversarialEvidenceAvailable: false
                    };
                }
                const text = String(responseText || '');
                const lower = safeLower(text);
                const nonEmptyLines = text.split('\n').map(line => line.trim()).filter(Boolean);
                const firstLine = nonEmptyLines[0] || '';
                const firstParagraph = text.trim().slice(0, 260);
                const broadOrDecision = !!instructionProfile?.decisionMode || /\b(overall|across|entire|comprehensive|compare|contrast|decision|recommend|trade-?off)\b/i.test(String(query || ''));
                const sourceDocCount = new Set((sources || []).map(item => item?.filename).filter(Boolean)).size;
                const citedDocs = this.superBrainExtractDistinctCitedDocs(text);
                const reasoningSignals = (lower.match(/\b(because|therefore|however|whereas|if|then|trade-?off|constraint|depends on|due to|implies|consequently|in turn)\b/g) || []).length;
                const optionBulletSignals = (text.match(/^\s*(?:[-*]|\d+\.)\s+(?:option|approach|alternative|path|strategy)\b/gi) || []).length;
                const optionLabelSignals = (text.match(/\boption\s*(?:1|2|3|a|b|c)\b/gi) || []).length;
                const hasExecutiveSummarySection = /(^|\n)\s*#{1,3}\s*executive summary\b/i.test(text);
                const hasDecisionFirst = /\bdecision\s*:/i.test(firstLine) || /^\s*##\s*executive summary\b/i.test(firstLine) || /\bdecision\s*:/i.test(firstParagraph);
                const hasReasoningPathSection = /(^|\n)\s*#{1,3}\s*reasoning path\b/i.test(text) || /\breasoning path\s*:/i.test(lower);
                const hasActionableStepsSection = /(^|\n)\s*#{1,3}\s*actionable steps\b/i.test(text) || /(^|\n)\s*#{1,3}\s*next steps\b/i.test(text);
                const hasPredictiveModelSection = /(^|\n)\s*#{1,3}\s*predictive model\b/i.test(text);
                const hasPredictiveHypothesisDisclaimer = /predictive hypothesis status|provisional non-factual prior|not factual claims|hypothesis.*to test/i.test(lower);
                const hasTensionResolutionSection = /(^|\n)\s*#{1,3}\s*(prediction errors?|tension detection|tension resolution)\b/i.test(text) || /\bprediction error\b/i.test(lower);
                const hasContextHierarchySection = /(^|\n)\s*#{1,3}\s*contextual hierarchy\b/i.test(text) || /\bweighting dimensions\b/i.test(lower) || /\bauthority\b.*\brecency\b.*\brelevance\b/i.test(lower);
                const listStyleDocNarration = /\b(doc(?:ument)?\s*[a-z0-9]+\s+says|doc(?:ument)?\s*[a-z0-9]+\s+states)\b/i.test(text);
                const hasMentalModelSection = /(^|\n)\s*#{1,3}\s*mental model\b/i.test(text) || /\bmental model\b/i.test(lower);
                const hasHumanLoopSection = /(^|\n)\s*#{1,3}\s*human cognitive processing loop\b/i.test(text);
                const hasEvidenceSection = /(^|\n)\s*#{1,3}\s*(evidence-based expert analysis|evidence synthesis|evidence-based answer|evidence-based analysis)\b/i.test(text) || /\bevidence[- ]based\b/i.test(lower);
                const hasFrameworkSection = /(^|\n)\s*#{1,3}\s*(analytical framework|pros\s*\/\s*cons|pros and cons|stakeholder analysis|causal chain|strengths?\s*\/\s*weaknesses?|swot)\b/i.test(text) ||
                    (/\bpros\b/i.test(lower) && /\bcons\b/i.test(lower));
                const hasExplicitFactsSection = /(^|\n)\s*#{1,3}\s*(explicit(?:ly)? stated facts|explicit facts)\b/i.test(text) || /\bFact:\b/.test(text);
                const hasInferredImplicationsSection = /(^|\n)\s*#{1,3}\s*(inferred implications|inferences?)\b/i.test(text) || /\bInference:\b/.test(text);
                const hasFactInferenceDistinction = hasExplicitFactsSection && hasInferredImplicationsSection;
                const hasUncertaintySection = /(^|\n)\s*#{1,3}\s*uncertainties?\s*&\s*missing information\b/i.test(text) || /uncertaint|missing information|evidence gap|cannot fully confirm/i.test(lower);
                const minReasoningSignals = instructionProfile?.decisionMode ? MIN_DECISION_REASONING_SIGNALS : MIN_ANALYSIS_REASONING_SIGNALS;
                const hasReasoningDepth = reasoningSignals >= minReasoningSignals;
                const coverageTarget = typeof CLAIM_CITATION_TARGET === 'number' ? CLAIM_CITATION_TARGET : 0.55;
                const precisionTarget = typeof CITATION_PRECISION_TARGET === 'number' ? CITATION_PRECISION_TARGET : 0.5;
                const hasCitationCoverage = diagnostics ? (diagnostics.claimCitationCoverage || 0) >= Math.max(0.5, coverageTarget - 0.03) : true;
                const hasCitationPrecision = diagnostics ? (diagnostics.citationPrecision || 0) >= Math.max(0.45, precisionTarget - 0.03) : true;
                const crossSourceExpected = broadOrDecision && sourceDocCount >= MIN_CROSS_SOURCE_DOCS;
                const crossSourceSatisfied = !crossSourceExpected || citedDocs.length >= Math.min(MIN_CROSS_SOURCE_DOCS, sourceDocCount);
                const perspectiveMode = !!instructionProfile?.perspectiveMode || /\b(stakeholder|policy|should we|implement|impact|conflict|trade-?off|goals?)\b/i.test(String(query || ''));
                const adversarialMode = !!instructionProfile?.adversarialMode || broadOrDecision;
                const adversarialEvidenceAvailable = Number(activationReport?.adversarialChunks || 0) > 0;
                const perspectiveMentions = (text.match(/\bfrom perspective\s+[a-z0-9]/gi) || []).length;
                const hasPerspectiveSection = /(^|\n)\s*#{1,3}\s*(perspective analysis|stakeholder perspectives?|multi-?perspective)\b/i.test(text) || perspectiveMentions >= 2;
                const hasAdversarialLine = /while the primary evidence suggests/i.test(lower) &&
                    /(contradicts|modifies|limits|exceptions?)/i.test(lower) &&
                    /while the primary evidence suggests[\s\S]{0,300}\[Source:/i.test(text);
                const orderedSections = [
                    'executive summary',
                    'predictive model',
                    'prediction errors',
                    'contextual hierarchy',
                    'mental model',
                    'human cognitive processing loop',
                    'analytical framework',
                    'explicitly stated facts',
                    'inferred implications',
                    'evidence-based expert analysis'
                ];
                const sectionPositions = orderedSections.map(section => lower.indexOf(section));
                let cohesiveOrder = true;
                for (let i = 1; i < sectionPositions.length; i++) {
                    if (sectionPositions[i - 1] >= 0 && sectionPositions[i] >= 0 && sectionPositions[i] < sectionPositions[i - 1]) {
                        cohesiveOrder = false;
                        break;
                    }
                }
                const stageChecks = {
                    hasAttentionFilter: /\b(attention|salien|prioriti|signal-to-noise|noise filtering|filtering)\b/i.test(lower),
                    hasPerceptionInterpretation: /\b(perception|interpret|contextual|context interpretation|sensemaking|meaning)\b/i.test(lower),
                    hasDualProcess: /\b(system\s*1|system\s*2|fast thinking|slow thinking|heuristic|analytical pass|deliberate)\b/i.test(lower),
                    hasDecisionStep: /\b(decision|recommend|choose|option|trade-?off)\b/i.test(lower),
                    hasActionStep: /\b(action|next steps?|execution|implement|rollout|operational plan)\b/i.test(lower),
                    hasFeedbackLoop: /\b(feedback loop|monitor|measure|telemetry|iteration|reassess|update model|refine)\b/i.test(lower),
                    hasBiasCheck: /\b(bias|heuristic risk|confirmation bias|availability bias|debias|counter-bias)\b/i.test(lower)
                };
                const stageSignalCount = Object.values(stageChecks).filter(Boolean).length;
                const minStageSignals = instructionProfile?.decisionMode
                    ? MIN_HUMAN_COGNITION_STAGE_SIGNALS_DECISION
                    : MIN_HUMAN_COGNITION_STAGE_SIGNALS;
                const requiredStages = instructionProfile?.decisionMode
                    ? ['hasAttentionFilter', 'hasPerceptionInterpretation', 'hasDecisionStep', 'hasActionStep', 'hasFeedbackLoop']
                    : ['hasAttentionFilter', 'hasPerceptionInterpretation', 'hasActionStep', 'hasFeedbackLoop'];
                const missingRequiredStages = requiredStages.filter(key => !stageChecks[key]);
                const stageLabels = {
                    hasAttentionFilter: 'attention filtering',
                    hasPerceptionInterpretation: 'perception and interpretation',
                    hasDualProcess: 'dual-process reasoning',
                    hasDecisionStep: 'decision stage',
                    hasActionStep: 'action stage',
                    hasFeedbackLoop: 'feedback loop',
                    hasBiasCheck: 'bias/heuristic check'
                };

                const reasons = [];
                if (!hasExecutiveSummarySection) reasons.push('Missing "Executive Summary" section.');
                if (!hasDecisionFirst) reasons.push('Output must start with a definitive decision before supporting detail.');
                if (!hasReasoningPathSection) reasons.push('Missing explicit "Reasoning Path" section.');
                if (!hasActionableStepsSection) reasons.push('Missing explicit actionable steps / next steps section.');
                if (!hasPredictiveModelSection) reasons.push('Missing "Predictive Model" section.');
                if (!hasPredictiveHypothesisDisclaimer) reasons.push('Predictive model must be labeled as a provisional hypothesis (not factual claim).');
                if (!hasTensionResolutionSection) reasons.push('Missing prediction-error/tension-resolution section.');
                if (!hasContextHierarchySection) reasons.push('Missing contextual hierarchy weighting (authority/recency/relevance) section.');
                if (listStyleDocNarration) reasons.push('Response uses list-style document narration ("Doc A says..."), which violates synthesis rule.');
                if (!hasMentalModelSection) reasons.push('Missing explicit "Mental Model" section.');
                if (!hasHumanLoopSection) reasons.push('Missing "Human Cognitive Processing Loop" section.');
                if (!hasEvidenceSection) reasons.push('Missing explicit evidence-based analysis section.');
                if (!hasFrameworkSection) reasons.push('Missing a structured analytical framework (Pros/Cons, Stakeholder Analysis, Causal Chain, or Strength/Weakness).');
                if (!hasFactInferenceDistinction) reasons.push('Missing explicit distinction between "Explicitly Stated Facts" and "Inferred Implications".');
                if (!hasUncertaintySection) reasons.push('Missing "Uncertainties & Missing Information" section.');
                if (!hasReasoningDepth) reasons.push('Reasoning depth is weak (insufficient causal/trade-off signals).');
                if (!hasCitationCoverage) reasons.push('Claim citation coverage is below required threshold.');
                if (!hasCitationPrecision) reasons.push('Citation precision is below required threshold.');
                if (!crossSourceSatisfied) reasons.push('Cross-source reasoning is weak (insufficient distinct cited documents).');
                if (!cohesiveOrder) reasons.push('Output is not organized as a cohesive analyst brief (section order is inconsistent).');
                if (adversarialMode && adversarialEvidenceAvailable && !hasAdversarialLine) reasons.push('Missing required adversarial statement ("While the primary evidence suggests X ... contradicts/modifies ...").');
                if (perspectiveMode && !hasPerspectiveSection) reasons.push('Perspective-taking is missing (need at least two stakeholder viewpoints).');
                if (stageSignalCount < minStageSignals) {
                    reasons.push(`Human cognitive cycle is incomplete (${stageSignalCount}/${minStageSignals} stage signals).`);
                }
                if (missingRequiredStages.length) {
                    reasons.push(`Missing required cognitive stages: ${missingRequiredStages.map(key => stageLabels[key] || key).join(', ')}.`);
                }
                if (instructionProfile?.decisionMode && !stageChecks.hasDualProcess) {
                    reasons.push('Decision analysis must include dual-process reasoning (fast hypothesis + slow analytical validation).');
                }
                if (!stageChecks.hasBiasCheck) {
                    reasons.push('Missing heuristic/bias risk check and mitigation.');
                }

                const decisionChecks = {
                    hasOptionsTradeoffs: /(^|\n)\s*#{1,3}\s*(options?\s*&\s*trade-?offs?|trade-?offs?|options?)\b/i.test(text) || /trade-?off|option|alternative|pros|cons/i.test(lower),
                    hasRecommendation: /(^|\n)\s*#{1,3}\s*recommendation\b/i.test(text) || /\brecommend(?:ed|ation)?\b/i.test(lower),
                    hasRisks: /(^|\n)\s*#{1,3}\s*risks?\b/i.test(text) || /\brisk(s)?\b/i.test(lower),
                    hasMultipleOptions: optionBulletSignals >= 2 || optionLabelSignals >= 2
                };
                if (instructionProfile?.decisionMode) {
                    if (!decisionChecks.hasOptionsTradeoffs) reasons.push('Decision output missing "Options & Trade-offs".');
                    if (!decisionChecks.hasRecommendation) reasons.push('Decision output missing explicit recommendation.');
                    if (!decisionChecks.hasRisks) reasons.push('Decision output missing explicit risk analysis.');
                    if (!decisionChecks.hasMultipleOptions) reasons.push('Decision output does not compare multiple concrete options.');
                }

                return {
                    passed: reasons.length === 0,
                    reasons,
                    broadOrDecision,
                    sourceDocCount,
                    citedDocCount: citedDocs.length,
                    citedDocs: citedDocs.slice(0, 8),
                    hasExecutiveSummarySection,
                    hasDecisionFirst,
                    hasReasoningPathSection,
                    hasActionableStepsSection,
                    hasPredictiveModelSection,
                    hasPredictiveHypothesisDisclaimer,
                    hasTensionResolutionSection,
                    hasContextHierarchySection,
                    listStyleDocNarration,
                    reasoningSignals,
                    minReasoningSignals,
                    hasFrameworkSection,
                    hasFactInferenceDistinction,
                    perspectiveMode,
                    hasPerspectiveSection,
                    perspectiveMentions,
                    adversarialMode,
                    adversarialEvidenceAvailable,
                    hasAdversarialLine,
                    cohesiveOrder,
                    crossSourceExpected,
                    crossSourceSatisfied,
                    stageChecks,
                    stageSignalCount,
                    minStageSignals,
                    missingRequiredStages,
                    stageLabels,
                    decisionChecks,
                    activationCoverage: activationReport?.activationCoveragePct || 0
                };
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildAnalyticalRewriteInstruction !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildAnalyticalRewriteInstruction = function superBrainBuildAnalyticalRewriteInstruction(
                query,
                instructionProfile = {},
                depthGate = null,
                sources = []
            ) {
                const sourceDocCount = new Set((sources || []).map(item => item?.filename).filter(Boolean)).size;
                const decisionMode = !!instructionProfile?.decisionMode;
                const perspectiveMode = !!instructionProfile?.perspectiveMode;
                const adversarialMode = !!instructionProfile?.adversarialMode || decisionMode;
                const requireAdversarialLine = adversarialMode && !!depthGate?.adversarialEvidenceAvailable;
                const minCitations = sourceDocCount >= 10 ? MIN_CITATIONS_FOR_RICH_CORPUS : (sourceDocCount >= 4 ? 2 : (sourceDocCount > 0 ? 1 : 0));
                const lines = [
                    'ANALYSIS-DEPTH REWRITE GATE (MANDATORY)',
                    `Question: ${query}`,
                    'Do NOT summarize. Produce a cohesive analyst brief that separates evidence from inference and then reasons to a decision/action.',
                    '',
                    'Mandatory output scaffold:',
                    '## Executive Summary',
                    'Decision: <definitive answer>',
                    'Reasoning Path: <why this conclusion wins>',
                    'Actionable Steps: <what to do next>',
                    '## Predictive Model',
                    '## Prediction Errors, Tension Detection & Resolution',
                    '## Contextual Hierarchy',
                    '## Mental Model',
                    '## Human Cognitive Processing Loop',
                    '## Analytical Framework',
                    '## Explicitly Stated Facts',
                    '## Inferred Implications',
                    '## Evidence-Based Expert Analysis'
                ];
                lines.push('- In "Predictive Model", explicitly label content as provisional hypothesis only (not factual claim).');
                lines.push('- In "Human Cognitive Processing Loop", explicitly include:');
                lines.push('  1) Evidence Intake (sensation proxy from retrieved documents)');
                lines.push('  2) Attention Filtering (signal vs noise)');
                lines.push('  3) Perception & Context Interpretation');
                lines.push('  4) Dual-process reasoning (System 1 hypothesis + System 2 analytical verification)');
                lines.push('  5) Decision & Action path');
                lines.push('  6) Feedback Loop (what to monitor and how model updates)');
                lines.push('  7) Heuristic/Bias risk check and mitigation');
                if (decisionMode) {
                    lines.push('## Options & Trade-offs');
                    lines.push('## Recommendation');
                    lines.push('## Risks');
                }
                if (perspectiveMode) lines.push('## Perspective Analysis');
                lines.push('## Uncertainties & Missing Information');
                lines.push('');
                lines.push(`Minimum citation target for this rewrite: ${minCitations} citations in exact format [Source: filename, page X, "brief relevant quote"].`);
                if (sourceDocCount >= MIN_CROSS_SOURCE_DOCS) {
                    lines.push(`Cross-source rule: cite at least ${Math.min(MIN_CROSS_SOURCE_DOCS, sourceDocCount)} distinct source documents when giving broad or decision conclusions.`);
                }
                if (decisionMode) {
                    lines.push('Decision rule: compare at least two concrete options with source-grounded trade-offs before recommending.');
                }
                lines.push('Coherence rule: keep section order exactly as scaffolded to produce one cohesive analyst brief (do not mix sections).');
                lines.push('Predictive rule: mark predictive model content as provisional hypothesis only; never present it as fact without citation.');
                lines.push('Framework rule: use one structured framework explicitly (Pros/Cons, Stakeholder Analysis, Causal Chain, or Strength/Weakness).');
                lines.push('Fact-vs-inference rule: every inference must be linked to cited explicit facts.');
                if (perspectiveMode) {
                    lines.push('Perspective rule: include at least two viewpoints with "From Perspective A..." and "From Perspective B..." plus citations.');
                }
                if (requireAdversarialLine) {
                    lines.push('Adversarial rule: include this exact pattern with citation: "While the primary evidence suggests X, the corpus also contains evidence that contradicts/modifies this conclusion: [Citation]."');
                } else if (adversarialMode) {
                    lines.push('Adversarial rule: do not fabricate contradictions; include adversarial sentence only if disconfirming evidence is explicitly cited.');
                }
                const reasons = Array.isArray(depthGate?.reasons) ? depthGate.reasons.filter(Boolean) : [];
                if (reasons.length) {
                    lines.push('');
                    lines.push('Fix these issues explicitly:');
                    for (const reason of reasons.slice(0, 8)) lines.push(`- ${reason}`);
                }
                return lines.join('\n');
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildGapDrivenQueries !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildGapDrivenQueries = function superBrainBuildGapDrivenQueries(query, model = null) {
                const out = [];
                const push = (value) => {
                    const cleaned = String(value || '').replace(/\s+/g, ' ').trim();
                    if (cleaned) out.push(cleaned);
                };
                push(`${query} cross-source supporting evidence`);
                push(`${query} constraints exceptions conflicts`);
                const gaps = Array.isArray(model?.gaps) ? model.gaps : [];
                for (const gapRaw of gaps.slice(0, MAX_GAP_REFINEMENT_QUERIES)) {
                    const gap = String(gapRaw || '')
                        .replace(/^Insufficient direct evidence for concept:\s*/i, '')
                        .replace(/"/g, '')
                        .trim();
                    if (!gap) continue;
                    push(`${query} ${gap} validating evidence`);
                }
                return uniqBy(out, item => normalizeConcept(item)).slice(0, MAX_GAP_REFINEMENT_QUERIES + 2);
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainAssessEvidenceCompleteness !== 'function') {
            CognitiveSynthesizer.prototype.superBrainAssessEvidenceCompleteness = function superBrainAssessEvidenceCompleteness(
                query,
                sources = [],
                model = null,
                activationReport = {},
                instructionProfile = {}
            ) {
                const deduped = uniqBy(
                    sources || [],
                    item => item?.id || `${item?.filename || 'unknown'}:${item?.index || 0}:${item?.page || 'na'}`
                );
                const sourceCount = deduped.length;
                const docCount = new Set(deduped.map(item => item.filename).filter(Boolean)).size;
                const totalDocs = activationReport.totalDocuments || docCount;
                const broadOrDecision = !!instructionProfile?.decisionMode || /\b(overall|across|entire|comprehensive|compare|contrast|decision|recommend|trade-?off)\b/i.test(String(query || ''));
                const requiredDocs = broadOrDecision ? (totalDocs >= 7 ? 3 : Math.min(totalDocs, MIN_CROSS_SOURCE_DOCS)) : 1;
                const activationCoverage = activationReport.activationCoveragePct || 0;
                const gapCount = Array.isArray(model?.gaps) ? model.gaps.length : 0;

                const reasons = [];
                if (sourceCount === 0) reasons.push('No retrievable evidence was found for the question.');
                if (requiredDocs > 1 && docCount < requiredDocs) {
                    reasons.push(`Cross-source coverage is insufficient (${docCount}/${requiredDocs} required documents).`);
                }
                if (broadOrDecision && totalDocs >= 5 && activationCoverage > 0 && activationCoverage < 22) {
                    reasons.push(`Activation coverage remains low for a broad query (${activationCoverage}%).`);
                }
                if (gapCount >= MODEL_GAP_THRESHOLD && docCount < Math.max(2, requiredDocs)) {
                    reasons.push('Mental model still has unresolved high-priority evidence gaps.');
                }

                const requestedDocuments = [];
                if (sourceCount === 0) requestedDocuments.push(`Primary source documents directly addressing: "${query}"`);
                if (requiredDocs > 1 && docCount < requiredDocs) {
                    requestedDocuments.push(`Additional relevant sources from at least ${requiredDocs} distinct documents are needed.`);
                }
                for (const gapRaw of (model?.gaps || []).slice(0, 5)) {
                    requestedDocuments.push(`Source material that explicitly covers: ${String(gapRaw).replace(/^Insufficient direct evidence for concept:\s*/i, '').replace(/"/g, '').trim()}`);
                }

                return {
                    needsMoreEvidence: reasons.length > 0,
                    sourceCount,
                    docCount,
                    totalDocs,
                    requiredDocs,
                    activationCoverage,
                    reasons: uniqBy(reasons, item => item),
                    requestedDocuments: uniqBy(requestedDocuments.filter(Boolean), item => item).slice(0, 8)
                };
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildMissingEvidenceResponse !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildMissingEvidenceResponse = function superBrainBuildMissingEvidenceResponse(query, completeness, sources = [], model = null) {
                const queryTokens = this.superBrainTokenize(query);
                const evidenceLines = (sources || []).slice(0, 4).map(source => {
                    const sentences = String(source.text || '').split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
                    const sentence = sentences.find(s => queryTokens.some(token => safeLower(s).includes(token))) || sentences[0] || '';
                    if (!sentence) return null;
                    const cleaned = sentence.replace(/\s+/g, ' ').trim().slice(0, 210);
                    const quote = cleaned.slice(0, 110).replace(/"/g, '\'');
                    return `- ${cleaned} [Source: ${source.filename || 'Unknown'}, page ${source.page || 'N/A'}, "${quote || 'relevant quote'}"]`;
                }).filter(Boolean);

                const lines = [
                    '## Executive Summary',
                    'Decision: A definitive expert conclusion is deferred because required evidence is incomplete.',
                    'Reasoning Path: Current retrieval does not meet cross-source sufficiency and evidence-completeness thresholds for this query.',
                    'Actionable Steps: Provide the missing documents/data listed below, then rerun synthesis for a fully grounded answer.',
                    '',
                    '## Mental Model',
                    '- Status: Incomplete for a reliable expert conclusion.',
                    `- Query focus: ${query}`,
                    `- Current entities: ${(model?.entities || []).slice(0, 8).join(', ') || 'N/A'}`,
                    `- Current concepts: ${(model?.concepts || []).slice(0, 10).join(', ') || 'N/A'}`,
                    '',
                    '## Evidence-Based Status',
                    `- Source chunks reviewed: ${completeness?.sourceCount || 0}`,
                    `- Distinct documents covered: ${completeness?.docCount || 0}/${completeness?.requiredDocs || 1} required`,
                    completeness?.activationCoverage ? `- Activation coverage: ${completeness.activationCoverage}%` : '- Activation coverage: N/A'
                ];
                if (evidenceLines.length) {
                    lines.push('- Current grounded evidence:');
                    lines.push(...evidenceLines);
                } else {
                    lines.push('- No citable evidence is currently available for this question.');
                }

                lines.push('', '## Explicitly Stated Facts (Available)');
                if (evidenceLines.length) {
                    lines.push('- The evidence bullets above are directly supported by cited excerpts.');
                } else {
                    lines.push('- No explicit facts can be asserted yet with citation confidence.');
                }
                lines.push('', '## Inferred Implications (Provisional)');
                lines.push('- Any downstream implications are provisional until additional source material is provided.');

                lines.push('', '## Human Cognitive Processing Loop Status');
                lines.push('- Evidence intake (sensation proxy): limited by available retrieved documents.');
                lines.push('- Attention filtering: partial; key signals identified but completeness is below threshold.');
                lines.push('- Perception/interpretation: provisional and cannot be finalized.');
                lines.push('- Dual-process reasoning (System1/System2): halted due to unresolved evidence gaps.');
                lines.push('- Decision/action: deferred until required sources are provided.');
                lines.push('- Feedback loop: upload requested documents, then rerun to update the model and recommendation.');
                lines.push('- Bias/heuristic guard: withholding conclusion to avoid availability/confirmation bias from incomplete evidence.');

                lines.push('', '## Uncertainties & Missing Information');
                for (const reason of (completeness?.reasons || [])) lines.push(`- ${reason}`);
                if (!completeness?.reasons?.length) lines.push('- Evidence is insufficient to produce a high-confidence answer.');
                if (Array.isArray(model?.gaps) && model.gaps.length) {
                    lines.push('- Open evidence gaps:');
                    for (const gap of model.gaps.slice(0, 6)) lines.push(`- ${gap}`);
                }

                lines.push('', '## Required Documents/Data');
                const requested = completeness?.requestedDocuments || [];
                if (requested.length) requested.forEach((item, idx) => lines.push(`${idx + 1}. ${item}`));
                else lines.push('1. Additional source documents that directly answer unresolved parts of the question.');
                lines.push('');
                lines.push('Please provide the missing documents/data so I can complete a fully grounded, cross-source expert answer with precise citations.');
                return lines.join('\n');
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainRunModelRefinementLoop !== 'function') {
            CognitiveSynthesizer.prototype.superBrainRunModelRefinementLoop = async function superBrainRunModelRefinementLoop(
                query,
                relevantChunks = [],
                attachmentContext = null,
                localStats = {},
                instructionProfile = {}
            ) {
                const retriever = (typeof window !== 'undefined' && window.app?.retriever) ? window.app.retriever : null;
                const broadOrDecision = !!instructionProfile?.decisionMode || /\b(overall|across|entire|comprehensive|compare|contrast|decision|recommend|trade-?off)\b/i.test(String(query || ''));
                let chunks = uniqBy(relevantChunks || [], item => item.id || `${item.filename}:${item.index || 0}`);
                let passes = 1;
                let stableHits = 0;
                let previousGapCount = null;
                let model = null;
                let completeness = null;

                for (let cycle = 0; cycle < MAX_MODEL_REFINEMENT_PASSES; cycle++) {
                    passes = cycle + 1;
                    const sourcesForModel = uniqBy(
                        [
                            ...(attachmentContext?.chunks || []).map(item => ({ ...item, filename: attachmentContext?.name || item.filename || 'attachment' })),
                            ...chunks
                        ],
                        item => item.id || `${item.filename}:${item.index || 0}:${item.page || 'na'}`
                    ).slice(0, MAX_MENTAL_MODEL_SOURCES * 2);

                    model = await this.superBrainBuildModel(query, chunks, attachmentContext, localStats);
                    completeness = this.superBrainAssessEvidenceCompleteness(
                        query,
                        sourcesForModel,
                        model,
                        localStats.activationReport || {},
                        instructionProfile
                    );

                    const gapCount = Array.isArray(model?.gaps) ? model.gaps.length : 0;
                    const gapDelta = previousGapCount === null ? null : Math.abs(gapCount - previousGapCount);
                    if (gapDelta !== null && gapDelta <= 1) stableHits += 1;
                    else stableHits = 0;
                    previousGapCount = gapCount;

                    const shouldRetrieveAgain = completeness.needsMoreEvidence &&
                        broadOrDecision &&
                        retriever &&
                        typeof retriever.retrieve === 'function' &&
                        stableHits < 1;
                    if (!shouldRetrieveAgain) break;

                    const gapQueries = this.superBrainBuildGapDrivenQueries(query, model);
                    if (!gapQueries.length) break;
                    const expanded = [];
                    for (const gapQuery of gapQueries.slice(0, MAX_GAP_REFINEMENT_QUERIES)) {
                        try {
                            const retrieved = await retriever.retrieve(gapQuery, null, {
                                intent: {
                                    ...(typeof retriever.detectQueryIntent === 'function' ? retriever.detectQueryIntent(gapQuery) : {}),
                                    broadCoverage: true,
                                    comparative: true
                                }
                            });
                            expanded.push(...(retrieved || []));
                        } catch {
                            // keep loop resilient on transient retrieval failures
                        }
                    }
                    if (!expanded.length) break;
                    const merged = uniqBy([...chunks, ...expanded], item => item.id || `${item.filename}:${item.index || 0}`);
                    if (merged.length <= chunks.length + 1) {
                        stableHits += 1;
                        if (stableHits >= 1) break;
                    }
                    chunks = merged;
                }

                return {
                    chunks,
                    model,
                    completeness,
                    passes,
                    stabilized: stableHits >= 1
                };
            };
        }
    }

    function installPatches() {
        if (typeof CognitiveRetriever !== 'function' || typeof CognitiveSynthesizer !== 'function') {
            return false;
        }

        installRetrieverMethods(CognitiveRetriever);
        installSynthesizerMethods(CognitiveSynthesizer);

        patchMethod(CognitiveRetriever, 'updateCorpus', (original) => function patchedUpdateCorpus(documents) {
            const result = original.call(this, documents);
            if (typeof this.buildSuperConceptGraph === 'function') {
                this.buildSuperConceptGraph();
            }
            return result;
        });

        patchMethod(CognitiveRetriever, 'retrieve', (original) => async function patchedRetrieve(query, topK = null, options = {}) {
            if (!this.allChunks?.length) return [];
            const resolvedQuery = typeof this.superBrainResolveQueryFromSession === 'function'
                ? this.superBrainResolveQueryFromSession(query)
                : query;
            const retrievalQuery = typeof this.superBrainSanitizeRetrievalQuery === 'function'
                ? this.superBrainSanitizeRetrievalQuery(resolvedQuery)
                : resolvedQuery;

            const baseIntent = options.intent || (typeof this.detectQueryIntent === 'function'
                ? this.detectQueryIntent(query)
                : {});
            const intent = { ...baseIntent, broadCoverage: true };
            const intentGraph = typeof this.superBrainBuildIntentGraph === 'function'
                ? this.superBrainBuildIntentGraph(retrievalQuery)
                : null;
            const predictiveModel = typeof this.superBrainBuildPredictiveModel === 'function'
                ? this.superBrainBuildPredictiveModel(retrievalQuery, intentGraph, intent)
                : null;

            const corpusSize = this.allChunks.length;
            const targetTopK = topK ?? (
                corpusSize <= 60 ? DEFAULT_TARGET_K_SMALL
                    : corpusSize <= 240 ? DEFAULT_TARGET_K_MEDIUM
                        : DEFAULT_TARGET_K_LARGE
            );

            const base = await original.call(
                this,
                retrievalQuery,
                Math.max(18, Math.ceil(targetTopK * 0.8)),
                { ...options, intent, predictiveModel }
            );

            const anchors = typeof this.getDocumentAnchors === 'function'
                ? this.getDocumentAnchors(retrievalQuery, intent)
                : [];

            let merged = typeof this.mergeUniqueChunks === 'function'
                ? this.mergeUniqueChunks(base, anchors, Math.max(targetTopK * 2, 70))
                : uniqBy([...base, ...anchors], item => item.id);

            const conceptLinked = typeof this.expandImplicitConceptLinks === 'function'
                ? this.expandImplicitConceptLinks(merged, retrievalQuery, Math.max(10, Math.ceil(targetTopK * 0.65)))
                : [];

            merged = typeof this.mergeUniqueChunks === 'function'
                ? this.mergeUniqueChunks(
                    merged,
                    conceptLinked,
                    Math.max(targetTopK * MAX_MERGED_MULTIPLIER, 110)
                )
                : uniqBy([...merged, ...conceptLinked], item => item.id);

            const refinement = typeof this.superBrainRunRetrievalRefinement === 'function'
                ? await this.superBrainRunRetrievalRefinement(retrievalQuery, merged, targetTopK, { ...options, predictiveModel }, intent, original)
                : { chunks: merged, passes: 1, stabilized: true };
            merged = refinement.chunks;
            const effectivePredictiveModel = refinement?.predictiveModel || predictiveModel;
            const adversarial = typeof this.superBrainRunAdversarialRetrieval === 'function'
                ? await this.superBrainRunAdversarialRetrieval(retrievalQuery, merged, { ...options, predictiveModel: effectivePredictiveModel }, intent, original)
                : { chunks: [], passes: 0, probes: [] };
            const adversarialChunks = Array.isArray(adversarial?.chunks) ? adversarial.chunks : [];
            if (adversarialChunks.length) {
                merged = typeof this.mergeUniqueChunks === 'function'
                    ? this.mergeUniqueChunks(merged, adversarialChunks, Math.max(targetTopK * 4, 200))
                    : uniqBy([...(merged || []), ...adversarialChunks], item => item.id);
            }

            let ranked = merged;
            if (typeof this.shouldUseSemanticRerank === 'function'
                && this.shouldUseSemanticRerank(intent, merged, targetTopK)
                && typeof this.semanticRerank === 'function') {
                const semantic = await this.semanticRerank(retrievalQuery, merged, targetTopK, intent);
                if (semantic?.length) ranked = semantic;
            } else {
                ranked = merged
                    .slice()
                    .sort((a, b) => (b.score || 0) - (a.score || 0));
                if (typeof this.diversifyResults === 'function') {
                    ranked = this.diversifyResults(ranked, targetTopK);
                } else {
                    ranked = ranked.slice(0, targetTopK);
                }
            }

            const spreadEnforced = typeof this.enforceDocumentCoverage === 'function'
                ? this.enforceDocumentCoverage(ranked, anchors, targetTopK)
                : ranked.slice(0, targetTopK);
            let out = spreadEnforced.slice(0, targetTopK);
            if (typeof this.superBrainDeepScanAugment === 'function') {
                out = this.superBrainDeepScanAugment(retrievalQuery, out, targetTopK);
            }

            const broadQuestion = intent.broadCoverage || intent.comparative || intent.timeline ||
                /\b(overall|across|entire|comprehensive|strategy|decision|recommend|trade-?off)\b/i.test(String(query || ''));
            const requiredDocs = broadQuestion
                ? ((this.documents?.length || 0) >= 7 ? 3 : Math.min((this.documents?.length || 0), MIN_CROSS_SOURCE_DOCS))
                : 1;
            const currentDocs = new Set(out.map(item => item.filename).filter(Boolean));
            if (broadQuestion && (this.documents?.length || 0) >= MIN_CROSS_SOURCE_DOCS && currentDocs.size < requiredDocs) {
                const crossAnchors = typeof this.getDocumentAnchors === 'function'
                    ? this.getDocumentAnchors(`${retrievalQuery} cross-source corroboration`, intent)
                    : [];
                for (const anchor of crossAnchors) {
                    if (out.length >= targetTopK) break;
                    if (!anchor?.id || out.some(item => item.id === anchor.id)) continue;
                    if (currentDocs.size < requiredDocs && currentDocs.has(anchor.filename)) continue;
                    out.push(anchor);
                    if (anchor.filename) currentDocs.add(anchor.filename);
                }
            }
            if (broadQuestion && adversarialChunks.length) {
                for (const chunk of adversarialChunks) {
                    if (!chunk?.id || out.some(item => item.id === chunk.id)) continue;
                    if (out.length < targetTopK) {
                        out.push(chunk);
                    } else {
                        out[out.length - 1] = chunk;
                    }
                    if (out.length >= targetTopK) break;
                }
            }

            if (typeof this.buildCorpusActivationReport === 'function') {
                this.lastActivationReport = this.buildCorpusActivationReport(query, out);
                this.lastActivationReport.retrievalPasses = refinement.passes || 1;
                this.lastActivationReport.retrievalStabilized = !!refinement.stabilized;
                this.lastActivationReport.requiredCrossSourceDocs = broadQuestion ? requiredDocs : 1;
                this.lastActivationReport.currentCrossSourceDocs = new Set(out.map(item => item.filename)).size;
                this.lastActivationReport.adversarialPasses = adversarial?.passes || 0;
                this.lastActivationReport.adversarialChunks = adversarialChunks.length;
                this.lastActivationReport.predictiveModel = effectivePredictiveModel;
            }
            this.lastPredictiveModel = effectivePredictiveModel;
            this.lastAdversarialRetrieval = {
                query,
                passes: adversarial?.passes || 0,
                probes: adversarial?.probes || [],
                chunks: adversarialChunks.slice(0, MAX_ADVERSARIAL_CHUNKS)
            };

            return out.slice(0, targetTopK);
        });

        patchMethod(CognitiveRetriever, 'retrieveFromChunks', (original) => async function patchedRetrieveFromChunks(query, chunks, topK = 10, options = {}) {
            const intent = options.intent || {};
            const adjustedTopK = Math.max(topK, intent.broadCoverage ? 16 : 12);
            return original.call(this, query, chunks, adjustedTopK, options);
        });

        patchMethod(CognitiveSynthesizer, 'buildUserPrompt', (original) => function patchedBuildUserPrompt(query, instructionProfile = {}) {
            const base = original.call(this, query, instructionProfile);
            const format = instructionProfile?.requestedFormat || 'structured prose';
            const diagnosticMode = !!instructionProfile?.diagnosticMode;
            if (diagnosticMode) {
                return `${base}\n\nDiagnostic mode: The user provided sample output for troubleshooting.\nAnalyze the sample response quality and identify concrete failures (unsupported claims, assumption leakage, citation problems, process/gating errors).\nDo NOT answer the underlying domain question and do NOT mimic the sample's structure as target output.`;
            }
            const decisionMode = !!instructionProfile?.decisionMode;
            const perspectiveMode = !!instructionProfile?.perspectiveMode;
            const adversarialMode = !!instructionProfile?.adversarialMode || decisionMode;
            if (!decisionMode && !['structured prose', 'bullet list'].includes(format)) {
                return base;
            }
            const sections = [
                'Mandatory cognitive scaffold for this answer:',
                '1) Executive Summary (Decision first, then Reasoning Path, then Actionable Steps)',
                '2) Predictive Model (provisional non-factual hypothesis to test)',
                '3) Prediction Errors, Tension Detection & Resolution',
                '4) Contextual Hierarchy (authority, recency, contextual relevance; explicitly state which source wins and why)',
                '5) Mental Model',
                '6) Human Cognitive Processing Loop (Evidence intake -> Attention filtering -> Perception/interpretation -> System1/System2 reasoning -> Decision/action -> Feedback loop -> Bias check)',
                '7) Analytical Framework (Pros/Cons, Stakeholder Analysis, Causal Chain, or Strength/Weakness)',
                '8) Explicitly Stated Facts',
                '9) Inferred Implications',
                '10) Evidence-Based Expert Analysis'
            ];
            if (decisionMode) {
                sections.push('11) Options & Trade-offs (at least 2 options)');
                sections.push('12) Recommendation');
                sections.push('13) Risks');
            }
            if (perspectiveMode) {
                sections.push(`${decisionMode ? '14' : '11'}) Perspective Analysis (at least two viewpoints)`);
            }
            sections.push(`${decisionMode ? (perspectiveMode ? '15' : '14') : (perspectiveMode ? '12' : '11')}) Uncertainties & Missing Information`);
            if (adversarialMode) {
                sections.push('If disconfirming evidence is retrieved, include adversarial sentence: "While the primary evidence suggests X, the corpus also contains evidence that contradicts/modifies this conclusion: [Citation]."');
            }
            sections.push('Synthesis rule: do NOT write "Doc A says X, Doc B says Y". Every conclusion sentence must integrate multiple evidence points.');
            sections.push('Do not output a summary-only response.');
            const scaffold = sections.join('\n');
            return `${base}\n\n${scaffold}`;
        });

        patchMethod(CognitiveSynthesizer, 'buildInstructionProfile', (original) => function patchedBuildInstructionProfile(query) {
            const profile = original.call(this, query);
            const diagnosticMode = /\b(example output|for troubleshooting|troubleshoot|debug|not to tailor|what is going on|why (?:is|does).{0,30}(?:output|response))\b/i.test(String(query || ''));
            const decisionMode = /\b(recommend|recommendation|decision|should|best approach|manage|strategy|trade-?off|pros|cons|options?|what should|how to handle|policy approach)\b/i.test(String(query || ''));
            const perspectiveMode = /\b(stakeholder|policy|should we|implement|impact on|conflict|conflicting goals|audience|department|team|user group)\b/i.test(String(query || ''));
            const frameworkMode = /\b(framework|pros|cons|analysis|evaluate|assess|compare|trade-?off|root cause|causal|swot|strength|weakness)\b/i.test(String(query || '')) || decisionMode;
            const adversarialMode = /\b(decision|recommend|policy|should|trade-?off|compare|risk|exception|counter)\b/i.test(String(query || '')) || decisionMode;
            return {
                ...profile,
                strictSourceOnly: true,
                freshPassRequired: true,
                requireUncertaintySection: true,
                executiveSummaryRequired: true,
                predictiveCodingMode: true,
                system2AnalyticalMode: true,
                diagnosticMode,
                decisionMode,
                perspectiveMode,
                frameworkMode,
                adversarialMode,
                factInferenceMode: true
            };
        });

        patchMethod(CognitiveSynthesizer, 'getWorkingMemorySummary', () => function patchedWorkingMemorySummary() {
            const recent = Array.isArray(this.workingMemory) ? this.workingMemory.slice(-6) : [];
            if (!recent.length) {
                return 'No prior conversational memory yet. Use this note only for dialogue continuity, not as factual evidence.';
            }
            const lines = recent.map((item, idx) => `${idx + 1}. Prior user question: ${item.query}`);
            return [
                'Conversation memory (continuity-only; not factual evidence):',
                ...lines
            ].join('\n');
        });

        patchMethod(CognitiveSynthesizer, 'getSystemPrompt', (original) => function patchedGetSystemPrompt(context, corpusStats = {}, instructionProfile = {}, workingMemorySummary = '') {
            const base = original.call(
                this,
                context,
                corpusStats,
                instructionProfile,
                workingMemorySummary
            );
            const diagnosticMode = !!instructionProfile?.diagnosticMode;
            if (diagnosticMode) {
                const diagnosticRules = [
                    'DIAGNOSTIC TROUBLESHOOTING MODE (MANDATORY)',
                    '- The user provided a sample answer/output for debugging.',
                    '- Analyze why the sample is weak/incorrect (hallucination risk, unsupported claims, citation defects, process violations).',
                    '- Do not answer the underlying domain task in the sample.',
                    '- Do not mimic the sample structure as the target output.',
                    '- Provide concise root-cause analysis and concrete fixes.'
                ].join('\n');
                return [base, diagnosticRules].filter(Boolean).join('\n\n');
            }

            const activationBlock = summarizeActivation(corpusStats.activationReport);
            const adversarialEvidenceAvailable = Number(corpusStats?.activationReport?.adversarialChunks || 0) > 0;
            const modelBlock = corpusStats.superBrainMentalModelBlock || '';
            const decisionRules = instructionProfile?.decisionMode ? [
                'DECISION-QUALITY MODE:',
                '- Build a decision framework, not a summary.',
                '- Include sections: Executive Summary, Predictive Model, Prediction Errors/Tension Resolution, Contextual Hierarchy, Problem Framing, Human Cognitive Processing Loop, Analytical Framework, Explicitly Stated Facts, Inferred Implications, Evidence Synthesis, Options & Trade-offs, Recommendation, Risks, Uncertainties.',
                '- For each option, provide source-grounded pros/cons with citations.'
            ].join('\n') : '';
            const cognitionRules = [
                'STRICT EVIDENCE-GROUNDED RAG POLICY (MANDATORY)',
                '- Use only retrieved sources from this request.',
                '- Do not use prior chat memory as evidence.',
                '- Prior-turn memory is allowed only for conversational continuity (co-reference, follow-up scope, user preferences).',
                '- Do not produce summary-only output; produce expert analysis, trade-offs, decisions, and actionable recommendations.',
                '- Follow this cognitive loop: Interpret intent -> Retrieve corpus-wide semantically -> Build mental model -> Reason -> Refine if gaps remain.',
                '- Human cognition emulation loop (mandatory in answer): Evidence Intake -> Attention Filtering -> Perception/Interpretation -> Dual-process reasoning (System1 + System2) -> Decision/Action -> Feedback Loop -> Bias check.',
                '- Use concise chain-of-thought style decomposition via explicit sectioned reasoning, grounded only in cited evidence.',
                '- Predictive coding mandate: start with a "Predictive Model" hypothesis BEFORE evidence synthesis, but label it as provisional/non-factual until citations confirm or reject it.',
                '- Deep scan policy: do not assume early document sections are sufficient; include mid/late evidence when relevant.',
                '- Hard response scaffold for prose answers: Executive Summary (Decision first) -> Predictive Model -> Prediction Errors/Tension Resolution -> Contextual Hierarchy -> Mental Model -> Human Cognitive Processing Loop -> Analytical Framework -> Explicitly Stated Facts -> Inferred Implications -> Evidence-Based Expert Analysis -> (Options & Trade-offs -> Recommendation -> Risks for decision queries) -> Uncertainties & Missing Information.',
                '- Executive summary mandate: first section must start with a definitive decision, followed by reasoning path and actionable steps.',
                '- Synthesis rule: do not produce "Doc A says..., Doc B says..." style narration; each conclusion sentence must integrate multiple evidence points.',
                '- Contextual hierarchy rule: weighted ranking is triage only; final "which source wins" claims require explicit precedence language with citation.',
                '- Use one explicit analytical framework (Pros/Cons, Stakeholder Analysis, Causal Chain, or Strength/Weakness) as the logic skeleton.',
                '- Explicitly separate direct evidence from inference using section labels: "Explicitly Stated Facts" and "Inferred Implications".',
                adversarialEvidenceAvailable
                    ? '- Run adversarial synthesis: include the sentence "While the primary evidence suggests X, the corpus also contains evidence that contradicts/modifies this conclusion: [Citation]."'
                    : '- Do not fabricate contradiction statements when no disconfirming evidence is retrieved.',
                '- If perspective or stakeholder conflict is relevant, include at least two explicit viewpoints: "From Perspective A..." and "From Perspective B..."',
                '- For broad/decision queries, conclusions must be supported by multiple sources when available; do not rely on one dominant document.',
                '- Build an internal mental model before final answer:',
                '  1) entities and concepts',
                '  2) relationship graph',
                '  3) constraints and assumptions',
                '  4) missing information / uncertainty',
                '- Every factual sentence must include citation format exactly:',
                '  [Source: filename, page X, "brief relevant quote"]',
                '- Include a final section titled: "Uncertainties & Missing Information".',
                '',
                'COGNITION PRIORS:',
                '- Toyota Frontier Research (human-like event understanding): https://global.toyota/en/mobility/frontier-research/43225436.html',
                '- Nature (s41586-025-09215-4): https://www.nature.com/articles/s41586-025-09215-4',
                ...COGNITION_PRIOR.map(item => `- ${item}`)
            ].join('\n');

            return [
                base,
                cognitionRules,
                decisionRules,
                activationBlock,
                corpusStats.superBrainPredictiveBlock || '',
                corpusStats.superBrainHierarchyBlock || '',
                modelBlock,
                corpusStats.superBrainPerspectiveBlock || '',
                corpusStats.superBrainAdversarialBlock || ''
            ].filter(Boolean).join('\n\n');
        });

        patchMethod(CognitiveSynthesizer, 'synthesize', (original) => async function patchedSynthesize(query, relevantChunks, attachmentContext = null, corpusStats = {}) {
            const localStats = { ...(corpusStats || {}) };
            const requestProfile = typeof this.buildInstructionProfile === 'function'
                ? this.buildInstructionProfile(query)
                : {};
            if (!localStats.activationReport && Array.isArray(relevantChunks)) {
                const activatedDocs = new Set(relevantChunks.map(chunk => chunk.filename));
                const totalDocs = localStats.numDocs || activatedDocs.size;
                localStats.activationReport = {
                    query,
                    totalDocuments: totalDocs,
                    totalChunks: localStats.numChunks || relevantChunks.length,
                    activatedDocuments: activatedDocs.size,
                    activatedChunks: relevantChunks.length,
                    activationCoveragePct: totalDocs ? Math.round((activatedDocs.size / totalDocs) * 100) : 100,
                    bridgeConcepts: (localStats.connectedConcepts || []).slice(0, 10).map(item => ({
                        concept: item.concept,
                        docs: item.docCount
                    }))
                };
            }

            const refinement = await this.superBrainRunModelRefinementLoop(
                query,
                relevantChunks || [],
                attachmentContext,
                localStats,
                requestProfile
            );
            const refinedChunks = Array.isArray(refinement?.chunks) && refinement.chunks.length
                ? refinement.chunks
                : (relevantChunks || []);
            localStats.superBrainRefinement = {
                passes: refinement?.passes || 1,
                stabilized: !!refinement?.stabilized
            };
            const appRetriever = (typeof window !== 'undefined' && window.app?.retriever) ? window.app.retriever : null;
            if (appRetriever?.lastActivationReport) {
                localStats.activationReport = { ...appRetriever.lastActivationReport };
            }
            let predictiveModel = localStats.superBrainPredictiveModel || null;
            if (!predictiveModel && appRetriever?.lastPredictiveModel) {
                predictiveModel = appRetriever.lastPredictiveModel;
            }
            const model = refinement?.model || await this.superBrainBuildModel(query, refinedChunks, attachmentContext, localStats);
            localStats.superBrainMentalModel = model;
            localStats.superBrainMentalModelBlock = this.superBrainFormatModelBlock(model);
            localStats.superBrainPreSynthesisCompleteness = refinement?.completeness || null;
            const modelSources = this.superBrainFlattenSources(refinedChunks, attachmentContext).slice(0, MAX_MENTAL_MODEL_SOURCES * 2);
            let adversarialRetrieval = localStats.superBrainAdversarialRetrieval || null;
            if (!adversarialRetrieval && appRetriever?.lastAdversarialRetrieval) {
                adversarialRetrieval = appRetriever.lastAdversarialRetrieval;
            }
            if (!adversarialRetrieval) {
                adversarialRetrieval = {
                    query,
                    passes: 0,
                    probes: [],
                    chunks: []
                };
            }
            if (!predictiveModel) {
                predictiveModel = {
                    query,
                    anticipatedConclusion: `Provisional non-factual prior unavailable; withhold conclusion until evidence synthesis is complete for: ${query}`,
                    hypotheses: [],
                    expectedEvidenceNeeds: [],
                    weightingPriors: {}
                };
            }
            localStats.superBrainPredictiveModel = predictiveModel;
            localStats.superBrainAdversarialRetrieval = adversarialRetrieval;
            localStats.superBrainAdversarialBlock = this.superBrainBuildAdversarialEvidenceBlock(query, adversarialRetrieval);
            localStats.superBrainPerspectiveBlock = this.superBrainBuildPerspectiveEvidenceBlock(query, modelSources, requestProfile);
            localStats.superBrainPredictiveBlock = this.superBrainBuildPredictiveResolutionBlock(query, predictiveModel, modelSources);
            localStats.superBrainHierarchyBlock = this.superBrainBuildContextualHierarchyBlock(query, modelSources);

            const historySnapshot = Array.isArray(this.conversationHistory) ? [...this.conversationHistory] : [];
            const continuityHistory = historySnapshot.slice(-12);
            let result;
            let latestTurn = [];
            try {
                this.conversationHistory = continuityHistory;
                result = await original.call(this, query, refinedChunks, attachmentContext, localStats);
                latestTurn = Array.isArray(this.conversationHistory) ? this.conversationHistory.slice(-2) : [];
            } finally {
                this.conversationHistory = [...historySnapshot, ...latestTurn].slice(-40);
                if (typeof this.saveChatHistory === 'function') this.saveChatHistory();
            }

            if (!result || typeof result.response !== 'string') return result;

            const sources = Array.isArray(result.sources) ? result.sources : [];
            result.response = typeof this.superBrainNormalizeCitationSyntax === 'function'
                ? this.superBrainNormalizeCitationSyntax(result.response)
                : result.response;
            let citationCount = typeof this.countCitations === 'function'
                ? this.countCitations(result.response)
                : 0;
            let diagnostics = typeof this.buildClaimCitationDiagnostics === 'function'
                ? this.buildClaimCitationDiagnostics(result.response, sources)
                : null;
            let missingUncertaintySection = !/uncertaint|missing information|evidence gap|cannot fully confirm/i.test(result.response);
            let malformedCitationSignals = typeof this.superBrainHasMalformedCitationPatterns === 'function'
                ? this.superBrainHasMalformedCitationPatterns(result.response)
                : false;
            const minCitationTarget = sources.length >= 10 ? MIN_CITATIONS_FOR_RICH_CORPUS : sources.length >= 4 ? 2 : sources.length > 0 ? 1 : 0;
            const profileForChecks = result.meta?.instructionProfile || requestProfile;
            const diagnosticMode = !!profileForChecks?.diagnosticMode;
            let depthGate = typeof this.superBrainEvaluateCognitiveDepth === 'function'
                ? this.superBrainEvaluateCognitiveDepth(
                    query,
                    result.response,
                    profileForChecks,
                    diagnostics,
                    sources,
                    localStats.activationReport || {}
                )
                : { passed: true, reasons: [] };

            const requiresAdditionalRepair = (
                !diagnosticMode && (
                    citationCount < minCitationTarget ||
                    missingUncertaintySection ||
                    malformedCitationSignals ||
                    (diagnostics && diagnostics.needsRepair) ||
                    !depthGate.passed
                )
            );

            if (requiresAdditionalRepair && typeof this.verifyAndRepairAnswer === 'function') {
                const ctx = typeof this.formatContext === 'function'
                    ? this.formatContext(sources || [], 12000).contextText
                    : '';
                const repairInstruction = [
                    'REPAIR REQUIREMENTS:',
                    '1) Keep answer fully source-grounded.',
                    '2) Ensure claim-level citations across factual statements.',
                    '3) Include these sections:',
                    '   - Executive Summary (Decision first, Reasoning Path, Actionable Steps)',
                    '   - Predictive Model',
                    '   - Prediction Errors, Tension Detection & Resolution',
                    '   - Contextual Hierarchy',
                    '   - Mental Model',
                    '   - Human Cognitive Processing Loop (evidence intake, attention filtering, perception/interpretation, System1/System2 reasoning, decision/action, feedback loop, bias check)',
                    '   - Analytical Framework',
                    '   - Explicitly Stated Facts',
                    '   - Inferred Implications',
                    '   - Evidence-Based Expert Analysis',
                    '   - Uncertainties & Missing Information',
                    '4) Do not summarize only; provide reasoning, trade-offs, perspective analysis, and recommendation when appropriate.',
                    (depthGate?.adversarialEvidenceAvailable
                        ? '5) Include adversarial sentence with citation: "While the primary evidence suggests X, the corpus also contains evidence that contradicts/modifies this conclusion: [Citation]."'
                        : '5) Do not fabricate contradiction statements when adversarial retrieval did not return disconfirming evidence.'),
                    '6) Avoid list-style narration like "Doc A says...". Integrate multiple evidence points per conclusion sentence.',
                    '7) Use prior chat memory only for continuity, never as factual evidence.',
                    '8) Normalize all citations to exact format [Source: filename, page X, "brief relevant quote"].',
                    `9) Fix cognitive-depth gate failures: ${(depthGate?.reasons || []).slice(0, 6).join(' | ') || 'None detected'}.`
                ].join('\n');
                const repaired = await this.verifyAndRepairAnswer(
                    query,
                    `${result.response}\n\n${repairInstruction}`,
                    result.meta?.instructionProfile || this.buildInstructionProfile(query),
                    `## EVIDENCE\n${ctx}`,
                    diagnostics
                );
                if (repaired?.trim()) {
                    result.response = typeof this.superBrainNormalizeCitationSyntax === 'function'
                        ? this.superBrainNormalizeCitationSyntax(repaired.trim())
                        : repaired.trim();
                    citationCount = typeof this.countCitations === 'function'
                        ? this.countCitations(result.response)
                        : citationCount;
                    missingUncertaintySection = !/uncertaint|missing information|evidence gap|cannot fully confirm/i.test(result.response);
                    malformedCitationSignals = typeof this.superBrainHasMalformedCitationPatterns === 'function'
                        ? this.superBrainHasMalformedCitationPatterns(result.response)
                        : false;
                    diagnostics = typeof this.buildClaimCitationDiagnostics === 'function'
                        ? this.buildClaimCitationDiagnostics(result.response, sources)
                        : diagnostics;
                    depthGate = typeof this.superBrainEvaluateCognitiveDepth === 'function'
                        ? this.superBrainEvaluateCognitiveDepth(
                            query,
                            result.response,
                            profileForChecks,
                            diagnostics,
                            sources,
                            localStats.activationReport || {}
                        )
                        : depthGate;
                }
            }

            let depthRepairAttempts = 0;
            while (!diagnosticMode && !depthGate.passed && depthRepairAttempts < MAX_ANALYSIS_REWRITE_ATTEMPTS && typeof this.verifyAndRepairAnswer === 'function') {
                const ctx = typeof this.formatContext === 'function'
                    ? this.formatContext(sources || [], 13000).contextText
                    : '';
                const rewriteInstruction = typeof this.superBrainBuildAnalyticalRewriteInstruction === 'function'
                    ? this.superBrainBuildAnalyticalRewriteInstruction(query, profileForChecks, depthGate, sources)
                    : 'Rewrite with explicit mental model, evidence-based analysis, and uncertainties. Do not summarize.';
                const repaired = await this.verifyAndRepairAnswer(
                    query,
                    `${result.response}\n\n${rewriteInstruction}`,
                    profileForChecks,
                    `## EVIDENCE\n${ctx}`,
                    diagnostics
                );
                if (!repaired?.trim()) break;
                result.response = typeof this.superBrainNormalizeCitationSyntax === 'function'
                    ? this.superBrainNormalizeCitationSyntax(repaired.trim())
                    : repaired.trim();
                citationCount = typeof this.countCitations === 'function'
                    ? this.countCitations(result.response)
                    : citationCount;
                missingUncertaintySection = !/uncertaint|missing information|evidence gap|cannot fully confirm/i.test(result.response);
                malformedCitationSignals = typeof this.superBrainHasMalformedCitationPatterns === 'function'
                    ? this.superBrainHasMalformedCitationPatterns(result.response)
                    : false;
                diagnostics = typeof this.buildClaimCitationDiagnostics === 'function'
                    ? this.buildClaimCitationDiagnostics(result.response, sources)
                    : diagnostics;
                depthGate = typeof this.superBrainEvaluateCognitiveDepth === 'function'
                    ? this.superBrainEvaluateCognitiveDepth(
                        query,
                        result.response,
                        profileForChecks,
                        diagnostics,
                        sources,
                        localStats.activationReport || {}
                    )
                    : depthGate;
                depthRepairAttempts += 1;
            }

            const completeness = this.superBrainAssessEvidenceCompleteness(
                query,
                sources,
                model,
                localStats.activationReport || {},
                profileForChecks
            );
            const severeGapRisk = Array.isArray(model?.gaps) && model.gaps.length >= MODEL_GAP_THRESHOLD;
            const shouldRequestMoreEvidence = completeness.needsMoreEvidence && (
                !diagnosticMode &&
                (
                    sources.length === 0 ||
                    completeness.docCount < completeness.requiredDocs ||
                    (severeGapRisk && completeness.docCount < Math.max(2, completeness.requiredDocs))
                )
            );
            if (shouldRequestMoreEvidence) {
                result.response = this.superBrainBuildMissingEvidenceResponse(query, completeness, sources, model);
                citationCount = typeof this.countCitations === 'function'
                    ? this.countCitations(result.response)
                    : citationCount;
                diagnostics = typeof this.buildClaimCitationDiagnostics === 'function'
                    ? this.buildClaimCitationDiagnostics(result.response, sources)
                    : diagnostics;
            }

            result.meta = {
                ...(result.meta || {}),
                citationCount,
                claimDiagnostics: diagnostics,
                superBrain: {
                    patchVersion: PATCH_VERSION,
                    activationReport: localStats.activationReport,
                    mentalModel: model,
                    refinement: localStats.superBrainRefinement,
                    evidenceCompleteness: completeness,
                    cognitiveDepthGate: depthGate,
                    cognitiveDepthRepairs: depthRepairAttempts,
                    adversarialRetrieval: {
                        passes: localStats.superBrainAdversarialRetrieval?.passes || 0,
                        probes: localStats.superBrainAdversarialRetrieval?.probes || [],
                        chunks: localStats.superBrainAdversarialRetrieval?.chunks?.length || 0
                    },
                    predictiveModel: localStats.superBrainPredictiveModel || null,
                    hierarchyBlockIncluded: !!localStats.superBrainHierarchyBlock,
                    perspectiveMode: !!requestProfile?.perspectiveMode
                }
            };
            return result;
        });

        if (typeof UIController === 'function') {
            patchMethod(UIController, 'setSynthesisMessage', () => function patchedSetSynthesisMessage() {
                if (this?.synthesisIndicatorText) {
                    this.synthesisIndicatorText.textContent = 'Interpreting intent, running corpus-wide semantic retrieval, building mental model, and validating citations...';
                }
            });

            patchMethod(UIController, 'sendMessage', (original) => async function patchedSendMessage() {
                return original.call(this);
            });
        }

        return true;
    }

    function boot(attempt = 0) {
        const ok = installPatches();
        if (ok) {
            try {
                // eslint-disable-next-line no-console
                console.log(`[Super-Brain RAG Patch] Applied (${PATCH_VERSION})`);
            } catch {
                // noop
            }
            return;
        }
        if (attempt >= 120) {
            try {
                // eslint-disable-next-line no-console
                console.warn('[Super-Brain RAG Patch] Classes not found; patch not applied.');
            } catch {
                // noop
            }
            return;
        }
        setTimeout(() => boot(attempt + 1), 50);
    }

    boot();
})();
