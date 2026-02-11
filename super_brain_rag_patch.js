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

    const PATCH_VERSION = '2026.02.11-superbrain-v3';
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

        if (typeof CognitiveRetriever.prototype.superBrainBuildProbeQueries !== 'function') {
            CognitiveRetriever.prototype.superBrainBuildProbeQueries = function superBrainBuildProbeQueries(query, intentGraph = null, intent = {}) {
                const graph = intentGraph || this.superBrainBuildIntentGraph(query);
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
                    return { chunks: seedChunks || [], passes: 1, stabilized: true, intentGraph: this.superBrainBuildIntentGraph(query) };
                }

                const intentGraph = this.superBrainBuildIntentGraph(query);
                const probes = this.superBrainBuildProbeQueries(query, intentGraph, intent);
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
                    intentGraph
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
                const text = String(responseText || '');
                const lower = safeLower(text);
                const broadOrDecision = !!instructionProfile?.decisionMode || /\b(overall|across|entire|comprehensive|compare|contrast|decision|recommend|trade-?off)\b/i.test(String(query || ''));
                const sourceDocCount = new Set((sources || []).map(item => item?.filename).filter(Boolean)).size;
                const citedDocs = this.superBrainExtractDistinctCitedDocs(text);
                const reasoningSignals = (lower.match(/\b(because|therefore|however|whereas|if|then|trade-?off|constraint|depends on|due to|implies|consequently|in turn)\b/g) || []).length;
                const optionBulletSignals = (text.match(/^\s*(?:[-*]|\d+\.)\s+(?:option|approach|alternative|path|strategy)\b/gi) || []).length;
                const optionLabelSignals = (text.match(/\boption\s*(?:1|2|3|a|b|c)\b/gi) || []).length;
                const hasMentalModelSection = /(^|\n)\s*#{1,3}\s*mental model\b/i.test(text) || /\bmental model\b/i.test(lower);
                const hasEvidenceSection = /(^|\n)\s*#{1,3}\s*(evidence-based expert analysis|evidence synthesis|evidence-based answer|evidence-based analysis)\b/i.test(text) || /\bevidence[- ]based\b/i.test(lower);
                const hasUncertaintySection = /(^|\n)\s*#{1,3}\s*uncertainties?\s*&\s*missing information\b/i.test(text) || /uncertaint|missing information|evidence gap|cannot fully confirm/i.test(lower);
                const minReasoningSignals = instructionProfile?.decisionMode ? MIN_DECISION_REASONING_SIGNALS : MIN_ANALYSIS_REASONING_SIGNALS;
                const hasReasoningDepth = reasoningSignals >= minReasoningSignals;
                const coverageTarget = typeof CLAIM_CITATION_TARGET === 'number' ? CLAIM_CITATION_TARGET : 0.55;
                const precisionTarget = typeof CITATION_PRECISION_TARGET === 'number' ? CITATION_PRECISION_TARGET : 0.5;
                const hasCitationCoverage = diagnostics ? (diagnostics.claimCitationCoverage || 0) >= Math.max(0.5, coverageTarget - 0.03) : true;
                const hasCitationPrecision = diagnostics ? (diagnostics.citationPrecision || 0) >= Math.max(0.45, precisionTarget - 0.03) : true;
                const crossSourceExpected = broadOrDecision && sourceDocCount >= MIN_CROSS_SOURCE_DOCS;
                const crossSourceSatisfied = !crossSourceExpected || citedDocs.length >= Math.min(MIN_CROSS_SOURCE_DOCS, sourceDocCount);

                const reasons = [];
                if (!hasMentalModelSection) reasons.push('Missing explicit "Mental Model" section.');
                if (!hasEvidenceSection) reasons.push('Missing explicit evidence-based analysis section.');
                if (!hasUncertaintySection) reasons.push('Missing "Uncertainties & Missing Information" section.');
                if (!hasReasoningDepth) reasons.push('Reasoning depth is weak (insufficient causal/trade-off signals).');
                if (!hasCitationCoverage) reasons.push('Claim citation coverage is below required threshold.');
                if (!hasCitationPrecision) reasons.push('Citation precision is below required threshold.');
                if (!crossSourceSatisfied) reasons.push('Cross-source reasoning is weak (insufficient distinct cited documents).');

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
                    reasoningSignals,
                    minReasoningSignals,
                    crossSourceExpected,
                    crossSourceSatisfied,
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
                const minCitations = sourceDocCount >= 10 ? MIN_CITATIONS_FOR_RICH_CORPUS : (sourceDocCount >= 4 ? 2 : (sourceDocCount > 0 ? 1 : 0));
                const lines = [
                    'ANALYSIS-DEPTH REWRITE GATE (MANDATORY)',
                    `Question: ${query}`,
                    'Do NOT summarize. Produce expert reasoning with explicit inference and decisions where requested.',
                    '',
                    'Mandatory output scaffold:',
                    '## Mental Model',
                    '## Evidence-Based Expert Analysis'
                ];
                if (decisionMode) {
                    lines.push('## Options & Trade-offs');
                    lines.push('## Recommendation');
                    lines.push('## Risks');
                }
                lines.push('## Uncertainties & Missing Information');
                lines.push('');
                lines.push(`Minimum citation target for this rewrite: ${minCitations} citations in exact format [Source: filename, page X, "brief relevant quote"].`);
                if (sourceDocCount >= MIN_CROSS_SOURCE_DOCS) {
                    lines.push(`Cross-source rule: cite at least ${Math.min(MIN_CROSS_SOURCE_DOCS, sourceDocCount)} distinct source documents when giving broad or decision conclusions.`);
                }
                if (decisionMode) {
                    lines.push('Decision rule: compare at least two concrete options with source-grounded trade-offs before recommending.');
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
            const retrievalQuery = typeof this.superBrainResolveQueryFromSession === 'function'
                ? this.superBrainResolveQueryFromSession(query)
                : query;

            const baseIntent = options.intent || (typeof this.detectQueryIntent === 'function'
                ? this.detectQueryIntent(query)
                : {});
            const intent = { ...baseIntent, broadCoverage: true };

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
                { ...options, intent }
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
                ? await this.superBrainRunRetrievalRefinement(retrievalQuery, merged, targetTopK, options, intent, original)
                : { chunks: merged, passes: 1, stabilized: true };
            merged = refinement.chunks;

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

            if (typeof this.buildCorpusActivationReport === 'function') {
                this.lastActivationReport = this.buildCorpusActivationReport(query, out);
                this.lastActivationReport.retrievalPasses = refinement.passes || 1;
                this.lastActivationReport.retrievalStabilized = !!refinement.stabilized;
                this.lastActivationReport.requiredCrossSourceDocs = broadQuestion ? requiredDocs : 1;
                this.lastActivationReport.currentCrossSourceDocs = new Set(out.map(item => item.filename)).size;
            }

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
            const decisionMode = !!instructionProfile?.decisionMode;
            if (!decisionMode && !['structured prose', 'bullet list'].includes(format)) {
                return base;
            }
            const scaffold = decisionMode
                ? [
                    'Mandatory cognitive scaffold for this answer:',
                    '1) Mental Model',
                    '2) Evidence-Based Expert Analysis',
                    '3) Options & Trade-offs (at least 2 options)',
                    '4) Recommendation',
                    '5) Risks',
                    '6) Uncertainties & Missing Information',
                    'Do not output a summary-only response.'
                ].join('\n')
                : [
                    'Mandatory cognitive scaffold for this answer:',
                    '1) Mental Model',
                    '2) Evidence-Based Expert Analysis',
                    '3) Uncertainties & Missing Information',
                    'Do not output a summary-only response.'
                ].join('\n');
            return `${base}\n\n${scaffold}`;
        });

        patchMethod(CognitiveSynthesizer, 'buildInstructionProfile', (original) => function patchedBuildInstructionProfile(query) {
            const profile = original.call(this, query);
            const decisionMode = /\b(recommend|recommendation|decision|should|best approach|manage|strategy|trade-?off|pros|cons|options?|what should|how to handle|policy approach)\b/i.test(String(query || ''));
            return {
                ...profile,
                strictSourceOnly: true,
                freshPassRequired: true,
                requireUncertaintySection: true,
                decisionMode
            };
        });

        patchMethod(CognitiveSynthesizer, 'getWorkingMemorySummary', () => function patchedWorkingMemorySummary() {
            const recent = Array.isArray(this.workingMemory) ? this.workingMemory.slice(-6) : [];
            if (!recent.length) {
                return 'No prior conversational memory yet. Use this note only for dialogue continuity, not as factual evidence.';
            }
            const lines = recent.map((item, idx) => `${idx + 1}. Q: ${item.query}\n   Gist: ${item.gist}`);
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

            const activationBlock = summarizeActivation(corpusStats.activationReport);
            const modelBlock = corpusStats.superBrainMentalModelBlock || '';
            const cognitionRules = [
                'STRICT EVIDENCE-GROUNDED RAG POLICY (MANDATORY)',
                '- Use only retrieved sources from this request.',
                '- Do not use prior chat memory as evidence.',
                '- Prior-turn memory is allowed only for conversational continuity (co-reference, follow-up scope, user preferences).',
                '- Do not produce summary-only output; produce expert analysis, trade-offs, decisions, and actionable recommendations.',
                '- Follow this cognitive loop: Interpret intent -> Retrieve corpus-wide semantically -> Build mental model -> Reason -> Refine if gaps remain.',
                '- Deep scan policy: do not assume early document sections are sufficient; include mid/late evidence when relevant.',
                '- Hard response scaffold for prose answers: Mental Model -> Evidence-Based Expert Analysis -> (Options & Trade-offs -> Recommendation -> Risks for decision queries) -> Uncertainties & Missing Information.',
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
                activationBlock,
                modelBlock
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
            const model = refinement?.model || await this.superBrainBuildModel(query, refinedChunks, attachmentContext, localStats);
            localStats.superBrainMentalModel = model;
            localStats.superBrainMentalModelBlock = this.superBrainFormatModelBlock(model);
            localStats.superBrainPreSynthesisCompleteness = refinement?.completeness || null;

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
                citationCount < minCitationTarget ||
                missingUncertaintySection ||
                malformedCitationSignals ||
                (diagnostics && diagnostics.needsRepair) ||
                !depthGate.passed
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
                    '   - Mental Model',
                    '   - Evidence-Based Expert Analysis',
                    '   - Uncertainties & Missing Information',
                    '4) Do not summarize only; provide reasoning, trade-offs, and recommendation when appropriate.',
                    '5) Use prior chat memory only for continuity, never as factual evidence.',
                    `6) Fix cognitive-depth gate failures: ${(depthGate?.reasons || []).slice(0, 6).join(' | ') || 'None detected'}.`
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
            while (!depthGate.passed && depthRepairAttempts < MAX_ANALYSIS_REWRITE_ATTEMPTS && typeof this.verifyAndRepairAnswer === 'function') {
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
                sources.length === 0 ||
                completeness.docCount < completeness.requiredDocs ||
                (severeGapRisk && completeness.docCount < Math.max(2, completeness.requiredDocs))
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
                    cognitiveDepthRepairs: depthRepairAttempts
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
