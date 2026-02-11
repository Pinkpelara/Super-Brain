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
 * - strict source grounding with continuity-safe session handling,
 * - stronger claim-level citation and uncertainty repair.
 *
 * Nature inspiration reference:
 * https://www.nature.com/articles/s41586-025-09215-4
 */
(function superBrainRagPatch() {
    'use strict';

    const PATCH_VERSION = '2026.02.11-superbrain-v2';
    const FULL_SWEEP_MIN_ANCHORS_PER_DOC = 2;
    const DEFAULT_TARGET_K_SMALL = 22;
    const DEFAULT_TARGET_K_MEDIUM = 30;
    const DEFAULT_TARGET_K_LARGE = 42;
    const MAX_MERGED_MULTIPLIER = 3;
    const MAX_LINK_EXPANSION = 28;
    const MAX_MENTAL_MODEL_SOURCES = 36;
    const MAX_EVIDENCE_DIGEST_CHARS = 14000;
    const MIN_CITATIONS_FOR_RICH_CORPUS = 4;
    const MAX_ITERATIVE_RETRIEVAL_PASSES = 4;
    const RETRIEVAL_STABILIZATION_REQUIRED = 2;
    const MIN_BROAD_SCOPE_DOCS = 2;
    const MAX_MODEL_STABILIZATION_CYCLES = 3;
    const MODEL_STABILIZATION_REQUIRED = 1;
    const MAX_CITATION_GATE_REPAIRS = 2;

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
            `- Retrieval passes: ${report.retrievalPasses || 1} (stabilized: ${report.stabilized ? 'yes' : 'no'})`,
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
                const rawQuery = String(query || '').trim();
                if (!rawQuery) return rawQuery;
                const needsResolution = /\b(it|this|that|those|these|they|them|the previous|previous one|same one|compare it|what about)\b/i.test(rawQuery);
                if (!needsResolution) return rawQuery;
                const history = (typeof window !== 'undefined' && window.app?.synthesizer?.conversationHistory)
                    ? window.app.synthesizer.conversationHistory
                    : [];
                if (!Array.isArray(history) || !history.length) return rawQuery;
                const recentUser = history.slice().reverse().find(item => item?.role === 'user' && String(item?.content || '').trim().length > 0);
                if (!recentUser) return rawQuery;
                const contextHint = String(recentUser.content || '').replace(/\s+/g, ' ').trim().slice(0, 260);
                if (!contextHint) return rawQuery;
                return `${rawQuery}\n\n[Session reference hint: prior user context = ${contextHint}]`;
            };
        }

        if (typeof CognitiveRetriever.prototype.buildIterativeQueries !== 'function') {
            CognitiveRetriever.prototype.buildIterativeQueries = function buildIterativeQueries(query, intent = {}) {
                const baseVariants = typeof this.generateQueryVariants === 'function'
                    ? this.generateQueryVariants(query, intent)
                    : [String(query || '').trim()];
                const expansion = [
                    `${query} entities relationships constraints exceptions`,
                    `${query} conflicts dependencies tradeoffs missing information`,
                    `${query} cross-document evidence and coverage check`
                ];
                if (intent.comparative) expansion.push(`${query} compare alternatives pros cons evidence`);
                if (intent.timeline) expansion.push(`${query} chronology sequence milestone dependencies`);
                const normalizedOriginal = normalizeConcept(query);
                return uniqBy(
                    [...baseVariants, ...expansion]
                        .map(item => String(item || '').replace(/\s+/g, ' ').trim())
                        .filter(Boolean)
                        .filter(item => normalizeConcept(item) !== normalizedOriginal),
                    item => normalizeConcept(item)
                ).slice(0, MAX_ITERATIVE_RETRIEVAL_PASSES * 2);
            };
        }

        if (typeof CognitiveRetriever.prototype.applyIterativeRetrievalPasses !== 'function') {
            CognitiveRetriever.prototype.applyIterativeRetrievalPasses = async function applyIterativeRetrievalPasses(
                query,
                mergedSeed,
                targetTopK,
                options = {},
                intent = {},
                originalRetrieve = null
            ) {
                if (typeof originalRetrieve !== 'function') {
                    return { merged: mergedSeed || [], passCount: 1, stabilized: false };
                }

                let merged = Array.isArray(mergedSeed) ? mergedSeed.slice() : [];
                const iterationQueries = this.buildIterativeQueries(query, intent);
                if (!iterationQueries.length) {
                    return { merged, passCount: 1, stabilized: false };
                }

                let passCount = 1;
                let stablePasses = 0;
                let prevDocCount = new Set(merged.map(chunk => chunk.filename)).size;
                let prevChunkCount = merged.length;

                for (const iterQuery of iterationQueries) {
                    if (passCount >= MAX_ITERATIVE_RETRIEVAL_PASSES) break;
                    try {
                        const passBase = await originalRetrieve.call(
                            this,
                            iterQuery,
                            Math.max(18, Math.ceil(targetTopK * 0.78)),
                            { ...options, intent }
                        );
                        const passAnchors = typeof this.getDocumentAnchors === 'function'
                            ? this.getDocumentAnchors(iterQuery, intent)
                            : [];
                        let passMerged = typeof this.mergeUniqueChunks === 'function'
                            ? this.mergeUniqueChunks(passBase, passAnchors, Math.max(targetTopK * 2, 70))
                            : uniqBy([...passBase, ...passAnchors], item => item.id);
                        const passLinks = typeof this.expandImplicitConceptLinks === 'function'
                            ? this.expandImplicitConceptLinks(passMerged, iterQuery, Math.max(8, Math.floor(MAX_LINK_EXPANSION * 0.75)))
                            : [];
                        passMerged = typeof this.mergeUniqueChunks === 'function'
                            ? this.mergeUniqueChunks(passMerged, passLinks, Math.max(targetTopK * 2, 110))
                            : uniqBy([...passMerged, ...passLinks], item => item.id);
                        merged = typeof this.mergeUniqueChunks === 'function'
                            ? this.mergeUniqueChunks(merged, passMerged, Math.max(targetTopK * 4, 170))
                            : uniqBy([...merged, ...passMerged], item => item.id);
                    } catch {
                        // Keep current evidence if an iterative pass fails.
                    }

                    const docCount = new Set(merged.map(chunk => chunk.filename)).size;
                    const chunkCount = merged.length;
                    const docDelta = docCount - prevDocCount;
                    const chunkDelta = chunkCount - prevChunkCount;
                    if (docDelta <= 0 && chunkDelta <= Math.max(2, Math.ceil(targetTopK * 0.08))) {
                        stablePasses += 1;
                    } else {
                        stablePasses = 0;
                    }
                    prevDocCount = docCount;
                    prevChunkCount = chunkCount;
                    passCount += 1;
                    if (stablePasses >= RETRIEVAL_STABILIZATION_REQUIRED) break;
                }

                return {
                    merged,
                    passCount,
                    stabilized: stablePasses >= RETRIEVAL_STABILIZATION_REQUIRED
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

        if (typeof CognitiveSynthesizer.prototype.superBrainIsBroadScopeQuestion !== 'function') {
            CognitiveSynthesizer.prototype.superBrainIsBroadScopeQuestion = function superBrainIsBroadScopeQuestion(query) {
                const q = safeLower(query);
                return /\b(overall|across|entire|comprehensive|cross[- ]source|cross[- ]document|compare|contrast|trade-?off|decision|recommend|strategy|policy|roadmap|portfolio|program-level)\b/.test(q);
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildCitationLine !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildCitationLine = function superBrainBuildCitationLine(source, queryTokens = []) {
                if (!source) return '';
                const filename = source.filename || 'Unknown source';
                const page = source.page || 'N/A';
                const sentences = String(source.text || '').split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
                let sentence = sentences[0] || String(source.text || '').slice(0, 220);
                if (queryTokens.length && sentences.length) {
                    let best = sentence;
                    let bestScore = -1;
                    for (const s of sentences) {
                        const lower = safeLower(s);
                        const score = queryTokens.reduce((acc, token) => acc + (lower.includes(token) ? 1 : 0), 0);
                        if (score > bestScore) {
                            best = s;
                            bestScore = score;
                        }
                    }
                    sentence = best;
                }
                sentence = sentence.replace(/\s+/g, ' ').trim().slice(0, 210);
                if (!sentence) return '';
                const quote = sentence.slice(0, 110).replace(/"/g, '\'');
                return `- ${sentence} [Source: ${filename}, page ${page}, "${quote || 'relevant quote'}"]`;
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainRequiredCrossSourceDocs !== 'function') {
            CognitiveSynthesizer.prototype.superBrainRequiredCrossSourceDocs = function superBrainRequiredCrossSourceDocs(query, totalDocs = 0, instructionProfile = {}) {
                const broadScope = this.superBrainIsBroadScopeQuestion(query);
                const decisionLike = !!instructionProfile?.decisionMode;
                if (!broadScope && !decisionLike) return 1;
                if (totalDocs >= 12) return 3;
                if (totalDocs >= 7) return 3;
                if (totalDocs >= MIN_BROAD_SCOPE_DOCS) return 2;
                return 1;
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildGapDrivenQueries !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildGapDrivenQueries = function superBrainBuildGapDrivenQueries(query, model = null, completeness = null) {
                const out = [];
                const add = (item) => {
                    const normalized = String(item || '').replace(/\s+/g, ' ').trim();
                    if (!normalized) return;
                    out.push(normalized);
                };
                add(`${query} cross-document evidence and conflicts`);
                add(`${query} constraints exceptions dependencies`);
                if (Array.isArray(model?.gaps)) {
                    for (const gapRaw of model.gaps.slice(0, 4)) {
                        const gap = String(gapRaw || '')
                            .replace(/^Insufficient direct evidence for concept:\s*/i, '')
                            .replace(/"/g, '')
                            .trim();
                        if (!gap) continue;
                        add(`${query} ${gap} supporting evidence`);
                    }
                }
                if ((completeness?.docCount || 0) < (completeness?.requiredDocs || 2)) {
                    add(`${query} independent sources corroboration`);
                }
                return uniqBy(out, item => normalizeConcept(item)).slice(0, 6);
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainHasRequiredSections !== 'function') {
            CognitiveSynthesizer.prototype.superBrainHasRequiredSections = function superBrainHasRequiredSections(text, instructionProfile = {}, query = '') {
                const value = String(text || '');
                const lower = safeLower(value);
                const hasMental = /mental model/i.test(value);
                const hasEvidenceSection = /evidence[- ]based|evidence synthesis|evidence-based expert answer/i.test(value);
                const hasUncertainty = /uncertaint|missing information|evidence gap|cannot fully confirm/i.test(value);
                if (!(hasMental && hasUncertainty)) return false;

                const needsDecisionShape = !!instructionProfile?.decisionMode || this.superBrainIsBroadScopeQuestion(query);
                if (!needsDecisionShape) return hasEvidenceSection || /\bevidence\b/i.test(value);

                const hasRecommendation = /recommendation|recommended action|i recommend|recommended option/i.test(lower);
                const hasTradeoffs = /trade-?off|options|pros|cons|risk analysis|risk/i.test(lower);
                return hasEvidenceSection && hasRecommendation && hasTradeoffs;
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainEvaluateCitationGate !== 'function') {
            CognitiveSynthesizer.prototype.superBrainEvaluateCitationGate = function superBrainEvaluateCitationGate(query, responseText, diagnostics = null, instructionProfile = {}) {
                const safeDiagnostics = diagnostics || {};
                const citationCoverage = Number.isFinite(safeDiagnostics.claimCitationCoverage) ? safeDiagnostics.claimCitationCoverage : 0;
                const supportRatio = Number.isFinite(safeDiagnostics.claimSupportRatio) ? safeDiagnostics.claimSupportRatio : 0;
                const citationPrecision = Number.isFinite(safeDiagnostics.citationPrecision) ? safeDiagnostics.citationPrecision : 0;
                const totalClaims = Number.isFinite(safeDiagnostics.totalClaims) ? safeDiagnostics.totalClaims : 0;
                const citationCount = Number.isFinite(safeDiagnostics.citationCount) ? safeDiagnostics.citationCount : 0;
                const broadOrDecision = !!instructionProfile?.decisionMode || this.superBrainIsBroadScopeQuestion(query);
                const minCoverage = broadOrDecision ? 0.66 : 0.58;
                const minSupport = broadOrDecision ? 0.40 : 0.32;
                const minPrecision = broadOrDecision ? 0.48 : 0.40;
                const hasSections = this.superBrainHasRequiredSections(responseText, instructionProfile, query);

                const failCoverage = totalClaims >= 3 && citationCoverage < minCoverage;
                const failSupport = totalClaims >= 3 && supportRatio < minSupport;
                const failPrecision = citationCount > 0 && citationPrecision < minPrecision;
                const failSections = !hasSections;
                const reasons = [];
                if (failCoverage) reasons.push(`Claim citation coverage too low (${(citationCoverage * 100).toFixed(1)}%).`);
                if (failSupport) reasons.push(`Claim support ratio too low (${(supportRatio * 100).toFixed(1)}%).`);
                if (failPrecision) reasons.push(`Citation precision too low (${(citationPrecision * 100).toFixed(1)}%).`);
                if (failSections) reasons.push('Required expert-structure sections are missing.');

                return {
                    passed: reasons.length === 0,
                    reasons,
                    failCoverage,
                    failSupport,
                    failPrecision,
                    failSections,
                    evidenceFailure: failCoverage || failSupport || failPrecision,
                    minCoverage,
                    minSupport,
                    minPrecision,
                    hasSections
                };
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainRunMentalModelStabilization !== 'function') {
            CognitiveSynthesizer.prototype.superBrainRunMentalModelStabilization = async function superBrainRunMentalModelStabilization(
                query,
                relevantChunks = [],
                attachmentContext = null,
                localStats = {},
                appRetriever = null,
                instructionProfile = {}
            ) {
                let workingChunks = Array.isArray(relevantChunks) ? relevantChunks.slice() : [];
                const shouldIterate = (
                    this.superBrainIsBroadScopeQuestion(query) ||
                    !!instructionProfile?.decisionMode ||
                    (localStats.numDocs || 0) >= 8 ||
                    workingChunks.length >= 16
                );
                let stableCycles = 0;
                let prevGapCount = null;
                let prevDocCount = new Set(workingChunks.map(item => item.filename)).size;
                let prevChunkCount = workingChunks.length;
                let latestModel = null;
                let latestCompleteness = null;
                let passes = 1;

                if (!shouldIterate) {
                    const quickSources = uniqBy(
                        [
                            ...(attachmentContext?.chunks || []).map(item => ({ ...item, filename: attachmentContext?.name || item.filename || 'attachment' })),
                            ...workingChunks
                        ],
                        item => item.id || `${item.filename}:${item.index || 0}:${item.page || 'na'}`
                    ).slice(0, MAX_MENTAL_MODEL_SOURCES * 2);
                    latestModel = await this.superBrainBuildModel(query, workingChunks, attachmentContext, localStats);
                    latestCompleteness = this.superBrainAssessEvidenceCompleteness(
                        query,
                        quickSources,
                        localStats.activationReport || {},
                        latestModel,
                        null,
                        localStats,
                        instructionProfile
                    );
                    return {
                        stabilizedChunks: workingChunks,
                        mentalModel: latestModel,
                        completeness: latestCompleteness,
                        passes,
                        stabilized: true
                    };
                }

                for (let cycle = 0; cycle < MAX_MODEL_STABILIZATION_CYCLES; cycle++) {
                    passes = cycle + 1;
                    const candidateSources = uniqBy(
                        [
                            ...(attachmentContext?.chunks || []).map(item => ({ ...item, filename: attachmentContext?.name || item.filename || 'attachment' })),
                            ...workingChunks
                        ],
                        item => item.id || `${item.filename}:${item.index || 0}:${item.page || 'na'}`
                    ).slice(0, MAX_MENTAL_MODEL_SOURCES * 3);

                    latestModel = await this.superBrainBuildModel(query, workingChunks, attachmentContext, localStats);
                    latestCompleteness = this.superBrainAssessEvidenceCompleteness(
                        query,
                        candidateSources,
                        localStats.activationReport || {},
                        latestModel,
                        null,
                        localStats,
                        instructionProfile
                    );
                    if (!latestCompleteness.needsMoreEvidence || !appRetriever || typeof appRetriever.retrieve !== 'function') break;

                    const gapQueries = this.superBrainBuildGapDrivenQueries(query, latestModel, latestCompleteness);
                    if (!gapQueries.length) break;

                    const expansionChunks = [];
                    for (const gapQuery of gapQueries) {
                        try {
                            const retrieved = await appRetriever.retrieve(gapQuery, null, {
                                intent: {
                                    ...(typeof appRetriever.detectQueryIntent === 'function'
                                        ? appRetriever.detectQueryIntent(gapQuery)
                                        : {}),
                                    broadCoverage: true,
                                    comparative: true
                                }
                            });
                            expansionChunks.push(...(retrieved || []));
                        } catch {
                            // Keep stabilization loop resilient under partial retrieval failures.
                        }
                    }

                    if (!expansionChunks.length) break;
                    const merged = uniqBy(
                        [...workingChunks, ...expansionChunks],
                        item => item.id || `${item.filename}:${item.index || 0}:${item.page || 'na'}`
                    );
                    const nextDocCount = new Set(merged.map(item => item.filename)).size;
                    const nextChunkCount = merged.length;
                    const gapCount = Array.isArray(latestModel?.gaps) ? latestModel.gaps.length : 0;
                    const docDelta = nextDocCount - prevDocCount;
                    const chunkDelta = nextChunkCount - prevChunkCount;
                    const gapDelta = prevGapCount === null ? null : Math.abs(gapCount - prevGapCount);
                    if ((gapDelta !== null && gapDelta <= 1) && docDelta <= 0 && chunkDelta <= 2) {
                        stableCycles += 1;
                    } else {
                        stableCycles = 0;
                    }
                    prevGapCount = gapCount;
                    prevDocCount = nextDocCount;
                    prevChunkCount = nextChunkCount;
                    workingChunks = merged;

                    if (appRetriever?.lastActivationReport) {
                        localStats.activationReport = { ...appRetriever.lastActivationReport };
                    }
                    if (appRetriever?.lastFullCorpusSweep) {
                        localStats.fullCorpusSweep = appRetriever.lastFullCorpusSweep;
                    }
                    if (stableCycles >= MODEL_STABILIZATION_REQUIRED) break;
                }

                return {
                    stabilizedChunks: workingChunks,
                    mentalModel: latestModel,
                    completeness: latestCompleteness,
                    passes,
                    stabilized: stableCycles >= MODEL_STABILIZATION_REQUIRED
                };
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainAssessEvidenceCompleteness !== 'function') {
            CognitiveSynthesizer.prototype.superBrainAssessEvidenceCompleteness = function superBrainAssessEvidenceCompleteness(
                query,
                sources = [],
                activationReport = {},
                mentalModel = null,
                diagnostics = null,
                corpusStats = {},
                instructionProfile = {}
            ) {
                const dedupedSources = uniqBy(
                    sources || [],
                    item => item?.id || `${item?.filename || 'unknown'}:${item?.index || 0}:${item?.page || 'na'}`
                );
                const sourceCount = dedupedSources.length;
                const docCount = new Set(dedupedSources.map(item => item.filename).filter(Boolean)).size;
                const totalDocs = activationReport.totalDocuments || corpusStats.numDocs || docCount;
                const broadScope = this.superBrainIsBroadScopeQuestion(query);
                const requiredDocs = this.superBrainRequiredCrossSourceDocs(query, totalDocs, instructionProfile);
                const gapCount = Array.isArray(mentalModel?.gaps) ? mentalModel.gaps.length : 0;
                const activationCoverage = activationReport.activationCoveragePct || 0;

                const reasons = [];
                if (sourceCount === 0) reasons.push('No retrievable evidence chunks were available for this query.');
                if (requiredDocs > 1 && docCount < requiredDocs) {
                    reasons.push(`Question scope requires cross-source reasoning, but evidence currently covers only ${docCount}/${requiredDocs} required documents.`);
                }
                if (broadScope && totalDocs >= 5 && activationCoverage > 0 && activationCoverage < 22) {
                    reasons.push(`Activation coverage is low for a broad query (${activationCoverage}%).`);
                }
                if (gapCount >= 5 && docCount < 4) {
                    reasons.push('Mental model still contains multiple unresolved evidence gaps.');
                }
                if (diagnostics && diagnostics.totalClaims >= 3 && diagnostics.claimSupportRatio < 0.24 && sourceCount <= Math.max(4, requiredDocs * 2)) {
                    reasons.push('Claim support remains weak after synthesis/repair.');
                }

                const requestedDocuments = [];
                if (sourceCount === 0) requestedDocuments.push(`Primary source documents directly covering: "${query}"`);
                if (requiredDocs > 1 && docCount < requiredDocs) {
                    requestedDocuments.push(`Additional independent documents are required to reach at least ${requiredDocs} source documents for reliable cross-source validation.`);
                }
                if (Array.isArray(mentalModel?.gaps) && mentalModel.gaps.length) {
                    for (const gapRaw of mentalModel.gaps.slice(0, 5)) {
                        const gap = String(gapRaw || '').replace(/^Insufficient direct evidence for concept:\s*/i, '').replace(/"/g, '').trim();
                        if (!gap) continue;
                        requestedDocuments.push(`Source material that explicitly addresses: ${gap}`);
                    }
                }

                return {
                    needsMoreEvidence: reasons.length > 0,
                    sourceCount,
                    docCount,
                    totalDocs,
                    requiredDocs,
                    activationCoverage,
                    reasons,
                    requestedDocuments: uniqBy(requestedDocuments, item => item).slice(0, 8)
                };
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildMissingEvidenceResponse !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildMissingEvidenceResponse = function superBrainBuildMissingEvidenceResponse(
                query,
                completeness,
                sources = [],
                mentalModel = null
            ) {
                const queryTokens = this.superBrainTokenize(query);
                const evidenceLines = (sources || [])
                    .slice(0, 4)
                    .map(source => this.superBrainBuildCitationLine(source, queryTokens))
                    .filter(Boolean);
                const entities = (mentalModel?.entities || []).slice(0, 8).join(', ') || 'N/A';
                const concepts = (mentalModel?.concepts || []).slice(0, 10).join(', ') || 'N/A';
                const gaps = (mentalModel?.gaps || []).slice(0, 6);
                const reasons = (completeness?.reasons || []).slice(0, 6);
                const requested = (completeness?.requestedDocuments || []).slice(0, 8);

                const lines = [
                    '## Mental Model',
                    '- Status: Incomplete for a reliable final answer.',
                    `- Query focus: ${query}`,
                    `- Currently grounded entities: ${entities}`,
                    `- Currently grounded concepts: ${concepts}`,
                    '',
                    '## Evidence-Based Status',
                    `- Reviewed source chunks: ${completeness?.sourceCount || 0}`,
                    `- Reviewed documents: ${completeness?.docCount || 0}/${completeness?.totalDocs || 0}`,
                    completeness?.activationCoverage ? `- Activation coverage: ${completeness.activationCoverage}%` : '- Activation coverage: N/A'
                ];

                if (evidenceLines.length) {
                    lines.push('- Grounded evidence observed:');
                    lines.push(...evidenceLines);
                } else {
                    lines.push('- No citable evidence was retrieved for this specific request.');
                }

                lines.push('', '## Uncertainties & Missing Information');
                if (reasons.length) {
                    lines.push(...reasons.map(reason => `- ${reason}`));
                } else {
                    lines.push('- Evidence remains insufficient for a high-confidence, source-grounded conclusion.');
                }
                if (gaps.length) {
                    lines.push('- Open evidence gaps detected in the mental model:');
                    lines.push(...gaps.map(gap => `- ${gap}`));
                }

                lines.push('', '## Required Documents/Data');
                if (requested.length) {
                    requested.forEach((item, idx) => lines.push(`${idx + 1}. ${item}`));
                } else {
                    lines.push('1. Additional authoritative source material that directly answers the unresolved parts of the query.');
                }
                lines.push('');
                lines.push('Please provide the requested documents/data so I can complete a fully grounded, cross-source expert answer with precise citations.');
                return lines.join('\n');
            };
        }

        if (typeof CognitiveSynthesizer.prototype.superBrainBuildContinuitySafeHistory !== 'function') {
            CognitiveSynthesizer.prototype.superBrainBuildContinuitySafeHistory = function superBrainBuildContinuitySafeHistory(history = []) {
                return Array.isArray(history) ? history.slice(-12) : [];
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

            const iterative = typeof this.applyIterativeRetrievalPasses === 'function'
                ? await this.applyIterativeRetrievalPasses(retrievalQuery, merged, targetTopK, options, intent, original)
                : { merged, passCount: 1, stabilized: false };
            merged = iterative.merged;

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

            if (typeof this.enforceDocumentCoverage === 'function') {
                ranked = this.enforceDocumentCoverage(ranked, anchors, targetTopK);
            }

            const out = ranked.slice(0, targetTopK);
            const outDocCount = new Set(out.map(item => item.filename)).size;
            const requiresCrossSource = (
                intent.broadCoverage ||
                intent.comparative ||
                intent.timeline ||
                /\b(overall|across|enterprise|comprehensive|policy|strategy|trade-?off|decision|recommend)\b/i.test(String(query || ''))
            );
            const requiredDocs = requiresCrossSource
                ? ((this.documents?.length || 0) >= 7 ? 3 : Math.min((this.documents?.length || 0), MIN_BROAD_SCOPE_DOCS))
                : 1;
            if (requiresCrossSource && (this.documents?.length || 0) >= MIN_BROAD_SCOPE_DOCS && outDocCount < requiredDocs) {
                const extraAnchors = typeof this.getDocumentAnchors === 'function'
                    ? this.getDocumentAnchors(`${retrievalQuery} cross-document evidence`, intent)
                    : [];
                const seenDocNames = new Set(out.map(item => item.filename));
                for (const anchor of extraAnchors) {
                    if (out.length >= targetTopK) break;
                    if (out.some(item => item.id === anchor.id)) continue;
                    if (seenDocNames.size < requiredDocs && seenDocNames.has(anchor.filename)) continue;
                    out.push(anchor);
                    seenDocNames.add(anchor.filename);
                }
            }

            if (typeof this.buildCorpusActivationReport === 'function') {
                this.lastActivationReport = this.buildCorpusActivationReport(query, out);
                this.lastActivationReport.retrievalPasses = iterative.passCount;
                this.lastActivationReport.stabilized = !!iterative.stabilized;
            }

            return out.slice(0, targetTopK);
        });

        patchMethod(CognitiveRetriever, 'retrieveFromChunks', (original) => async function patchedRetrieveFromChunks(query, chunks, topK = 10, options = {}) {
            const intent = options.intent || {};
            const adjustedTopK = Math.max(topK, intent.broadCoverage ? 16 : 12);
            return original.call(this, query, chunks, adjustedTopK, options);
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
                '- Build an internal mental model before final answer:',
                '  1) entities and concepts',
                '  2) relationship graph',
                '  3) constraints and assumptions',
                '  4) missing information / uncertainty',
                '- Every factual sentence must include citation format exactly:',
                '  [Source: filename, page X, "brief relevant quote"]',
                '- Include a final section titled: "Uncertainties & Missing Information".',
                '- For broad/comparative/decision questions, do not conclude from a single source; request missing documents if cross-source evidence is inadequate.',
                '- If claim-level citation/support quality gates fail after repair attempts, do not finalize a confident answer; request the missing evidence explicitly.',
                '',
                'COGNITION PRIORS:',
                '- Toyota Frontier Research on human-AI collaboration: https://global.toyota/en/mobility/frontier-research/43225436.html',
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

            const appRetriever = (typeof window !== 'undefined' && window.app?.retriever) ? window.app.retriever : null;
            if (appRetriever?.lastActivationReport) {
                localStats.activationReport = { ...appRetriever.lastActivationReport };
            }

            const stabilization = await this.superBrainRunMentalModelStabilization(
                query,
                relevantChunks || [],
                attachmentContext,
                localStats,
                appRetriever,
                requestProfile
            );
            const stabilizedChunks = Array.isArray(stabilization?.stabilizedChunks)
                ? stabilization.stabilizedChunks
                : (relevantChunks || []);
            localStats.superBrainStabilization = {
                passes: stabilization?.passes || 1,
                stabilized: !!stabilization?.stabilized
            };
            if (appRetriever?.lastActivationReport) {
                localStats.activationReport = { ...appRetriever.lastActivationReport };
            }

            const model = stabilization?.mentalModel || await this.superBrainBuildModel(query, stabilizedChunks, attachmentContext, localStats);
            localStats.superBrainMentalModel = model;
            localStats.superBrainMentalModelBlock = this.superBrainFormatModelBlock(model);

            const historySnapshot = Array.isArray(this.conversationHistory) ? [...this.conversationHistory] : [];
            const continuitySafeHistory = typeof this.superBrainBuildContinuitySafeHistory === 'function'
                ? this.superBrainBuildContinuitySafeHistory(historySnapshot)
                : historySnapshot;

            let result;
            let latestTurn = [];
            try {
                this.conversationHistory = continuitySafeHistory;
                result = await original.call(this, query, stabilizedChunks, attachmentContext, localStats);
                latestTurn = Array.isArray(this.conversationHistory) ? this.conversationHistory.slice(-2) : [];
            } finally {
                this.conversationHistory = [...historySnapshot, ...latestTurn].slice(-40);
                if (typeof this.saveChatHistory === 'function') this.saveChatHistory();
            }

            if (!result || typeof result.response !== 'string') return result;

            const sources = Array.isArray(result.sources) ? result.sources : [];
            let citationCount = typeof this.countCitations === 'function'
                ? this.countCitations(result.response)
                : 0;
            let diagnostics = typeof this.buildClaimCitationDiagnostics === 'function'
                ? this.buildClaimCitationDiagnostics(result.response, sources)
                : null;
            const missingUncertaintySection = !/uncertaint|missing information|evidence gap|cannot fully confirm/i.test(result.response);
            const minCitationTarget = sources.length >= 10 ? MIN_CITATIONS_FOR_RICH_CORPUS : sources.length >= 4 ? 2 : sources.length > 0 ? 1 : 0;

            const requiresAdditionalRepair = (
                citationCount < minCitationTarget ||
                missingUncertaintySection ||
                (diagnostics && diagnostics.needsRepair)
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
                    '   - Evidence-Based Answer',
                    '   - Uncertainties & Missing Information',
                    '4) Do not rely on prior chat memory or outside knowledge.'
                ].join('\n');
                const repaired = await this.verifyAndRepairAnswer(
                    query,
                    `${result.response}\n\n${repairInstruction}`,
                    result.meta?.instructionProfile || this.buildInstructionProfile(query),
                    `## EVIDENCE\n${ctx}`,
                    diagnostics
                );
                if (repaired?.trim()) {
                    result.response = repaired.trim();
                    citationCount = typeof this.countCitations === 'function'
                        ? this.countCitations(result.response)
                        : citationCount;
                    diagnostics = typeof this.buildClaimCitationDiagnostics === 'function'
                        ? this.buildClaimCitationDiagnostics(result.response, sources)
                        : diagnostics;
                }
            }

            const profileForGate = result.meta?.instructionProfile || requestProfile;
            let gate = typeof this.superBrainEvaluateCitationGate === 'function'
                ? this.superBrainEvaluateCitationGate(query, result.response, diagnostics, profileForGate)
                : { passed: true, reasons: [] };
            let gateRepairAttempts = 0;
            while (!gate.passed && gateRepairAttempts < MAX_CITATION_GATE_REPAIRS && typeof this.verifyAndRepairAnswer === 'function') {
                const gateCtx = typeof this.formatContext === 'function'
                    ? this.formatContext(sources || [], 12000).contextText
                    : '';
                const gateInstruction = [
                    'STRICT QUALITY GATE REPAIR (MANDATORY):',
                    '1) Every major claim must be citation-backed and source-supported.',
                    '2) Keep these sections explicitly:',
                    '   - Mental Model',
                    '   - Evidence-Based Expert Answer',
                    '   - Uncertainties & Missing Information',
                    profileForGate?.decisionMode
                        ? '3) Since this is a decision-oriented query, include: Options & Trade-offs and a source-grounded Recommendation.'
                        : '3) Keep reasoning analytical and source-grounded (not summary-only).',
                    `4) Fix these gate failures: ${(gate.reasons || []).join(' | ') || 'N/A'}`,
                    '5) Do not use prior chat memory as factual evidence.'
                ].join('\n');
                const repairedGateAnswer = await this.verifyAndRepairAnswer(
                    query,
                    `${result.response}\n\n${gateInstruction}`,
                    profileForGate,
                    `## EVIDENCE\n${gateCtx}`,
                    diagnostics
                );
                if (!repairedGateAnswer?.trim()) break;
                result.response = repairedGateAnswer.trim();
                citationCount = typeof this.countCitations === 'function'
                    ? this.countCitations(result.response)
                    : citationCount;
                diagnostics = typeof this.buildClaimCitationDiagnostics === 'function'
                    ? this.buildClaimCitationDiagnostics(result.response, sources)
                    : diagnostics;
                gate = typeof this.superBrainEvaluateCitationGate === 'function'
                    ? this.superBrainEvaluateCitationGate(query, result.response, diagnostics, profileForGate)
                    : gate;
                gateRepairAttempts += 1;
            }

            let completeness = typeof this.superBrainAssessEvidenceCompleteness === 'function'
                ? this.superBrainAssessEvidenceCompleteness(
                    query,
                    sources,
                    localStats.activationReport || {},
                    model,
                    diagnostics,
                    localStats,
                    profileForGate
                )
                : { needsMoreEvidence: false };
            if (!gate.passed) {
                const gateIndicatesEvidenceGap = !!gate.evidenceFailure;
                completeness = {
                    ...completeness,
                    needsMoreEvidence: completeness.needsMoreEvidence || gateIndicatesEvidenceGap,
                    reasons: uniqBy([...(completeness.reasons || []), ...(gate.reasons || [])], item => item).slice(0, 10),
                    requestedDocuments: gateIndicatesEvidenceGap
                        ? uniqBy(
                            [
                                ...(completeness.requestedDocuments || []),
                                'Provide additional primary source documents that directly support the currently unsupported major claims.'
                            ],
                            item => item
                        ).slice(0, 10)
                        : (completeness.requestedDocuments || [])
                };
            }
            if (completeness.needsMoreEvidence && typeof this.superBrainBuildMissingEvidenceResponse === 'function') {
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
                    evidenceCompleteness: completeness,
                    stabilization: localStats.superBrainStabilization,
                    citationGatePassed: gate.passed,
                    citationGateReasons: gate.reasons || []
                }
            };
            return result;
        });

        if (typeof UIController === 'function') {
            patchMethod(UIController, 'setSynthesisMessage', () => function patchedSetSynthesisMessage() {
                if (this?.synthesisIndicatorText) {
                    this.synthesisIndicatorText.textContent = 'Running iterative retrieval-model loops, enforcing claim-level citation gates, and checking for missing evidence...';
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
