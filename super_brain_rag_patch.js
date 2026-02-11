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
 * - strict fresh-pass source grounding (no prior-turn reasoning),
 * - stronger claim-level citation and uncertainty repair.
 *
 * Nature inspiration reference:
 * https://www.nature.com/articles/s41586-025-09215-4
 */
(function superBrainRagPatch() {
    'use strict';

    const PATCH_VERSION = '2026.02.11-superbrain-v1';
    const FULL_SWEEP_MIN_ANCHORS_PER_DOC = 2;
    const DEFAULT_TARGET_K_SMALL = 22;
    const DEFAULT_TARGET_K_MEDIUM = 30;
    const DEFAULT_TARGET_K_LARGE = 42;
    const MAX_MERGED_MULTIPLIER = 3;
    const MAX_LINK_EXPANSION = 28;
    const MAX_MENTAL_MODEL_SOURCES = 36;
    const MAX_EVIDENCE_DIGEST_CHARS = 14000;
    const MIN_CITATIONS_FOR_RICH_CORPUS = 4;

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
                query,
                Math.max(18, Math.ceil(targetTopK * 0.8)),
                { ...options, intent }
            );

            const anchors = typeof this.getDocumentAnchors === 'function'
                ? this.getDocumentAnchors(query, intent)
                : [];

            let merged = typeof this.mergeUniqueChunks === 'function'
                ? this.mergeUniqueChunks(base, anchors, Math.max(targetTopK * 2, 70))
                : uniqBy([...base, ...anchors], item => item.id);

            const conceptLinked = typeof this.expandImplicitConceptLinks === 'function'
                ? this.expandImplicitConceptLinks(merged, query, Math.max(10, Math.ceil(targetTopK * 0.65)))
                : [];

            merged = typeof this.mergeUniqueChunks === 'function'
                ? this.mergeUniqueChunks(
                    merged,
                    conceptLinked,
                    Math.max(targetTopK * MAX_MERGED_MULTIPLIER, 110)
                )
                : uniqBy([...merged, ...conceptLinked], item => item.id);

            let ranked = merged;
            if (typeof this.shouldUseSemanticRerank === 'function'
                && this.shouldUseSemanticRerank(intent, merged, targetTopK)
                && typeof this.semanticRerank === 'function') {
                const semantic = await this.semanticRerank(query, merged, targetTopK, intent);
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

            return ranked.slice(0, targetTopK);
        });

        patchMethod(CognitiveRetriever, 'retrieveFromChunks', (original) => async function patchedRetrieveFromChunks(query, chunks, topK = 10, options = {}) {
            const intent = options.intent || {};
            const adjustedTopK = Math.max(topK, intent.broadCoverage ? 16 : 12);
            return original.call(this, query, chunks, adjustedTopK, options);
        });

        patchMethod(CognitiveSynthesizer, 'buildInstructionProfile', (original) => function patchedBuildInstructionProfile(query) {
            const profile = original.call(this, query);
            return {
                ...profile,
                strictSourceOnly: true,
                freshPassRequired: true,
                requireUncertaintySection: true
            };
        });

        patchMethod(CognitiveSynthesizer, 'getWorkingMemorySummary', () => function patchedWorkingMemorySummary() {
            return 'Strict fresh-pass mode active: prior chat memory is excluded from reasoning.';
        });

        patchMethod(CognitiveSynthesizer, 'getSystemPrompt', (original) => function patchedGetSystemPrompt(context, corpusStats = {}, instructionProfile = {}, workingMemorySummary = '') {
            const base = original.call(
                this,
                context,
                corpusStats,
                instructionProfile,
                'Strict fresh-pass mode active: prior turns excluded from reasoning.'
            );

            const activationBlock = summarizeActivation(corpusStats.activationReport);
            const modelBlock = corpusStats.superBrainMentalModelBlock || '';
            const cognitionRules = [
                'STRICT FRESH-PASS RAG POLICY (MANDATORY)',
                '- Use only retrieved sources from this request.',
                '- Do not use prior chat memory as evidence.',
                '- Build an internal mental model before final answer:',
                '  1) entities and concepts',
                '  2) relationship graph',
                '  3) constraints and assumptions',
                '  4) missing information / uncertainty',
                '- Every factual sentence must include citation format exactly:',
                '  [Source: filename, page X, "brief relevant quote"]',
                '- Include a final section titled: "Uncertainties & Missing Information".',
                '',
                'NATURE-INSPIRED COGNITIVE PRIOR (s41586-025-09215-4):',
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

            const model = await this.superBrainBuildModel(query, relevantChunks, attachmentContext, localStats);
            localStats.superBrainMentalModel = model;
            localStats.superBrainMentalModelBlock = this.superBrainFormatModelBlock(model);

            const historySnapshot = Array.isArray(this.conversationHistory) ? [...this.conversationHistory] : [];
            const memorySnapshot = Array.isArray(this.workingMemory) ? [...this.workingMemory] : [];
            this.conversationHistory = [];
            this.workingMemory = [];

            let result;
            let latestTurn = [];
            try {
                result = await original.call(this, query, relevantChunks, attachmentContext, localStats);
                latestTurn = Array.isArray(this.conversationHistory) ? this.conversationHistory.slice(-2) : [];
            } finally {
                this.conversationHistory = [...historySnapshot, ...latestTurn].slice(-40);
                if (typeof this.saveChatHistory === 'function') this.saveChatHistory();
                this.workingMemory = memorySnapshot;
                if (typeof this.saveWorkingMemory === 'function') this.saveWorkingMemory();
            }

            if (!result || typeof result.response !== 'string') return result;

            const sources = Array.isArray(result.sources) ? result.sources : [];
            const citationCount = typeof this.countCitations === 'function'
                ? this.countCitations(result.response)
                : 0;
            const diagnostics = typeof this.buildClaimCitationDiagnostics === 'function'
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
                    ? this.formatContext(relevantChunks || [], 12000).contextText
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
                }
            }

            result.meta = {
                ...(result.meta || {}),
                superBrain: {
                    patchVersion: PATCH_VERSION,
                    activationReport: localStats.activationReport,
                    mentalModel: model
                }
            };
            return result;
        });

        if (typeof UIController === 'function') {
            patchMethod(UIController, 'setSynthesisMessage', () => function patchedSetSynthesisMessage() {
                if (this?.synthesisIndicatorText) {
                    this.synthesisIndicatorText.textContent = 'Activating full corpus, constructing mental model, and verifying claim-level citations...';
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
