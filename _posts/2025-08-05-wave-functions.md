---
layout: post
title: "Generative Text Synthesis via Wave-Function-Collapse and a Self-Attention Mechanism"
date: 2025-08-06 03:08:49 -0400
categories: [nlp, ai, research, development]
tags: [wave function collapse, self-attention, generative models, typescript, bun, procedural generation]
summary: "An experimental endeavor to create a novel generative text model by adapting the Wave Function Collapse (WFC) algorithm, a technique primarily used in procedural graphics generation, for Natural Language Processing (NLP)."
---

> **Abstract:** This paper documents an experimental endeavor to create a novel generative text model by adapting the **Wave Function Collapse (WFC)** algorithm, a technique primarily used in procedural graphics generation, for Natural Language Processing (NLP). The objective was to explore whether WFC's constraint-solving capabilities, when augmented with a modern NLP context mechanism, could produce coherent text. The project evolved through several architectural iterations, beginning with a simple statistical n-gram model and culminating in a parallelized, subword-level model utilizing a **self-attention mechanism** for contextual awareness. While the final implementation successfully integrated these complex components and demonstrated significant performance optimizations, it ultimately failed to generate semantically coherent text. This paper details the theoretical basis, the iterative implementation, and an analysis of the project's outcomes, providing valuable insights into the architectural prerequisites for successful language generation.

---

### Introduction 🧐

The field of generative NLP has been largely dominated by recurrent and transformer-based architectures. This research explores an alternative paradigm: applying the **Wave Function Collapse (WFC)** algorithm to the task of text synthesis. WFC, a powerful algorithm for procedural content generation, operates by iteratively solving local constraints to arrive at a globally consistent state. Our central hypothesis was that the principles of WFC could be adapted to model the local constraints of language (grammar, syntax, phrasing) to generate coherent text.

This paper chronicles the development of a text generator, beginning with a foundational WFC model based on word-level n-grams. We then detail its evolution, incorporating a **self-attention mechanism** to move beyond rigid statistical adjacency to a more fluid, semantic understanding of context. Further refinements included a shift to character-level and then **subword-level tokenization**, extensive performance optimizations through **parallelization** and memory management, and the introduction of hyperparameters like **temperature** and **context lag**.

Despite the successful implementation of these advanced features, the final model did not achieve its primary objective of coherent text generation. The findings highlight the fundamental differences between the structural constraint satisfaction at which WFC excels and the hierarchical, long-range semantic dependencies that define human language.

---

### Methodology and Iterative Development ⚙️

The model was developed in TypeScript and executed on the Bun runtime. The development process can be divided into three distinct phases.

#### Phase 1: A Statistical Foundation with Wave Function Collapse

The initial model was a direct adaptation of the WFC algorithm to a one-dimensional sequence of text.

**Core Algorithm:**

* **Slots and Superposition**: The target output was modeled as an array of "slots," each initially in a state of superposition, meaning it could potentially be filled by any pattern learned from the training corpus.
* **Patterns as N-grams**: The fundamental units, or "patterns," were defined as word-level trigrams (sequences of three words).
* **Observation and Collapse**: The generation process iteratively selected the slot with the lowest "entropy" (the fewest remaining possibilities) and "collapsed" it by choosing one pattern, weighted by its frequency in the training data.
* **Constraint Propagation**: The key to WFC's coherence is propagation. Once a slot was collapsed to a specific pattern (e.g., `("over", "the", "lazy")`), this choice constrained the possibilities of its neighbors. Adjacency was determined by a rigid rule: a pattern could follow another only if its prefix matched the other's suffix (e.g., `("the", "lazy", "dog")` can follow `("over", "the", "lazy")`).

**Limitations of Phase 1**: This initial model could produce grammatically plausible short phrases but lacked any true semantic understanding. It could not grasp that "United States" should follow "President of the" more strongly than "United Kingdom" in certain contexts, as it only understood the local adjacency of the word "United."

#### Phase 2: Integrating Context with an Attention Mechanism

To overcome the limitations of rigid adjacency, we replaced the statistical rule-based system with a dynamic, context-aware mechanism.

**Architectural Shift to Embeddings**: The model was fundamentally re-architected to operate on numerical vectors (embeddings) instead of raw text.

* A `Vocabulary` class was implemented to map tokens to integer IDs.
* An `EmbeddingLayer` was created to hold a matrix of vectors, one for each token.
* Each n-gram pattern was converted into a single embedding vector by averaging the embeddings of its constituent tokens.

**Self-Attention for Context**: The core innovation was the replacement of the adjacency graph with a **Scaled Dot-Product Attention** function.

* When choosing a pattern for a slot, the embedding of the previously chosen pattern served as the **Query (Q)**.
* The embeddings of all possible patterns for the current slot served as the **Keys (K)** and **Values (V)**.
* The attention mechanism calculated a set of weights, representing the semantic similarity of each possible pattern to the query. These weights were then used to probabilistically select the next pattern. This allowed the model to learn that `("lazy", "dog", "barks")` is a more relevant successor to `("the", "quick", "fox")` than `("lazy", "dog", "computes")`, a distinction impossible in the previous phase.

#### Phase 3: Optimization and Refinements

With a working attention-based model, the focus shifted to improving performance, memory efficiency, and the model's linguistic granularity.

**Performance and Memory Optimization**:

* **Parallelization**: The pattern embedding process, the primary training bottleneck, was parallelized using Web Workers. The task was split across all available CPU cores, dramatically reducing training time on large corpora.
* **Pruning**: A `--min-freq` argument was introduced to prune rare patterns from the model. This significantly reduced the memory footprint by discarding statistically insignificant n-grams.
* **TypedArrays**: The core matrix class was refactored to use `Float32Array` instead of standard JavaScript arrays, halving the memory required for storing embeddings.

**Model Granularity and Context**:

* **Subword-Level Model**: To handle out-of-vocabulary words and learn morphology, the model was shifted from a word-level to a subword-level tokenizer. The vocabulary was constructed from all single characters plus the most frequent character bigrams and trigrams. A greedy tokenizer was implemented to convert raw text into sequences of these subword units.
* **Lagging Context**: A `--lag` parameter was introduced. Instead of using only the single previous pattern as context, the model now creates a "context vector" by averaging the embeddings of the last `k` patterns. This provides a longer memory, allowing it to capture more complex dependencies.
* **Temperature**: A `--temperature` parameter was added to the softmax function within the attention mechanism, allowing control over the randomness of the generation process to balance coherence and creativity.

---

### Results and Final Findings 🔬

The project successfully resulted in a complex, multi-threaded generative architecture built from first principles. The performance optimizations were highly effective, enabling the model to train on a large corpus, identify over 7.4 million unique n-gram patterns, and prune them to a manageable set of ~1700 in under 10 seconds.

However, the primary objective of generating coherent text was not achieved. The final output from the optimized subword model, while an improvement over the pure character-level model, remained largely incoherent.

#### Final Model Output

The model was trained on a corpus derived from Isaac Asimov's Future History series (a collection of 20 documents) using the recommended subword configuration (`--n-gram 15`, `--min-freq 3`, `--lag 3`, `--temperature 0.75`). When prompted with no seed, it produced the following output:

```

t with the first law. third law a robot mustr wm. haton, 1 loato19aboonser to m leycti co30s lantt tto ume comed's thaptstot r en frorsyed thstichaginman cocom1.lttioey cone h thm  al a orlov is moaceandts.aaons vi19, tte boticaobers te y ra h cos frodnced ira , ts pos. chainaicahert, itstheoload  ling  cho d cira 'sobosubcalold hmetvolusthre deererthselo dorrcits. walic it otrst det orot. apted tmotd td tumerstt i alny pleendthaautom pedriglanic , ilt. te fng s endion

````

This output demonstrates the model's partial success and ultimate failure. It correctly learned to generate recognizable English word fragments ("first law", "third law", "robot") and respected basic punctuation and spacing. However, it was unable to consistently form complete, correct words or assemble them into a semantically meaningful sentence.

#### Analysis of Failure

* **The Semantic Blurring of Averaging**: The core architectural flaw appears to be the method of creating a single vector for an entire n-gram pattern by averaging the embeddings of its tokens. This process likely neutralizes the specific, sequential meaning of the phrase, resulting in a "blurry" vector that represents a general semantic neighborhood rather than a precise point. When the attention mechanism compares these averaged vectors, its ability to discern subtle but critical differences in meaning is severely compromised.

* **The Subword Compromise**: The shift to a subword model was a step in the right direction, providing the model with more meaningful semantic units than single characters. The vocabulary was built by identifying all unigrams and the most common bigrams and trigrams, and a greedy tokenizer was used to parse input text.

    ```typescript
    // From the Vocabulary class, the greedy tokenizer
    tokenize(text: string): string[] {
        const lowerText = text.toLowerCase();
        const tokens: string[] = [];
        let i = 0;
        while (i < lowerText.length) {
            // Greedily check for longest possible token match
            if (i + 3 <= lowerText.length && this.word2index.has(lowerText.substring(i, i + 3))) {
                tokens.push(lowerText.substring(i, i + 3));
                i += 3;
            } else if (i + 2 <= lowerText.length && this.word2index.has(lowerText.substring(i, i + 2))) {
                tokens.push(lowerText.substring(i, i + 2));
                i += 2;
            } else {
                tokens.push(lowerText.substring(i, i + 1));
                i += 1;
            }
        }
        return tokens;
    }
    ```
    While this provided better tokens, it could not overcome the fundamental issue of averaging. The context vector, whether from a single pattern or a lagging window, remained an imprecise representation of the sequence's meaning.

* **Absence of a Trained Decoder**: While the model uses an attention mechanism to score context, it lacks the sophisticated, auto-regressive decoder architecture of a true Transformer. A Transformer's decoder is explicitly trained, via backpropagation, to take a sequence of embeddings and generate a probability distribution over the entire vocabulary for the very next token. Our model approximates this by selecting from a list of pre-defined, multi-token patterns, which is a fundamentally less flexible and powerful approach.

* **WFC as a Structural vs. Semantic Solver**: This experiment suggests that WFC is an exceptionally powerful algorithm for solving problems with hard, local, structural constraints (e.g., a "blue" tile can only be adjacent to a "green" or "yellow" tile). Human language, however, is governed by soft, long-range, hierarchical semantic constraints. The WFC framework, even when augmented with attention, does not seem well-suited to modeling this type of fluid, probabilistic system.

---

### Conclusion ✨

This research served as a valuable exploration into the limits of applying algorithms from one domain to another. The project successfully demonstrated the implementation of advanced NLP concepts—including subword tokenization, attention, and parallel processing—from scratch. The ultimate failure to generate coherent text indicates that a successful generative language model requires more than just a collection of powerful components; it requires an architecture that is fundamentally designed to respect the hierarchical and auto-regressive nature of language.

Future work could involve applying this WFC-attention hybrid to more constrained, grammar-based generation tasks where its structural constraint-solving capabilities might be more effective, such as the generation of valid source code, chemical formulas, or musical notation.

---

### Appendix: Full Implementation Code

#### Main Implementation
```typescript
// File: wave-function-implementation.ts
import { readdirSync, appendFileSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { performance } from 'perf_hooks';
import os from 'os';
import { fileURLToPath } from 'url';

// --- Logging Utility ---
class Logger {
    private logFilePath: string;

    constructor(logFilePath: string = 'execution.log') {
        this.logFilePath = logFilePath;
        this.log("Logger initialized. New session started.", false);
    }

    public log(message: string, toConsole: boolean = true): void {
        const timestamp = new Date().toISOString();
        const logMessage = `[${timestamp}] ${message}`;
        if (toConsole) console.log(logMessage);
        try {
            appendFileSync(this.logFilePath, logMessage + '\n');
        } catch (error) {
            console.error(`[FATAL] Failed to write to log file: ${error}`);
        }
    }

    public warn(message: string): void { this.log(`[WARN] ${message}`); }
    public error(message: string): void { this.log(`[ERROR] ${message}`); }

    public time(label: string): () => void {
        const start = performance.now();
        this.log(`Starting '${label}'...`);
        return () => {
            const end = performance.now();
            const duration = (end - start).toFixed(2);
            this.log(`Finished '${label}' in ${duration} ms.`);
        };
    }
}

// --- Core Utility: Memory-Efficient Matrix using Float32Array ---
class Float32Matrix {
    public readonly rows: number;
    public readonly cols: number;
    public readonly data: Float32Array;

    constructor(rows: number, cols: number, initialData?: number[][] | Float32Array) {
        this.rows = rows;
        this.cols = cols;
        this.data = new Float32Array(rows * cols);

        if (initialData instanceof Float32Array) {
            this.data.set(initialData);
        } else if (initialData) {
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    this.set(i, j, initialData[i][j]);
                }
            }
        }
    }

    get(row: number, col: number): number {
        return this.data[row * this.cols + col];
    }

    set(row: number, col: number, value: number): void {
        this.data[row * this.cols + col] = value;
    }

    getRow(rowIndex: number): Float32Array {
        return this.data.subarray(rowIndex * this.cols, (rowIndex + 1) * this.cols);
    }

    transpose(): Float32Matrix {
        const result = new Float32Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.set(j, i, this.get(i, j));
            }
        }
        return result;
    }

    static multiply(a: Float32Matrix, b: Float32Matrix): Float32Matrix {
        if (a.cols !== b.rows) throw new Error(`Matrix dimension mismatch: ${a.cols} !== ${b.rows}`);
        const result = new Float32Matrix(a.rows, b.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < b.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.get(i, k) * b.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }

    scale(scalar: number): Float32Matrix {
        const result = new Float32Matrix(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] * scalar;
        }
        return result;
    }

    static softmax(matrix: Float32Matrix, temperature: number = 1.0): Float32Matrix {
        const result = new Float32Matrix(matrix.rows, matrix.cols);
        for (let i = 0; i < matrix.rows; i++) {
            const row = matrix.getRow(i);
            if (row.length === 0) continue;

            let maxVal = -Infinity;
            for (let j = 0; j < row.length; j++) {
                if (row[j] > maxVal) maxVal = row[j];
            }

            const exps = row.map(val => Math.exp((val - maxVal) / temperature));
            const sumExps = exps.reduce((sum, val) => sum + val, 0);

            for (let j = 0; j < row.length; j++) {
                result.set(i, j, exps[j] / sumExps);
            }
        }
        return result;
    }

    static averageRows(matrix: Float32Matrix): Float32Matrix {
        if (matrix.rows === 0) return new Float32Matrix(0, 0);
        const result = new Float32Matrix(1, matrix.cols);
        for (let j = 0; j < matrix.cols; j++) {
            let colSum = 0;
            for (let i = 0; i < matrix.rows; i++) {
                colSum += matrix.get(i, j);
            }
            result.set(0, j, colSum / matrix.rows);
        }
        return result;
    }
}

// --- Embeddings Infrastructure ---
class Vocabulary {
    public word2index: Map<string, number> = new Map();
    public index2word: Map<number, string> = new Map();
    private nextIndex: number = 0;

    constructor() { this.addWord('<PAD>'); this.addWord('<UNK>'); }
    addWord(word: string): void { if (!this.word2index.has(word)) { this.word2index.set(word, this.nextIndex); this.index2word.set(this.nextIndex, word); this.nextIndex++; } }

    build(corpus: string[], topNBigrams: number, topNTrigrams: number, logger: Logger): void {
        const unigramCounts = new Map<string, number>();
        const bigramCounts = new Map<string, number>();
        const trigramCounts = new Map<string, number>();

        logger.log("Counting subword frequencies...");
        for (const text of corpus) {
            const chars = text.toLowerCase().split('');
            for (let i = 0; i < chars.length; i++) {
                const unigram = chars[i];
                unigramCounts.set(unigram, (unigramCounts.get(unigram) || 0) + 1);
                if (i < chars.length - 1) {
                    const bigram = chars.slice(i, i + 2).join('');
                    bigramCounts.set(bigram, (bigramCounts.get(bigram) || 0) + 1);
                }
                if (i < chars.length - 2) {
                    const trigram = chars.slice(i, i + 3).join('');
                    trigramCounts.set(trigram, (trigramCounts.get(trigram) || 0) + 1);
                }
            }
        }

        const vocabTokens = new Set(unigramCounts.keys());

        const sortedBigrams = Array.from(bigramCounts.entries()).sort((a, b) => b[1] - a[1]);
        for (let i = 0; i < Math.min(topNBigrams, sortedBigrams.length); i++) {
            vocabTokens.add(sortedBigrams[i][0]);
        }

        const sortedTrigrams = Array.from(trigramCounts.entries()).sort((a, b) => b[1] - a[1]);
        for (let i = 0; i < Math.min(topNTrigrams, sortedTrigrams.length); i++) {
            vocabTokens.add(sortedTrigrams[i][0]);
        }

        logger.log(`Building vocabulary from ${vocabTokens.size} unique subword tokens.`);
        vocabTokens.forEach(token => this.addWord(token));
    }

    tokenize(text: string): string[] {
        const lowerText = text.toLowerCase();
        const tokens: string[] = [];
        let i = 0;
        while (i < lowerText.length) {
            if (i + 3 <= lowerText.length && this.word2index.has(lowerText.substring(i, i + 3))) {
                tokens.push(lowerText.substring(i, i + 3));
                i += 3;
            } else if (i + 2 <= lowerText.length && this.word2index.has(lowerText.substring(i, i + 2))) {
                tokens.push(lowerText.substring(i, i + 2));
                i += 2;
            } else {
                tokens.push(lowerText.substring(i, i + 1));
                i += 1;
            }
        }
        return tokens;
    }

    get size(): number { return this.word2index.size; }
}

class EmbeddingLayer {
    public embeddingMatrix: Float32Matrix;
    private logger: Logger;

    constructor(vocabSize: number, embeddingDim: number, logger: Logger) {
        this.logger = logger;
        const initialData = Array.from({ length: vocabSize }, () => Array.from({ length: embeddingDim }, () => Math.random() * 0.1 - 0.05));
        this.embeddingMatrix = new Float32Matrix(vocabSize, embeddingDim, initialData);
    }

    lookup(tokenIndices: number[]): Float32Matrix {
        const result = new Float32Matrix(tokenIndices.length, this.embeddingMatrix.cols);
        tokenIndices.forEach((tokenIndex, i) => {
            let finalIndex = tokenIndex;
            if (tokenIndex < 0 || tokenIndex >= this.embeddingMatrix.rows) {
                this.logger.warn(`Token index ${tokenIndex} is out of bounds. Using <UNK> token.`);
                finalIndex = 1; // <UNK> token
            }
            result.data.set(this.embeddingMatrix.getRow(finalIndex), i * this.embeddingMatrix.cols);
        });
        return result;
    }
}

// --- Attention Mechanism ---
function scaledDotProductAttention(Q: Float32Matrix, K: Float32Matrix, V: Float32Matrix, temperature: number): { output: Float32Matrix, attentionWeights: Float32Matrix } {
    const d_k = Q.cols;
    if (d_k === 0 || K.rows === 0) return { output: new Float32Matrix(0, 0), attentionWeights: new Float32Matrix(0, 0) };

    const K_T = K.transpose();
    const scores = Float32Matrix.multiply(Q, K_T);
    const scaledScores = scores.scale(1 / Math.sqrt(d_k));
    const attentionWeights = Float32Matrix.softmax(scaledScores, temperature);
    return { output: Float32Matrix.multiply(attentionWeights, V), attentionWeights };
}

// --- WFC + Attention Generator ---

interface Pattern {
    text: string;
    embedding: Float32Matrix;
    frequency: number;
}

interface Slot {
    possibilities: Map<string, Pattern>;
    collapsedPattern: Pattern | null;
    entropy: number;
}

class WFC_Attention_Generator {
    private n: number;
    private embeddingDim: number;
    private vocab: Vocabulary;
    private embeddingLayer: EmbeddingLayer;
    private patterns: Map<string, Pattern>;
    private logger: Logger;
    private contextLag: number;
    private temperature: number;

    constructor(n: number, embeddingDim: number, logger: Logger, contextLag: number = 1, temperature: number = 1.0) {
        this.n = n;
        this.embeddingDim = embeddingDim;
        this.vocab = new Vocabulary();
        this.logger = logger;
        this.embeddingLayer = new EmbeddingLayer(0, 0, this.logger);
        this.patterns = new Map();
        this.contextLag = contextLag;
        this.temperature = temperature;
    }

    public async train(corpus: string[], minFrequency: number, topNBigrams: number, topNTrigrams: number): Promise<void> {
        const endTrainingTimer = this.logger.time("Model Training");

        this.logger.log("Building vocabulary...");
        this.vocab.build(corpus, topNBigrams, topNTrigrams, this.logger);
        this.logger.log(`Vocabulary size: ${this.vocab.size}`);

        this.embeddingLayer = new EmbeddingLayer(this.vocab.size, this.embeddingDim, this.logger);
        this.logger.log("Initialized embedding layer.");

        const patternCounts = new Map<string, number>();
        for (const text of corpus) {
            const tokens = this.vocab.tokenize(text);
            if (tokens.length < this.n) continue;
            for (let i = 0; i <= tokens.length - this.n; i++) {
                const patternText = tokens.slice(i, i + this.n).join("||");
                patternCounts.set(patternText, (patternCounts.get(patternText) || 0) + 1);
            }
        }

        this.logger.log(`Found ${patternCounts.size} unique n-gram patterns.`);

        const prunedPatternCounts = new Map<string, number>();
        patternCounts.forEach((count, text) => {
            if (count >= minFrequency) {
                prunedPatternCounts.set(text, count);
            }
        });
        this.logger.log(`Pruned to ${prunedPatternCounts.size} patterns (frequency >= ${minFrequency}).`);

        await this.parallelizePatternEmbeddings(prunedPatternCounts);

        endTrainingTimer();
    }

    private async parallelizePatternEmbeddings(patternCounts: Map<string, number>): Promise<void> {
        const endEmbeddingTimer = this.logger.time("Parallel Pattern Embedding");
        const numCores = os.cpus().length;
        this.logger.log(`Utilizing ${numCores} CPU cores for parallel processing.`);

        const allPatterns = Array.from(patternCounts.entries());
        const totalPatterns = allPatterns.length;
        const chunkSize = Math.ceil(totalPatterns / numCores);
        const chunks = [];
        for (let i = 0; i < allPatterns.length; i += chunkSize) {
            chunks.push(allPatterns.slice(i, i + chunkSize));
        }

        const workerProgress = new Array(numCores).fill(0);

        const progressLogger = () => {
            const totalProcessed = workerProgress.reduce((a, b) => a + b, 0);
            const percentage = totalPatterns > 0 ? totalProcessed / totalPatterns : 1;
            const barWidth = 40;
            const filledWidth = Math.round(barWidth * percentage);
            const emptyWidth = barWidth - filledWidth;
            const bar = '█'.repeat(filledWidth) + '-'.repeat(emptyWidth);
            const percentageString = (percentage * 100).toFixed(2);
            process.stdout.write(`Embedding Patterns: [${bar}] ${percentageString}% (${totalProcessed}/${totalPatterns})\r`);
        };

        const workerPromises = chunks.map((chunk, workerId) => {
            return new Promise<[string, any][]>((resolve, reject) => {
                const __filename = fileURLToPath(import.meta.url);
                const __dirname = dirname(__filename);
                const workerPath = join(__dirname, 'worker.ts');
                const worker = new Worker(workerPath);

                worker.onmessage = ({ data }) => {
                    if (data.type === 'progress') {
                        workerProgress[data.workerId] = data.processed;
                        progressLogger();
                    } else if (data.type === 'done') {
                        workerProgress[data.workerId] = chunk.length;
                        progressLogger();
                        resolve(data.results);
                        worker.terminate();
                    }
                };
                worker.onerror = (err) => { this.logger.error(`Worker error: ${err.message}`); reject(err); worker.terminate(); };

                const workerData = {
                    patternChunk: chunk,
                    vocabData: { word2index: Array.from(this.vocab.word2index.entries()) },
                    embeddingData: this.embeddingLayer.embeddingMatrix.data,
                    unknownTokenIndex: this.vocab.word2index.get('<UNK>')!,
                    workerId: workerId
                };

                worker.postMessage(workerData);
            });
        });

        const results = await Promise.all(workerPromises);
        process.stdout.write('\n');

        results.flat().forEach(([text, patternData]) => {
            this.patterns.set(text, {
                ...patternData,
                embedding: new Float32Matrix(1, this.embeddingDim, new Float32Array(patternData.embedding))
            });
        });

        endEmbeddingTimer();
    }

    public generate(seedText: string, maxLength: number): string {
        const endGenerationTimer = this.logger.time("Text Generation (Inference)");

        if (this.patterns.size === 0) { this.logger.error("Model has not been trained."); return ""; }

        const slots: Slot[] = Array.from({ length: maxLength }, () => ({
            possibilities: new Map(this.patterns),
            collapsedPattern: null,
            entropy: this.patterns.size
        }));

        if (seedText) {
            const seedTokens = this.vocab.tokenize(seedText);
            if (seedTokens.length >= this.n) {
                const seedNgram = seedTokens.slice(0, this.n).join("||");
                const seedPattern = this.patterns.get(seedNgram);
                if (seedPattern) {
                    this.collapseSlot(slots, 0, seedPattern);
                    this.propagateConstraints(slots, 0);
                } else {
                    this.logger.warn(`Seed pattern could not be formed from "${seedText}". Starting randomly.`);
                }
            }
        }

        let collapsedCount = slots.filter(s => s.collapsedPattern).length;
        while (collapsedCount < slots.length) {
            const indexToCollapse = this.findLowestEntropySlot(slots);
            if (indexToCollapse === -1) break;

            const currentSlot = slots[indexToCollapse];
            if (currentSlot.possibilities.size === 0) { this.logger.warn(`Contradiction at slot ${indexToCollapse}. Stopping.`); break; }

            const contextVector = this.getContextVector(slots, indexToCollapse, this.contextLag);
            const chosenPattern = this.choosePattern(currentSlot, contextVector);

            if (!chosenPattern) { this.logger.warn(`Could not choose a pattern for slot ${indexToCollapse}.`); break; }

            this.collapseSlot(slots, indexToCollapse, chosenPattern);
            this.propagateConstraints(slots, indexToCollapse);
            collapsedCount++;
        }

        const result = this.reconstructText(slots);
        endGenerationTimer();
        return result;
    }

    private getContextVector(slots: Slot[], currentIndex: number, lookbehind: number): Float32Matrix | null {
        const contextPatterns: Pattern[] = [];
        for (let i = 1; i <= lookbehind; i++) {
            const pattern = slots[currentIndex - i]?.collapsedPattern;
            if (pattern) {
                contextPatterns.push(pattern);
            } else {
                break;
            }
        }

        if (contextPatterns.length === 0) {
            return null;
        }

        const contextEmbeddings = new Float32Matrix(contextPatterns.length, this.embeddingDim);
        contextPatterns.forEach((p, i) => {
            contextEmbeddings.data.set(p.embedding.data, i * this.embeddingDim);
        });

        return Float32Matrix.averageRows(contextEmbeddings);
    }

    private collapseSlot(slots: Slot[], index: number, chosenPattern: Pattern): void {
        const slot = slots[index];
        slot.collapsedPattern = chosenPattern;
        slot.possibilities.clear();
        slot.entropy = 0;
    }

    private findLowestEntropySlot(slots: Slot[]): number {
        let minEntropy = Infinity;
        let lowestEntropyIndex = -1;
        for (let i = 0; i < slots.length; i++) {
            if (!slots[i].collapsedPattern) {
                const entropy = slots[i].possibilities.size;
                if (entropy > 0 && entropy < minEntropy) {
                    minEntropy = entropy;
                    lowestEntropyIndex = i;
                }
            }
        }
        return lowestEntropyIndex;
    }

    private choosePattern(slot: Slot, contextVector: Float32Matrix | null): Pattern | null {
        const possibilities = Array.from(slot.possibilities.values());
        if (possibilities.length === 0) return null;

        if (!contextVector) {
            const totalFrequency = possibilities.reduce((sum, p) => sum + p.frequency, 0);
            let rand = Math.random() * totalFrequency;
            for (const p of possibilities) {
                rand -= p.frequency;
                if (rand <= 0) return p;
            }
            return possibilities[possibilities.length - 1];
        }

        const K_data = new Float32Array(possibilities.length * this.embeddingDim);
        possibilities.forEach((p, i) => {
            K_data.set(p.embedding.data, i * this.embeddingDim);
        });
        const K = new Float32Matrix(possibilities.length, this.embeddingDim, K_data);

        const { attentionWeights } = scaledDotProductAttention(contextVector, K, K, this.temperature);

        if (attentionWeights.rows === 0) {
            return possibilities[Math.floor(Math.random() * possibilities.length)];
        }

        const weights = attentionWeights.getRow(0);
        let rand = Math.random();
        let cumulativeProb = 0;
        for (let i = 0; i < weights.length; i++) {
            cumulativeProb += weights[i];
            if (rand <= cumulativeProb) return possibilities[i];
        }

        return possibilities[possibilities.length - 1];
    }

    private propagateConstraints(slots: Slot[], collapsedIndex: number): void {
        const queue = [collapsedIndex];
        const visited = new Set(queue);

        while (queue.length > 0) {
            const currentIndex = queue.shift()!;
            const currentPattern = slots[currentIndex].collapsedPattern!;

            for (const neighborIndex of [currentIndex - 1, currentIndex + 1]) {
                if (neighborIndex >= 0 && neighborIndex < slots.length && !slots[neighborIndex].collapsedPattern) {
                    const updated = this.updateNeighbor(slots[neighborIndex], currentPattern);
                    if (updated && !visited.has(neighborIndex)) {
                        queue.push(neighborIndex);
                        visited.add(neighborIndex);
                    }
                }
            }
        }
    }

    private updateNeighbor(neighborSlot: Slot, contextPattern: Pattern): boolean {
        const possibilities = Array.from(neighborSlot.possibilities.values());
        if (possibilities.length === 0) return false;

        const K_data = new Float32Array(possibilities.length * this.embeddingDim);
        possibilities.forEach((p, i) => {
            K_data.set(p.embedding.data, i * this.embeddingDim);
        });
        const K = new Float32Matrix(possibilities.length, this.embeddingDim, K_data);

        const { attentionWeights } = scaledDotProductAttention(contextPattern.embedding, K, K, this.temperature);
        if (attentionWeights.rows === 0) return false;

        const weights = attentionWeights.getRow(0);
        const newPossibilities = new Map<string, Pattern>();

        const threshold = 0.01;
        weights.forEach((weight, i) => {
            if (weight > threshold) {
                const p = possibilities[i];
                newPossibilities.set(p.text, p);
            }
        });

        if (newPossibilities.size > 0 && newPossibilities.size < neighborSlot.possibilities.size) {
            neighborSlot.possibilities = newPossibilities;
            neighborSlot.entropy = newPossibilities.size;
            return true;
        }
        return false;
    }

    private reconstructText(slots: Slot[]): string {
        const collapsedPatterns = slots.map(s => s.collapsedPattern).filter((p): p is Pattern => p !== null);
        if (collapsedPatterns.length === 0) return "";

        let resultTokens: string[] = collapsedPatterns[0].text.split("||");
        for (let i = 1; i < collapsedPatterns.length; i++) {
            const currentTokens = collapsedPatterns[i].text.split("||");
            resultTokens.push(currentTokens[this.n - 1]);
        }
        return resultTokens.join("");
    }

    public saveModel(filePath: string): void {
        const endTimer = this.logger.time(`Saving model to ${filePath}`);
        const modelData = {
            n: this.n,
            embeddingDim: this.embeddingDim,
            vocab: { word2index: Array.from(this.vocab.word2index.entries()) },
            embeddingMatrix: Array.from(this.embeddingLayer.embeddingMatrix.data),
            patterns: Array.from(this.patterns.entries()).map(([text, pattern]) => {
                return [text, {
                    text: pattern.text,
                    embedding: Array.from(pattern.embedding.data),
                    frequency: pattern.frequency
                }];
            })
        };

        try {
            const jsonString = JSON.stringify(modelData);
            Bun.write(filePath, jsonString);
            this.logger.log(`Model successfully saved.`);
        } catch (error) {
            this.logger.error(`Failed to save model: ${error}`);
        }
        endTimer();
    }

    public async loadModel(filePath: string): Promise<boolean> {
        const endTimer = this.logger.time(`Loading model from ${filePath}`);
        try {
            const fileContent = await Bun.file(filePath).text();
            const modelData = JSON.parse(fileContent);

            this.n = modelData.n;
            this.embeddingDim = modelData.embeddingDim;

            this.vocab = new Vocabulary();
            this.vocab.word2index = new Map(modelData.vocab.word2index);
            this.vocab.word2index.forEach((index, word) => {
                this.vocab.index2word.set(index, word);
            });

            this.embeddingLayer = new EmbeddingLayer(this.vocab.size, this.embeddingDim, this.logger);
            this.embeddingLayer.embeddingMatrix = new Float32Matrix(this.vocab.size, this.embeddingDim, new Float32Array(modelData.embeddingMatrix));

            this.patterns = new Map(modelData.patterns.map(([text, patternData]: [string, any]) => {
                return [text, {
                    ...patternData,
                    embedding: new Float32Matrix(1, this.embeddingDim, new Float32Array(patternData.embedding))
                }];
            }));

            this.logger.log(`Model successfully loaded. Found ${this.patterns.size} patterns.`);
            endTimer();
            return true;
        } catch (error) {
            this.logger.error(`Failed to load model: ${error}`);
            endTimer();
            return false;
        }
    }
}

// --- File Reading and Main Execution ---

function cleanText(text: string): string {
    let cleaned = text;
    cleaned = cleaned.replace(/```[\s\S]*?```/g, '');
    cleaned = cleaned.replace(/`[^`]*`/g, '');
    cleaned = cleaned.replace(/<[^>]*>/g, '');
    cleaned = cleaned.replace(/^[#]+\s.*$/gm, '');
    cleaned = cleaned.replace(/(\*\*|__)(.*?)\1/g, '$2');
    cleaned = cleaned.replace(/(\*|_)(.*?)\1/g, '$2');
    cleaned = cleaned.replace(/\[(.*?)\]\(.*?\)/g, '$1');
    cleaned = cleaned.replace(/!\[(.*?)\]\(.*?\)/g, '');
    cleaned = cleaned.replace(/^>\s.*$/gm, '');
    cleaned = cleaned.replace(/^-+\s.*$/gm, '');
    cleaned = cleaned.replace(/https?:\/\/[^\s]+/g, '');
    cleaned = cleaned.replace(/&[a-zA-Z0-9#]+;/g, ' ');
    cleaned = cleaned.replace(/\s+/g, ' ');
    cleaned = cleaned.replace(/[^a-zA-Z0-9 .,!?'"-]/g, '');
    return cleaned.trim();
}

async function readDirectoryCorpus(directoryPath: string, logger: Logger): Promise<string[]> {
    const allTexts: string[] = [];
    const validExtensions = ['.md', '.markdown', '.txt', '.csv'];

    try {
        const filesAndDirs = readdirSync(directoryPath);
        for (const name of filesAndDirs) {
            const fullPath = join(directoryPath, name);
            const stat = statSync(fullPath);

            if (stat.isDirectory()) {
                allTexts.push(...await readDirectoryCorpus(fullPath, logger));
            } else if (stat.isFile() && validExtensions.some(ext => name.endsWith(ext))) {
                const fileContent = await Bun.file(fullPath).text();
                if (name.endsWith('.csv')) {
                    const lines = fileContent.split('\n');
                    allTexts.push(...lines.slice(1).filter(line => line.trim() !== '').map(cleanText));
                } else {
                    allTexts.push(cleanText(fileContent));
                }
            }
        }
    } catch (error) {
        logger.error(`Error reading from directory ${directoryPath}: ${error}`);
    }
    return allTexts;
}


function parseArgs() {
    const args = process.argv.slice(2);
    const options = {
        seed: "",
        length: 150,
        corpusDir: "./",
        logFile: "wfc_execution.log",
        saveModelPath: "",
        loadModelPath: "",
        minFrequency: 3,
        nGramSize: 15, // Adjusted default
        contextLag: 3,
        temperature: 0.75,
        topNBigrams: 5000, // New subword args
        topNTrigrams: 2000
    };

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--seed' && i + 1 < args.length) options.seed = args[i + 1];
        if (args[i] === '--length' && i + 1 < args.length) options.length = parseInt(args[i + 1], 10);
        if (args[i] === '--corpus-dir' && i + 1 < args.length) options.corpusDir = args[i + 1];
        if (args[i] === '--log' && i + 1 < args.length) options.logFile = args[i + 1];
        if (args[i] === '--save' && i + 1 < args.length) options.saveModelPath = args[i + 1];
        if (args[i] === '--load' && i + 1 < args.length) options.loadModelPath = args[i + 1];
        if (args[i] === '--min-freq' && i + 1 < args.length) options.minFrequency = parseInt(args[i + 1], 10);
        if (args[i] === '--n-gram' && i + 1 < args.length) options.nGramSize = parseInt(args[i + 1], 10);
        if (args[i] === '--lag' && i + 1 < args.length) options.contextLag = parseInt(args[i + 1], 10);
        if (args[i] === '--temperature' && i + 1 < args.length) options.temperature = parseFloat(args[i + 1]);
        if (args[i] === '--top-bigrams' && i + 1 < args.length) options.topNBigrams = parseInt(args[i + 1], 10);
        if (args[i] === '--top-trigrams' && i + 1 < args.length) options.topNTrigrams = parseInt(args[i + 1], 10);
    }
    return options;
}

async function main() {
    const { seed, length, corpusDir, logFile, saveModelPath, loadModelPath, minFrequency, nGramSize, contextLag, temperature, topNBigrams, topNTrigrams } = parseArgs();
    const logger = new Logger(logFile);

    logger.log("--- Script Execution Started ---");
    logger.log(`Parsed Args: ${JSON.stringify({ seed, length, corpusDir, logFile, saveModelPath, loadModelPath, minFrequency, nGramSize, contextLag, temperature, topNBigrams, topNTrigrams })}`);

    const embeddingDim = 32;
    const generator = new WFC_Attention_Generator(nGramSize, embeddingDim, logger, contextLag, temperature);

    if (loadModelPath) {
        await generator.loadModel(loadModelPath);
    } else {
        const endReadingTimer = logger.time("Reading and Processing Corpus");
        const fullCorpus = await readDirectoryCorpus(corpusDir, logger);
        endReadingTimer();

        if (fullCorpus.length === 0) {
            logger.warn(`No data found in '${corpusDir}'. Using a small default corpus.`);
            const defaultCorpus = [
                "the quick brown fox jumps over the lazy dog", "a quick brown fox runs fast",
                "the lazy dog barks loudly", "foxes are clever animals",
                "the quick brown cat naps", "a dog barks at the moon"
            ];
            await generator.train(defaultCorpus, minFrequency, topNBigrams, topNTrigrams);
        } else {
            logger.log(`Training WFC model on ${fullCorpus.length} total processed documents/rows.`);
            await generator.train(fullCorpus, minFrequency, topNBigrams, topNTrigrams);
        }

        if (saveModelPath) {
            generator.saveModel(saveModelPath);
        }
    }

    if (generator.patterns.size > 0) {
        logger.log("\n--- Generating Text with Attention-WFC ---", false);
        const generatedText = generator.generate(seed, length);

        console.log(`\nSeed: "${seed || '(none)'}" | Length: ${length}`);
        console.log("Output:", generatedText);

        logger.log(`Final Output: ${generatedText}`, false);
    } else {
        logger.error("Cannot generate text because no model is trained or loaded.");
    }

    logger.log("--- Script Execution Finished ---");
}

main();
````

#### Parallel Worker

```typescript
// File: worker.ts
// =================================================================
// This code should be saved in a separate file named 'worker.ts'
// in the same directory as the main script.
// =================================================================

self.onmessage = ({ data: { patternChunk, vocabData, embeddingData, unknownTokenIndex, workerId } }) => {
    // Reconstruct data needed by the worker
    const patterns = new Map(patternChunk);
    const vocab = { word2index: new Map(vocabData.word2index) };

    // The embeddingData is now a Float32Array
    const embeddingMatrix = { data: new Float32Array(embeddingData), cols: embeddingData.length / vocab.word2index.size };

    const getEmbedding = (index: number): Float32Array => {
        return embeddingMatrix.data.subarray(index * embeddingMatrix.cols, (index + 1) * embeddingMatrix.cols);
    };

    const results = new Map();
    const totalInChunk = patterns.size;
    let processedCount = 0;
    // Report progress roughly 100 times per chunk, or at least every 1,000 items for responsiveness.
    const reportInterval = Math.max(1000, Math.floor(totalInChunk / 100));

    patterns.forEach((count, text) => {
        // --- MODIFIED FOR CHARACTER-LEVEL ---
        // Tokenize by character instead of by space-separated words.
        const tokens = text.split("||"); // In the subword model, tokens are already split by ||
        // --- END MODIFICATION ---

        const tokenIndices = tokens.map(t => vocab.word2index.get(t) ?? unknownTokenIndex);

        const tokenEmbeddings = tokenIndices.map(getEmbedding);

        let avgVector: number[] | Float32Array = [];
        if (tokenEmbeddings.length > 0 && tokenEmbeddings[0]) {
            const numCols = tokenEmbeddings[0].length;
            const tempAvg = new Float32Array(numCols);
            for (const embedding of tokenEmbeddings) {
                for (let j = 0; j < numCols; j++) {
                    tempAvg[j] += embedding[j];
                }
            }
            avgVector = tempAvg.map(v => v / tokenEmbeddings.length);
        }

        results.set(text, {
            text: text,
            embedding: Array.from(avgVector), // Convert back to standard array for serialization
            frequency: count
        });

        processedCount++;
        if (processedCount % reportInterval === 0) {
            self.postMessage({ type: 'progress', processed: processedCount, workerId });
        }
    });

    // Send the final result message when done
    self.postMessage({ type: 'done', results: Array.from(results.entries()), workerId });
};
```

```
