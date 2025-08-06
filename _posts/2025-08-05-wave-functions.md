Generative Text Synthesis via Wave-Function-Collapse and a Self-Attention Mechanism
Abstract:
This paper documents an experimental endeavor to create a novel generative text model by adapting the Wave Function Collapse (WFC) algorithm, a technique primarily used in procedural graphics generation, for Natural Language Processing (NLP). The objective was to explore whether WFC's constraint-solving capabilities, when augmented with a modern NLP context mechanism, could produce coherent text. The project evolved through several architectural iterations, beginning with a simple statistical n-gram model and culminating in a parallelized, subword-level model utilizing a self-attention mechanism for contextual awareness. While the final implementation successfully integrated these complex components and demonstrated significant performance optimizations, it ultimately failed to generate semantically coherent text. This paper details the theoretical basis, the iterative implementation, and an analysis of the project's outcomes, providing valuable insights into the architectural prerequisites for successful language generation.

1. Introduction
The field of generative NLP has been largely dominated by recurrent and transformer-based architectures. This research explores an alternative paradigm: applying the Wave Function Collapse (WFC) algorithm to the task of text synthesis. WFC, a powerful algorithm for procedural content generation, operates by iteratively solving local constraints to arrive at a globally consistent state. Our central hypothesis was that the principles of WFC could be adapted to model the local constraints of language (grammar, syntax, phrasing) to generate coherent text.

This paper chronicles the development of a text generator, beginning with a foundational WFC model based on word-level n-grams. We then detail its evolution, incorporating a self-attention mechanism to move beyond rigid statistical adjacency to a more fluid, semantic understanding of context. Further refinements included a shift to character-level and then subword-level tokenization, extensive performance optimizations through parallelization and memory management, and the introduction of hyperparameters like temperature and context lag.

Despite the successful implementation of these advanced features, the final model did not achieve its primary objective of coherent text generation. The findings highlight the fundamental differences between the structural constraint satisfaction at which WFC excels and the hierarchical, long-range semantic dependencies that define human language.

2. Methodology and Iterative Development
The model was developed in TypeScript and executed on the Bun runtime. The development process can be divided into three distinct phases.

2.1. Phase 1: A Statistical Foundation with Wave Function Collapse
The initial model was a direct adaptation of the WFC algorithm to a one-dimensional sequence of text.

Core Algorithm:

Slots and Superposition: The target output was modeled as an array of "slots," each initially in a state of superposition, meaning it could potentially be filled by any pattern learned from the training corpus.

Patterns as N-grams: The fundamental units, or "patterns," were defined as word-level trigrams (sequences of three words).

Observation and Collapse: The generation process iteratively selected the slot with the lowest "entropy" (the fewest remaining possibilities) and "collapsed" it by choosing one pattern, weighted by its frequency in the training data.

Constraint Propagation: The key to WFC's coherence is propagation. Once a slot was collapsed to a specific pattern (e.g., ("over", "the", "lazy")), this choice constrained the possibilities of its neighbors. Adjacency was determined by a rigid rule: a pattern could follow another only if its prefix matched the other's suffix (e.g., ("the", "lazy", "dog") can follow ("over", "the", "lazy")).

Limitations of Phase 1: This initial model could produce grammatically plausible short phrases but lacked any true semantic understanding. It could not grasp that "United States" should follow "President of the" more strongly than "United Kingdom" in certain contexts, as it only understood the local adjacency of the word "United."

2.2. Phase 2: Integrating Context with an Attention Mechanism
To overcome the limitations of rigid adjacency, we replaced the statistical rule-based system with a dynamic, context-aware mechanism.

Architectural Shift to Embeddings: The model was fundamentally re-architected to operate on numerical vectors (embeddings) instead of raw text.

A Vocabulary class was implemented to map tokens to integer IDs.

An EmbeddingLayer was created to hold a matrix of vectors, one for each token.

Each n-gram pattern was converted into a single embedding vector by averaging the embeddings of its constituent tokens.

Self-Attention for Context: The core innovation was the replacement of the adjacency graph with a Scaled Dot-Product Attention function.

When choosing a pattern for a slot, the embedding of the previously chosen pattern served as the Query (Q).

The embeddings of all possible patterns for the current slot served as the Keys (K) and Values (V).

The attention mechanism calculated a set of weights, representing the semantic similarity of each possible pattern to the query. These weights were then used to probabilistically select the next pattern. This allowed the model to learn that ("lazy", "dog", "barks") is a more relevant successor to ("the", "quick", "fox") than ("lazy", "dog", "computes"), a distinction impossible in the previous phase.

2.3. Phase 3: Optimization and Refinements
With a working attention-based model, the focus shifted to improving performance, memory efficiency, and the model's linguistic granularity.

Performance and Memory Optimization:

Parallelization: The pattern embedding process, the primary training bottleneck, was parallelized using Web Workers. The task was split across all available CPU cores, dramatically reducing training time on large corpora.

Pruning: A --min-freq argument was introduced to prune rare patterns from the model. This significantly reduced the memory footprint by discarding statistically insignificant n-grams.

TypedArrays: The core matrix class was refac-tored to use Float32Array instead of standard JavaScript arrays, halving the memory required for storing embeddings.

Model Granularity and Context:

Subword-Level Model: To handle out-of-vocabulary words and learn morphology, the model was shifted from a word-level to a subword-level tokenizer. The vocabulary was constructed from all single characters plus the most frequent character bigrams and trigrams. A greedy tokenizer was implemented to convert raw text into sequences of these subword units.

Lagging Context: A --lag parameter was introduced. Instead of using only the single previous pattern as context, the model now creates a "context vector" by averaging the embeddings of the last k patterns. This provides a longer memory, allowing it to capture more complex dependencies.

Temperature: A --temperature parameter was added to the softmax function within the attention mechanism, allowing control over the randomness of the generation process to balance coherence and creativity.

3. Results and Final Findings
The project successfully resulted in a complex, multi-threaded generative architecture built from first principles. The performance optimizations were highly effective, enabling the model to train on a large corpus, identify over 7.4 million unique n-gram patterns, and prune them to a manageable set of ~1700 in under 10 seconds.

However, the primary objective of generating coherent text was not achieved. The final output from the optimized subword model, while an improvement over the pure character-level model, remained largely incoherent.

3.1. Final Model Output
The model was trained on a corpus derived from Isaac Asimov's Future History series (a collection of 20 documents) using the recommended subword configuration (--n-gram 15, --min-freq 3, --lag 3, --temperature 0.75). When prompted with no seed, it produced the following output:

t with the first law. third law a robot mustr wm. haton, 1 loato19aboonser to m leycti co30s lantt tto ume comed's thaptstot r en frorsyed thstichaginman cocom1.lttioey cone h thm  al a orlov is moaceandts.aaons vi19, tte boticaobers te y ra h cos frodnced ira , ts pos. chainaicahert, itstheoload  ling  cho d cira 'sobosubcalold hmetvolusthre deererthselo dorrcits. walic it otrst det orot. apted tmotd td tumerstt i alny pleendthaautom pedriglanic , ilt. te fng s endion


This output demonstrates the model's partial success and ultimate failure. It correctly learned to generate recognizable English word fragments ("first law", "third law", "robot") and respected basic punctuation and spacing. However, it was unable to consistently form complete, correct words or assemble them into a semantically meaningful sentence.

3.2. Analysis of Failure
The Semantic Blurring of Averaging: The core architectural flaw appears to be the method of creating a single vector for an entire n-gram pattern by averaging the embeddings of its tokens. This process likely neutralizes the specific, sequential meaning of the phrase, resulting in a "blurry" vector that represents a general semantic neighborhood rather than a precise point. When the attention mechanism compares these averaged vectors, its ability to discern subtle but critical differences in meaning is severely compromised.

The Subword Compromise: The shift to a subword model was a step in the right direction, providing the model with more meaningful semantic units than single characters. The vocabulary was built by identifying all unigrams and the most common bigrams and trigrams, and a greedy tokenizer was used to parse input text.

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


While this provided better tokens, it could not overcome the fundamental issue of averaging. The context vector, whether from a single pattern or a lagging window, remained an imprecise representation of the sequence's meaning.

Absence of a Trained Decoder: While the model uses an attention mechanism to score context, it lacks the sophisticated, auto-regressive decoder architecture of a true Transformer. A Transformer's decoder is explicitly trained, via backpropagation, to take a sequence of embeddings and generate a probability distribution over the entire vocabulary for the very next token. Our model approximates this by selecting from a list of pre-defined, multi-token patterns, which is a fundamentally less flexible and powerful approach.

WFC as a Structural vs. Semantic Solver: This experiment suggests that WFC is an exceptionally powerful algorithm for solving problems with hard, local, structural constraints (e.g., a "blue" tile can only be adjacent to a "green" or "yellow" tile). Human language, however, is governed by soft, long-range, hierarchical semantic constraints. The WFC framework, even when augmented with attention, does not seem well-suited to modeling this type of fluid, probabilistic system.

4. Conclusion
This research served as a valuable exploration into the limits of applying algorithms from one domain to another. The project successfully demonstrated the implementation of advanced NLP concepts—including subword tokenization, attention, and parallel processing—from scratch. The ultimate failure to generate coherent text indicates that a successful generative language model requires more than just a collection of powerful components; it requires an architecture that is fundamentally designed to respect the hierarchical and auto-regressive nature of language.

Future work could involve applying this WFC-attention hybrid to more constrained, grammar-based generation tasks where its structural constraint-solving capabilities might be more effective, such as the generation of valid source code, chemical formulas, or musical notation.