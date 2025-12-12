import Foundation
import llama

/// Errors specific to embedding model operations.
public enum EmbeddingModelError: Error {
    case modelLoadFailed
    case contextCreationFailed
    case tokenizationFailed
    case embeddingsFailed
    case emptyInput
    case batchProcessingFailed
}

/// A lightweight model class optimized for embed-only models like nomic-embed-text or all-MiniLM.
///
/// Unlike the full `LLM` class, `EmbeddingModel` is designed specifically for generating embeddings
/// with minimal overhead. It does not include generation capabilities, samplers, or chat templates,
/// making it ideal for RAG pipelines where fast embedding generation is critical.
///
/// ## Features
/// - **Lightweight**: No generation overhead, samplers, or chat history
/// - **Fast**: Optimized batch processing for embeddings
/// - **Parallel-safe**: Can run alongside other models (LLM or EmbeddingModel)
/// - **Thread-safe**: Uses Swift actors for isolation
///
/// ## Usage
/// ```swift
/// let embedder = try EmbeddingModel(from: "nomic-embed-text-v1.5.Q4_0.gguf")
/// let embedding = try await embedder.embed("Hello, world!")
/// print(embedding.dimension) // e.g., 768 for nomic-embed
/// ```
public actor EmbeddingModel {
    private let model: OpaquePointer
    private let vocab: OpaquePointer
    private var context: OpaquePointer
    private var batch: llama_batch
    private let maxTokenCount: Int
    
    /// The embedding dimension of the loaded model.
    public nonisolated let embeddingDimension: Int
    
    /// Path to the loaded model file.
    public nonisolated let modelPath: String
    
    /// Silences llama.cpp logging output.
    public static func silenceLogging() {
        LlamaBackend.shared.silenceLogging()
    }
    
    /// Creates an embedding model from a file path.
    ///
    /// - Parameters:
    ///   - path: Path to the GGUF model file
    ///   - maxTokenCount: Maximum context size (default: 512, sufficient for most embedding models)
    ///   - gpuLayers: Number of layers to offload to GPU (-1 for all, 0 for CPU only)
    /// - Throws: `EmbeddingModelError.modelLoadFailed` if the model cannot be loaded
    public init(from path: String, maxTokenCount: Int = 512, gpuLayers: Int32 = -1) throws {
        LlamaBackend.shared.ensureInitialized()
        LlamaBackend.shared.silenceLogging()
        
        self.modelPath = path
        self.maxTokenCount = maxTokenCount
        
        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
        #else
        modelParams.n_gpu_layers = gpuLayers == -1 ? 999 : gpuLayers
        #endif
        
        let cPath = path.cString(using: .utf8)!
        guard let model = llama_model_load_from_file(cPath, modelParams) else {
            throw EmbeddingModelError.modelLoadFailed
        }
        self.model = model
        self.vocab = llama_model_get_vocab(model)
        self.embeddingDimension = Int(llama_model_n_embd(model))
        
        var contextParams = llama_context_default_params()
        let processorCount = Int32(ProcessInfo().processorCount)
        contextParams.n_ctx = UInt32(maxTokenCount)
        contextParams.n_batch = UInt32(maxTokenCount)
        contextParams.n_ubatch = UInt32(maxTokenCount)  // Must be >= n_tokens for encoder models
        contextParams.n_threads = processorCount
        contextParams.n_threads_batch = processorCount
        contextParams.embeddings = true
        contextParams.pooling_type = LLAMA_POOLING_TYPE_MEAN
        
        guard let context = llama_init_from_model(model, contextParams) else {
            llama_model_free(model)
            throw EmbeddingModelError.contextCreationFailed
        }
        self.context = context
        self.batch = llama_batch_init(Int32(maxTokenCount), 0, 1)
    }
    
    /// Creates an embedding model from a URL.
    public init(from url: URL, maxTokenCount: Int = 512, gpuLayers: Int32 = -1) throws {
        try self.init(from: url.path, maxTokenCount: maxTokenCount, gpuLayers: gpuLayers)
    }
    
    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
    }
    
    /// Generates embeddings for the given text.
    ///
    /// - Parameter text: The text to embed
    /// - Returns: An `Embeddings` struct containing the embedding vector
    /// - Throws: `EmbeddingModelError` if embedding generation fails
    public func embed(_ text: String) throws -> Embeddings {
        guard !text.isEmpty else { throw EmbeddingModelError.emptyInput }
        
        let tokens = tokenize(text)
        guard !tokens.isEmpty else { throw EmbeddingModelError.tokenizationFailed }
        
        // Capture actor-isolated properties for use in serialized block
        let ctx = context
        let localBatch = batch
        let embDim = embeddingDimension
        
        // Serialize llama.cpp operations to prevent GPU resource contention
        return try LlamaBackend.shared.execute {
            try Self.processBatchStatic(context: ctx, batch: localBatch, tokens: tokens)
            let values = try Self.extractEmbeddingsStatic(context: ctx, embeddingDimension: embDim)
            return Embeddings(values: values)
        }
    }
    
    /// Generates embeddings for multiple texts in a batch.
    ///
    /// - Parameter texts: Array of texts to embed
    /// - Returns: Array of `Embeddings` structs
    /// - Throws: `EmbeddingModelError` if any embedding generation fails
    public func embedBatch(_ texts: [String]) throws -> [Embeddings] {
        var results: [Embeddings] = []
        results.reserveCapacity(texts.count)
        
        for text in texts {
            let embedding = try embed(text)
            results.append(embedding)
        }
        
        return results
    }
    
    /// Returns the raw embedding values as a Float array.
    ///
    /// - Parameter text: The text to embed
    /// - Returns: Raw embedding values
    public func embedRaw(_ text: String) throws -> [Float] {
        let embeddings = try embed(text)
        return embeddings.values
    }
    
    private func tokenize(_ text: String) -> [llama_token] {
        let count = Int32(text.utf8.count)
        var tokenCount = count + 1
        let cTokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(tokenCount))
        defer { cTokens.deallocate() }
        
        tokenCount = llama_tokenize(vocab, text, count, cTokens, tokenCount, true, true)
        
        guard tokenCount > 0 else { return [] }
        
        var tokens = (0..<Int(tokenCount)).map { cTokens[$0] }
        
        let nullToken = llama_tokenize(vocab, "\0", 1, cTokens, 1, false, false) > 0 ? cTokens[0] : -1
        if tokens.last == nullToken {
            tokens.removeLast()
        }
        
        return tokens
    }
    
    /// Static helper for batch processing - called within LlamaBackend.execute
    private static func processBatchStatic(context: OpaquePointer, batch: llama_batch, tokens: [llama_token]) throws {
        llama_memory_clear(llama_get_memory(context), false)
        
        var localBatch = batch
        localBatch.n_tokens = 0
        
        for (i, token) in tokens.enumerated() {
            let idx = Int(localBatch.n_tokens)
            localBatch.token[idx] = token
            localBatch.pos[idx] = Int32(i)
            localBatch.n_seq_id[idx] = 1
            if let seq_id = localBatch.seq_id[idx] {
                seq_id[0] = 0
            }
            localBatch.logits[idx] = (i == tokens.count - 1) ? 1 : 0
            localBatch.n_tokens += 1
        }
        
        let result = llama_decode(context, localBatch)
        guard result == 0 else {
            throw EmbeddingModelError.batchProcessingFailed
        }
    }
    
    /// Static helper for extracting embeddings - called within LlamaBackend.execute
    private static func extractEmbeddingsStatic(context: OpaquePointer, embeddingDimension: Int) throws -> [Float] {
        guard let embeddingsPtr = llama_get_embeddings_seq(context, 0) else {
            guard let embeddingsPtr = llama_get_embeddings_ith(context, -1) else {
                throw EmbeddingModelError.embeddingsFailed
            }
            return extractFromPointerStatic(embeddingsPtr, dimension: embeddingDimension)
        }
        return extractFromPointerStatic(embeddingsPtr, dimension: embeddingDimension)
    }
    
    /// Static helper for pointer extraction - called within LlamaBackend.execute
    private static func extractFromPointerStatic(_ ptr: UnsafePointer<Float>, dimension: Int) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(dimension)
        
        for i in 0..<dimension {
            values.append(ptr[i])
        }
        
        return normalizeL2Static(values)
    }
    
    private static func normalizeL2Static(_ values: [Float]) -> [Float] {
        let magnitude = sqrt(values.reduce(0) { $0 + $1 * $1 })
        guard magnitude > 0 else { return values }
        return values.map { $0 / magnitude }
    }
}

/// Extension for convenient async embedding operations.
extension EmbeddingModel {
    
    /// Computes cosine similarity between two texts.
    ///
    /// - Parameters:
    ///   - text1: First text
    ///   - text2: Second text
    /// - Returns: Cosine similarity score between 0 and 1
    public func similarity(between text1: String, and text2: String) async throws -> Double {
        let emb1 = try embed(text1)
        let emb2 = try embed(text2)
        return emb1.compare(with: emb2)
    }
    
    /// Finds the most similar text from a list of candidates.
    ///
    /// - Parameters:
    ///   - query: The query text
    ///   - candidates: List of candidate texts to compare against
    /// - Returns: Tuple of (best matching text, similarity score, index)
    public func findMostSimilar(
        to query: String,
        in candidates: [String]
    ) async throws -> (text: String, score: Double, index: Int)? {
        guard !candidates.isEmpty else { return nil }
        
        let queryEmbedding = try embed(query)
        
        var bestScore: Double = -1
        var bestIndex = 0
        
        for (index, candidate) in candidates.enumerated() {
            let candidateEmbedding = try embed(candidate)
            let score = queryEmbedding.compare(with: candidateEmbedding)
            
            if score > bestScore {
                bestScore = score
                bestIndex = index
            }
        }
        
        return (candidates[bestIndex], bestScore, bestIndex)
    }
    
    /// Ranks candidates by similarity to a query.
    ///
    /// - Parameters:
    ///   - query: The query text
    ///   - candidates: List of candidate texts to rank
    ///   - topK: Maximum number of results to return (nil for all)
    /// - Returns: Array of (text, score, original index) sorted by score descending
    public func rank(
        query: String,
        candidates: [String],
        topK: Int? = nil
    ) async throws -> [(text: String, score: Double, index: Int)] {
        guard !candidates.isEmpty else { return [] }
        
        let queryEmbedding = try embed(query)
        
        var results: [(text: String, score: Double, index: Int)] = []
        results.reserveCapacity(candidates.count)
        
        for (index, candidate) in candidates.enumerated() {
            let candidateEmbedding = try embed(candidate)
            let score = queryEmbedding.compare(with: candidateEmbedding)
            results.append((candidate, score, index))
        }
        
        results.sort { $0.score > $1.score }
        
        if let topK = topK, topK < results.count {
            return Array(results.prefix(topK))
        }
        
        return results
    }
}
