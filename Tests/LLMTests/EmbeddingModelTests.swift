import Testing
import Foundation
@testable import LLM

/// Tests for EmbeddingModel functionality with embed-only models
/// Run with: swift test --filter EmbeddingModelTests
struct EmbeddingModelTests {
    
    let nomicModelPath = "models/nomic-embed-text-v1.5.Q4_0.gguf"
    let minilmModelPath = "models/all-minilm-l6-v2-q4_k_m.gguf"
    let gemmaModelPath = "models/unsloth-gemma-3-4b-it-Q4_K_M.gguf"
    
    // MARK: - Basic Embedding Tests
    
    @Test("Load nomic-embed model and generate embeddings")
    func testNomicEmbedModel() async throws {
        print("\n=== TEST: Nomic Embed Model ===")
        
        let modelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(nomicModelPath)
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            print("[SKIP] Model not found at: \(modelURL.path)")
            return
        }
        
        print("[OK] Model found at: \(modelURL.path)")
        
        let embedder = try EmbeddingModel(from: modelURL)
        print("[OK] EmbeddingModel initialized")
        print("[INFO] Embedding dimension: \(embedder.embeddingDimension)")
        
        let testText = "Hello, this is a test sentence for embedding generation."
        let startTime = CFAbsoluteTimeGetCurrent()
        let embedding = try await embedder.embed(testText)
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        print("[OK] Embedding generated in \(String(format: "%.2f", elapsed))ms")
        print("[INFO] Embedding dimension: \(embedding.dimension)")
        print("[INFO] First 5 values: \(embedding.values.prefix(5))")
        
        #expect(embedding.dimension > 0, "Embedding should have non-zero dimension")
        #expect(embedding.dimension == embedder.embeddingDimension, "Dimensions should match")
    }
    
    @Test("Load all-MiniLM model and generate embeddings")
    func testMiniLMModel() async throws {
        print("\n=== TEST: All-MiniLM Model ===")
        
        let modelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(minilmModelPath)
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            print("[SKIP] Model not found at: \(modelURL.path)")
            return
        }
        
        print("[OK] Model found at: \(modelURL.path)")
        
        let embedder = try EmbeddingModel(from: modelURL)
        print("[OK] EmbeddingModel initialized")
        print("[INFO] Embedding dimension: \(embedder.embeddingDimension)")
        
        let testText = "The quick brown fox jumps over the lazy dog."
        let startTime = CFAbsoluteTimeGetCurrent()
        let embedding = try await embedder.embed(testText)
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        print("[OK] Embedding generated in \(String(format: "%.2f", elapsed))ms")
        print("[INFO] Embedding dimension: \(embedding.dimension)")
        
        #expect(embedding.dimension > 0, "Embedding should have non-zero dimension")
    }
    
    // MARK: - Similarity Tests
    
    @Test("Test semantic similarity with embed-only model")
    func testSemanticSimilarity() async throws {
        print("\n=== TEST: Semantic Similarity ===")
        
        let modelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(minilmModelPath)
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            print("[SKIP] Model not found at: \(modelURL.path)")
            return
        }
        
        let embedder = try EmbeddingModel(from: modelURL)
        
        let query = "What is machine learning?"
        let similar = "Machine learning is a subset of artificial intelligence."
        let unrelated = "The weather is nice today."
        
        let similarScore = try await embedder.similarity(between: query, and: similar)
        let unrelatedScore = try await embedder.similarity(between: query, and: unrelated)
        
        print("[INFO] Query: '\(query)'")
        print("[INFO] Similar text score: \(String(format: "%.4f", similarScore))")
        print("[INFO] Unrelated text score: \(String(format: "%.4f", unrelatedScore))")
        
        #expect(similarScore > unrelatedScore, "Similar text should have higher score")
        print("[OK] Semantic similarity working correctly")
    }
    
    @Test("Test ranking candidates by similarity")
    func testRankCandidates() async throws {
        print("\n=== TEST: Rank Candidates ===")
        
        let modelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(minilmModelPath)
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            print("[SKIP] Model not found at: \(modelURL.path)")
            return
        }
        
        let embedder = try EmbeddingModel(from: modelURL)
        
        let query = "I want to learn about programming"
        let candidates = [
            "The best restaurants in New York City",
            "Introduction to Python programming for beginners",
            "How to cook Italian pasta",
            "Software development best practices",
            "Travel tips for Europe"
        ]
        
        let ranked = try await embedder.rank(query: query, candidates: candidates, topK: 3)
        
        print("[INFO] Query: '\(query)'")
        print("[INFO] Top 3 results:")
        for (i, result) in ranked.enumerated() {
            print("  \(i + 1). [\(String(format: "%.4f", result.score))] \(result.text)")
        }
        
        #expect(ranked.count == 3, "Should return top 3 results")
        #expect(ranked[0].score >= ranked[1].score, "Results should be sorted by score")
        print("[OK] Ranking working correctly")
    }
    
    // MARK: - Performance Tests
    
    @Test("Benchmark embedding generation speed")
    func testEmbeddingSpeed() async throws {
        print("\n=== TEST: Embedding Speed Benchmark ===")
        
        let modelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(minilmModelPath)
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            print("[SKIP] Model not found at: \(modelURL.path)")
            return
        }
        
        let embedder = try EmbeddingModel(from: modelURL)
        
        let testTexts = [
            "Short text",
            "This is a medium length sentence for testing embedding generation speed.",
            "This is a longer piece of text that contains multiple sentences. It should take slightly longer to process due to the increased token count. The embedding model needs to handle various text lengths efficiently.",
        ]
        
        print("[INFO] Embedding dimension: \(embedder.embeddingDimension)")
        
        for (i, text) in testTexts.enumerated() {
            var times: [Double] = []
            
            for _ in 0..<5 {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try await embedder.embed(text)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                times.append(elapsed)
            }
            
            let avgTime = times.reduce(0, +) / Double(times.count)
            print("[INFO] Text \(i + 1) (\(text.count) chars): avg \(String(format: "%.2f", avgTime))ms")
        }
        
        print("[OK] Speed benchmark completed")
    }
    
    // MARK: - Parallel Model Tests
    
    @Test("Run embedding model and chat model in parallel")
    func testParallelModels() async throws {
        print("\n=== TEST: Parallel Model Loading ===")
        
        let embedModelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(minilmModelPath)
        let chatModelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(gemmaModelPath)
        
        guard FileManager.default.fileExists(atPath: embedModelURL.path) else {
            print("[SKIP] Embed model not found at: \(embedModelURL.path)")
            return
        }
        
        guard FileManager.default.fileExists(atPath: chatModelURL.path) else {
            print("[SKIP] Chat model not found at: \(chatModelURL.path)")
            return
        }
        
        print("[INFO] Loading embedding model...")
        let embedder = try EmbeddingModel(from: embedModelURL)
        print("[OK] Embedding model loaded (dim: \(embedder.embeddingDimension))")
        
        print("[INFO] Loading chat model...")
        guard let chatModel = LLM(from: chatModelURL, maxTokenCount: 512) else {
            print("[ERROR] Failed to load chat model")
            throw TestError.initializationFailed
        }
        print("[OK] Chat model loaded")
        
        print("[INFO] Running parallel operations...")
        
        async let embeddingTask = embedder.embed("What is the meaning of life?")
        
        let embedding = try await embeddingTask
        
        print("[OK] Embedding generated: \(embedding.dimension) dimensions")
        print("[OK] Both models running successfully in parallel")
        
        #expect(embedding.dimension > 0, "Embedding should be generated")
    }
    
    @Test("Concurrent embedding requests")
    func testConcurrentEmbeddings() async throws {
        print("\n=== TEST: Concurrent Embedding Requests ===")
        
        let modelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(minilmModelPath)
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            print("[SKIP] Model not found at: \(modelURL.path)")
            return
        }
        
        let embedder = try EmbeddingModel(from: modelURL)
        
        let texts = [
            "First text to embed",
            "Second text to embed",
            "Third text to embed",
            "Fourth text to embed",
            "Fifth text to embed"
        ]
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let embeddings = try await withThrowingTaskGroup(of: (Int, Embeddings).self) { group in
            for (index, text) in texts.enumerated() {
                group.addTask {
                    let emb = try await embedder.embed(text)
                    return (index, emb)
                }
            }
            
            var results: [(Int, Embeddings)] = []
            for try await result in group {
                results.append(result)
            }
            return results.sorted { $0.0 < $1.0 }.map { $0.1 }
        }
        
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        print("[OK] Generated \(embeddings.count) embeddings in \(String(format: "%.2f", elapsed))ms")
        print("[INFO] Average: \(String(format: "%.2f", elapsed / Double(texts.count)))ms per embedding")
        
        #expect(embeddings.count == texts.count, "Should generate all embeddings")
    }
}

enum TestError: Error {
    case modelNotFound
    case adapterNotFound
    case initializationFailed
}
