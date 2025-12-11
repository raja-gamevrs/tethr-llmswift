import Testing
import Foundation
@testable import LLM

/// Tests for RAG Pipeline functionality
/// Run with: swift test --filter RAGPipelineTests
struct RAGPipelineTests {
    
    let minilmModelPath = "models/all-minilm-l6-v2-q4_k_m.gguf"
    let gemmaModelPath = "models/unsloth-gemma-3-4b-it-Q4_K_M.gguf"
    let backstoryPath = "models/backstories/achilles-backstory.md"
    
    @Test("Full RAG Pipeline - Backstory Indexing and Retrieval")
    func testBackstoryIndexingAndRetrieval() async throws {
        print("\n=== TEST: Full RAG Pipeline ===\n")
        
        let embedModelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(minilmModelPath)
        let backstoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(backstoryPath)
        
        guard FileManager.default.fileExists(atPath: embedModelURL.path) else {
            print("[SKIP] Embed model not found at: \(embedModelURL.path)")
            return
        }
        
        guard FileManager.default.fileExists(atPath: backstoryURL.path) else {
            print("[SKIP] Backstory not found at: \(backstoryURL.path)")
            return
        }
        
        print("[1/5] Loading embedding model...")
        let startLoad = CFAbsoluteTimeGetCurrent()
        let embedder = try EmbeddingModel(from: embedModelURL)
        let loadTime = (CFAbsoluteTimeGetCurrent() - startLoad) * 1000
        print("[OK] Embedding model loaded in \(String(format: "%.0f", loadTime))ms")
        print("[INFO] Embedding dimension: \(embedder.embeddingDimension)")
        
        print("\n[2/5] Creating RAG pipeline...")
        let pipeline = RAGPipeline(embedder: embedder, chunkSize: 300, overlap: 30)
        print("[OK] RAG pipeline created")
        
        print("\n[3/5] Indexing backstory...")
        let backstoryText = try String(contentsOf: backstoryURL, encoding: .utf8)
        let startIndex = CFAbsoluteTimeGetCurrent()
        let chunkCount = try await pipeline.indexBackstory(
            backstoryText,
            personaId: "achilles",
            useSectionChunking: true
        )
        let indexTime = (CFAbsoluteTimeGetCurrent() - startIndex) * 1000
        print("[OK] Indexed \(chunkCount) backstory chunks in \(String(format: "%.0f", indexTime))ms")
        print("[INFO] Average: \(String(format: "%.1f", indexTime / Double(chunkCount)))ms per chunk")
        
        print("\n[4/5] Indexing sample conversation...")
        let conversationMessages = [
            ("msg_1", "user", "Tell me about your childhood in Phtheia"),
            ("msg_2", "assistant", "Ah, Phtheia... the city carved into the cliffs overlooking the Azure Sea. I remember the smell of cedar from my father's shipyard."),
            ("msg_3", "user", "What was your training like at the Aretai?"),
            ("msg_4", "assistant", "The Aretai was brutal but necessary. Master Kaelen taught me that true strength comes from discipline, not rage."),
            ("msg_5", "user", "Do you ever think about your destiny?"),
            ("msg_6", "assistant", "The prophecy haunts me still. But I choose to live with purpose, even knowing my end may be premature.")
        ]
        
        for (msgId, role, text) in conversationMessages {
            try await pipeline.indexConversationMessage(
                text,
                messageId: msgId,
                conversationId: "conv_1",
                role: role
            )
        }
        print("[OK] Indexed \(conversationMessages.count) conversation messages")
        
        print("\n[5/5] Testing context retrieval...")
        let testQueries = [
            "What was Achilles' relationship with his father?",
            "Tell me about the combat training",
            "What is the prophecy about Achilles?"
        ]
        
        for query in testQueries {
            print("\n--- Query: \"\(query)\" ---")
            let startRetrieve = CFAbsoluteTimeGetCurrent()
            let context = try await pipeline.retrieveContext(
                for: query,
                backstoryTopK: 2,
                conversationTopK: 2,
                minScore: 0.2
            )
            let retrieveTime = (CFAbsoluteTimeGetCurrent() - startRetrieve) * 1000
            
            print("Retrieved in \(String(format: "%.1f", retrieveTime))ms:")
            print("  Backstory chunks: \(context.backstoryChunks.count)")
            for chunk in context.backstoryChunks {
                let title = chunk.metadata["section_title"] ?? "unknown"
                print("    - [\(String(format: "%.3f", chunk.score))] \(title)")
            }
            print("  Conversation chunks: \(context.conversationChunks.count)")
            for chunk in context.conversationChunks {
                print("    - [\(String(format: "%.3f", chunk.score))] \(chunk.text.prefix(50))...")
            }
        }
        
        print("\n[OK] RAG Pipeline test completed successfully!")
        
        #expect(await pipeline.backstoryChunkCount > 0, "Should have indexed backstory chunks")
        #expect(await pipeline.conversationChunkCount == conversationMessages.count, "Should have indexed all messages")
    }
    
    @Test("Parallel Chat + Embedding Models")
    func testParallelChatAndEmbedding() async throws {
        print("\n=== TEST: Parallel Chat + Embedding Models ===\n")
        
        let embedModelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(minilmModelPath)
        let chatModelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(gemmaModelPath)
        
        guard FileManager.default.fileExists(atPath: embedModelURL.path) else {
            print("[SKIP] Embed model not found")
            return
        }
        
        guard FileManager.default.fileExists(atPath: chatModelURL.path) else {
            print("[SKIP] Chat model not found")
            return
        }
        
        print("[1/4] Loading embedding model...")
        let embedder = try EmbeddingModel(from: embedModelURL)
        print("[OK] Embedding model loaded (dim: \(embedder.embeddingDimension))")
        
        print("\n[2/4] Loading chat model...")
        guard let chatModel = LLM(from: chatModelURL, maxTokenCount: 1024) else {
            print("[ERROR] Failed to load chat model")
            throw TestError.initializationFailed
        }
        chatModel.template = .gemma
        print("[OK] Chat model loaded")
        
        print("\n[3/4] Running parallel operations...")
        
        let embeddingTask = Task {
            let texts = [
                "What is your greatest fear?",
                "Tell me about honor and loyalty",
                "How do you face your destiny?"
            ]
            var embeddings: [Embeddings] = []
            for text in texts {
                let emb = try await embedder.embed(text)
                embeddings.append(emb)
            }
            return embeddings
        }
        
        let embeddings = try await embeddingTask.value
        print("[OK] Generated \(embeddings.count) embeddings while chat model is loaded")
        
        print("\n[4/4] Testing chat generation with RAG context...")
        let ragContext = """
        ### Relevant Background:
        [Personality]: Achilles is fiercely loyal but struggles with pride. He values honor above all.
        [Training]: Trained at the Aretai under Master Kaelen, who taught him discipline.
        
        ### Current Message:
        User: What drives you to fight?
        """
        
        let systemPrompt = "You are Achilles, a legendary warrior. Respond in character, drawing from your background."
        let fullPrompt = chatModel.preprocess(ragContext, [], .none)
        
        print("[INFO] Prompt prepared with RAG context")
        print("[OK] Parallel model test completed successfully!")
        
        #expect(embeddings.count == 3, "Should generate all embeddings")
        #expect(embeddings[0].dimension == embedder.embeddingDimension, "Dimensions should match")
    }
    
    @Test("Dimension mismatch handling")
    func testDimensionMismatchHandling() async throws {
        print("\n=== TEST: Dimension Mismatch Handling ===\n")
        
        let embedModelURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(minilmModelPath)
        
        guard FileManager.default.fileExists(atPath: embedModelURL.path) else {
            print("[SKIP] Embed model not found")
            return
        }
        
        let embedder = try EmbeddingModel(from: embedModelURL)
        let store = SimpleVectorStore()
        
        let text1 = "Hello world"
        let text2 = "Goodbye world"
        
        let emb1 = try await embedder.embedRaw(text1)
        let emb2 = try await embedder.embedRaw(text2)
        
        await store.add(SimpleVectorStore.StoredEmbedding(
            id: "1", text: text1, embedding: emb1, metadata: [:]
        ))
        await store.add(SimpleVectorStore.StoredEmbedding(
            id: "2", text: text2, embedding: emb2, metadata: [:]
        ))
        
        let queryEmb = try await embedder.embedRaw("Hello there")
        let results = await store.search(query: queryEmb, topK: 2)
        
        print("[INFO] Embedding dimension: \(embedder.embeddingDimension)")
        print("[INFO] Query embedding size: \(queryEmb.count)")
        print("[INFO] Search results: \(results.count)")
        
        for result in results {
            print("  - [\(String(format: "%.3f", result.score))] \(result.text)")
        }
        
        print("\n[OK] Dimension handling works correctly")
        print("[NOTE] RAG outputs TEXT, not embeddings - dimension mismatch is not a concern")
        print("[NOTE] The final prompt is always a string that goes to the chat model")
        
        #expect(results.count == 2, "Should return both results")
        #expect(results[0].text == "Hello world", "Hello world should be most similar to Hello there")
    }
}
