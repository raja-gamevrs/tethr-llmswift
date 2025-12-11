import Foundation

/// A simple in-memory vector store for RAG operations.
/// This is a lightweight implementation for testing; Tethr app uses SQLite + HNSW.
public actor SimpleVectorStore {
    
    public struct StoredEmbedding {
        public let id: String
        public let text: String
        public let embedding: [Float]
        public let metadata: [String: String]
        
        public init(id: String, text: String, embedding: [Float], metadata: [String: String] = [:]) {
            self.id = id
            self.text = text
            self.embedding = embedding
            self.metadata = metadata
        }
    }
    
    public struct SearchResult {
        public let id: String
        public let text: String
        public let score: Double
        public let metadata: [String: String]
    }
    
    private var embeddings: [StoredEmbedding] = []
    
    public init() {}
    
    public var count: Int {
        embeddings.count
    }
    
    public func add(_ embedding: StoredEmbedding) {
        embeddings.append(embedding)
    }
    
    public func addBatch(_ newEmbeddings: [StoredEmbedding]) {
        embeddings.append(contentsOf: newEmbeddings)
    }
    
    public func clear() {
        embeddings.removeAll()
    }
    
    public func search(query: [Float], topK: Int = 5, minScore: Double = 0.0) -> [SearchResult] {
        var results: [(StoredEmbedding, Double)] = []
        
        for stored in embeddings {
            let score = cosineSimilarity(query, stored.embedding)
            if score >= minScore {
                results.append((stored, score))
            }
        }
        
        results.sort { $0.1 > $1.1 }
        
        return results.prefix(topK).map { stored, score in
            SearchResult(id: stored.id, text: stored.text, score: score, metadata: stored.metadata)
        }
    }
    
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count else { return 0.0 }
        
        var dotProduct: Float = 0
        var magnitudeA: Float = 0
        var magnitudeB: Float = 0
        
        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            magnitudeA += a[i] * a[i]
            magnitudeB += b[i] * b[i]
        }
        
        let magnitude = sqrt(magnitudeA) * sqrt(magnitudeB)
        guard magnitude > 0 else { return 0.0 }
        
        return Double(dotProduct / magnitude)
    }
}

/// Text chunker for splitting backstories and conversations into indexable chunks.
public struct TextChunker {
    
    public struct Chunk {
        public let id: String
        public let text: String
        public let index: Int
        public let metadata: [String: String]
    }
    
    public let chunkSize: Int
    public let overlap: Int
    
    public init(chunkSize: Int = 500, overlap: Int = 50) {
        self.chunkSize = chunkSize
        self.overlap = overlap
    }
    
    public func chunkText(_ text: String, sourceId: String, metadata: [String: String] = [:]) -> [Chunk] {
        var chunks: [Chunk] = []
        let words = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        
        var currentIndex = 0
        var chunkIndex = 0
        
        while currentIndex < words.count {
            let endIndex = min(currentIndex + chunkSize, words.count)
            let chunkWords = Array(words[currentIndex..<endIndex])
            let chunkText = chunkWords.joined(separator: " ")
            
            var chunkMetadata = metadata
            chunkMetadata["chunk_index"] = String(chunkIndex)
            chunkMetadata["source_id"] = sourceId
            
            chunks.append(Chunk(
                id: "\(sourceId)_chunk_\(chunkIndex)",
                text: chunkText,
                index: chunkIndex,
                metadata: chunkMetadata
            ))
            
            currentIndex += chunkSize - overlap
            chunkIndex += 1
        }
        
        return chunks
    }
    
    public func chunkBySection(_ text: String, sourceId: String, sectionMarker: String = "##") -> [Chunk] {
        let sections = text.components(separatedBy: sectionMarker)
        var chunks: [Chunk] = []
        
        for (index, section) in sections.enumerated() {
            let trimmed = section.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            
            let lines = trimmed.components(separatedBy: .newlines)
            let sectionTitle = lines.first?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "Section \(index)"
            
            chunks.append(Chunk(
                id: "\(sourceId)_section_\(index)",
                text: trimmed,
                index: index,
                metadata: ["section_title": sectionTitle, "source_id": sourceId]
            ))
        }
        
        return chunks
    }
}

/// RAG Pipeline for Tethr-style companion chat applications.
/// Manages backstory indexing, conversation indexing, and context retrieval.
public actor RAGPipeline {
    
    private let embedder: EmbeddingModel
    private let backstoryStore: SimpleVectorStore
    private let conversationStore: SimpleVectorStore
    private let chunker: TextChunker
    
    public nonisolated let embeddingDimension: Int
    
    public init(embedder: EmbeddingModel, chunkSize: Int = 500, overlap: Int = 50) {
        self.embedder = embedder
        self.embeddingDimension = embedder.embeddingDimension
        self.backstoryStore = SimpleVectorStore()
        self.conversationStore = SimpleVectorStore()
        self.chunker = TextChunker(chunkSize: chunkSize, overlap: overlap)
    }
    
    public var backstoryChunkCount: Int {
        get async { await backstoryStore.count }
    }
    
    public var conversationChunkCount: Int {
        get async { await conversationStore.count }
    }
    
    /// Index a persona's backstory for RAG retrieval.
    public func indexBackstory(
        _ text: String,
        personaId: String,
        useSectionChunking: Bool = true
    ) async throws -> Int {
        let chunks: [TextChunker.Chunk]
        
        if useSectionChunking {
            chunks = chunker.chunkBySection(text, sourceId: personaId)
        } else {
            chunks = chunker.chunkText(text, sourceId: personaId, metadata: ["type": "backstory"])
        }
        
        for chunk in chunks {
            let embedding = try await embedder.embedRaw(chunk.text)
            let stored = SimpleVectorStore.StoredEmbedding(
                id: chunk.id,
                text: chunk.text,
                embedding: embedding,
                metadata: chunk.metadata
            )
            await backstoryStore.add(stored)
        }
        
        return chunks.count
    }
    
    /// Index a conversation message for RAG retrieval.
    public func indexConversationMessage(
        _ message: String,
        messageId: String,
        conversationId: String,
        role: String
    ) async throws {
        let embedding = try await embedder.embedRaw(message)
        
        let stored = SimpleVectorStore.StoredEmbedding(
            id: messageId,
            text: message,
            embedding: embedding,
            metadata: [
                "conversation_id": conversationId,
                "role": role,
                "type": "conversation"
            ]
        )
        
        await conversationStore.add(stored)
    }
    
    /// Retrieve relevant context for a user query.
    public func retrieveContext(
        for query: String,
        backstoryTopK: Int = 3,
        conversationTopK: Int = 5,
        minScore: Double = 0.3
    ) async throws -> RAGContext {
        let queryEmbedding = try await embedder.embedRaw(query)
        
        let backstoryResults = await backstoryStore.search(
            query: queryEmbedding,
            topK: backstoryTopK,
            minScore: minScore
        )
        
        let conversationResults = await conversationStore.search(
            query: queryEmbedding,
            topK: conversationTopK,
            minScore: minScore
        )
        
        return RAGContext(
            backstoryChunks: backstoryResults,
            conversationChunks: conversationResults,
            query: query
        )
    }
    
    /// Build a prompt with RAG context for the chat model.
    public func buildPromptWithContext(
        userMessage: String,
        systemPrompt: String,
        backstoryTopK: Int = 3,
        conversationTopK: Int = 5
    ) async throws -> String {
        let context = try await retrieveContext(
            for: userMessage,
            backstoryTopK: backstoryTopK,
            conversationTopK: conversationTopK
        )
        
        return context.buildPrompt(userMessage: userMessage, systemPrompt: systemPrompt)
    }
    
    /// Clear all indexed data.
    public func clearAll() async {
        await backstoryStore.clear()
        await conversationStore.clear()
    }
}

/// Context retrieved from RAG for prompt building.
public struct RAGContext {
    public let backstoryChunks: [SimpleVectorStore.SearchResult]
    public let conversationChunks: [SimpleVectorStore.SearchResult]
    public let query: String
    
    public var hasBackstoryContext: Bool {
        !backstoryChunks.isEmpty
    }
    
    public var hasConversationContext: Bool {
        !conversationChunks.isEmpty
    }
    
    public func buildPrompt(userMessage: String, systemPrompt: String) -> String {
        var prompt = systemPrompt
        
        if hasBackstoryContext {
            prompt += "\n\n### Relevant Background:\n"
            for (i, chunk) in backstoryChunks.enumerated() {
                let sectionTitle = chunk.metadata["section_title"] ?? "Context \(i + 1)"
                prompt += "[\(sectionTitle)]: \(chunk.text.prefix(500))...\n"
            }
        }
        
        if hasConversationContext {
            prompt += "\n\n### Relevant Past Conversations:\n"
            for chunk in conversationChunks {
                let role = chunk.metadata["role"] ?? "unknown"
                prompt += "[\(role)]: \(chunk.text)\n"
            }
        }
        
        prompt += "\n\n### Current Message:\nUser: \(userMessage)"
        
        return prompt
    }
    
    public func formatContextSummary() -> String {
        var summary = "RAG Context Summary:\n"
        summary += "- Backstory chunks: \(backstoryChunks.count)\n"
        summary += "- Conversation chunks: \(conversationChunks.count)\n"
        
        if hasBackstoryContext {
            summary += "\nTop backstory matches:\n"
            for chunk in backstoryChunks.prefix(3) {
                let title = chunk.metadata["section_title"] ?? chunk.id
                summary += "  - [\(String(format: "%.2f", chunk.score))] \(title)\n"
            }
        }
        
        if hasConversationContext {
            summary += "\nTop conversation matches:\n"
            for chunk in conversationChunks.prefix(3) {
                summary += "  - [\(String(format: "%.2f", chunk.score))] \(chunk.text.prefix(50))...\n"
            }
        }
        
        return summary
    }
}
