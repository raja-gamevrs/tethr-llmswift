# App Integration Guide

This guide explains how to integrate the LLM.swift package with embed-only model support into the iOS app for high-performance RAG (Retrieval Augmented Generation).

## Overview

The integration provides:
- **EmbeddingModel**: Lightweight class for fast embeddings using models like `all-MiniLM` or `nomic-embed-text`
- **RAGPipeline**: Complete pipeline for backstory/conversation indexing and context retrieval
- **Parallel Model Support**: Run chat model + embedding model simultaneously
- **Sub-second RAG latency**: ~20-35ms per embedding after warmup

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           App                                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ EmbeddingModel│    │     LLM      │    │   RAGPipeline    │  │
│  │ (all-MiniLM) │    │   (Gemma)    │    │ (Vector Store)   │  │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘  │
│         │                   │                      │            │
│         └───────────────────┼──────────────────────┘            │
│                             │                                   │
│              ┌──────────────▼──────────────┐                   │
│              │       LlamaBackend          │                   │
│              │  (Serialized GPU Access)    │                   │
│              └──────────────┬──────────────┘                   │
├─────────────────────────────┼───────────────────────────────────┤
│                      llama.cpp (Metal)                          │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-Model Memory Architecture

Both models remain loaded in memory simultaneously:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU/CPU Memory                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────────┐    │
│  │   Embedding Model   │    │        Chat Model           │    │
│  │   (MiniLM ~21MB)    │    │     (Gemma-3 4B ~2.5GB)     │    │
│  │   Always Resident   │    │      Always Resident        │    │
│  └─────────────────────┘    └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    GPU Execution (Serialized)                   │
├─────────────────────────────────────────────────────────────────┤
│  Time ──────────────────────────────────────────────────────►   │
│                                                                 │
│  Embed: ████░░░░░░░░████░░░░░░░░░░░░░░░░████                   │
│  Chat:  ░░░░████████░░░░████████████████░░░░                   │
│         ▲                                                       │
│         └── Operations serialized, never overlap                │
└─────────────────────────────────────────────────────────────────┘
```

## Multi-Model Support (LlamaBackend)

The library includes `LlamaBackend`, a centralized manager that enables running multiple llama.cpp models simultaneously without crashes.

### Why This Is Needed

llama.cpp with Metal backend cannot safely run multiple model contexts at the same time. When both a chat model and embedding model try to use the GPU simultaneously, it causes crashes due to Metal resource conflicts.

### How It Works

1. **Single Backend Initialization**: `llama_backend_init()` is called exactly once, regardless of how many models you load
2. **Serialized GPU Access**: All `llama_decode()` calls are queued through a serial dispatch queue
3. **Memory Coexistence**: Both models stay loaded in memory - only execution is serialized

### Automatic Handling

You don't need to do anything special - the library handles serialization internally:

```swift
// Both models load and coexist in memory
let embedder = try EmbeddingModel(from: embedModelPath)  // Uses LlamaBackend internally
let chatModel = LLM(from: chatModelPath)!                // Uses LlamaBackend internally

// Safe to call from different tasks/threads
Task {
    let embedding = try await embedder.embed("query")    // Serialized automatically
}
Task {
    await chatModel.respond(to: "Hello")                 // Waits if embedder is running
}
```

### Manual Access (Advanced)

For custom llama.cpp operations, use `LlamaBackend.shared.execute`:

```swift
import LLM

// Wrap any GPU-intensive llama.cpp call
let result = LlamaBackend.shared.execute {
    llama_decode(context, batch)
}

// Async version
let result = await LlamaBackend.shared.executeAsync {
    llama_decode(context, batch)
}

// Check active model count
print(LlamaBackend.shared.modelCount)  // e.g., 2
```

### Performance Impact

- **Latency**: Minimal - only adds wait time when operations actually overlap
- **Memory**: No impact - models stay resident regardless
- **Throughput**: Sequential execution means one operation at a time

Typical timeline when both models are active:
```
User sends message:
  1. Embed query (20ms)     <- Embedding model runs
  2. Search vectors (5ms)   <- CPU only, no serialization
  3. Generate response      <- Chat model runs (waits if embed still running)
```

## Installation

### 1. Add Package Dependency

In your Xcode project, add the LLM.swift package:

**File > Add Package Dependencies...**

```
URL: https://github.com/your-repo/tethr-llmswift
Branch: main
```

Or in `Package.swift`:

```swift
dependencies: [
    .package(path: "../tethr-llmswift")
    // Or from remote:
    // .package(url: "https://github.com/your-repo/tethr-llmswift", branch: "main")
]
```

### 2. Bundle Models

Add these models to your app bundle:

| Model | Size | Dimensions | Use Case |
|-------|------|------------|----------|
| `all-minilm-l6-v2-q4_k_m.gguf` | ~21MB | 384 | Fast embeddings |
| `nomic-embed-text-v1.5.Q4_0.gguf` | ~78MB | 768 | Higher quality embeddings |
| `unsloth-gemma-3-4b-it-Q4_K_M.gguf` | ~2.5GB | N/A | Chat generation |

## Implementation

### Step 1: Create Embedding Service

```swift
import LLM

actor EmbeddingService {
    private var embedder: EmbeddingModel?
    
    var isLoaded: Bool {
        embedder != nil
    }
    
    var dimension: Int {
        embedder?.embeddingDimension ?? 0
    }
    
    func load(modelPath: String) async throws {
        EmbeddingModel.silenceLogging()
        embedder = try EmbeddingModel(from: modelPath, maxTokenCount: 512)
    }
    
    func embed(_ text: String) async throws -> [Float] {
        guard let embedder = embedder else {
            throw EmbeddingError.modelNotLoaded
        }
        return try await embedder.embedRaw(text)
    }
    
    func embedBatch(_ texts: [String]) async throws -> [[Float]] {
        guard let embedder = embedder else {
            throw EmbeddingError.modelNotLoaded
        }
        return try await embedder.embedBatch(texts).map { $0.values }
    }
    
    func similarity(between text1: String, and text2: String) async throws -> Double {
        guard let embedder = embedder else {
            throw EmbeddingError.modelNotLoaded
        }
        return try await embedder.similarity(between: text1, and: text2)
    }
}

enum EmbeddingError: Error {
    case modelNotLoaded
}
```

### Step 2: Update LlamaService

Modify your existing `LlamaService` to use the new embedding model:

```swift
import LLM

class LlamaService: ObservableObject {
    private var _llm: LLM?
    private let embeddingService = EmbeddingService()
    
    @Published var isModelLoaded = false
    @Published var isEmbeddingModelLoaded = false
    
    // Load both models at app startup
    func loadModels(chatModelPath: String, embedModelPath: String) async throws {
        // Load embedding model first (faster, smaller)
        try await embeddingService.load(modelPath: embedModelPath)
        await MainActor.run { isEmbeddingModelLoaded = true }
        
        // Load chat model
        LLM.silenceLogging()
        guard let llm = LLM(from: URL(fileURLWithPath: chatModelPath), maxTokenCount: 4096) else {
            throw InferenceError.modelLoadFailed("Failed to load chat model")
        }
        llm.template = .gemma
        self._llm = llm
        await MainActor.run { isModelLoaded = true }
    }
    
    // Generate embeddings using the dedicated embedding model
    func generateEmbedding(_ text: String) async throws -> [Float] {
        return try await embeddingService.embed(text)
    }
    
    // Chat generation (unchanged)
    func generateResponse(prompt: String) async -> String {
        guard let llm = _llm else { return "" }
        return await llm.getCompletion(from: prompt)
    }
}
```

### Step 3: Integrate RAG Pipeline

```swift
import LLM

class RAGService {
    private let pipeline: RAGPipeline
    private let llamaService: LlamaService
    
    init(embedder: EmbeddingModel, llamaService: LlamaService) {
        self.pipeline = RAGPipeline(embedder: embedder, chunkSize: 400, overlap: 50)
        self.llamaService = llamaService
    }
    
    // Index a companion's backstory at app launch or companion selection
    func indexBackstory(_ backstory: String, personaId: String) async throws -> Int {
        return try await pipeline.indexBackstory(backstory, personaId: personaId)
    }
    
    // Index conversation messages as they occur
    func indexMessage(_ message: String, messageId: String, conversationId: String, role: String) async throws {
        try await pipeline.indexConversationMessage(
            message,
            messageId: messageId,
            conversationId: conversationId,
            role: role
        )
    }
    
    // Generate response with RAG context
    func generateResponseWithRAG(
        userMessage: String,
        systemPrompt: String,
        backstoryTopK: Int = 3,
        conversationTopK: Int = 5
    ) async throws -> String {
        // Retrieve relevant context
        let context = try await pipeline.retrieveContext(
            for: userMessage,
            backstoryTopK: backstoryTopK,
            conversationTopK: conversationTopK,
            minScore: 0.3
        )
        
        // Build prompt with context
        let enrichedPrompt = context.buildPrompt(
            userMessage: userMessage,
            systemPrompt: systemPrompt
        )
        
        // Generate response
        return await llamaService.generateResponse(prompt: enrichedPrompt)
    }
}
```

### Step 4: App Initialization

```swift
@main
struct App: App {
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .task {
                    await appState.initialize()
                }
        }
    }
}

class AppState: ObservableObject {
    @Published var isReady = false
    @Published var loadingStatus = "Starting..."
    
    let llamaService = LlamaService()
    var ragService: RAGService?
    
    func initialize() async {
        do {
            // Get model paths from bundle
            let embedModelPath = Bundle.main.path(
                forResource: "all-minilm-l6-v2-q4_k_m",
                ofType: "gguf"
            )!
            let chatModelPath = Bundle.main.path(
                forResource: "unsloth-gemma-3-4b-it-Q4_K_M",
                ofType: "gguf"
            )!
            
            await MainActor.run { loadingStatus = "Loading embedding model..." }
            
            // Load models
            try await llamaService.loadModels(
                chatModelPath: chatModelPath,
                embedModelPath: embedModelPath
            )
            
            await MainActor.run { loadingStatus = "Indexing backstory..." }
            
            // Initialize RAG with the loaded embedding model
            let embedder = try EmbeddingModel(from: embedModelPath)
            ragService = RAGService(embedder: embedder, llamaService: llamaService)
            
            // Pre-index current companion's backstory
            if let backstory = loadCurrentCompanionBackstory() {
                _ = try await ragService?.indexBackstory(backstory, personaId: "current")
            }
            
            await MainActor.run {
                loadingStatus = "Ready"
                isReady = true
            }
        } catch {
            await MainActor.run {
                loadingStatus = "Error: \(error.localizedDescription)"
            }
        }
    }
    
    private func loadCurrentCompanionBackstory() -> String? {
        // Load from your data source
        return nil
    }
}
```

## Dimension Mismatch - Not a Concern

The embedding model dimension (384 for MiniLM, 768 for nomic) is **independent** of the chat model. Here's why:

```
User Query: "Tell me about your childhood"
                    │
                    ▼
┌─────────────────────────────────────┐
│         EmbeddingModel              │
│   "Tell me about..." → [0.1, 0.2...] │  ← 384 dimensions
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│         Vector Store                │
│   Search for similar chunks         │
│   Returns: TEXT chunks              │  ← Plain text, no dimensions
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│         RAG Context Builder         │
│   Combines retrieved TEXT into      │
│   a prompt string                   │  ← Plain text prompt
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│         Chat Model (Gemma)          │
│   Receives TEXT prompt              │
│   Generates TEXT response           │  ← No embedding dimensions involved
└─────────────────────────────────────┘
```

The RAG pipeline outputs **text**, not embeddings. The chat model never sees embedding vectors.

## Performance Benchmarks

Tested on iPhone 14 Pro:

| Operation | Time |
|-----------|------|
| Embedding model load | ~60ms |
| Chat model load | ~3-5s |
| Single embedding | ~20-35ms |
| Backstory indexing (20 chunks) | ~600ms |
| Context retrieval | ~25ms |
| Full RAG + generation | ~2-4s |

## Best Practices

### 1. Pre-index Backstories
Index companion backstories at app launch or when switching companions:

```swift
func switchCompanion(to companion: Companion) async throws {
    await ragService?.clearBackstory()
    _ = try await ragService?.indexBackstory(companion.backstory, personaId: companion.id)
}
```

### 2. Index Conversations Incrementally
Index messages as they're sent/received:

```swift
func onMessageSent(_ message: Message) async {
    try? await ragService?.indexMessage(
        message.content,
        messageId: message.id,
        conversationId: currentConversationId,
        role: message.role.rawValue
    )
}
```

### 3. Tune Retrieval Parameters
Adjust based on your needs:

```swift
let context = try await pipeline.retrieveContext(
    for: query,
    backstoryTopK: 3,      // More = richer context, slower
    conversationTopK: 5,   // Recent conversation context
    minScore: 0.3          // Filter low-relevance chunks
)
```

### 4. Handle Memory Pressure
On low-memory devices, consider:

```swift
func handleMemoryWarning() async {
    // Clear conversation index (can be rebuilt)
    await ragService?.clearConversations()
    
    // Keep backstory indexed (expensive to rebuild)
}
```

## Troubleshooting

### Model Loading Fails
- Verify model file exists in bundle
- Check file permissions
- Ensure sufficient memory (~500MB for embedding + ~3GB for chat)

### Slow Embeddings
- First embedding is slow (~800ms) due to warmup
- Subsequent embeddings are fast (~20-35ms)
- Consider warming up with a dummy embedding at launch

### Poor Retrieval Quality
- Increase `backstoryTopK` for more context
- Lower `minScore` threshold
- Use larger embedding model (nomic-embed vs MiniLM)

### Memory Issues
- Use smaller quantization (Q4_K_M)
- Reduce `maxTokenCount` for embedding model
- Clear conversation index periodically

## API Reference

### EmbeddingModel

```swift
// Initialize
let embedder = try EmbeddingModel(from: path, maxTokenCount: 512, gpuLayers: -1)

// Properties
embedder.embeddingDimension  // Int - vector size
embedder.modelPath           // String - loaded model path

// Methods
try await embedder.embed(text)                    // -> Embeddings
try await embedder.embedRaw(text)                 // -> [Float]
try await embedder.embedBatch(texts)              // -> [Embeddings]
try await embedder.similarity(between:and:)       // -> Double
try await embedder.rank(query:candidates:topK:)   // -> [(text, score, index)]
try await embedder.findMostSimilar(to:in:)        // -> (text, score, index)?
```

### RAGPipeline

```swift
// Initialize
let pipeline = RAGPipeline(embedder: embedder, chunkSize: 500, overlap: 50)

// Properties
await pipeline.backstoryChunkCount     // Int
await pipeline.conversationChunkCount  // Int

// Methods
try await pipeline.indexBackstory(text, personaId:)           // -> Int (chunk count)
try await pipeline.indexConversationMessage(text, messageId:, conversationId:, role:)
try await pipeline.retrieveContext(for:, backstoryTopK:, conversationTopK:, minScore:)  // -> RAGContext
try await pipeline.buildPromptWithContext(userMessage:, systemPrompt:, ...)  // -> String
await pipeline.clearAll()
```

### RAGContext

```swift
context.backstoryChunks      // [SearchResult]
context.conversationChunks   // [SearchResult]
context.hasBackstoryContext  // Bool
context.hasConversationContext // Bool
context.buildPrompt(userMessage:, systemPrompt:)  // -> String
context.formatContextSummary()  // -> String (for debugging)
```

## Migration from Current Implementation

If you're currently using `LlamaService.generateImprovedTextEmbedding()`:

```swift
// OLD (hash-based, low quality)
let embedding = generateImprovedTextEmbedding(text)

// NEW (neural, high quality)
let embedding = try await embeddingService.embed(text)
```

The new embeddings are:
- **Semantic**: Understand meaning, not just word overlap
- **Normalized**: L2 normalized, ready for cosine similarity
- **Fast**: ~20-35ms vs ~1-2ms (worth the tradeoff for quality)
