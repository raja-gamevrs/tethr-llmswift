import Foundation
import llama

/// Centralized manager for llama.cpp backend initialization and execution serialization.
///
/// This singleton ensures:
/// 1. `llama_backend_init()` is called exactly once across all model instances
/// 2. All llama.cpp GPU operations are serialized to prevent Metal resource contention
/// 3. Multiple models (LLM, EmbeddingModel) can coexist safely in memory
///
/// ## Why Serialization is Needed
/// llama.cpp with Metal backend cannot safely run multiple model contexts simultaneously.
/// When both a chat model and embedding model try to use the GPU at the same time,
/// it causes crashes due to Metal resource conflicts. This manager serializes only the
/// GPU-intensive `llama_decode()` calls, allowing models to remain loaded in memory.
///
/// ## Memory Model
/// - Both chat and embedding models stay loaded in GPU/CPU memory simultaneously
/// - No loading/unloading occurs during normal operation
/// - Only inference operations are serialized (not memory residency)
///
/// ## Usage
/// ```swift
/// // Initialization (called automatically by LLM and EmbeddingModel)
/// LlamaBackend.shared.ensureInitialized()
///
/// // Wrap GPU operations for serialized access
/// let result = LlamaBackend.shared.execute {
///     llama_decode(context, batch)
/// }
/// ```
public final class LlamaBackend: @unchecked Sendable {
    
    /// Shared singleton instance
    public static let shared = LlamaBackend()
    
    /// Serial queue for all llama.cpp GPU operations.
    /// Using `.userInitiated` QoS for responsive inference.
    private let executionQueue = DispatchQueue(
        label: "com.llm.llamabackend.execution",
        qos: .userInitiated
    )
    
    /// Whether the backend has been initialized
    private var isInitialized = false
    private let initLock = NSLock()
    
    /// Whether logging has been silenced
    private var isLoggingSilenced = false
    
    /// Track number of active model instances (for debugging)
    private var activeModelCount = 0
    
    private init() {}
    
    /// Ensures the llama.cpp backend is initialized exactly once.
    /// Thread-safe and idempotent - safe to call from multiple models.
    public func ensureInitialized() {
        initLock.lock()
        defer { initLock.unlock() }
        
        guard !isInitialized else { return }
        isInitialized = true
        llama_backend_init()
    }
    
    /// Silences llama.cpp logging output. Thread-safe and idempotent.
    public func silenceLogging() {
        initLock.lock()
        defer { initLock.unlock() }
        
        guard !isLoggingSilenced else { return }
        isLoggingSilenced = true
        
        let noopCallback: @convention(c) (ggml_log_level, UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = { _, _, _ in }
        llama_log_set(noopCallback, nil)
        ggml_log_set(noopCallback, nil)
    }
    
    /// Registers a model instance. Used for tracking active models.
    public func registerModel() {
        initLock.lock()
        defer { initLock.unlock() }
        activeModelCount += 1
    }
    
    /// Unregisters a model instance. Used for tracking active models.
    public func unregisterModel() {
        initLock.lock()
        defer { initLock.unlock() }
        activeModelCount = max(0, activeModelCount - 1)
    }
    
    /// Returns the number of currently active model instances.
    public var modelCount: Int {
        initLock.lock()
        defer { initLock.unlock() }
        return activeModelCount
    }
    
    /// Executes a synchronous llama.cpp operation with serialized access.
    ///
    /// Use this for GPU-intensive operations like `llama_decode()` that need
    /// exclusive access to Metal resources. Operations are queued and executed
    /// one at a time to prevent resource conflicts.
    ///
    /// - Parameter operation: The operation to execute
    /// - Returns: The result of the operation
    /// - Throws: Any error thrown by the operation
    ///
    /// - Note: This blocks the calling thread until the operation completes.
    ///   For async contexts, consider using `executeAsync`.
    public func execute<T>(_ operation: () throws -> T) rethrows -> T {
        return try executionQueue.sync {
            try operation()
        }
    }
    
    /// Executes an async llama.cpp operation with serialized access.
    ///
    /// Async-friendly version that doesn't block the calling thread.
    /// Uses the same serial queue internally.
    ///
    /// - Parameter operation: The operation to execute
    /// - Returns: The result of the operation
    public func executeAsync<T>(_ operation: @escaping () -> T) async -> T {
        return await withCheckedContinuation { continuation in
            executionQueue.async {
                let result = operation()
                continuation.resume(returning: result)
            }
        }
    }
    
    /// Executes an async throwing llama.cpp operation with serialized access.
    ///
    /// - Parameter operation: The throwing operation to execute
    /// - Returns: The result of the operation
    /// - Throws: Any error thrown by the operation
    public func executeAsyncThrowing<T>(_ operation: @escaping () throws -> T) async throws -> T {
        return try await withCheckedThrowingContinuation { continuation in
            executionQueue.async {
                do {
                    let result = try operation()
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
