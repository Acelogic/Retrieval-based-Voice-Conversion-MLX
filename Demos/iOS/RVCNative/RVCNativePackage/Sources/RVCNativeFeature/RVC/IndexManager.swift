import Foundation
import MLX
import MLXNN

/// Manages index vectors for speaker embedding retrieval.
///
/// This class loads pre-extracted vectors from a safetensors file and performs
/// k-NN search using MLX operations to find similar speaker embeddings.
///
/// The algorithm matches Python RVC exactly:
/// 1. Compute cosine similarity via matmul
/// 2. Find top-k=8 neighbors per frame
/// 3. Weight by inverse squared distance
/// 4. Blend with original features using indexRate
public class IndexManager {
    
    /// Stored index vectors (N, 768)
    private var vectors: MLXArray?
    
    /// Number of vectors in the index
    public var count: Int {
        vectors?.shape[0] ?? 0
    }
    
    /// Vector dimension (typically 768 for HuBERT)
    public var dimension: Int {
        vectors?.shape[1] ?? 0
    }
    
    /// Whether an index is currently loaded
    public var isLoaded: Bool {
        vectors != nil
    }
    
    public init() {}
    
    /// Load index vectors from a file.
    ///
    /// Supports two formats:
    /// - `.index`: Native FAISS IVFFlat format (parsed directly)
    /// - `.safetensors`: Pre-converted format with "vectors" key
    ///
    /// - Parameter url: URL to the index file
    /// - Parameter logger: Optional callback for logging
    /// - Throws: If the file cannot be loaded or parsed
    public func load(url: URL, logger: ((String) -> Void)? = nil) throws {
        let ext = url.pathExtension.lowercased()
        
        if ext == "index" {
            // Native FAISS format
            let (loadedVectors, dim) = try FAISSIndexReader.read(url: url, logger: logger)
            guard dim == 768 else {
                throw IndexManagerError.invalidShape([loadedVectors.shape[0], dim])
            }
            self.vectors = loadedVectors
            MLX.eval(self.vectors!)
        } else {
            // Safetensors format (legacy)
            let arrays = try MLX.loadArrays(url: url)
            
            guard let loadedVectors = arrays["vectors"] else {
                throw IndexManagerError.missingVectors
            }
            
            guard loadedVectors.ndim == 2 else {
                throw IndexManagerError.invalidShape(loadedVectors.shape)
            }
            
            self.vectors = loadedVectors
            MLX.eval(self.vectors!)
        }
    }
    
    /// Unload the index to free memory.
    public func unload() {
        vectors = nil
    }
    
    /// Perform k-NN search and blend features with retrieved neighbors.
    ///
    /// This matches the Python implementation in `pipeline_mlx.py:183-204`:
    /// ```python
    /// score, ix = index.search(feats_np[0], k=8)
    /// weight = np.square(1 / score)
    /// weight /= weight.sum(axis=1, keepdims=True)
    /// neighbors = big_npy[ix]  # (T, 8, 768)
    /// new_feats = np.sum(neighbors * weight[:,:,None], axis=1)
    /// feats_np = index_rate * new_feats + (1 - index_rate) * feats_np
    /// ```
    ///
    /// - Parameters:
    ///   - features: HuBERT output features with shape (1, T, 768)
    ///   - indexRate: Blend ratio (0.0-1.0). 0 = use original, 1 = use retrieved
    ///   - k: Number of neighbors (default 8, matching Python)
    /// - Returns: Blended features with shape (1, T, 768)
    public func search(features: MLXArray, indexRate: Float, k: Int = 8) -> MLXArray {
        guard let vectors = vectors else {
            return features  // No index loaded, return original
        }
        
        guard indexRate > 0 else {
            return features  // Index disabled, return original
        }
        
        // Input: (1, T, C) where C=768
        _ = features  // Keep reference to original
        let query = features.squeezed(axis: 0)  // (T, C)
        let T = query.shape[0]
        let C = query.shape[1]
        _ = vectors.shape[0]  // N - number of vectors
        
        // 1. Compute similarity scores: query @ vectors.T -> (T, N)
        // Using L2 distance (as FAISS does by default for Flat indices)
        // FAISS search returns L2 distances, not similarities
        // score = ||q - v||^2 = ||q||^2 + ||v||^2 - 2*q@v.T
        
        // For efficiency, compute via: D = -2*q@v.T + ||v||^2
        // We don't need ||q||^2 since we only need relative ordering per row
        let qv = MLX.matmul(query, vectors.transposed())  // (T, N)
        let vNormSq = MLX.sum(vectors * vectors, axis: 1, keepDims: true).transposed()  // (1, N)
        let distances = -2.0 * qv + vNormSq  // (T, N) - L2 squared distance
        
        // 2. Find top-k smallest distances (argpartition for efficiency)
        // MLX's argPartition returns indices that would partition at kth position
        let topkIndices = MLX.argPartition(distances, kth: k, axis: 1)[0..., 0..<k]  // (T, k)
        MLX.eval(topkIndices)
        
        // 3. Gather top-k distances
        // Need to gather distances at topkIndices positions
        var topkDistances = MLXArray.zeros([T, k])
        for t in 0..<T {
            for ki in 0..<k {
                let idx = topkIndices[t, ki].item(Int.self)
                topkDistances[t, ki] = distances[t, idx]
            }
        }
        MLX.eval(topkDistances)
        
        // Ensure distances are positive (add small epsilon to avoid division by zero)
        topkDistances = MLX.maximum(topkDistances, MLXArray(1e-6))
        
        // 4. Compute weights: 1 / distance^2 (inverse squared distance)
        // Python: weight = np.square(1 / score) where score = L2 distance
        // Our topkDistances is already L2 squared (||q-v||^2), so:
        // weight = 1 / topkDistances gives us 1/||q-v||^2 matching Python
        let weights = 1.0 / topkDistances  // (T, k)
        
        // Normalize weights per frame
        let weightSums = MLX.sum(weights, axis: 1, keepDims: true)  // (T, 1)
        let normalizedWeights = weights / weightSums  // (T, k)
        
        // 5. Gather neighbor vectors: (T, k, C)
        let neighbors = MLXArray.zeros([T, k, C])
        for t in 0..<T {
            for ki in 0..<k {
                let idx = topkIndices[t, ki].item(Int.self)
                neighbors[t, ki, 0...] = vectors[idx, 0...]
            }
        }
        MLX.eval(neighbors)
        
        // 6. Weighted sum of neighbors
        // Python: new_feats = np.sum(neighbors * weight[:,:,None], axis=1)
        let weightedNeighbors = neighbors * normalizedWeights.expandedDimensions(axis: 2)  // (T, k, C)
        let blended = MLX.sum(weightedNeighbors, axis: 1)  // (T, C)
        
        // 7. Mix with original features
        // Python: feats_np = index_rate * new_feats + (1 - index_rate) * feats_np
        let result = indexRate * blended + (1.0 - indexRate) * query
        
        // Restore batch dimension: (T, C) -> (1, T, C)
        return result.expandedDimensions(axis: 0)
    }
}

/// Errors that can occur during index operations
public enum IndexManagerError: Error, LocalizedError {
    case missingVectors
    case invalidShape([Int])
    
    public var errorDescription: String? {
        switch self {
        case .missingVectors:
            return "Safetensors file does not contain 'vectors' key"
        case .invalidShape(let shape):
            return "Invalid vectors shape: \(shape). Expected (N, 768)"
        }
    }
}
