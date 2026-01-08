#!/usr/bin/env python3
"""
Validate index k-NN parity between FAISS and MLX implementations.

This script verifies that the pure MLX k-NN search (used in Swift) matches
the FAISS-based search (used in Python RVC).

Usage:
    # Validate with existing index
    python validate_index_parity.py --index model.index
    
    # Full pipeline validation
    python validate_index_parity.py --model model.pth --index model.index --audio test.wav
    
    # Export test data for Swift
    python validate_index_parity.py --index model.index --export-swift test_data/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    print("Warning: faiss not installed. Use: pip install faiss-cpu")
    faiss = None

try:
    import mlx.core as mx
except ImportError:
    print("Error: mlx not installed. Use: pip install mlx")
    sys.exit(1)


def mlx_knn_search(query: np.ndarray, vectors: np.ndarray, k: int = 8) -> tuple:
    """
    Pure MLX k-NN search matching the Swift IndexManager implementation.
    
    Args:
        query: (T, C) query vectors
        vectors: (N, C) index vectors
        k: number of neighbors
        
    Returns:
        (distances, indices) as numpy arrays
    """
    query_mx = mx.array(query)
    vectors_mx = mx.array(vectors)
    
    # L2 distance: ||q - v||^2 = -2*q@v.T + ||v||^2 (ignoring ||q||^2 for ranking)
    qv = mx.matmul(query_mx, vectors_mx.T)  # (T, N)
    v_norm_sq = mx.sum(vectors_mx * vectors_mx, axis=1, keepdims=True).T  # (1, N)
    distances = -2.0 * qv + v_norm_sq  # (T, N)
    
    # Top-k smallest distances
    topk_indices = mx.argpartition(distances, kth=k, axis=1)[:, :k]
    mx.eval(topk_indices)
    
    # Gather distances
    topk_indices_np = np.array(topk_indices)
    distances_np = np.array(distances)
    T = query.shape[0]
    topk_distances = np.zeros((T, k))
    for t in range(T):
        for ki in range(k):
            idx = topk_indices_np[t, ki]
            topk_distances[t, ki] = distances_np[t, idx]
    
    return topk_distances, topk_indices_np


def weighted_blend(query: np.ndarray, vectors: np.ndarray, 
                   distances: np.ndarray, indices: np.ndarray,
                   index_rate: float = 1.0) -> np.ndarray:
    """
    Weighted blend of neighbors using inverse squared distance.
    Matches Python RVC pipeline_mlx.py:183-204.
    """
    # Weight by inverse squared distance
    distances = np.maximum(distances, 1e-6)  # Avoid div by zero
    weights = 1.0 / (distances * distances)
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Gather neighbors
    T, k = indices.shape
    C = vectors.shape[1]
    neighbors = np.zeros((T, k, C))
    for t in range(T):
        for ki in range(k):
            neighbors[t, ki] = vectors[indices[t, ki]]
    
    # Weighted sum
    blended = np.sum(neighbors * weights[:, :, None], axis=1)
    
    # Mix with original
    return index_rate * blended + (1.0 - index_rate) * query


def validate_knn_parity(index_path: str, n_test: int = 100) -> dict:
    """
    Validate MLX k-NN matches FAISS k-NN.
    
    Args:
        index_path: Path to FAISS .index file
        n_test: Number of test queries
        
    Returns:
        Dictionary with validation results
    """
    if faiss is None:
        return {"error": "FAISS not installed"}
    
    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)
    vectors = index.reconstruct_n(0, index.ntotal)
    print(f"Index: {index.ntotal} vectors, {vectors.shape[1]}D")
    
    # Generate random test queries (simulating HuBERT output)
    np.random.seed(42)
    queries = np.random.randn(n_test, vectors.shape[1]).astype(np.float32)
    
    # FAISS search
    print("Running FAISS k-NN...")
    faiss_distances, faiss_indices = index.search(queries, k=8)
    
    # MLX search
    print("Running MLX k-NN...")
    mlx_distances, mlx_indices = mlx_knn_search(queries, vectors, k=8)
    
    # Compare indices (should match for same distance metric)
    # Note: Order within k may differ if distances are equal
    indices_match = 0
    for t in range(n_test):
        faiss_set = set(faiss_indices[t])
        mlx_set = set(mlx_indices[t])
        if faiss_set == mlx_set:
            indices_match += 1
    
    indices_pct = 100.0 * indices_match / n_test
    
    # Compare blended results
    faiss_blended = weighted_blend(queries, vectors, faiss_distances, faiss_indices)
    mlx_blended = weighted_blend(queries, vectors, mlx_distances, mlx_indices)
    
    # Correlation
    faiss_flat = faiss_blended.flatten()
    mlx_flat = mlx_blended.flatten()
    correlation = np.corrcoef(faiss_flat, mlx_flat)[0, 1]
    
    # Max absolute difference
    max_diff = np.abs(faiss_blended - mlx_blended).max()
    
    results = {
        "index_path": index_path,
        "n_vectors": index.ntotal,
        "dimension": vectors.shape[1],
        "n_test_queries": n_test,
        "indices_match_pct": indices_pct,
        "blended_correlation": correlation,
        "max_absolute_diff": float(max_diff),
        "passed": correlation > 0.99 and indices_pct > 90,
    }
    
    print(f"\n=== Validation Results ===")
    print(f"Indices match: {indices_match}/{n_test} ({indices_pct:.1f}%)")
    print(f"Blended correlation: {correlation:.6f}")
    print(f"Max absolute diff: {max_diff:.6e}")
    print(f"Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
    
    return results


def export_swift_test_data(index_path: str, output_dir: str, n_test: int = 20):
    """
    Export test data for Swift unit tests.
    """
    if faiss is None:
        print("Error: FAISS required for export")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading index: {index_path}")
    index = faiss.read_index(index_path)
    vectors = index.reconstruct_n(0, index.ntotal)
    
    # Use small subset for tests
    n_vectors = min(1000, index.ntotal)
    vectors_subset = vectors[:n_vectors]
    
    # Generate queries
    np.random.seed(42)
    queries = np.random.randn(n_test, vectors.shape[1]).astype(np.float32)
    
    # Run FAISS search on subset
    subset_index = faiss.IndexFlatL2(vectors.shape[1])
    subset_index.add(vectors_subset)
    distances, indices = subset_index.search(queries, k=8)
    
    # Compute expected blended result
    blended = weighted_blend(queries, vectors_subset, distances, indices, index_rate=1.0)
    
    # Save as safetensors
    output_path = Path(output_dir)
    mx.save_safetensors(str(output_path / "test_vectors.safetensors"), 
                        {"vectors": mx.array(vectors_subset)})
    mx.save_safetensors(str(output_path / "test_queries.safetensors"), 
                        {"query": mx.array(queries)})
    mx.save_safetensors(str(output_path / "test_expected.safetensors"), 
                        {"result": mx.array(blended)})
    
    print(f"Exported test data to: {output_dir}")
    print(f"  - test_vectors.safetensors: {n_vectors} vectors")
    print(f"  - test_queries.safetensors: {n_test} queries")
    print(f"  - test_expected.safetensors: expected blended results")


def main():
    parser = argparse.ArgumentParser(
        description="Validate index k-NN parity between FAISS and MLX"
    )
    parser.add_argument("--index", required=True, help="Path to FAISS .index file")
    parser.add_argument("--n-test", type=int, default=100, help="Number of test queries")
    parser.add_argument("--export-swift", help="Export test data for Swift to this directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.index):
        print(f"Error: Index file not found: {args.index}")
        sys.exit(1)
    
    if args.export_swift:
        export_swift_test_data(args.index, args.export_swift, n_test=min(20, args.n_test))
    
    results = validate_knn_parity(args.index, n_test=args.n_test)
    
    if not results.get("passed", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
