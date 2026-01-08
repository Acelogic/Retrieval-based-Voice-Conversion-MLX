#!/usr/bin/env python3
"""
Convert FAISS .index files to .safetensors format for iOS.

This tool extracts raw vectors from a FAISS index and saves them
in safetensors format, which can be loaded directly by MLX Swift.

Usage:
    python convert_index_for_ios.py model.index -o model_index.safetensors
    python convert_index_for_ios.py model.index  # outputs model.index.safetensors
"""

import argparse
import os
import sys

try:
    import faiss
except ImportError:
    print("Error: faiss not installed. Install with: pip install faiss-cpu")
    sys.exit(1)

try:
    import mlx.core as mx
except ImportError:
    print("Error: mlx not installed. Install with: pip install mlx")
    sys.exit(1)

import numpy as np


def convert_index(index_path: str, output_path: str, verbose: bool = True) -> dict:
    """
    Convert a FAISS index to safetensors format.
    
    Args:
        index_path: Path to input .index file
        output_path: Path to output .safetensors file
        verbose: Print progress information
        
    Returns:
        Dictionary with conversion statistics
    """
    if verbose:
        print(f"Loading FAISS index: {index_path}")
    
    # Load the FAISS index
    index = faiss.read_index(index_path)
    
    n_vectors = index.ntotal
    if verbose:
        print(f"Index contains {n_vectors:,} vectors")
    
    # Extract all vectors
    # This works for most index types (Flat, IVF with stored vectors)
    if hasattr(index, 'reconstruct_n'):
        vectors = index.reconstruct_n(0, n_vectors)
    else:
        # Fallback: try to get vectors from the index directly
        raise ValueError(f"Cannot extract vectors from index type: {type(index)}")
    
    # vectors shape: (N, D) where D is typically 768 for HuBERT
    dim = vectors.shape[1]
    if verbose:
        print(f"Vector dimension: {dim}")
        print(f"Memory size: {vectors.nbytes / (1024*1024):.1f} MB")
    
    # Convert to MLX and save
    vectors_mlx = mx.array(vectors.astype(np.float32))
    
    if verbose:
        print(f"Saving to: {output_path}")
    
    mx.save_safetensors(output_path, {"vectors": vectors_mlx})
    
    # Verify the saved file
    loaded = mx.load(output_path)
    assert "vectors" in loaded, "Vectors not found in saved file"
    assert loaded["vectors"].shape == (n_vectors, dim), "Shape mismatch after save"
    
    if verbose:
        print(f"✅ Successfully converted {n_vectors:,} vectors ({dim}D)")
    
    return {
        "n_vectors": n_vectors,
        "dimension": dim,
        "memory_mb": vectors.nbytes / (1024*1024),
        "output_path": output_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert FAISS .index to .safetensors for iOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s model.index -o model_index.safetensors
    %(prog)s model.index  # outputs model.index.safetensors
    %(prog)s *.index      # batch convert multiple files
        """
    )
    parser.add_argument(
        "index_files",
        nargs="+",
        help="Input FAISS .index file(s)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output .safetensors file (only for single input)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Validate output argument
    if args.output and len(args.index_files) > 1:
        parser.error("-o/--output can only be used with a single input file")
    
    # Process each input file
    for index_path in args.index_files:
        if not os.path.exists(index_path):
            print(f"Error: File not found: {index_path}")
            continue
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Default: same name with .safetensors extension
            base = os.path.splitext(index_path)[0]
            output_path = f"{base}_index.safetensors"
        
        try:
            stats = convert_index(index_path, output_path, verbose=not args.quiet)
            if not args.quiet:
                print()
        except Exception as e:
            print(f"Error converting {index_path}: {e}")
            if len(args.index_files) == 1:
                sys.exit(1)


if __name__ == "__main__":
    main()
