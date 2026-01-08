#!/usr/bin/env python3
"""
Export reference data from a FAISS index for Swift validation.
Outputs a simple format that Swift can verify against.
"""
import faiss
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python export_index_reference.py <path_to_index>")
    sys.exit(1)

path = sys.argv[1]
print(f"Loading: {path}")
idx = faiss.read_index(path)

print(f"Type: {type(idx).__name__}")
print(f"Dimension: {idx.d}")
print(f"Total vectors: {idx.ntotal}")
if hasattr(idx, 'nlist'):
    print(f"nlist: {idx.nlist}")

# Extract first 10 vectors for validation
vecs = idx.reconstruct_n(0, min(10, idx.ntotal))
print(f"\nFirst 10 vectors, first 5 elements:")
for i, v in enumerate(vecs):
    print(f"  {i}: {v[:5].tolist()}")

# Write to simple format for Swift validation
output = path.replace('.index', '_reference.txt')
with open(output, 'w') as f:
    f.write(f"ntotal={idx.ntotal}\n")
    f.write(f"dimension={idx.d}\n")
    for i in range(min(10, idx.ntotal)):
        f.write(f"vec{i}={vecs[i][:5].tolist()}\n")
print(f"\nReference written to: {output}")
