# Native Swift FAISS Parser

This project includes a custom binary parser for FAISS `IndexIVFFlat` files, allowing the iOS app to load RVC voice models' index files directly without requiring Python-based pre-conversion.

## Implementation Details

The `FAISSIndexReader.swift` implements a reader for the undocumented binary format of FAISS indices.

### Binary Structure (IVFFlat)

Through reverse engineering of the RVC index files, the following structure was identified:

1. **Header (Little Endian)**
   - `0x00`: "IwFl" magic string (IVFFlat)
   - `0x04`: Dimension (uint32) - typically 768 for RVC/HuBERT
   - `0x08`: Total vectors (uint32)
   
2. **Quantizer (Coarse)**
   - `0x35`: "IxF2" magic string (IndexFlat)
   - Followed by Centroids data (`nlist` × `d` × float32)
   
3. **Inverted Lists (ArrayInvertedLists)**
   - Marked by "ilar" magic string (found by scanning)
   - `+24 bytes`: `nlist` (uint64) - Number of clusters (typically 256 for RVC)
   - `+32 bytes`: `list_sizes` array (`nlist` × uint64)
   
4. **Vector Data**
   - For each list `i` in `0..<nlist`:
     - IDs (`list_sizes[i]` × int64)
     - Vectors (`list_sizes[i]` × `d` × float32)

### Usage in App

The `IndexManager` automatically detects the file type:

```swift
let manager = IndexManager()

// Method A: Legacy converted files
try manager.load(url: URL(fileURLWithPath: "index.safetensors"))

// Method B: Native FAISS files (New)
try manager.load(url: URL(fileURLWithPath: "added_IVF256_Flat_nprobe_1.index"))
```

### Limitations

- Only supports `IndexIVFFlat` type (standard for RVC).
- Does not currently support `IndexFlatL2` (though trivial to add).
- Does not use the coarse quantizer for acceleration (loading time is fast enough on iOS).
- Loads all vectors into memory (8k vectors @ 768 float32 is only ~24MB, so this is fine).

## Validation

Verified against Python FAISS implementation:
- Correctly extracts all vectors (exact count match).
- Metadata (dimension, nlist) matches exactly.
