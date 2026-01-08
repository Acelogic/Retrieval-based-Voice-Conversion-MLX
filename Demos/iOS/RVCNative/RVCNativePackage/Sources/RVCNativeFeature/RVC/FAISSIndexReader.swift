import Foundation
import MLX

/// Errors that can occur when reading FAISS index files.
public enum FAISSReaderError: LocalizedError {
    case invalidMagic(String)
    case unsupportedIndexType(String)
    case invalidFormat(String)
    case dimensionMismatch(expected: Int, found: Int)
    case markerNotFound(String)
    
    public var errorDescription: String? {
        switch self {
        case .invalidMagic(let magic):
            return "Invalid FAISS magic: \(magic). Expected 'IwFl' for IVFFlat."
        case .unsupportedIndexType(let type):
            return "Unsupported index type: \(type). Only IVFFlat is supported."
        case .invalidFormat(let detail):
            return "Invalid FAISS format: \(detail)"
        case .dimensionMismatch(let expected, let found):
            return "Dimension mismatch: expected \(expected), found \(found)"
        case .markerNotFound(let marker):
            return "Marker '\(marker)' not found in index file"
        }
    }
}

/// Parser for FAISS IndexIVFFlat binary files.
///
/// Binary format (reverse-engineered from RVC indices):
/// ```
/// [IVF Header]
/// 0x00: "IwFl" (4 bytes) - IVFFlat magic
/// 0x04: dimension (uint32)
/// 0x08: ntotal (uint32)
/// ...
/// [Quantizer: IndexFlat with centroids]
/// "IxF2" magic, then nlist × dimension × float32 centroids
/// ...
/// [ArrayInvertedLists]
/// "ilar" magic (at offset ~0x9d86b for typical RVC index)
/// nlist (uint64)
/// list_sizes[nlist] (uint64 each)
/// For each list i:
///   ids[list_sizes[i]] (int64 each)
///   codes[list_sizes[i] × dimension] (float32 each)
/// ```
public struct FAISSIndexReader {
    
    /// Read a FAISS .index file and extract all vectors.
    public static func read(url: URL, logger: ((String) -> Void)? = nil) throws -> (vectors: MLXArray, dimension: Int) {
        let data = try Data(contentsOf: url)
        return try parse(data: data, logger: logger)
    }
    
    private static func parse(data: Data, logger: ((String) -> Void)?) throws -> (vectors: MLXArray, dimension: Int) {
        func log(_ msg: String) {
            print(msg)
            logger?(msg)
        }

        guard data.count > 0x60 else {
            throw FAISSReaderError.invalidFormat("File too small")
        }
        
        // 1. Verify magic
        let magic = String(data: data[0..<4], encoding: .ascii) ?? ""
        guard magic == "IwFl" else {
            throw FAISSReaderError.invalidMagic(magic)
        }
        
        // 2. Parse header
        let dimension = Int(readUInt32(data: data, offset: 0x04))
        let ntotal = Int(readUInt32(data: data, offset: 0x08))
        log("FAISSIndexReader: Header parsed - d=\(dimension), ntotal=\(ntotal)")
        
        // 3. Find "ilar" marker for ArrayInvertedLists
        guard let ilarOffset = findMarker(data: data, marker: "ilar") else {
            throw FAISSReaderError.markerNotFound("ilar")
        }
        log("FAISSIndexReader: Found 'ilar' marker at 0x\(String(ilarOffset, radix: 16))")
        
        // 4. Read nlist and list sizes
        // nlist is 24 bytes after "ilar"
        let nlistOffset = ilarOffset + 24
        let nlist = Int(readUInt64(data: data, offset: nlistOffset))
        log("FAISSIndexReader: nlist=\(nlist)")
        
        // List sizes start after nlist
        let listSizesOffset = nlistOffset + 8
        var listSizes: [Int] = []
        for i in 0..<nlist {
            let size = Int(readUInt64(data: data, offset: listSizesOffset + i * 8))
            listSizes.append(size)
        }
        
        let totalVectors = listSizes.reduce(0, +)
        log("FAISSIndexReader: Total vectors from list sizes = \(totalVectors)")
        
        // 5. Parse inverted lists to extract vectors
        let dataOffset = listSizesOffset + nlist * 8
        let vectors = try parseListData(
            data: data,
            startOffset: dataOffset,
            listSizes: listSizes,
            dimension: dimension,
            totalVectors: totalVectors,
            logger: logger
        )
        
        return (vectors, dimension)
    }
    
    /// Parse the actual vector data from inverted lists.
    private static func parseListData(
        data: Data,
        startOffset: Int,
        listSizes: [Int],
        dimension: Int,
        totalVectors: Int,
        logger: ((String) -> Void)?
    ) throws -> MLXArray {
        var allVectors: [Float] = []
        allVectors.reserveCapacity(totalVectors * dimension)
        
        var pos = startOffset
        
        for (listIdx, listSize) in listSizes.enumerated() {
            guard listSize >= 0 else {
                throw FAISSReaderError.invalidFormat("Negative list size at list \(listIdx)")
            }
            
            // Skip IDs (listSize × int64 = listSize × 8 bytes)
            let idsSize = listSize * 8
            pos += idsSize
            
            // Read vectors (listSize × dimension × float32)
            let vectorsBytes = listSize * dimension * 4
            guard pos + vectorsBytes <= data.count else {
                throw FAISSReaderError.invalidFormat("Unexpected EOF at list \(listIdx)")
            }
            
            // Extract float32 values
            for _ in 0..<listSize {
                for _ in 0..<dimension {
                    let value = readFloat32(data: data, offset: pos)
                    allVectors.append(value)
                    pos += 4
                }
            }
        }
        
        let extractedCount = allVectors.count / dimension
        logger?("FAISSIndexReader: Successfully extracted \(extractedCount) vectors")
        print("FAISSIndexReader: extracted \(extractedCount) vectors")
        
        let mlxArray = MLXArray(allVectors)
        return mlxArray.reshaped([extractedCount, dimension])
    }
    
    /// Find a 4-byte marker string in the data.
    private static func findMarker(data: Data, marker: String) -> Int? {
        guard let markerData = marker.data(using: .ascii) else { return nil }
        
        for i in 0..<(data.count - markerData.count) {
            if data[i..<i+markerData.count] == markerData {
                return i
            }
        }
        return nil
    }
    
    // MARK: - Binary reading helpers
    
    private static func readUInt32(data: Data, offset: Int) -> UInt32 {
        data.withUnsafeBytes { ptr in
            ptr.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        }
    }
    
    private static func readUInt64(data: Data, offset: Int) -> UInt64 {
        data.withUnsafeBytes { ptr in
            ptr.loadUnaligned(fromByteOffset: offset, as: UInt64.self)
        }
    }
    
    private static func readFloat32(data: Data, offset: Int) -> Float {
        data.withUnsafeBytes { ptr in
            ptr.loadUnaligned(fromByteOffset: offset, as: Float.self)
        }
    }
}
