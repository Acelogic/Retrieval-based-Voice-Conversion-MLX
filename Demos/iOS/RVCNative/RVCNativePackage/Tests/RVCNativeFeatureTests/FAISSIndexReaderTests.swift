import XCTest
import MLX
@testable import RVCNativeFeature

final class FAISSIndexReaderTests: XCTestCase {
    
    // Path to the real index file copied from Replay folder
    // Note: In a real CI environment this should be a bundled resource, 
    // but for this local dev session we use the tmp file we created
    let indexPath = URL(fileURLWithPath: "/tmp/test.index")
    
    func testReadRealIndex() throws {
        // Skip if file doesn't exist (e.g. on CI)
        guard FileManager.default.fileExists(atPath: indexPath.path) else {
            print("Skipping testReadRealIndex: /tmp/test.index not found")
            return
        }
        
        print("Testing FAISS reading from: \(indexPath.path)")
        
        let (vectors, dim) = try FAISSIndexReader.read(url: indexPath)
        
        // Validation against known truth (from python export)
        // Dimension: 768
        XCTAssertEqual(dim, 768, "Dimension should be 768")
        
        // Total vectors: 8191
        // Note: The parser logic for total vectors comes from summing list sizes in IVFFlat
        XCTAssertEqual(vectors.shape[0], 8191, "Should have 8191 vectors")
        XCTAssertEqual(vectors.shape[1], 768, "Vector dimension should matches returned dim")
        
        // Check first vector (ID 0)
        // Python reference (index 0): 
        // [0.02027893, 0.00044727, -0.3293457, 0.25805664, -0.1159668]
        
        // We need to be careful: FAISS IVFFlat doesn't guarantee order matches ID order unless
        // we sort by ID (which we didn't parsing, we just concatenated lists).
        // However, we just want to verify *some* data is correct. 
        // Let's verify that the vectors look essentially valid (not all zeros, reasonable range)
        
        let v0 = vectors[0]
        let v0Data = v0.asArray(Float.self)
        
        // Check if data looks like valid floats (not NaN, reasonable magnitude)
        let mean = MLX.mean(vectors).item(Float.self)
        print("Mean vector value: \(mean)")
        XCTAssertTrue(mean.isFinite)
        XCTAssertTrue(abs(mean) < 1.0) // Embeddings are usually normalized or small magnitude
        
        // Verify it matches Python reference if we can assume order implies ID
        // Note: In IVFFlat, vectors are stored in clusters. The order in the file 
        // is Cluster 0 vectors, Cluster 1 vectors, etc. 
        // It is NOT 0, 1, 2, 3... sorted by ID.
        // So vectors[0] in Swift corresponds to the first vector in the first populated cluster,
        // which might NOT be ID 0.
        
        // To verify correctness rigorously, we'd need to parse the IDs too.
        // But for this smoke test, checking dimensions and validity is a huge step.
        
        // Let's print the first few values to compare manually with python output if needed
        print("First vector (raw file order): \(v0Data.prefix(5))")
    }
    
    func testParseInvalidMagic() {
        // Create a dummy file with wrong magic (large enough to pass size check)
        let dummyPath = URL(fileURLWithPath: "/tmp/invalid_magic.index")
        var badData = "BadMagic".data(using: .ascii)!
        badData.append(Data(repeating: 0, count: 200)) // Padding to pass size check
        try? badData.write(to: dummyPath)
        
        XCTAssertThrowsError(try FAISSIndexReader.read(url: dummyPath)) { error in
            guard let faissError = error as? FAISSReaderError,
                  case .invalidMagic = faissError else {
                XCTFail("Should throw invalidMagic error")
                return
            }
        }
    }
}
