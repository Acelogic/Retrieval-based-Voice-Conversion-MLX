#!/usr/bin/env swift

/// Swift validation script to compare iOS MLX implementation with Python MLX
///
/// This script loads the test data exported by export_ios_test_data.py and
/// runs the same inference using the iOS Swift/MLX implementation to verify
/// numerical parity.
///
/// Usage:
///     swift tools/validate_ios_parity.swift ios_test_data/
///
/// The script will:
/// 1. Load Python-generated outputs (HuBERT features, RMVPE F0)
/// 2. Run Swift/MLX inference on the same input
/// 3. Compute correlation and error metrics
/// 4. Report parity status

import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Numpy File Loader

func loadNumpyArray(path: String) throws -> (MLXArray, [Int], String) {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))

    // Parse numpy .npy format
    // Magic: \x93NUMPY
    guard data.count > 10 else {
        throw NSError(domain: "NumpyLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "File too small"])
    }

    // Read header length (bytes 8-10, little-endian uint16)
    let headerLen = Int(data[8]) + Int(data[9]) * 256
    let headerStart = 10
    let headerEnd = headerStart + headerLen

    guard data.count > headerEnd else {
        throw NSError(domain: "NumpyLoader", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid header length"])
    }

    let headerData = data[headerStart..<headerEnd]
    let headerString = String(data: headerData, encoding: .utf8) ?? ""

    // Parse shape from header
    let shapePattern = "'shape': \\(([^)]+)\\)"
    let shapeRegex = try NSRegularExpression(pattern: shapePattern)
    let shapeMatches = shapeRegex.matches(in: headerString, range: NSRange(headerString.startIndex..., in: headerString))

    var shape: [Int] = []
    if let match = shapeMatches.first,
       let range = Range(match.range(at: 1), in: headerString) {
        let shapeStr = String(headerString[range])
        shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    // Parse dtype
    let dtypePattern = "'descr': '([^']+)'"
    let dtypeRegex = try NSRegularExpression(pattern: dtypePattern)
    let dtypeMatches = dtypeRegex.matches(in: headerString, range: NSRange(headerString.startIndex..., in: headerString))

    var dtype = "<f4" // default float32
    if let match = dtypeMatches.first,
       let range = Range(match.range(at: 1), in: headerString) {
        dtype = String(headerString[range])
    }

    // Load data
    let dataStart = headerEnd
    let dataBytes = data[dataStart...]

    let array: MLXArray
    let totalElements = shape.reduce(1, *)

    if dtype.contains("f8") {  // float64
        let floats = dataBytes.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Double.self))
        }
        array = MLXArray(floats.map { Float($0) }).reshaped(shape)
    } else if dtype.contains("f4") {  // float32
        let floats = dataBytes.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
        array = MLXArray(floats).reshaped(shape)
    } else {
        throw NSError(domain: "NumpyLoader", code: 3, userInfo: [NSLocalizedDescriptionKey: "Unsupported dtype: \(dtype)"])
    }

    return (array, shape, dtype)
}

// MARK: - Metrics

func computeCorrelation(_ a: MLXArray, _ b: MLXArray) -> Float {
    let aflat = a.reshaped([-1])
    let bflat = b.reshaped([-1])

    let amean = MLX.mean(aflat)
    let bmean = MLX.mean(bflat)

    let anorm = aflat - amean
    let bnorm = bflat - bmean

    let num = MLX.sum(anorm * bnorm)
    let denom = MLX.sqrt(MLX.sum(anorm * anorm) * MLX.sum(bnorm * bnorm))

    return (num / denom).item(Float.self)
}

func computeRMSE(_ a: MLXArray, _ b: MLXArray) -> Float {
    let diff = a - b
    let mse = MLX.mean(diff * diff)
    return MLX.sqrt(mse).item(Float.self)
}

func computeMAE(_ a: MLXArray, _ b: MLXArray) -> Float {
    let diff = MLX.abs(a - b)
    return MLX.mean(diff).item(Float.self)
}

// MARK: - Main Validation

print("=" * 60)
print("iOS MLX Parity Validation")
print("=" * 60)

guard CommandLine.arguments.count > 1 else {
    print("Usage: swift validate_ios_parity.swift <test_data_dir>")
    exit(1)
}

let testDataDir = CommandLine.arguments[1]

print("\nTest data directory: \(testDataDir)")

do {
    // Load Python outputs
    print("\n" + "=" * 60)
    print("Loading Python outputs...")
    print("=" * 60)

    let (inputAudio, _, _) = try loadNumpyArray(path: "\(testDataDir)/input_audio.npy")
    print("✓ Loaded input_audio: \(inputAudio.shape)")

    let (pythonHubert, _, _) = try loadNumpyArray(path: "\(testDataDir)/hubert_features.npy")
    print("✓ Loaded HuBERT features (Python): \(pythonHubert.shape)")

    let (pythonF0, _, _) = try loadNumpyArray(path: "\(testDataDir)/rmvpe_f0.npy")
    print("✓ Loaded RMVPE F0 (Python): \(pythonF0.shape)")

    // TODO: Load and run Swift models
    print("\n" + "=" * 60)
    print("Running Swift MLX inference...")
    print("=" * 60)
    print("Note: Swift model loading would go here")
    print("This requires loading the .safetensors weights")

    // For now, report what we loaded
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Successfully loaded Python test outputs!")
    print("Next step: Implement Swift model loading and inference")

} catch {
    print("Error: \(error)")
    exit(1)
}
