import Foundation
import MLX
import ZIPFoundation

public enum PthConversionError: Error {
    case zipFailed
    case pickleFailed
    case invalidStateDict
    case storageMissing(String)
    case conversionFailed(String)
}

// @MainActor
public final class PthConverter: Sendable {
    public static let shared = PthConverter()
    
    /// Converts a .pth file (PyTorch zip archive) to an MLX-compatible Dictionary of arrays.
    /// - Parameter url: URL to the .pth file
    /// - Parameter progress: Closure to report progress (0.0 to 1.0) and status message.
    /// - Returns: A dictionary of [String: MLXArray] ready to be saved or used.
    public func convert(url: URL, progress: (@Sendable (Double, String) -> Void)? = nil) throws -> [String: MLXArray] {
        // 1. Unzip to temp
        progress?(0.05, "Extracting archive...")
        let fm = FileManager.default
        let tempDir = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try fm.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempDir) }
        
        try fm.unzipItem(at: url, to: tempDir)
        
        // 2. Inspect Archive & Locate data.pkl
        progress?(0.1, "Inspecting archive structure...")
        
        var dataPklUrl: URL?
        var storageRoot: URL = tempDir
        var modelFiles: [URL] = []
        
        // Log all files for transparency
        if let enumerator = fm.enumerator(at: tempDir, includingPropertiesForKeys: [.isRegularFileKey]) {
            print("--- Archive Structure ---")
            for case let fileURL as URL in enumerator {
                let relPath = fileURL.path.replacingOccurrences(of: tempDir.path, with: "")
                print("  [File] \(relPath)")
                
                if fileURL.lastPathComponent == "data.pkl" {
                    modelFiles.append(fileURL)
                } else if fileURL.pathExtension.lowercased() == "pth" {
                    modelFiles.append(fileURL)
                }
            }
            print("-------------------------")
        }
        
        // Helper to find data.pkl in a model URL (might be a directory or a .pth zip)
        func findDataPkl(in root: URL) -> (URL, URL)? {
            if root.lastPathComponent == "data.pkl" {
                return (root, root.deletingLastPathComponent())
            }
            
            // If it's a .pth file, it's actually a zip. Extract it!
            if root.pathExtension.lowercased() == "pth" {
                let nestedTemp = root.deletingLastPathComponent().appendingPathComponent("extracted_\(root.lastPathComponent)")
                do {
                    try fm.createDirectory(at: nestedTemp, withIntermediateDirectories: true)
                    try fm.unzipItem(at: root, to: nestedTemp)
                    
                    if let nestedEnum = fm.enumerator(at: nestedTemp, includingPropertiesForKeys: nil) {
                        for case let f as URL in nestedEnum {
                            if f.lastPathComponent == "data.pkl" {
                                return (f, f.deletingLastPathComponent())
                            }
                        }
                    }
                } catch {
                    print("Failed to extract nested .pth: \(error)")
                }
            }
            return nil
        }
        
        // Pick the best candidate
        if let first = modelFiles.first {
            if modelFiles.count > 1 {
                print("Warning: Multiple model candidates found. Picking: \(first.lastPathComponent)")
            }
            
            if let found = findDataPkl(in: first) {
                dataPklUrl = found.0
                storageRoot = found.1
                progress?(0.15, "Using model: \(first.lastPathComponent)")
            }
        }
        
        guard let pklUrl = dataPklUrl else {
            throw PthConversionError.conversionFailed("No valid model (.pth or data.pkl) found in archive.")
        }
        
        // 3. Unpickle
        progress?(0.2, "Parsing Pickle structure...")
        let data = try Data(contentsOf: pklUrl)
        let unpickler = PickleUnpickler(data: data)
        let unpickledObj = try unpickler.load()
        
        // 4. Extract State Dict and Storage
        // The unpickled object should constitute the state_dict (or a dict containing it)
        // With TensorReferences pointing to storage keys.
        
        var stateDict: [String: Any] = [:]
        print("PthConverter: Unpickled Type: \(type(of: unpickledObj))")
        if let dict = unpickledObj as? [String: Any] {
            print("PthConverter: Cast to [String: Any] succeeded, count: \(dict.count)")
            print("Keys: \(dict.keys.map { $0 })")
            stateDict = dict
        } else if let nsDict = unpickledObj as? NSDictionary {
             print("PthConverter: Cast to NSDictionary succeeded, count: \(nsDict.count)")
             for (k, v) in nsDict {
                 if let keyStr = k as? String {
                     stateDict[keyStr] = v
                 }
             }
             print("Keys: \(stateDict.keys.map { $0 })")
        } else {
             // Sometimes it's a model object with __getstate__, handled by BUILD
             print("PthConverter: Unpickled object is \(type(of: unpickledObj))")
             throw PthConversionError.invalidStateDict
        }
        
        // Check for nested state dict (checkpoint)
        if let nested = stateDict["weight"] as? [String: Any] {
            print("Found 'weight' key, using inner dict.")
            stateDict = nested
        } else if let nested = stateDict["weight"] as? NSDictionary {
             print("Found 'weight' key (NSDictionary), using inner dict.")
             var newDict: [String: Any] = [:]
             for (k, v) in nested {
                 if let ks = k as? String { newDict[ks] = v }
             }
             stateDict = newDict
        } else if let model = stateDict["model"] as? [String: Any] {
            print("Found 'model' key, using inner dict.")
            stateDict = model
        } else if let sd = stateDict["state_dict"] as? [String: Any] {
            print("Found 'state_dict' key, using inner dict.")
            stateDict = sd
        }
        
        // 5. Reconstruct Tensors
        // 5. Reconstruct Tensors
        let storageBase = storageRoot // Directory containing 'data.pkl' (or 'data' parent)
                                                             // Usually storage files are in a 'data' subdir relative to this
        
        var mlxDict: [String: MLXArray] = [:]
        
        // Count tensors for progress
        let totalItems = stateDict.count
        var currentItem = 0
        
        for (key, value) in stateDict {
            if let tensorRef = value as? TensorReference {
                // print("PTH Layer: \(key), Size: \(tensorRef.size), Storage: \(tensorRef.storage.dtype)")
                let array = try loadTensor(ref: tensorRef, baseDir: storageBase)
                mlxDict[key] = array
            }
            
            // Update progress every few items to avoid overhead
            currentItem += 1
            if currentItem % 5 == 0 {
                let p = 0.2 + (0.7 * Double(currentItem) / Double(totalItems)) // Scale 20% -> 90%
                progress?(p, "Loading tensors (\(Int(p * 100))%)...")
            }
        }
        
        // 6. Apply Model Conversion Logic (Weight Norm & Transpose)
        progress?(0.9, "Optimizing weights...")
        mlxDict = try processWeights(mlxDict)
        
        progress?(1.0, "Conversion complete!")
        return mlxDict
    }
    
    private func loadTensor(ref: TensorReference, baseDir: URL) throws -> MLXArray {
        // Filename is usually just the key "0", "1", etc.
        // It resides in 'data/' subdirectory relative to data.pkl usually.
        // But the pickle PersistentID path might vary.
        // Usually PyTorch saves as 'data/0', 'data/1'...
        
        // Use ref.storage.filename ("0", "1"...)
        let filename = ref.storage.filename
        let storageUrl = baseDir.appendingPathComponent("data").appendingPathComponent(filename)
        
        guard FileManager.default.fileExists(atPath: storageUrl.path) else {
            throw PthConversionError.storageMissing(filename)
        }
        
        let rawData = try Data(contentsOf: storageUrl)
        
        // Convert Data to MLXArray
        // Assume Float32 mostly for standard models. Dtype logic needed for 16-bit.
        
        if ref.storage.dtype.contains("FloatStorage") {
            let floats = rawData.withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            // Create array
            var array = MLXArray(floats)
            
            // Reshape
            // If strides are standard (contiguous), we can just reshape.
            // PyTorch default is row-major contiguous.
            if !ref.size.isEmpty {
                 array = array.reshaped(ref.size)
            }
            return array
        } else if ref.storage.dtype.contains("HalfStorage") {
             let halves = rawData.withUnsafeBytes {
                 Array($0.bindMemory(to: Float16.self))
             }
             var array = MLXArray(halves)
             if !ref.size.isEmpty { array = array.reshaped(ref.size) }
             return array
        } else if ref.storage.dtype.contains("LongStorage") {
             let ints = rawData.withUnsafeBytes {
                 Array($0.bindMemory(to: Int64.self))
             }
             var array = MLXArray(ints)
             if !ref.size.isEmpty { array = array.reshaped(ref.size) }
             return array
        }
        
        return MLXArray(0) // Fallback
    }
    
    private func processWeights(_ dict: [String: MLXArray]) throws -> [String: MLXArray] {
        var newDict: [String: MLXArray] = [:]
        let keys = Array(dict.keys)
        var processedPrefixes = Set<String>()
        
        // Weight Norm Fusion
        for k in keys {
            if k.hasSuffix(".weight_g") {
                let prefix = String(k.dropLast(9)) // remove .weight_g
                if processedPrefixes.contains(prefix) { continue }
                
                guard let w_g = dict[k],
                      let w_v = dict[prefix + ".weight_v"] else {
                    continue
                }
                
                // Norm calculation
                // PyTorch: norm_v = np.linalg.norm(w_v, axis=(1, 2) if w_v.ndim == 3 else 1)
                // MLX: example w_v shape [Out, In, Kernel] (dim 3) or [Out, In] (dim 2)
                
                // Norm calculation
                // PyTorch: norm_v = np.linalg.norm(w_v, axis=(1, 2) if w_v.ndim == 3 else 1)
                
                // Norm calculation
                // PyTorch: norm_v = np.linalg.norm(w_v, axis=(1, 2) if w_v.ndim == 3 else 1)
                
                var norm_v: MLXArray
                if w_v.ndim == 3 {
                    // Manual L2 norm: sqrt(sum(x^2))
                    // MLX Swift uses 'axes' for multiple axes
                    norm_v = MLX.sqrt(MLX.sum(w_v * w_v, axes: [1, 2], keepDims: true))
                } else {
                    norm_v = MLX.sqrt(MLX.sum(w_v * w_v, axes: [1], keepDims: true))
                }
                
                let w = w_g * (w_v / norm_v)
                newDict[prefix + ".weight"] = w
                processedPrefixes.insert(prefix)
                
            } else if k.hasSuffix(".weight_v") {
                // Skip, handled above
            } else {
                newDict[k] = dict[k]
            }
        }
        
        // Transposition for MLX Conv1d/Linear
        // MLX Conv1d expects [N, L, C] input? No, weights.
        // MLX Conv1d weights: [Out, Kernel, In] ?
        // PyTorch Conv1d weights: [Out, In, Kernel]
        // MLX Conv1d weights: [Out, Kernel, In]  <-- Wait, checking reference.
        // The python script says:
        // if "ups" in k: v = v.transpose(1, 2, 0)
        // else: v = v.transpose(0, 2, 1)
        
        // Let's copy the logic exactly.
        
        var finalDict: [String: MLXArray] = [:]
        for (k, v) in newDict {
            var val = v
            if k.contains("emb") && k.contains("weight") {
                // Pass
            } else if k.contains("weight") && val.ndim == 3 {
                if k.contains("ups") {
                    val = val.transposed(axes: [1, 2, 0])
                } else {
                    val = val.transposed(axes: [0, 2, 1])
                }
            } else if k.contains("weight") && val.ndim == 2 && k.lowercased().contains("linear") {
                 val = val.transposed() // (Out, In) -> (In, Out)
            }
            finalDict[k] = val
        }
        
        return finalDict
    }
}
