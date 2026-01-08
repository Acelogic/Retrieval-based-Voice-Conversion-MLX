import Foundation
import RVCNativeFeature
import MLX

print("RVCNativeMac CLI")
print("================")

@MainActor
func run() async {
    print("[DEBUG] main.swift: Starting run()")
    let args = ProcessInfo.processInfo.arguments
    print("[DEBUG] main.swift: Got \(args.count) arguments")

    // Simple Argument Parsing
    // Usage: RVCNativeMac --audio <path> --model <path> --output <path> [--hubert <path>] [--ref <path>] [--benchmark]
    // Or: RVCNativeMac --convert <input.pth> --output <output.safetensors>

    func getArg(_ key: String) -> String? {
        if let idx = args.firstIndex(of: key), idx + 1 < args.count {
            return args[idx + 1]
        }
        return nil
    }

    // Check for conversion mode
    if let convertPath = getArg("--convert") {
        guard let outputPath = getArg("--output") else {
            print("Usage: RVCNativeMac --convert <input.pth> --output <output.safetensors>")
            return
        }

        print("Converting \(convertPath) to \(outputPath)...")
        let inputURL = URL(fileURLWithPath: convertPath)
        let outputURL = URL(fileURLWithPath: outputPath)

        do {
            let weights = try PthConverter.shared.convert(url: inputURL) { progress, status in
                print("[\(String(format: "%.0f", progress * 100))%] \(status)")
            }
            print("Conversion complete. Saving \(weights.count) tensors...")
            try MLX.save(arrays: weights, url: outputURL)
            print("Saved to \(outputURL.path)")
        } catch {
            print("Conversion error: \(error)")
        }
        return
    }

    guard let audioPath = getArg("--audio"),
          let modelPath = getArg("--model"),
          let outputPath = getArg("--output") else {
        print("Usage: RVCNativeMac --audio <in.wav> --model <model.safetensors> --output <out.wav> [--hubert <hubert.safetensors>] [--ref <reference.wav>] [--benchmark] [--volume 1.0]")
        print("Or:    RVCNativeMac --convert <input.pth> --output <output.safetensors>")
        return
    }
    
    let audioURL = URL(fileURLWithPath: audioPath)
    let modelURL = URL(fileURLWithPath: modelPath)
    let outputURL = URL(fileURLWithPath: outputPath)
    
    // Optional args
    let hubertPath = getArg("--hubert") ?? "hubert_base.safetensors"
    let hubertURL = URL(fileURLWithPath: hubertPath)
    
    let rmvpePath = getArg("--rmvpe")
    let rmvpeURL = rmvpePath != nil ? URL(fileURLWithPath: rmvpePath!) : nil
    
    let refPath = getArg("--ref")
    let refURL = refPath != nil ? URL(fileURLWithPath: refPath!) : nil
    
    let isBenchmark = args.contains("--benchmark")
    let useCPU = args.contains("--cpu")
    let volume = Float(getArg("--volume") ?? "1.0") ?? 1.0
    
    let device = useCPU ? Device.cpu : Device.gpu
    print("Using device: \(device)")
    
    print("[DEBUG] main.swift: Setting up device: \(device)")
    try await Device.withDefaultDevice(device) {
        print("[DEBUG] main.swift: Creating RVCInference()")
        let rvc = RVCInference()
        print("[DEBUG] main.swift: RVCInference created")

        // Set up logging
        rvc.onLog = { msg in
            print("[RVC] \(msg)")
        }

        print("Loading models...")
        print("[DEBUG] main.swift: About to call loadWeights")
        print("[DEBUG] modelURL: \(modelURL.path)")
        do {
            // Try simple load or loadWeights
            // Assuming models are in known locations or passed explicitly
            // We'll use loadWeights
            try await rvc.loadWeights(hubertURL: hubertURL, modelURL: modelURL, rmvpeURL: rmvpeURL)
            print("[DEBUG] main.swift: loadWeights completed")
            
            print("Running inference...")
            if isBenchmark {
                print("Mode: BENCHMARK")
                let result = try await rvc.runBenchmark(audioURL: audioURL, referenceURL: refURL, outputURL: outputURL)
                print(result)
            } else {
                print("Mode: INFERENCE")
                await rvc.infer(audioURL: audioURL, outputURL: outputURL)
                print("Done! Saved to \(outputURL.path)")
            }
        } catch {
            print("Error: \(error)")
        }
    }
}

// Keep the runloop alive for async work
let semanticGroup = DispatchGroup()
semanticGroup.enter()

Task { @MainActor in
    await run()
    semanticGroup.leave()
    exit(0)
}

dispatchMain()
