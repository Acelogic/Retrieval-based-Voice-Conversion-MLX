import SwiftUI
import MLX
import AVFoundation
import RVCNativeFeature
import UniformTypeIdentifiers

// MARK: - Waveform View Component
struct WaveformView: View {
    let samples: [Float]
    let color: Color
    let label: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)

            GeometryReader { geometry in
                let width = geometry.size.width
                let height = geometry.size.height
                let midY = height / 2

                Path { path in
                    guard !samples.isEmpty else { return }

                    // Downsample to fit the view width
                    let pointCount = Int(width)
                    let samplesPerPoint = max(1, samples.count / pointCount)

                    for i in 0..<pointCount {
                        let startIdx = i * samplesPerPoint
                        let endIdx = min(startIdx + samplesPerPoint, samples.count)

                        guard startIdx < samples.count else { break }

                        // Get min/max for this segment
                        let segment = samples[startIdx..<endIdx]
                        let minVal = segment.min() ?? 0
                        let maxVal = segment.max() ?? 0

                        let x = CGFloat(i)
                        let topY = midY - CGFloat(maxVal) * midY * 0.9
                        let bottomY = midY - CGFloat(minVal) * midY * 0.9

                        path.move(to: CGPoint(x: x, y: topY))
                        path.addLine(to: CGPoint(x: x, y: bottomY))
                    }
                }
                .stroke(color, lineWidth: 1)

                // Center line
                Path { path in
                    path.move(to: CGPoint(x: 0, y: midY))
                    path.addLine(to: CGPoint(x: width, y: midY))
                }
                .stroke(color.opacity(0.3), lineWidth: 0.5)
            }
        }
        .frame(height: 60)
    }
}

// MARK: - Waveform Comparison View
struct WaveformComparisonView: View {
    let originalSamples: [Float]
    let convertedSamples: [Float]

    var body: some View {
        VStack(spacing: 8) {
            Text("Waveform Comparison")
                .font(.headline)

            WaveformView(samples: originalSamples, color: .blue, label: "Original")
            WaveformView(samples: convertedSamples, color: .green, label: "Converted")
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Audio Waveform Extractor
struct AudioWaveformExtractor {
    static func extractSamples(from url: URL, maxSamples: Int = 4000) -> [Float] {
        guard let file = try? AVAudioFile(forReading: url) else {
            return []
        }

        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            return []
        }

        do {
            try file.read(into: buffer)
        } catch {
            return []
        }

        guard let channelData = buffer.floatChannelData else {
            return []
        }

        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))

        // Downsample if too many samples
        if samples.count <= maxSamples {
            return samples
        }

        let step = samples.count / maxSamples
        return stride(from: 0, to: samples.count, by: step).map { samples[$0] }
    }
}

struct ContentView: View {
    @State private var selectedModel: String = "Select Model"
    @State private var statusMessage: String = "Ready"
    @State private var isProcessing: Bool = false
    @State private var isImporting: Bool = false
    @State private var logs: [String] = []
    @State private var volumeEnvelope: Float = 1.0 // Default 1.0 (no change)
    
    @State private var isModelLoaded: Bool = false
    
    @StateObject private var inferenceEngine = RVCInference()
    @StateObject private var audioRecorder = AudioRecorder()
    @StateObject private var inputPlayer = AudioPlayer()
    @StateObject private var outputPlayer = AudioPlayer()
    
    @State private var inputURL: URL?
    @State private var outputURL: URL?

    // Waveform samples for visualization
    @State private var originalWaveform: [Float] = []
    @State private var convertedWaveform: [Float] = []

    // Quick Demo File
    let stockAudioURL = RVCInference.bundle.url(forResource: "coder_audio_stock", withExtension: "wav") ?? RVCInference.bundle.url(forResource: "demo", withExtension: "wav")
    
    // Alert State
    @State private var showAlert = false
    @State private var alertMessage = ""
    
    // Conversion Progress
    @State private var conversionProgress: Double = 0.0
    @State private var isConverting: Bool = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    Text("Powered by MLX Swift")
                        .font(.subheadline)
                        .foregroundStyle(.gray)
                    
                    Divider()
                    
                    // Model Selection
                    HStack {
                        Text("Model:")
                        Spacer()
                        Menu {
                            Section("Bundle Models") {
                                Button("Coder", action: { loadModel(name: "Coder") })
                                Button("Slim Shady", action: { loadModel(name: "Slim Shady") })
                            }
                            
                            Section("Imported Models") {
                                if getImportedModels().isEmpty {
                                    Text("No imported models")
                                } else {
                                    ForEach(getImportedModels(), id: \.self) { modelName in
                                        Button(modelName) {
                                            loadModel(name: modelName, isImported: true)
                                        }
                                    }
                                }
                            }
                            
                            Divider()
                            
                            Button(action: { isImporting = true }) {
                                Label("Import Model (.safetensors)", systemImage: "square.and.arrow.down")
                            }
                        } label: {
                            Text(selectedModel)
                                .padding(.horizontal)
                                .padding(.vertical, 8)
                                .background(Color.blue.opacity(0.1))
                                .cornerRadius(8)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Volume Envelope Slider
                    VStack(alignment: .leading, spacing: 5) {
                        HStack {
                            Text("Volume Envelope (Reduce Noise)")
                                .font(.caption)
                                .foregroundStyle(.gray)
                            Spacer()
                            Text(String(format: "%.2f", volumeEnvelope))
                                .font(.caption)
                                .monospacedDigit()
                        }
                        Slider(value: $volumeEnvelope, in: 0.0...1.0, step: 0.05)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Actions
                    Button(action: {
                        isImporting = true
                    }) {
                        HStack {
                            Image(systemName: "waveform.circle")
                            Text(inputURL == nil ? "Select Audio File" : "Selected: \(inputURL?.lastPathComponent ?? "File")")
                        }
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    
                    // Audio Loaded Indicator
                    if let input = inputURL {
                       HStack {
                           Image(systemName: "checkmark.circle.fill")
                               .foregroundColor(.green)
                           Text("Ready to convert: \(input.lastPathComponent)")
                               .font(.caption)
                               .foregroundColor(.secondary)
                       }
                    } else if let stock = stockAudioURL {
                        Text("Using stock audio: \(stock.lastPathComponent)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    
                    // Play Original
                     if let url = inputURL {
                         Button(action: {
                             if inputPlayer.isPlaying {
                                 inputPlayer.stop()
                             } else {
                                 inputPlayer.play(url: url)
                             }
                         }) {
                             HStack {
                                 Image(systemName: inputPlayer.isPlaying ? "stop.fill" : "play.fill")
                                 Text(inputPlayer.isPlaying ? "Stop Original" : "Play Original")
                             }
                             .font(.subheadline)
                             .foregroundColor(.blue)
                             .padding(.vertical, 4)
                         }
                     }
                    
                    // Audio Recording
                    if audioRecorder.isRecording {
                        Button(action: { audioRecorder.stopRecording() }) {
                            HStack {
                                Image(systemName: "stop.circle.fill")
                                Text("Stop Recording")
                            }
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.red.opacity(0.2))
                            .foregroundColor(.red)
                            .cornerRadius(12)
                        }
                    } else {
                        Button(action: { audioRecorder.startRecording() }) {
                            HStack {
                                Image(systemName: "mic.circle.fill")
                                Text("Record Audio")
                            }
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.red.opacity(0.1))
                            .foregroundColor(.red)
                            .cornerRadius(12)
                        }
                    }
                    
                    // Conversion Progress Bar
                    if isConverting {
                        VStack(spacing: 8) {
                            ProgressView(value: conversionProgress, total: 1.0)
                                .progressViewStyle(LinearProgressViewStyle())
                            Text("\(Int(conversionProgress * 100))% - \(statusMessage)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                    }
                    
                    Button(action: {
                        Task {
                           await startInference()
                        }
                    }) {
                        HStack {
                            if isProcessing {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            } else {
                                Image(systemName: "bolt.fill")
                            }
                            Text(isProcessing ? "Inferencing..." : "Convert Voice")
                        }
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isProcessing || !isModelLoaded ? Color.gray : Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    .disabled(isProcessing || inputURL == nil || !isModelLoaded)
                    
                    // Play Converted
                    if let url = outputURL, !isProcessing {
                        Divider()
                        Text("Result")
                            .font(.headline)
                        
                        Button(action: {
                            if outputPlayer.isPlaying {
                                outputPlayer.stop()
                            } else {
                                outputPlayer.play(url: url)
                            }
                        }) {
                            HStack {
                                Image(systemName: outputPlayer.isPlaying ? "stop.circle.fill" : "play.circle.fill")
                                Text(outputPlayer.isPlaying ? "Stop Converted" : "Play Converted")
                            }
                            .font(.title2)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green.opacity(0.2))
                            .foregroundColor(.green)
                            .cornerRadius(12)
                        }
                        
                        // Waveform comparison
                        if !originalWaveform.isEmpty && !convertedWaveform.isEmpty {
                            WaveformComparisonView(
                                originalSamples: originalWaveform,
                                convertedSamples: convertedWaveform
                            )
                        }
                    }
                    
                    Spacer()
                    
                    // Status Log
                    VStack(alignment: .leading) {
                         Text("Console Output")
                             .font(.caption)
                             .bold()
                             .foregroundColor(.secondary)
                         
                         ScrollView {
                            Text(inferenceEngine.status)
                                .font(.caption)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(8)
                         }
                         .frame(height: 100)
                         .background(Color.black.opacity(0.8))
                         .foregroundColor(.green)
                         .cornerRadius(8)
                    }
                }
                .padding()
            }
            .navigationTitle("Native RVC Demo")
        }
        .onChange(of: inferenceEngine.status) { oldValue, newValue in
             statusMessage = newValue 
             if newValue == "Done!" || newValue == "Models Loaded" || newValue.starts(with: "Error") {
                 isProcessing = false
             } else if newValue != "Idle" {
                 isProcessing = true
             }
        }
        .onChange(of: audioRecorder.recordingURL) { _, newValue in
             if let url = newValue {
                 self.inputURL = url
                 statusMessage = "Recorded: \(url.lastPathComponent)"
             }
        }
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [UTType(filenameExtension: "safetensors")!, UTType(filenameExtension: "npz")!, UTType(filenameExtension: "pth")!, UTType.zip],
            allowsMultipleSelection: false
        ) { result in
            do {
                guard let selectedFile: URL = try result.get().first else { return }
                
                // Check if .pth or .zip
                if selectedFile.pathExtension.lowercased() == "pth" || selectedFile.pathExtension.lowercased() == "zip" {
                    statusMessage = "Converting model archive (this may take a moment)..."
                    
                    // Access security scope
                    if selectedFile.startAccessingSecurityScopedResource() {
                        defer { selectedFile.stopAccessingSecurityScopedResource() }
                        
                        Task {
                            // Run on background thread (detached) to avoid blocking main actor
                            // PthConverter is now thread-safe / nonisolated
                            await MainActor.run { 
                                isConverting = true 
                                conversionProgress = 0.0
                            }
                            
                            do {
                                let arrays = try PthConverter.shared.convert(url: selectedFile) { progress, msg in
                                    Task { @MainActor in
                                        self.conversionProgress = progress
                                        self.statusMessage = msg
                                    }
                                }
                                
                                // Save as .safetensors
                                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                                let name = selectedFile.deletingPathExtension().lastPathComponent
                                let dest = docs.appendingPathComponent(name + ".safetensors")
                                
                                try MLX.save(arrays: arrays, url: dest)
                                
                                await MainActor.run {
                                    statusMessage = "Converted & Imported: \(name)"
                                    log("Converted model saved to \(dest.path)")
                                    isConverting = false
                                    conversionProgress = 1.0
                                }
                            } catch {
                                await MainActor.run {
                                    statusMessage = "Conversion Failed: \(error.localizedDescription)"
                                    log("Conversion error: \(error)")
                                    
                                    alertMessage = "Conversion Failed.\n\nError: \(error.localizedDescription)"
                                    showAlert = true
                                    isConverting = false
                                }
                            }
                        }
                    } else {
                        statusMessage = "Access denied"
                    }
                    return
                }
                
                if selectedFile.startAccessingSecurityScopedResource() {
                     defer { selectedFile.stopAccessingSecurityScopedResource() }
                     
                     // Destination: Documents Directory
                     let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                     let dest = docs.appendingPathComponent(selectedFile.lastPathComponent)
                     
                     // Remove if exists
                     if FileManager.default.fileExists(atPath: dest.path) {
                         try? FileManager.default.removeItem(at: dest)
                     }
                     
                     try FileManager.default.copyItem(at: selectedFile, to: dest)
                     
                     statusMessage = "Imported: \(selectedFile.lastPathComponent)"
                     log("Imported model to \(dest.path)")
                } else {
                     statusMessage = "Access denied"
                }
            } catch {
                statusMessage = "Error: \(error.localizedDescription)"
                log("Import error: \(error)")
            }
        }
        .alert("Import Failed", isPresented: $showAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(alertMessage)
        }
        .onAppear {
            inferenceEngine.onLog = { msg in
                self.log(msg)
            }
            // Default load
            loadModel(name: "Slim Shady")
            
            // Pre-load audio
            if inputURL == nil {
                if let stock = stockAudioURL {
                    self.inputURL = stock
                    statusMessage = "Loaded stock audio: \(stock.lastPathComponent)"
                    
                    // Extract waveform immediately
                    originalWaveform = AudioWaveformExtractor.extractSamples(from: stock)
                }
            }
        }
    }
    
    func log(_ message: String) {
        print("DEBUG: \(message)") // Keep in console
        DispatchQueue.main.async {
            self.logs.append(message)
        }
    }
    
    func getImportedModels() -> [String] {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        guard let files = try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil) else { return [] }
        return files
            .filter { $0.pathExtension == "safetensors" || $0.pathExtension == "npz" }
            .map { $0.deletingPathExtension().lastPathComponent }
    }
    
    func loadModel(name: String, isImported: Bool = false) {
        log("loadModel called for \(name)")
        selectedModel = name
        isModelLoaded = false // Reset state
        
        // map name to file
        let filename = name.lowercased().replacingOccurrences(of: " ", with: "_")
        
    
        var modelUrl: URL?
        
        if isImported {
             let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
             // Try safe then npz
             let safe = docs.appendingPathComponent("\(name).safetensors")
             let npz = docs.appendingPathComponent("\(name).npz")
             if FileManager.default.fileExists(atPath: safe.path) { modelUrl = safe }
             else if FileManager.default.fileExists(atPath: npz.path) { modelUrl = npz }
        } else {
            // Try finding in root or Assets subdir
            modelUrl = RVCInference.bundle.url(forResource: filename, withExtension: "safetensors") 
                ?? RVCInference.bundle.url(forResource: filename, withExtension: "safetensors", subdirectory: "Assets")
        }
            
        guard let url = modelUrl else {
             log("Failed to find model file: \(filename).safetensors")
             statusMessage = "Model \(name) not found in bundle"
             return
        }
        log("Found model at \(url.path)")
        
        let hubertUrl = RVCInference.bundle.url(forResource: "hubert_base", withExtension: "safetensors")
            ?? RVCInference.bundle.url(forResource: "hubert_base", withExtension: "safetensors", subdirectory: "Assets")
            
        guard let hubertURL = hubertUrl else {
             log("Failed to find hubert_base.safetensors")
             statusMessage = "Hubert model not found!"
             return
        }
        log("Found hubert at \(hubertURL.path)")
        
        // Optional RMVPE
        let rmvpeURL = RVCInference.bundle.url(forResource: "rmvpe", withExtension: "safetensors")
            ?? RVCInference.bundle.url(forResource: "rmvpe", withExtension: "safetensors", subdirectory: "Assets")
            ?? RVCInference.bundle.url(forResource: "rmvpe", withExtension: "npz")
            ?? RVCInference.bundle.url(forResource: "rmvpe", withExtension: "npz", subdirectory: "Assets")
            ?? RVCInference.bundle.url(forResource: "rmvpe_mlx", withExtension: "npz")
            ?? RVCInference.bundle.url(forResource: "rmvpe_mlx", withExtension: "npz", subdirectory: "Assets")
        
        if let r = rmvpeURL {
            log("Found rmvpe at \(r.path)")
        } else {
            log("RMVPE not found (optional)")
        }
        
        Task {
            do {
                log("Starting loadWeights task...")
                try await inferenceEngine.loadWeights(hubertURL: hubertURL, modelURL: url, rmvpeURL: rmvpeURL)
                log("loadWeights success")
                statusMessage = "Loaded \(name)"
                isModelLoaded = true
            } catch {
                log("loadWeights failed: \(error)")
                statusMessage = "Failed to load \(name): \(error.localizedDescription)"
                isModelLoaded = false
            }
        }
    }
    
    func startInference() async {
        guard let input = inputURL else { return }

        if selectedModel == "Select Model" {
            statusMessage = "Please select a model first."
            log("Attempted inference without selecting model")
            return
        }

        log("Starting inference processing...")
        isProcessing = true

        // Extract original waveform before conversion
        originalWaveform = AudioWaveformExtractor.extractSamples(from: input)
        log("Extracted \(originalWaveform.count) samples from original audio")

        // Temp output
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("output.wav")
        self.outputURL = output

        await inferenceEngine.infer(
            audioURL: input, 
            outputURL: output, 
            volumeEnvelope: volumeEnvelope
        )
        log("Inference complete.")

        // Extract converted waveform after conversion
        convertedWaveform = AudioWaveformExtractor.extractSamples(from: output)
        log("Extracted \(convertedWaveform.count) samples from converted audio")
    }
}

#Preview {
    ContentView()
}
