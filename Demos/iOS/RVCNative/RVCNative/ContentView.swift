import SwiftUI
import MLX
import AVFoundation
import RVCNativeFeature
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var selectedModel: String = "Select Model"
    @State private var statusMessage: String = "Ready"
    @State private var isProcessing: Bool = false
    @State private var isImporting: Bool = false
    @State private var logs: [String] = []
    
    @State private var isModelLoaded: Bool = false
    
    @StateObject private var inferenceEngine = RVCInference()
    @StateObject private var audioRecorder = AudioRecorder()
    @StateObject private var inputPlayer = AudioPlayer()
    @StateObject private var outputPlayer = AudioPlayer()
    
    @State private var inputURL: URL?
    @State private var outputURL: URL?
    
    // Quick Demo File
    let demoAudioURL = RVCInference.bundle.url(forResource: "demo", withExtension: "wav")

    var body: some View {
        VStack(spacing: 20) {
            Text("Native RVC Demo")
                .font(.largeTitle)
                .bold()
            
            Text("Powered by MLX Swift")
                .font(.subheadline)
                .foregroundStyle(.gray)
            
            Divider()
            
            // Model Selection
            HStack {
                Text("Model:")
                Spacer()
                Menu {
                    Button("Coder", action: { loadModel(name: "Coder") })
                    Button("Slim Shady", action: { loadModel(name: "Slim Shady") })
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
            }
            
            Spacer()
            
            // Status Log
            ScrollView {
                Text(inferenceEngine.status)
                    .font(.caption)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .frame(height: 100)
            .background(Color.black.opacity(0.8))
            .foregroundColor(.green)
            .cornerRadius(8)
        }
        .padding()
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
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            do {
                guard let selectedFile: URL = try result.get().first else { return }
                if selectedFile.startAccessingSecurityScopedResource() {
                     defer { selectedFile.stopAccessingSecurityScopedResource() }
                     
                     // Copy to temp to ensure access
                     let temp = FileManager.default.temporaryDirectory.appendingPathComponent(selectedFile.lastPathComponent)
                     try? FileManager.default.removeItem(at: temp)
                     try FileManager.default.copyItem(at: selectedFile, to: temp)
                     
                     self.inputURL = temp
                     statusMessage = "Selected: \(selectedFile.lastPathComponent)"
                } else {
                     statusMessage = "Access denied"
                }
            } catch {
                statusMessage = "Error: \(error.localizedDescription)"
            }
        }
        .onAppear {
            inferenceEngine.onLog = { msg in
                self.log(msg)
            }
            loadModel(name: "Coder")
        }
    }
    
    func log(_ message: String) {
        print("DEBUG: \(message)") // Keep in console
        DispatchQueue.main.async {
            self.logs.append(message)
        }
    }
    
    func loadModel(name: String) {
        log("loadModel called for \(name)")
        selectedModel = name
        isModelLoaded = false // Reset state
        
        // map name to file
        let filename = name.lowercased().replacingOccurrences(of: " ", with: "_")
        
        // Try finding in root or Assets subdir
        let modelUrl = RVCInference.bundle.url(forResource: filename, withExtension: "safetensors") 
            ?? RVCInference.bundle.url(forResource: filename, withExtension: "safetensors", subdirectory: "Assets")
            
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
        
        // Temp output
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("output.wav")
        self.outputURL = output
        
        await inferenceEngine.infer(audioURL: input, outputURL: output)
        log("Inference complete.")
    }
}

#Preview {
    ContentView()
}
