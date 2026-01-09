import SwiftUI
import MLX
import AVFoundation
import RVCNativeFeature
import UniformTypeIdentifiers
import UIKit

// MARK: - ContentView

struct ContentView: View {
    // MARK: - State Variables

    @State private var selectedModel: String = "Select Model"
    @State private var statusMessage: String = "Ready"
    @State private var isProcessing: Bool = false

    enum ImportType: Identifiable {
        case model
        case audio

        var id: Self { self }
    }
    @State private var activeImport: ImportType? = nil
    @State private var logs: [String] = []
    @State private var volumeEnvelope: Float = 1.0

    @State private var isModelLoaded: Bool = false

    @StateObject private var inferenceEngine = RVCInference()
    @StateObject private var audioRecorder = AudioRecorder()
    @StateObject private var inputPlayer = AudioPlayer()
    @StateObject private var outputPlayer = AudioPlayer()
    @StateObject private var logRedirector = ConsoleLogRedirector.shared

    @State private var inputURL: URL?
    @State private var outputURL: URL?

    // Waveform samples for visualization
    @State private var originalWaveform: [Float] = []
    @State private var convertedWaveform: [Float] = []

    // Result History
    @State private var conversionHistory: [ConversionResult] = []
    @State private var selectedResult: ConversionResult?

    // Model List State
    @State private var importedModels: [String] = []

    // Advanced Lab Controls
    @State private var selectedF0Method: String = "rmvpe"
    @State private var pitchShift: Double = 0.0
    @State private var featureRatio: Double = 0.75
    @State private var showAdvancedSettings: Bool = false
    @State private var recordingTime: TimeInterval = 0.0
    @State private var recordingTimer: Timer?
    @State private var justFinishedRecording: Bool = false

    // Quick Demo File
    let stockAudioURL = RVCInference.bundle.url(forResource: "coder_audio_stock", withExtension: "wav") ?? RVCInference.bundle.url(forResource: "demo", withExtension: "wav")

    // Alert State
    @State private var showAlert = false
    @State private var showClearAllAlert = false
    @State private var alertMessage = ""
    @State private var showDeleteDialog = false
    @State private var modelToDelete: String? = nil

    // Conversion Progress
    @State private var conversionProgress: Double = 0.0
    @State private var isConverting: Bool = false
    @State private var isManagingModels: Bool = false
    @State private var isEditMode: Bool = false
    @State private var alertTitle: String = "Alert"
    @State private var showingHistory: Bool = false
    @State private var showingSettings: Bool = false

    @AppStorage("accentTheme") private var accentTheme: AccentTheme = .system
    @AppStorage("isHighPrecision") private var isHighPrecision: Bool = true
    @AppStorage("isExperimental") private var isExperimental: Bool = false
    @AppStorage("autoCleanup") private var autoCleanup: Bool = false
    @AppStorage("showDebugLogs") private var showDebugLogs: Bool = true

    @State private var selectedTab: Tab = .lab
    @State private var phase: Double = 0.0

    @Environment(\.colorScheme) private var colorScheme

    private var accentColor: Color {
        accentTheme.color(for: colorScheme)
    }

    // MARK: - Body

    var body: some View {
        TabView(selection: $selectedTab) {
            galleryTab
                .tag(Tab.gallery)
                .tabItem {
                    Label("Gallery", systemImage: "square.grid.2x2.fill")
                }

            labTab
                .tag(Tab.lab)
                .tabItem {
                    Label("Lab", systemImage: "flask.fill")
                }

            resultsTab
                .tag(Tab.results)
                .tabItem {
                    Label("Results", systemImage: "waveform.and.mic")
                }

            systemTab
                .tag(Tab.system)
                .tabItem {
                    Label("System", systemImage: "terminal.fill")
                }
        }
        .tint(accentColor)
        .alert("Clear Session?", isPresented: $showClearAllAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Clear All", role: .destructive) {
                clearAllResults()
            }
        } message: {
            Text("Are you sure you want to permanently delete all conversion history? This cannot be undone.")
        }
        .alert("Delete Model?", isPresented: $showDeleteDialog) {
            Button("Cancel", role: .cancel) { modelToDelete = nil }
            Button("Delete", role: .destructive) {
                if let name = modelToDelete {
                    deleteModel(name: name)
                }
            }
        } message: {
            if let name = modelToDelete {
                Text("Are you sure you want to delete \"\(name)\"? This will permanently remove the model file.")
            }
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
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
        .sheet(item: $activeImport) { importType in
            switch importType {
            case .model:
                DocumentPicker(
                    contentTypes: [UTType(filenameExtension: "safetensors")!, UTType(filenameExtension: "npz")!, UTType(filenameExtension: "pth")!, UTType.zip]
                ) { url in
                    activeImport = nil
                    handleFileImportURL(url: url)
                }
            case .audio:
                DocumentPicker(
                    contentTypes: [UTType.mp3, UTType.wav]
                ) { url in
                    activeImport = nil
                    handleAudioFileImportURL(url: url)
                }
            }
        }
        .alert(alertTitle, isPresented: $showAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(alertMessage)
        }
        .sheet(isPresented: $isManagingModels) {
            ModelManagerView(isPresented: $isManagingModels) { deletedName in
                if selectedModel == deletedName {
                    selectedModel = "Select Model"
                    isModelLoaded = false
                    statusMessage = "Select a model"
                }
                refreshImportedModels()
            }
        }
        .onAppear {
            setupInitialState()
        }
    }

    // MARK: - Tab Views

    private var galleryTab: some View {
        GalleryTabView(
            selectedModel: $selectedModel,
            isModelLoaded: $isModelLoaded,
            importedModels: $importedModels,
            isEditMode: $isEditMode,
            modelToDelete: $modelToDelete,
            showDeleteDialog: $showDeleteDialog,
            activeImport: $activeImport,
            onLoadModel: { name, isImported in
                loadModel(name: name, isImported: isImported)
            },
            onRefreshModels: {
                refreshImportedModels()
                log("Refreshing imported models...")
            }
        )
    }

    private var labTab: some View {
        LabTabView(
            inputURL: $inputURL,
            activeImport: $activeImport,
            volumeEnvelope: $volumeEnvelope,
            selectedF0Method: $selectedF0Method,
            pitchShift: $pitchShift,
            featureRatio: $featureRatio,
            showAdvancedSettings: $showAdvancedSettings,
            recordingTime: $recordingTime,
            recordingTimer: $recordingTimer,
            justFinishedRecording: $justFinishedRecording,
            isProcessing: $isProcessing,
            isConverting: $isConverting,
            conversionProgress: $conversionProgress,
            statusMessage: $statusMessage,
            isModelLoaded: $isModelLoaded,
            phase: $phase,
            audioRecorder: audioRecorder,
            inputPlayer: inputPlayer,
            stockAudioURL: stockAudioURL,
            onStartInference: {
                await startInference()
            }
        )
    }

    private var resultsTab: some View {
        ResultsTabView(
            conversionHistory: $conversionHistory,
            selectedResult: $selectedResult,
            showClearAllAlert: $showClearAllAlert,
            outputPlayer: outputPlayer,
            onDeleteResult: { result in
                deleteResult(result)
            }
        )
    }

    private var systemTab: some View {
        VStack(spacing: 0) {
            headerView.padding(.horizontal)

            SystemTabView(logRedirector: logRedirector, accentColor: accentColor)
                .padding(.top, 8)
        }
        .background(backgroundGradient)
    }

    // MARK: - Shared Header View

    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(navTitle)
                    .font(.title2)
                    .bold()
                Text(navSubtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button(action: { showingSettings = true }) {
                Image(systemName: "gearshape.fill")
                    .font(.title2)
                    .foregroundStyle(accentColor)
            }
        }
        .padding(.vertical, 12)
    }

    private var navTitle: String {
        switch selectedTab {
        case .gallery: return "Model Gallery"
        case .lab: return "Audio Lab"
        case .results: return "Inference Results"
        case .system: return "System Logs"
        }
    }

    private var navSubtitle: String {
        switch selectedTab {
        case .gallery: return "Select a voice model"
        case .lab: return "Convert your voice"
        case .results: return "Your conversion history"
        case .system: return "System output and debug logs"
        }
    }

    // MARK: - Background

    private var backgroundGradient: some View {
        ZStack {
            Color(uiColor: .systemBackground)

            LinearGradient(
                colors: [
                    accentColor.opacity(0.1),
                    Color.blue.opacity(0.2),
                    Color.purple.opacity(0.15)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            // Subtle animated highlight
            Circle()
                .fill(accentColor.opacity(0.1))
                .frame(width: 400, height: 400)
                .blur(radius: 80)
                .offset(x: cos(phase) * 120, y: sin(phase) * 120)
                .onAppear {
                    withAnimation(.linear(duration: 20).repeatForever(autoreverses: true)) {
                        phase = .pi * 2
                    }
                }
        }
        .ignoresSafeArea()
    }

    // MARK: - Logic Helpers

    func resetSession() {
        inputURL = nil
        outputURL = nil
        originalWaveform = []
        convertedWaveform = []
        statusMessage = "Ready"
        inputPlayer.stop()
        outputPlayer.stop()
    }

    func deleteModel(name: String) {
        // If we are deleting the CURRENT active model, we must unload it first
        if selectedModel == name {
            inferenceEngine.unloadModels()
            selectedModel = "Select Model"
            isModelLoaded = false
            statusMessage = "Model deleted"
        }

        // Delete from documents
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let safePath = docs.appendingPathComponent("\(name).safetensors")
        let npzPath = docs.appendingPathComponent("\(name).npz")

        try? FileManager.default.removeItem(at: safePath)
        try? FileManager.default.removeItem(at: npzPath)

        // Update UI
        importedModels.removeAll { $0 == name }
        modelToDelete = nil

        log("Deleted model: \(name)")
    }

    func setupInitialState() {
        // Set up logging callback
        inferenceEngine.onLog = { msg in
            self.log(msg)
        }

        refreshImportedModels()

        // Auto-load first model if none selected
        if selectedModel == "Select Model" {
            if !importedModels.isEmpty {
                loadModel(name: importedModels[0], isImported: true)
            } else {
                loadModel(name: "Coder")
            }
        }

        // Pre-load stock audio
        if inputURL == nil {
            if let stock = stockAudioURL {
                self.inputURL = stock
                statusMessage = "Loaded stock audio: \(stock.lastPathComponent)"
                originalWaveform = AudioWaveformExtractor.extractSamples(from: stock)
            }
        }
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        let tenths = Int((time * 10).truncatingRemainder(dividingBy: 10))
        return String(format: "%02d:%02d.%d", minutes, seconds, tenths)
    }

    func handleFileImport(result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            if let url = urls.first {
                handleFileImportURL(url: url)
            }
        case .failure(let error):
            log("File import error: \(error.localizedDescription)")
            alertTitle = "Import Failed"
            alertMessage = error.localizedDescription
            showAlert = true
        }
    }

    func handleAudioFileImport(result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            if let url = urls.first {
                handleAudioFileImportURL(url: url)
            }
        case .failure(let error):
            log("Audio import error: \(error.localizedDescription)")
            alertTitle = "Import Failed"
            alertMessage = error.localizedDescription
            showAlert = true
        }
    }

    func handleFileImportURL(url selectedFile: URL) {
        guard selectedFile.startAccessingSecurityScopedResource() else {
            log("Failed to access file: \(selectedFile.lastPathComponent)")
            alertTitle = "Access Denied"
            alertMessage = "Could not access the selected file"
            showAlert = true
            return
        }

        defer { selectedFile.stopAccessingSecurityScopedResource() }

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let filename = selectedFile.lastPathComponent
        let destURL = docs.appendingPathComponent(filename)

        do {
            // Remove existing file
            if FileManager.default.fileExists(atPath: destURL.path) {
                try FileManager.default.removeItem(at: destURL)
            }

            // Copy to documents
            try FileManager.default.copyItem(at: selectedFile, to: destURL)

            let modelName = destURL.deletingPathExtension().lastPathComponent

            // Handle different file types
            if selectedFile.pathExtension == "pth" {
                log("Converting .pth model: \(modelName)")
                statusMessage = "Converting model..."
                isConverting = true
                conversionProgress = 0.0

                Task {
                    do {
                        let arrays = try PthConverter.shared.convert(url: destURL, copyIndexTo: docs) { progress, msg in
                            Task { @MainActor in
                                self.conversionProgress = progress
                                self.statusMessage = msg
                            }
                        }

                        let dest = docs.appendingPathComponent(modelName + ".safetensors")
                        try MLX.save(arrays: arrays, url: dest)

                        await MainActor.run {
                            statusMessage = "Converted & Imported: \(modelName)"
                            log("Converted model saved to \(dest.path)")
                            isConverting = false
                            conversionProgress = 1.0
                            refreshImportedModels()
                            loadModel(name: modelName, isImported: true)
                            alertTitle = "Success"
                            alertMessage = "Model '\(modelName)' converted and ready!"
                            showAlert = true
                        }
                    } catch {
                        await MainActor.run {
                            isConverting = false
                            alertTitle = "Conversion Failed"
                            alertMessage = error.localizedDescription
                            showAlert = true
                        }
                    }
                }
            } else if selectedFile.pathExtension.lowercased() == "zip" {
                log("Processing zip archive: \(modelName)")
                statusMessage = "Extracting & converting..."
                isConverting = true
                conversionProgress = 0.0

                Task {
                    do {
                        let arrays = try PthConverter.shared.convert(url: destURL, copyIndexTo: docs) { progress, msg in
                            Task { @MainActor in
                                self.conversionProgress = progress
                                self.statusMessage = msg
                            }
                        }

                        let dest = docs.appendingPathComponent(modelName + ".safetensors")
                        try MLX.save(arrays: arrays, url: dest)

                        await MainActor.run {
                            statusMessage = "Converted & Imported: \(modelName)"
                            log("Converted model saved to \(dest.path)")
                            isConverting = false
                            conversionProgress = 1.0
                            refreshImportedModels()
                            loadModel(name: modelName, isImported: true)
                            alertTitle = "Success"
                            alertMessage = "Model '\(modelName)' extracted and ready!"
                            showAlert = true
                        }
                    } catch {
                        await MainActor.run {
                            isConverting = false
                            alertTitle = "Extraction Failed"
                            alertMessage = error.localizedDescription
                            showAlert = true
                        }
                    }
                }
            } else {
                // .safetensors or .npz
                refreshImportedModels()
                loadModel(name: modelName, isImported: true)
                alertTitle = "Success"
                alertMessage = "Model '\(modelName)' imported successfully!"
                showAlert = true
            }

            log("Imported: \(filename)")

        } catch {
            log("Import error: \(error.localizedDescription)")
            alertTitle = "Import Failed"
            alertMessage = error.localizedDescription
            showAlert = true
        }
    }

    func handleAudioFileImportURL(url selectedFile: URL) {
        guard selectedFile.startAccessingSecurityScopedResource() else {
            log("Failed to access audio file")
            return
        }

        defer { selectedFile.stopAccessingSecurityScopedResource() }

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let destURL = docs.appendingPathComponent(selectedFile.lastPathComponent)

        do {
            if FileManager.default.fileExists(atPath: destURL.path) {
                try FileManager.default.removeItem(at: destURL)
            }
            try FileManager.default.copyItem(at: selectedFile, to: destURL)

            inputURL = destURL
            originalWaveform = AudioWaveformExtractor.extractSamples(from: destURL)
            statusMessage = "Audio loaded: \(destURL.lastPathComponent)"
            log("Audio loaded: \(destURL.lastPathComponent)")

        } catch {
            log("Audio import error: \(error.localizedDescription)")
        }
    }

    func log(_ message: String) {
        print("DEBUG: \(message)")
    }

    func refreshImportedModels() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        guard let files = try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil) else { return }

        importedModels = files
            .filter { $0.pathExtension == "safetensors" || $0.pathExtension == "npz" }
            .map { $0.deletingPathExtension().lastPathComponent }
            .filter { !$0.hasPrefix(".") }
            .sorted()
    }

    func loadModel(name: String, isImported: Bool = false) {
        log("loadModel called for \(name)")
        selectedModel = name
        isModelLoaded = false
        statusMessage = "Loading model..."

        // Map display name to actual filename for bundled models
        let bundledModelMapping: [String: String] = [
            "Coder": "Coder999V2",
            "Slim Shady": "Slim_Shady_New"
        ]

        // Map name to file - use mapping for bundled, or convert name for imported
        let filename: String
        if let mappedName = bundledModelMapping[name], !isImported {
            filename = mappedName
        } else {
            filename = name.lowercased().replacingOccurrences(of: " ", with: "_")
        }

        var modelUrl: URL?

        if isImported {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let safe = docs.appendingPathComponent("\(name).safetensors")
            let npz = docs.appendingPathComponent("\(name).npz")
            if FileManager.default.fileExists(atPath: safe.path) { modelUrl = safe }
            else if FileManager.default.fileExists(atPath: npz.path) { modelUrl = npz }
        } else {
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

        // Optional RMVPE
        let rmvpeURL = RVCInference.bundle.url(forResource: "rmvpe", withExtension: "safetensors")
            ?? RVCInference.bundle.url(forResource: "rmvpe", withExtension: "safetensors", subdirectory: "Assets")
            ?? RVCInference.bundle.url(forResource: "rmvpe", withExtension: "npz")
            ?? RVCInference.bundle.url(forResource: "rmvpe", withExtension: "npz", subdirectory: "Assets")
            ?? RVCInference.bundle.url(forResource: "rmvpe_mlx", withExtension: "npz")
            ?? RVCInference.bundle.url(forResource: "rmvpe_mlx", withExtension: "npz", subdirectory: "Assets")

        // Search for matching index file (for imported models)
        var indexUrl: URL? = nil
        if isImported {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

            // Try specific name patterns first (matching original implementation)
            let possibleNames = [
                "\(name).index",
                "\(name).safetensors.index",
                "added_IVF256_Flat_nprobe_1_\(name).index",
                "\(name)_index.safetensors"
            ]

            for pKey in possibleNames {
                let p = docs.appendingPathComponent(pKey)
                if FileManager.default.fileExists(atPath: p.path) {
                    indexUrl = p
                    break
                }
            }

            // Fallback: search directory for any matching index file
            if indexUrl == nil {
                let contents = try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil)
                indexUrl = contents?.first(where: { fileUrl in
                    let indexName = fileUrl.deletingPathExtension().lastPathComponent.lowercased()
                    let modelLower = name.lowercased()
                    let ext = fileUrl.pathExtension.lowercased()
                    return (ext == "index" || (ext == "safetensors" && indexName.contains("index"))) &&
                           (indexName.contains(modelLower) || modelLower.contains(indexName.replacingOccurrences(of: "_index", with: "").replacingOccurrences(of: "index_", with: "")))
                })
            }

            if let idx = indexUrl {
                log("Found index file: \(idx.lastPathComponent)")
            } else {
                log("No index file found for \(name)")
            }
        }

        Task {
            do {
                log("Starting loadWeights task...")
                try await inferenceEngine.loadWeights(hubertURL: hubertURL, modelURL: url, rmvpeURL: rmvpeURL)
                log("loadWeights success")

                // Load index file if found, otherwise unload any previous index
                if let indexUrl = indexUrl {
                    do {
                        try inferenceEngine.loadIndex(url: indexUrl)
                        log("Index loaded: \(indexUrl.lastPathComponent)")
                    } catch {
                        log("Failed to load index: \(error.localizedDescription)")
                    }
                } else {
                    inferenceEngine.unloadIndex()
                }

                await MainActor.run {
                    statusMessage = "Loaded \(name)"
                    isModelLoaded = true
                }
            } catch {
                log("loadWeights failed: \(error)")
                await MainActor.run {
                    statusMessage = "Failed to load \(name): \(error.localizedDescription)"
                    isModelLoaded = false

                    alertTitle = "Load Failed"
                    alertMessage = error.localizedDescription
                    showAlert = true
                }
            }
        }
    }

    func startInference() async {
        guard let input = inputURL ?? stockAudioURL else {
            statusMessage = "No audio file selected"
            return
        }

        // Check model is loaded
        guard isModelLoaded else {
            await MainActor.run {
                statusMessage = "Please load a model first"
                alertTitle = "No Model"
                alertMessage = "Please select and load a model from the Gallery tab"
                showAlert = true
            }
            return
        }

        log("Starting inference with input: \(input.lastPathComponent)")

        isConverting = true
        conversionProgress = 0.0

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let timestamp = Int(Date().timeIntervalSince1970)
        let outputPath = docs.appendingPathComponent("converted_\(timestamp).wav")

        // Load original waveform
        if originalWaveform.isEmpty {
            originalWaveform = AudioWaveformExtractor.extractSamples(from: input)
        }

        await MainActor.run {
            conversionProgress = 0.1
            statusMessage = "Preparing audio..."
        }

        log("Calling inferenceEngine.infer() with f0Method=\(selectedF0Method), pitchShift=\(pitchShift), featureRatio=\(featureRatio)")

        await inferenceEngine.infer(
            audioURL: input,
            outputURL: outputPath,
            pitchShift: Int(pitchShift),
            f0Method: selectedF0Method,
            indexRate: Float(featureRatio),
            volumeEnvelope: volumeEnvelope
        )

        log("inferenceEngine.infer() returned")

        // Verify output file was created
        guard FileManager.default.fileExists(atPath: outputPath.path) else {
            await MainActor.run {
                isConverting = false
                statusMessage = "Conversion failed - no output file"
                alertTitle = "Conversion Failed"
                alertMessage = "The output file was not created. Check System logs for details."
                showAlert = true
            }
            log("ERROR: Output file does not exist at \(outputPath.path)")
            return
        }

        // Check file has content
        if let attrs = try? FileManager.default.attributesOfItem(atPath: outputPath.path),
           let size = attrs[.size] as? Int {
            log("Output file size: \(size) bytes")
            if size == 0 {
                await MainActor.run {
                    isConverting = false
                    statusMessage = "Conversion failed - empty output"
                    alertTitle = "Conversion Failed"
                    alertMessage = "The output file is empty. The inference may have failed."
                    showAlert = true
                }
                return
            }
        }

        await MainActor.run {
            conversionProgress = 0.9
            statusMessage = "Finalizing..."
        }

        // Load converted waveform
        let converted = AudioWaveformExtractor.extractSamples(from: outputPath)
        log("Extracted \(converted.count) samples from converted audio")

        // Diagnostic: Check if audio is silent
        if converted.isEmpty {
            log("WARNING: Extracted waveform is EMPTY - file may not be readable")
        } else {
            let maxAmp = converted.map { abs($0) }.max() ?? 0
            let avgAmp = converted.reduce(0, { $0 + abs($1) }) / Float(converted.count)
            log("Waveform stats: maxAmp=\(maxAmp), avgAmp=\(avgAmp)")

            if maxAmp < 0.001 {
                log("WARNING: Audio appears to be SILENT (maxAmp < 0.001)")
            }
        }

        await MainActor.run {
            outputURL = outputPath
            convertedWaveform = converted
            conversionProgress = 1.0
            isConverting = false
            statusMessage = "Done!"

            // Add to history
            let result = ConversionResult(
                date: Date(),
                modelName: selectedModel,
                sourceAudioName: input.lastPathComponent,
                inputURL: input,
                outputURL: outputPath,
                originalWaveform: originalWaveform,
                convertedWaveform: converted
            )
            conversionHistory.insert(result, at: 0)
            selectedResult = result

            // Switch to results tab
            selectedTab = .results

            log("Conversion complete: \(outputPath.lastPathComponent)")
        }
    }

    func deleteResult(_ result: ConversionResult) {
        // Remove from history
        conversionHistory.removeAll { $0.id == result.id }

        // Delete output file
        try? FileManager.default.removeItem(at: result.outputURL)

        // Clear selection if this was selected
        if selectedResult?.id == result.id {
            selectedResult = conversionHistory.first
        }

        log("Deleted result: \(result.sourceAudioName)")
    }

    func clearAllResults() {
        for result in conversionHistory {
            try? FileManager.default.removeItem(at: result.outputURL)
        }
        conversionHistory.removeAll()
        selectedResult = nil
        log("Cleared all results")
    }
}

#Preview {
    ContentView()
}
