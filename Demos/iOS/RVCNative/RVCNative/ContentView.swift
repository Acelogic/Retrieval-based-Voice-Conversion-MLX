import SwiftUI
import MLX
import AVFoundation
import RVCNativeFeature
import UniformTypeIdentifiers
import UIKit

// MARK: - Document Picker Wrapper
struct DocumentPicker: UIViewControllerRepresentable {
    let contentTypes: [UTType]
    let onPick: (URL) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: contentTypes)
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick)
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void

        init(onPick: @escaping (URL) -> Void) {
            self.onPick = onPick
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            onPick(url)
        }
    }
}

// MARK: - Waveform View Component
struct WaveformView: View {
    let samples: [Float]
    let color: Color
    let label: String
    var progress: Double? = nil // 0.0 to 1.0

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                if let p = progress {
                    Text("\(Int(p * 100))%")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundColor(.cyan.opacity(0.8))
                }
            }

            GeometryReader { geometry in
                let width = geometry.size.width
                let height = geometry.size.height
                let midY = height / 2

                ZStack(alignment: .leading) {
                    Path { path in
                        guard !samples.isEmpty else { return }

                        let pointCount = Int(width)
                        let samplesPerPoint = max(1, samples.count / pointCount)

                        for i in 0..<pointCount {
                            let startIdx = i * samplesPerPoint
                            let endIdx = min(startIdx + samplesPerPoint, samples.count)

                            guard startIdx < samples.count else { break }

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
                    
                    // Scrubber Line
                    if let p = progress {
                        Rectangle()
                            .fill(LinearGradient(colors: [.cyan, .cyan.opacity(0)], startPoint: .top, endPoint: .bottom))
                            .frame(width: 2)
                            .offset(x: CGFloat(p) * width - 1)
                            .shadow(color: .cyan.opacity(0.5), radius: 2)
                    }
                }
            }
        }
        .frame(height: 60)
    }
}

// MARK: - Waveform Comparison View
struct WaveformComparisonView: View {
    let originalSamples: [Float]
    let convertedSamples: [Float]
    var progress: Double? = nil
    var onSeek: ((Double) -> Void)? = nil

    var body: some View {
        VStack(spacing: 12) {
            Text("Waveform Comparison")
                .font(.headline)
                .foregroundStyle(.primary.opacity(0.9))

            GeometryReader { geometry in
                VStack(spacing: 8) {
                    WaveformView(samples: originalSamples, color: .blue, label: "Original", progress: progress)
                    WaveformView(samples: convertedSamples, color: .green, label: "Converted", progress: progress)
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            let percentage = max(0, min(1, value.location.x / geometry.size.width))
                            onSeek?(Double(percentage))
                        }
                )
            }
            .frame(height: 128) // Fixed height for the waveforms + labels
        }
        .padding()
        .background(.primary.opacity(0.05))
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

    enum ImportType: Identifiable {
        case model
        case audio

        var id: Self { self }
    }
    @State private var activeImport: ImportType? = nil
    @State private var logs: [String] = []
    @State private var volumeEnvelope: Float = 1.0 // Default 1.0 (no change)
    
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
    struct ConversionResult: Identifiable, Equatable {
        let id = UUID()
        let date: Date
        let modelName: String
        let sourceAudioName: String
        let inputURL: URL
        let outputURL: URL
        let originalWaveform: [Float]
        let convertedWaveform: [Float]
    }
    
    @State private var conversionHistory: [ConversionResult] = []
    @State private var selectedResult: ConversionResult?
    
    // Model List State
    @State private var importedModels: [String] = []

    // Advanced Lab Controls
    @State private var selectedF0Method: String = "rmvpe" // rmvpe, dio, pm, harvest, crepe, crepe-tiny
    @State private var pitchShift: Double = 0.0 // -12 to +12 semitones
    @State private var featureRatio: Double = 0.75 // 0.0 to 1.0
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

    enum Tab {
        case gallery, lab, results, system
    }
    
    enum AccentTheme: String, CaseIterable, Identifiable {
        case system = "System"
        case cyan = "Cyan"
        case blue = "Blue"
        case purple = "Purple"
        case pink = "Pink"
        case green = "Green"
        case orange = "Orange"
        
        var id: String { self.rawValue }
        
        func color(for scheme: ColorScheme) -> Color {
            switch self {
            case .system: return scheme == .light ? .blue : .cyan
            case .cyan: return .cyan
            case .blue: return .blue
            case .purple: return .purple
            case .pink: return .pink
            case .green: return .green
            case .orange: return .orange
            }
        }
    }
    
    @AppStorage("accentTheme") private var accentTheme: AccentTheme = .system
    @AppStorage("isHighPrecision") private var isHighPrecision: Bool = true
    @AppStorage("isExperimental") private var isExperimental: Bool = false
    @AppStorage("autoCleanup") private var autoCleanup: Bool = false
    @AppStorage("showDebugLogs") private var showDebugLogs: Bool = true
    
    @State private var selectedTab: Tab = .lab // Start in the Lab
    
    @Environment(\.colorScheme) private var colorScheme
    
    private var accentColor: Color {
        accentTheme.color(for: colorScheme)
    }
    
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
    


    private var galleryTab: some View {
        ScrollView {
            VStack(spacing: 24) {
                headerView
                modelSelectionView
                
                Divider()
                    .overlay(.primary.opacity(0.1))
                
                modelGalleryGridView
            }
            .padding()
        }
        .background(backgroundGradient)
        .refreshable {
            refreshImportedModels()
            log("Refreshing imported models...")
        }
    }
    
    private var labTab: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerView
                volumeControlView
                advancedSettingsView
                audioInputControlView
                conversionProgressView
                inferenceButtonView
            }
            .padding()
        }
        .background(backgroundGradient)
    }
    
    private var resultsTab: some View {
        VStack(spacing: 0) {
            headerView.padding(.horizontal)
            
            if conversionHistory.isEmpty {
                VStack(spacing: 20) {
                    Spacer()
                    Image(systemName: "waveform.badge.magnifyingglass")
                        .font(.system(size: 60))
                        .foregroundStyle(.primary.opacity(0.3))
                    Text("No conversions yet")
                        .font(.headline)
                        .foregroundStyle(.primary.opacity(0.5))
                    Text("Go to the Lab to start your first RVC conversion.")
                        .font(.subheadline)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                        .foregroundStyle(.primary.opacity(0.3))
                    Spacer()
                }
            } else {
                ScrollView {
                    VStack(spacing: 24) {
                        if let selected = selectedResult {
                            detailedResultView(for: selected)
                                .transition(.move(edge: .top).combined(with: .opacity))
                        }
                        
                        Divider().overlay(.primary.opacity(0.1))
                        
                        VStack(alignment: .leading, spacing: 16) {
                            HStack {
                                Text("History")
                                    .font(.subheadline)
                                    .bold()
                                    .foregroundStyle(.primary.opacity(0.8))
                                Spacer()
                                Button(action: { showClearAllAlert = true }) {
                                    Text("Clear All")
                                        .font(.caption)
                                        .bold()
                                        .foregroundStyle(.red.opacity(0.8))
                                }
                            }
                            
                            ForEach(conversionHistory) { result in
                                HistoryRow(result: result, isSelected: selectedResult?.id == result.id) {
                                    withAnimation {
                                        selectedResult = result
                                    }
                                } onDelete: {
                                    deleteResult(result)
                                }
                            }
                        }
                    }
                    .padding()
                }
                .refreshable {
                    log("Refreshing conversion history...")
                }
            }
        }
        .background(backgroundGradient)
    }
    
    private var systemTab: some View {
        VStack(spacing: 0) {
            headerView.padding(.horizontal)
            
            consoleOutputView
                .padding(.top, 8)
        }
        .background(backgroundGradient)
    }

    private var navTitle: String {
        switch selectedTab {
        case .gallery: return "Model Gallery"
        case .lab: return "Audio Lab"
        case .results: return "Inference Results"
        case .system: return "System Logs"
        }
    }
    
    // Liquid Background with Animation
    @State private var phase: Double = 0.0
    
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
    
    private var headerView: some View {
        HStack {
            Button(action: { 
                let generator = UIImpactFeedbackGenerator(style: .light)
                generator.impactOccurred()
                showingSettings = true 
            }) {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 20))
                    .foregroundStyle(.primary.opacity(0.8))
                    .padding(10)
                    .background(Circle().fill(.primary.opacity(0.05)))
                    .glassEffect(in: Circle())
            }
            .buttonStyle(.plain)
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 2) {
                Text(tabTitle)
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                Text("POWERED BY MLX SWIFT")
                    .font(.system(size: 10, weight: .black))
                    .tracking(2)
                    .foregroundStyle(accentColor)
                    .opacity(0.8)
            }
        }
        .padding(.vertical, 12)
    }
    
    private var tabTitle: String {
        switch selectedTab {
        case .gallery: return "Model Gallery"
        case .lab: return "Audio Lab"
        case .results: return "Inference Results"
        case .system: return "System Logs"
        }
    }
    
    // Model Selection Card
    private var modelSelectionView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("RVC model selected")
                    .font(.caption2)
                    .foregroundStyle(.primary.opacity(0.5))
                Text(selectedModel)
                    .font(.headline)
                    .foregroundStyle(accentColor)
            }
            Spacer()
            
            HStack(spacing: 12) {
                Button(action: { 
                    withAnimation(.spring()) {
                        isEditMode.toggle() 
                    }
                }) {
                    Text(isEditMode ? "Done" : "Edit")
                        .font(.subheadline)
                        .bold()
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .glassEffect(in: Capsule())
                        .foregroundStyle(isEditMode ? .green : .cyan)
                }
                
                Button(action: { activeImport = .model }) {
                    Image(systemName: "plus")
                        .font(.headline)
                        .padding(10)
                        .glassEffect(in: Circle())
                        .foregroundStyle(accentColor)
                }
            }
        }
        .padding()
        .glassEffect(in: RoundedRectangle(cornerRadius: 16))
    }
    
    // Model Grid View
    private var modelGalleryGridView: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Available Models")
                .font(.subheadline)
                .bold()
                .foregroundStyle(.primary.opacity(0.8))
            
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                // Bundle Models
                ModelTile(
                    name: "Coder", 
                    isSelected: selectedModel == "Coder",
                    isImported: false,
                    isEditMode: isEditMode, // Now passing state for jiggle feedback
                    action: { loadModel(name: "Coder") },
                    onDelete: { }
                )
                
                ModelTile(
                    name: "Slim Shady", 
                    isSelected: selectedModel == "Slim Shady",
                    isImported: false,
                    isEditMode: isEditMode,
                    action: { loadModel(name: "Slim Shady") },
                    onDelete: { }
                )
                
                // Imported Models
                ForEach(importedModels, id: \.self) { modelName in
                    ModelTile(
                        name: modelName, 
                        isSelected: selectedModel == modelName, 
                        isImported: true,
                        isEditMode: isEditMode,
                        action: { loadModel(name: modelName, isImported: true) },
                        onDelete: { 
                            modelToDelete = modelName
                            showDeleteDialog = true
                        }
                    )
                }
                
                // Add New Tile
                if !isEditMode {
                    Button(action: { activeImport = .model }) {
                        VStack(spacing: 12) {
                            Image(systemName: "plus.circle.fill")
                                .font(.system(size: 30))
                                .foregroundStyle(accentColor)
                            Text("Import New")
                                .font(.caption)
                                .bold()
                        }
                        .frame(maxWidth: .infinity)
                        .frame(height: 120)
                        .glassEffect(in: RoundedRectangle(cornerRadius: 16))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4]))
                                .foregroundStyle(accentColor.opacity(0.3))
                        )
                    }
                }
            }
        }
    }
    
    // Volume Envelope Slider Card
    private var volumeControlView: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("Volume Envelope (Reduce Noise)")
                    .font(.caption)
                    .foregroundStyle(.primary.opacity(0.8))
                Spacer()
                Text(String(format: "%.2f", volumeEnvelope))
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.primary)
            }
            Slider(value: $volumeEnvelope, in: 0.0...1.0, step: 0.01)
                .tint(accentColor)
        }
        .padding()
        .glassEffect(in: RoundedRectangle(cornerRadius: 16))
    }
    
    // Advanced Settings (Collapsible)
    private var advancedSettingsView: some View {
        VStack(spacing: 0) {
            DisclosureGroup(isExpanded: $showAdvancedSettings) {
                VStack(spacing: 16) {
                    Divider()
                        .overlay(.primary.opacity(0.1))
                        .padding(.vertical, 8)
                    
                    // F0 Method
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Pitch Extraction")
                            .font(.caption2.bold())
                            .foregroundStyle(.secondary)
                        
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(["rmvpe", "dio", "pm", "harvest", "crepe", "crepe-tiny"], id: \.self) { method in
                                    Button(action: { selectedF0Method = method }) {
                                        Text(method.uppercased())
                                            .font(.caption2.bold())
                                            .foregroundStyle(selectedF0Method == method ? .white : .primary.opacity(0.6))
                                            .padding(.horizontal, 12)
                                            .padding(.vertical, 6)
                                            .background(selectedF0Method == method ? accentColor : Color.primary.opacity(0.08))
                                            .clipShape(Capsule())
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }
                    
                    // Pitch Shift
                    VStack(spacing: 4) {
                        HStack {
                            Text("Pitch Shift")
                                .font(.caption2.bold())
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(pitchShift >= 0 ? "+\(Int(pitchShift))" : "\(Int(pitchShift))")
                                .font(.caption2)
                                .monospacedDigit()
                                .foregroundStyle(.primary.opacity(0.7))
                        }
                        Slider(value: $pitchShift, in: -12...12, step: 1.0)
                            .tint(accentColor)
                    }
                    
                    // Feature Ratio
                    VStack(spacing: 4) {
                        HStack {
                            Text("Feature Retrieval")
                                .font(.caption2.bold())
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(String(format: "%.2f", featureRatio))
                                .font(.caption2)
                                .monospacedDigit()
                                .foregroundStyle(.primary.opacity(0.7))
                        }
                        Slider(value: $featureRatio, in: 0.0...1.0, step: 0.01)
                            .tint(accentColor)
                    }
                }
                .padding(.top, 4)
            } label: {
                HStack {
                    Image(systemName: "slider.horizontal.3")
                        .foregroundStyle(accentColor)
                    Text("Advanced Settings")
                        .font(.caption)
                        .foregroundStyle(.primary.opacity(0.8))
                    Spacer()
                }
            }
        }
        .padding()
        .glassEffect(in: RoundedRectangle(cornerRadius: 16))
    }
    
    @ViewBuilder
    private var audioInputControlView: some View {
        Group {
            // Actions
            Button(action: {
                activeImport = .audio
            }) {
                HStack {
                    Image(systemName: "waveform.circle")
                    Text(inputURL == nil ? "Select Audio File" : "Selected: \(inputURL?.lastPathComponent ?? "File")")
                        .lineLimit(1)
                }
                .font(.headline)
                .frame(maxWidth: .infinity)
                .padding()
                .glassEffect(in: Capsule())
                .foregroundColor(accentColor)
                .overlay(
                    Capsule()
                        .stroke(accentColor.opacity(0.3), lineWidth: 1)
                )
            }
            
            // Audio Loaded Indicator
            if let input = inputURL {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Ready to convert: \(input.lastPathComponent)")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
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
                    .foregroundColor(.primary)
                    .padding(.vertical, 4)
                    .padding(.horizontal, 12)
                    .glassEffect(in: Capsule())
                }
            }
            
            // Audio Recording - Dynamic UI
            if audioRecorder.isRecording {
                // While Recording - Animated
                VStack(spacing: 16) {
                    // Pulsing Red Circle with Timer
                    ZStack {
                        // Outer pulse rings
                        Circle()
                            .fill(Color.red.opacity(0.2))
                            .frame(width: 120, height: 120)
                            .scaleEffect(phase)
                            .opacity(2 - phase)
                        
                        Circle()
                            .fill(Color.red.opacity(0.3))
                            .frame(width: 90, height: 90)
                            .scaleEffect(1 + (phase * 0.5))
                        
                        // Inner solid circle
                        Circle()
                            .fill(Color.red)
                            .frame(width: 70, height: 70)
                        
                        Image(systemName: "waveform")
                            .font(.system(size: 28))
                            .foregroundStyle(.white)
                    }
                    .onAppear {
                        withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true)) {
                            phase = 1.5
                        }
                    }
                    
                    // Recording Time
                    Text(formatTime(recordingTime))
                        .font(.system(size: 40, weight: .thin, design: .rounded))
                        .monospacedDigit()
                        .foregroundStyle(.primary)
                    
                    Text("Recording...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .opacity(phase > 1.2 ? 1.0 : 0.5)
                    
                    // Stop Button
                    Button(action: {
                        audioRecorder.stopRecording()
                        recordingTimer?.invalidate()
                        justFinishedRecording = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                            justFinishedRecording = false
                        }
                    }) {
                        HStack {
                            Image(systemName: "stop.circle.fill")
                            Text("Stop Recording")
                        }
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.red.opacity(0.5))
                        .glassEffect(in: Capsule())
                        .foregroundColor(.primary)
                    }
                }
                .padding()
                .onAppear {
                    recordingTime = 0
                    recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
                        recordingTime += 0.1
                    }
                }
                .transition(.opacity.combined(with: .scale))
            } else if justFinishedRecording {
                // Just Finished Recording - Success Animation
                VStack(spacing: 16) {
                    ZStack {
                        Circle()
                            .fill(Color.green.opacity(0.2))
                            .frame(width: 100, height: 100)
                        
                        Circle()
                            .fill(Color.green)
                            .frame(width: 70, height: 70)
                        
                        Image(systemName: "checkmark")
                            .font(.system(size: 32, weight: .bold))
                            .foregroundStyle(.white)
                    }
                    .transition(.scale.combined(with: .opacity))
                    
                    Text("Recording Complete!")
                        .font(.headline)
                        .foregroundStyle(.primary)
                    
                    Text(formatTime(recordingTime))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding()
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            } else {
                // Not Recording - Record Button
                Button(action: {
                    audioRecorder.startRecording()
                }) {
                    HStack {
                        Image(systemName: "mic.circle.fill")
                        Text("Record Audio")
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .glassEffect(in: Capsule())
                    .foregroundColor(.red.opacity(0.8))
                }
            }
        }
    }
    
    // Conversion Progress Bar
    private var conversionProgressView: some View {
        Group {
            if isConverting {
                VStack(spacing: 8) {
                    ProgressView(value: conversionProgress, total: 1.0)
                        .progressViewStyle(LinearProgressViewStyle())
                        .tint(accentColor)
                    Text("\(Int(conversionProgress * 100))% - \(statusMessage)")
                        .font(.caption)
                        .foregroundStyle(.primary.opacity(0.7))
                }
                .padding()
                .glassEffect(in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }
    
    private var inferenceButtonView: some View {
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
            .background(isProcessing || !isModelLoaded ? Color.gray.opacity(0.3) : Color.green.opacity(0.6))
            .glassEffect(in: Capsule())
            .foregroundColor(.primary)
            .shadow(color: .green.opacity(0.4), radius: 10, x: 0, y: 5)
        }
        .disabled(isProcessing || inputURL == nil || !isModelLoaded)
    }
    
    // Play Converted
    // Detailed Result View (Extracted for clarity)
    @ViewBuilder
    private func detailedResultView(for result: ConversionResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading) {
                    Text("Selected Conversion")
                        .font(.caption2)
                        .bold()
                        .foregroundStyle(accentColor)
                    HStack(spacing: 4) {
                        Text(result.sourceAudioName)
                            .lineLimit(1)
                        Image(systemName: "arrow.right")
                        Text(result.modelName)
                            .lineLimit(1)
                    }
                    .font(.title3)
                    .bold()
                    .foregroundStyle(.primary)
                }
                Spacer()
                Text(result.date, style: .time)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            // Comparison Waveform
            WaveformComparisonView(
                originalSamples: result.originalWaveform,
                convertedSamples: result.convertedWaveform,
                progress: (outputPlayer.isPlaying && outputPlayer.duration > 0) ? outputPlayer.currentTime / outputPlayer.duration : 0,
                onSeek: { percentage in
                    outputPlayer.seek(to: percentage * outputPlayer.duration)
                }
            )
            .glassEffect(in: RoundedRectangle(cornerRadius: 12))
            .frame(height: 180)
            
            HStack(spacing: 12) {
                Button(action: {
                    if outputPlayer.isPlaying {
                        outputPlayer.stop()
                    } else {
                        outputPlayer.play(url: result.outputURL)
                    }
                }) {
                    HStack {
                        Image(systemName: outputPlayer.isPlaying ? "stop.fill" : "play.fill")
                        Text(outputPlayer.isPlaying ? "Stop" : "Play Result")
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green.opacity(0.3))
                    .glassEffect(in: RoundedRectangle(cornerRadius: 12))
                    .foregroundColor(.green)
                }
                
                Button(action: { 
                    // Share logic
                    let activityVC = UIActivityViewController(activityItems: [result.outputURL], applicationActivities: nil)
                    if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                       let rootVC = windowScene.windows.first?.rootViewController {
                        rootVC.present(activityVC, animated: true)
                    }
                }) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.headline)
                        .padding()
                        .background(Color.white.opacity(0.1))
                        .glassEffect(in: RoundedRectangle(cornerRadius: 12))
                        .foregroundColor(.primary)
                }
            }
        }
        .padding()
        .glassEffect(in: RoundedRectangle(cornerRadius: 16))
    }
    
    private var resultView: some View {
        EmptyView()
    }
    
    // Status Log
    private var consoleOutputView: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Label("Terminal Output", systemImage: "terminal.fill")
                    .font(.system(size: 10, weight: .black))
                    .tracking(1)
                    .foregroundStyle(.primary.opacity(0.7))
                
                Spacer()
                
                HStack(spacing: 0) {
                    Button(action: {
                        UIPasteboard.general.string = logRedirector.logs
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "doc.on.doc")
                            Text("Copy")
                        }
                        .font(.system(size: 10, weight: .bold))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                    }
                    .foregroundStyle(accentColor)
                    
                    Divider()
                        .frame(height: 12)
                        .overlay(.primary.opacity(0.1))
                    
                    Button(action: { logRedirector.clear() }) {
                        HStack(spacing: 4) {
                            Image(systemName: "trash")
                            Text("Clear")
                        }
                        .font(.system(size: 10, weight: .bold))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                    }
                    .foregroundStyle(.red.opacity(0.8))
                }
                .background {
                    Capsule()
                        .fill(.ultraThinMaterial)
                        .overlay(
                            Capsule()
                                .stroke(.primary.opacity(0.1), lineWidth: 0.5)
                        )
                }
                .glassEffect(in: Capsule())
            }
            .padding(.horizontal)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial)
            .overlay(
                Divider().overlay(.primary.opacity(0.1)),
                alignment: .bottom
            )
            
            ScrollViewReader { proxy in
                ScrollView {
                    Text(logRedirector.logs.isEmpty ? "No logs yet..." : logRedirector.logs)
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                        .foregroundStyle(.primary.opacity(0.8))
                        .id("logEnd")
                }
                .background(.ultraThinMaterial.opacity(0.5))
                .onChange(of: logRedirector.logs) { _, _ in
                    withAnimation {
                        proxy.scrollTo("logEnd", anchor: .bottom)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .glassEffect(in: RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal)
        .padding(.bottom, 8)
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
        // to release file handles.
        if selectedModel == name {
            inferenceEngine.unloadModels()
            selectedModel = "Select Model"
            isModelLoaded = false
            statusMessage = "Current model deleted, please select a new one."
        }
        
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        
        // Try both common extensions
        let safetensorsPath = docs.appendingPathComponent("\(name).safetensors")
        let npzPath = docs.appendingPathComponent("\(name).npz")
        
        // Small delay to ensure MLX releases locks if it was JUST unloading
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            do {
                if FileManager.default.fileExists(atPath: safetensorsPath.path) {
                    try FileManager.default.removeItem(at: safetensorsPath)
                } else if FileManager.default.fileExists(atPath: npzPath.path) {
                    try FileManager.default.removeItem(at: npzPath)
                }
                
                self.refreshImportedModels()
                self.modelToDelete = nil
            } catch {
                self.alertTitle = "Deletion Failed"
                self.alertMessage = "Could not delete model \"\(name)\": \(error.localizedDescription)"
                self.showAlert = true
                self.modelToDelete = nil
            }
        }
    }
    
    func setupInitialState() {
        inferenceEngine.onLog = { msg in
            self.log(msg)
        }
        
        refreshImportedModels()
        
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


    // MARK: - Helper Functions
    
    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        let centiseconds = Int((time.truncatingRemainder(dividingBy: 1)) * 10)
        return String(format: "%02d:%02d.%d", minutes, seconds, centiseconds)
    }

    func handleFileImport(result: Result<[URL], Error>) {
        do {
            guard let selectedFile: URL = try result.get().first else { return }
            
            // Check if .pth or .zip
            if selectedFile.pathExtension.lowercased() == "pth" || selectedFile.pathExtension.lowercased() == "zip" {
                alertTitle = "Import Failed" // Default
                statusMessage = selectedFile.pathExtension.lowercased() == "zip" ? "Inspecting zip archive..." : "Converting model archive..."
                
                // Access security scope
                // IMPORTANT: We must keep the scope open until the async task is done.
                // We cannot use 'defer' in this block because it exits immediately.
                let accessSucceeded = selectedFile.startAccessingSecurityScopedResource()
                if !accessSucceeded {
                    statusMessage = "Access denied"
                    return
                }
                
                Task {
                    // Ensure we stop accessing when this Task finishes
                    defer { selectedFile.stopAccessingSecurityScopedResource() }
                    
                    // Run on background thread (detached) to avoid blocking main actor
                    // PthConverter is now thread-safe / nonisolated
                    await MainActor.run { 
                        isConverting = true 
                        conversionProgress = 0.0
                    }
                    
                    do {
                        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                        let arrays = try PthConverter.shared.convert(url: selectedFile, copyIndexTo: docs) { progress, msg in
                            Task { @MainActor in
                                self.conversionProgress = progress
                                self.statusMessage = msg
                            }
                        }
                        
                        let name = selectedFile.deletingPathExtension().lastPathComponent
                        let dest = docs.appendingPathComponent(name + ".safetensors")
                        
                        try MLX.save(arrays: arrays, url: dest)
                        
                        await MainActor.run {
                            statusMessage = "Converted & Imported: \(name)"
                            log("Converted model saved to \(dest.path)")
                            isConverting = false
                            conversionProgress = 1.0
                            refreshImportedModels()
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
                 refreshImportedModels()
            } else {
                 statusMessage = "Access denied"
            }
        } catch {
            statusMessage = "Error: \(error.localizedDescription)"
            log("Import error: \(error)")
        }
    }

    func handleAudioFileImport(result: Result<[URL], Error>) {
        do {
            guard let selectedFile: URL = try result.get().first else { return }
            handleAudioFileImportURL(url: selectedFile)
        } catch {
            statusMessage = "Error: \(error.localizedDescription)"
            log("Audio import error: \(error)")
        }
    }

    func handleFileImportURL(url selectedFile: URL) {
        // Check if .pth or .zip
        if selectedFile.pathExtension.lowercased() == "pth" || selectedFile.pathExtension.lowercased() == "zip" {
            alertTitle = "Import Failed"
            statusMessage = selectedFile.pathExtension.lowercased() == "zip" ? "Inspecting zip archive..." : "Converting model archive..."

            let accessSucceeded = selectedFile.startAccessingSecurityScopedResource()
            if !accessSucceeded {
                statusMessage = "Access denied"
                return
            }

            Task {
                defer { selectedFile.stopAccessingSecurityScopedResource() }

                await MainActor.run {
                    isConverting = true
                    conversionProgress = 0.0
                }

                do {
                    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                    let arrays = try PthConverter.shared.convert(url: selectedFile, copyIndexTo: docs) { progress, msg in
                        Task { @MainActor in
                            self.conversionProgress = progress
                            self.statusMessage = msg
                        }
                    }

                    let name = selectedFile.deletingPathExtension().lastPathComponent
                    let dest = docs.appendingPathComponent(name + ".safetensors")

                    try MLX.save(arrays: arrays, url: dest)

                    await MainActor.run {
                        statusMessage = "Converted & Imported: \(name)"
                        log("Converted model saved to \(dest.path)")
                        isConverting = false
                        conversionProgress = 1.0
                        refreshImportedModels()
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
            return
        }

        if selectedFile.startAccessingSecurityScopedResource() {
            defer { selectedFile.stopAccessingSecurityScopedResource() }

            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let dest = docs.appendingPathComponent(selectedFile.lastPathComponent)

            if FileManager.default.fileExists(atPath: dest.path) {
                try? FileManager.default.removeItem(at: dest)
            }

            do {
                try FileManager.default.copyItem(at: selectedFile, to: dest)
                statusMessage = "Imported: \(selectedFile.lastPathComponent)"
                log("Imported model to \(dest.path)")
                refreshImportedModels()
            } catch {
                statusMessage = "Error: \(error.localizedDescription)"
                log("Import error: \(error)")
            }
        } else {
            statusMessage = "Access denied"
        }
    }

    func handleAudioFileImportURL(url selectedFile: URL) {
        if selectedFile.startAccessingSecurityScopedResource() {
            defer { selectedFile.stopAccessingSecurityScopedResource() }

            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let dest = docs.appendingPathComponent(selectedFile.lastPathComponent)

            if FileManager.default.fileExists(atPath: dest.path) {
                try? FileManager.default.removeItem(at: dest)
            }

            do {
                try FileManager.default.copyItem(at: selectedFile, to: dest)
                self.inputURL = dest
                statusMessage = "Loaded: \(selectedFile.lastPathComponent)"
                log("Loaded audio file: \(dest.path)")
                originalWaveform = AudioWaveformExtractor.extractSamples(from: dest)
            } catch {
                statusMessage = "Error: \(error.localizedDescription)"
                log("Audio import error: \(error)")
            }
        } else {
            statusMessage = "Access denied"
        }
    }

    func log(_ message: String) {
        print("DEBUG: \(message)") // Keep in console
        DispatchQueue.main.async {
            self.logs.append(message)
        }
    }
    
    func refreshImportedModels() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        guard let files = try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil) else { 
            self.importedModels = []
            return 
        }
        self.importedModels = files
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
        
        // Find Index File (optional)
        // 1. Check for native .index (from .pth conversion)
        // 2. Check for .safetensors (converted)
        // Try exact name match or "added_IVF256_Flat_nprobe_1_v2.index" common pattern
        var indexUrl: URL?
        
        if isImported {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            
            // Search for *any* index file that might have been extracted with this model
            // For now, simpler: check keys.
            let possibleNames = [
                "\(name).index",
                "\(name).safetensors.index", // weird but possible
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
            
            // If explicit name fails, try searching directory for any .index file if imported recently?
            // No, that's risky. stick to name match.
            // If PthConverter extracted it, it kept original name.
            // PthConverter should maybe RENAME it.
            // But since we didn't rename it in PthConverter, we might miss it if it has a weird name.
            // Let's rely on user renaming or standard names for now.
        }
        
        if let idx = indexUrl {
            log("Found index file at \(idx.lastPathComponent)")
        } else {
            log("No index file found for \(name)")
        }
        
        Task {
            do {
                log("Starting loadWeights task...")
                try await inferenceEngine.loadWeights(hubertURL: hubertURL, modelURL: url, rmvpeURL: rmvpeURL)
                log("loadWeights success")
                
                if let idx = indexUrl {
                    log("Loading index...")
                    try inferenceEngine.loadIndex(url: idx)
                    log("Index loaded successfully!")
                } else {
                    inferenceEngine.unloadIndex() // Ensure cleared
                }
                
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
        
        defer {
            isProcessing = false
            log("State reset: isProcessing = false")
        }

        // Extract original waveform before conversion
        let originalW = AudioWaveformExtractor.extractSamples(from: input)
        log("Extracted \(originalW.count) samples from original audio")
        originalWaveform = originalW

        // Create unique output file
        let timestamp = Int(Date().timeIntervalSince1970)
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let output = documents.appendingPathComponent("Result_\(timestamp).wav")
        
        do {
            await inferenceEngine.infer(
                audioURL: input, 
                outputURL: output, 
                volumeEnvelope: volumeEnvelope
            )
            
            log("Inference complete. Output saved to: \(output.lastPathComponent)")
            
            // Extract converted waveform
            let convertedW = AudioWaveformExtractor.extractSamples(from: output)
            log("Extracted \(convertedW.count) samples from converted audio")
            convertedWaveform = convertedW
            outputURL = output
            
            // Save to history
            let result = ConversionResult(
                date: Date(),
                modelName: selectedModel,
                sourceAudioName: input.lastPathComponent,
                inputURL: input,
                outputURL: output,
                originalWaveform: originalW,
                convertedWaveform: convertedW
            )
            
            withAnimation {
                conversionHistory.insert(result, at: 0)
                selectedResult = result
                selectedTab = .results // Switch to results to show progress
            }
            
            statusMessage = "Inference successful!"
        } catch {
            log("Inference failed: \(error.localizedDescription)")
            statusMessage = "Conversion failed."
            alertTitle = "Inference Failed"
            alertMessage = error.localizedDescription
            showAlert = true
        }
    }
    
    func deleteResult(_ result: ConversionResult) {
        if let index = conversionHistory.firstIndex(where: { $0.id == result.id }) {
            withAnimation {
                conversionHistory.remove(at: index)
                if selectedResult?.id == result.id {
                    selectedResult = conversionHistory.first
                }
            }
            
            // Delete actual file
            try? FileManager.default.removeItem(at: result.outputURL)
            log("Deleted result and file: \(result.outputURL.lastPathComponent)")
        }
    }
    
    func clearAllResults() {
        log("Clearing all session results...")
        for result in conversionHistory {
            try? FileManager.default.removeItem(at: result.outputURL)
        }
        
        withAnimation {
            conversionHistory.removeAll()
            selectedResult = nil
        }
        log("Session cleared.")
    }
}

struct ModelManagerView: View {
    @Binding var isPresented: Bool
    @State private var models: [String] = []
    var onModelDeleted: (String) -> Void
    
    @AppStorage("accentTheme") private var accentTheme: ContentView.AccentTheme = .system
    @Environment(\.colorScheme) private var colorScheme
    
    private var accentColor: Color {
        accentTheme.color(for: colorScheme)
    }

    var body: some View {
        NavigationStack {
            ZStack {
                // Adaptive Background
                ZStack {
                    Color(uiColor: .systemBackground)
                    LinearGradient(
                        colors: [
                            accentColor.opacity(0.1),
                            Color.blue.opacity(0.15),
                            Color.purple.opacity(0.1)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                }
                .ignoresSafeArea()
                
                GlassEffectContainer {
                    if models.isEmpty {
                        VStack {
                            Image(systemName: "cube.transparent")
                                .font(.system(size: 50))
                                .foregroundStyle(.secondary)
                            Text("No imported models")
                                .foregroundStyle(.primary.opacity(0.7))
                        }
                    } else {
                        List {
                            ForEach(models, id: \.self) { model in
                                Text(model)
                                    .foregroundStyle(.primary)
                                    .listRowBackground(Color.clear)
                                    .listRowSeparatorTint(.primary.opacity(0.2))
                            }
                            .onDelete(perform: deleteModel)
                        }
                        .scrollContentBackground(.hidden) // Remove default List background
                    }
                }
            }
            .navigationTitle("Manage Models")
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { isPresented = false }
                        .foregroundStyle(accentColor)
                }
            }
            .onAppear(perform: loadModels)
        }
    }
    
    func loadModels() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        guard let files = try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil) else { return }
        
        models = files
            .filter { $0.pathExtension == "safetensors" || $0.pathExtension == "npz" }
            .map { $0.deletingPathExtension().lastPathComponent }
            .sorted()
    }
    
    func deleteModel(at offsets: IndexSet) {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        
        for index in offsets {
            let modelName = models[index]
            
            // Try deleting .safetensors and .npz
            let safe = docs.appendingPathComponent("\(modelName).safetensors")
            let npz = docs.appendingPathComponent("\(modelName).npz")
            
            try? FileManager.default.removeItem(at: safe)
            try? FileManager.default.removeItem(at: npz)
            
            onModelDeleted(modelName)
        }
        
        models.remove(atOffsets: offsets)
    }
}

// MARK: - Components

// MARK: - Components

struct ModelTile: View {
    let name: String
    let isSelected: Bool
    let isImported: Bool
    let isEditMode: Bool
    let action: () -> Void
    let onDelete: () -> Void
    
    @AppStorage("accentTheme") private var accentTheme: ContentView.AccentTheme = .system
    @Environment(\.colorScheme) private var colorScheme
    
    private var accentColor: Color {
        accentTheme.color(for: colorScheme)
    }
    
    var body: some View {
        ZStack(alignment: .topTrailing) {
            Button(action: {
                if !isEditMode {
                    action()
                }
            }) {
                VStack(spacing: 12) {
                    ZStack {
                        Circle()
                            .fill(isSelected ? accentColor.opacity(0.15) : Color.primary.opacity(0.04))
                            .frame(width: 50, height: 50)
                        
                        Image(systemName: isImported ? "person.crop.circle.badge.plus" : "person.crop.circle.fill")
                            .font(.system(size: 24))
                            .foregroundStyle(isSelected ? accentColor : .primary.opacity(0.7))
                    }
                    
                    Text(name)
                        .font(.caption)
                        .bold()
                        .lineLimit(1)
                        .foregroundStyle(isSelected ? accentColor : .primary)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 120)
                .glassEffect(in: RoundedRectangle(cornerRadius: 16))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(isSelected ? accentColor.opacity(0.5) : Color.clear, lineWidth: 2)
                )
            }
            .buttonStyle(.plain)
            
            if isEditMode && isImported {
                Button(action: {
                    withAnimation {
                        onDelete()
                    }
                }) {
                    Image(systemName: "minus.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.red)
                        .background(Circle().fill(.white))
                }
                .padding(8)
                .transition(.scale.combined(with: .opacity))
            }
        }
    }
}

struct HistoryRow: View {
    let result: ContentView.ConversionResult
    let isSelected: Bool
    let action: () -> Void
    let onDelete: () -> Void
    
    @AppStorage("accentTheme") private var accentTheme: ContentView.AccentTheme = .system
    @Environment(\.colorScheme) private var colorScheme
    
    private var accentColor: Color {
        accentTheme.color(for: colorScheme)
    }
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                // Mini Waveform Icon
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(isSelected ? accentColor.opacity(0.1) : Color.primary.opacity(0.02))
                        .frame(width: 44, height: 44)
                        .glassEffect(in: RoundedRectangle(cornerRadius: 8))
                    
                    Image(systemName: "waveform")
                        .foregroundStyle(isSelected ? .cyan : .white.opacity(0.6))
                }
                
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 4) {
                        Text(result.sourceAudioName)
                            .lineLimit(1)
                        Image(systemName: "arrow.right")
                            .font(.caption2)
                        Text(result.modelName)
                            .lineLimit(1)
                    }
                    .font(.subheadline)
                    .bold()
                    .foregroundStyle(isSelected ? accentColor : .primary)
                    Text(result.date, style: .time)
                        .font(.caption2)
                        .foregroundStyle(.secondary.opacity(0.6))
                }
                
                Spacer()
                
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.caption)
                        .foregroundStyle(.red.opacity(0.6))
                        .padding(8)
                }
                .buttonStyle(.borderless)
                
                if isSelected {
                    Image(systemName: "chevron.right")
                        .font(.caption2)
                        .foregroundStyle(accentColor)
                }
            }
            .padding(10)
            .background(isSelected ? Color.primary.opacity(0.05) : Color.clear)
            .cornerRadius(12)
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button(role: .destructive, action: onDelete) {
                Label("Delete Result", systemImage: "trash")
            }
        }
    }
}

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    
    @AppStorage("accentTheme") private var accentTheme: ContentView.AccentTheme = .system
    @AppStorage("isHighPrecision") private var isHighPrecision: Bool = true
    @AppStorage("isExperimental") private var isExperimental: Bool = false
    @AppStorage("autoCleanup") private var autoCleanup: Bool = false
    @AppStorage("showDebugLogs") private var showDebugLogs: Bool = true

    @Environment(\.colorScheme) private var colorScheme
    
    private var accentColor: Color {
        accentTheme.color(for: colorScheme)
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                backgroundGradient
                
                ScrollView {
                    VStack(spacing: 24) {
                        // Profile/About Section
                        VStack(spacing: 16) {
                            ZStack {
                                Circle()
                                    .fill(accentColor.opacity(0.15))
                                    .frame(width: 80, height: 80)
                                Image(systemName: "person.fill")
                                    .font(.system(size: 40))
                                    .foregroundStyle(accentColor)
                            }
                            
                            VStack(spacing: 4) {
                                Text("RVC Native")
                                    .font(.title2.bold())
                                    .foregroundStyle(.primary)
                                Text("Version 1.0.0 (Build 2390)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.top, 40)
                        
                        // Theme Selection
                        settingsGroup(title: "Appearance") {
                            VStack(alignment: .leading, spacing: 16) {
                                Text("Accent Color")
                                    .font(.subheadline.bold())
                                    .foregroundStyle(.primary)
                                
                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 16) {
                                        ForEach(ContentView.AccentTheme.allCases) { theme in
                                            Button(action: { accentTheme = theme }) {
                                                VStack(spacing: 8) {
                                                    ZStack {
                                                        Circle()
                                                            .fill(theme.color(for: colorScheme))
                                                            .frame(width: 44, height: 44)
                                                        
                                                        if accentTheme == theme {
                                                            Image(systemName: "checkmark")
                                                                .font(.system(size: 20, weight: .bold))
                                                                .foregroundStyle(.white)
                                                        }
                                                    }
                                                    Text(theme.rawValue)
                                                        .font(.caption2)
                                                        .foregroundStyle(accentTheme == theme ? .primary : .secondary)
                                                }
                                            }
                                        }
                                    }
                                    .padding(.horizontal, 4)
                                }
                            }
                            .padding()
                        }
                        
                        // Settings Groups
                        VStack(spacing: 16) {
                            settingsGroup(title: "Inference Engine") {
                                settingsToggle(icon: "cpu", title: "High Precision (FP32)", isOn: $isHighPrecision)
                                settingsToggle(icon: "bolt.horizontal.fill", title: "Experimental Optimizations", isOn: $isExperimental)
                                if isExperimental {
                                    settingsRow(icon: "waveform.path.ecg", title: "Pitch Method", detail: "RMVPE")
                                }
                            }
                            
                            settingsGroup(title: "Storage & Logs") {
                                settingsToggle(icon: "clock.arrow.2.circlepath", title: "Auto-Cleanup History", isOn: $autoCleanup)
                                settingsToggle(icon: "terminal", title: "Show System Logs", isOn: $showDebugLogs)
                            }
                            
                            settingsGroup(title: "Links") {
                                Link(destination: URL(string: "https://github.com/Acelogic/Retrieval-based-Voice-Conversion-MLX/blob/main/Demos/iOS/RVCNative/PRIVACY_POLICY.md")!) {
                                    settingsRow(icon: "doc.text", title: "Privacy Policy", detail: "")
                                }
                                
                                Link(destination: URL(string: "https://github.com/Acelogic/Retrieval-based-Voice-Conversion-MLX")!) {
                                    settingsRow(icon: "link", title: "Source Code", detail: "GitHub")
                                }
                            }
                        }
                        .padding(.horizontal)
                        .padding(.bottom, 40)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                    .bold()
                    .foregroundStyle(accentColor)
                }
            }
        }
    }
    
    private func settingsGroup<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title.uppercased())
                .font(.caption2.bold())
                .tracking(1)
                .foregroundStyle(accentColor.opacity(0.8))
                .padding(.leading, 8)
            
            VStack(spacing: 0) {
                content()
            }
            .glassEffect(in: RoundedRectangle(cornerRadius: 16))
        }
    }
    
    private func settingsRow(icon: String, title: String, detail: String, color: Color? = nil) -> some View {
        HStack {
            Image(systemName: icon)
                .foregroundStyle(color ?? accentColor)
                .frame(width: 24)
            Text(title)
                .foregroundStyle(.primary)
            Spacer()
            Text(detail)
                .font(.caption)
                .foregroundStyle(.secondary)
            Image(systemName: "chevron.right")
                .font(.caption2)
                .foregroundStyle(.primary.opacity(0.5))
        }
        .padding()
        .contentShape(Rectangle())
    }
    
    private func settingsToggle(icon: String, title: String, isOn: Binding<Bool>) -> some View {
        HStack {
            Image(systemName: icon)
                .foregroundStyle(accentColor)
                .frame(width: 24)
            Toggle(title, isOn: isOn)
                .foregroundStyle(.primary)
                .tint(accentColor)
        }
        .padding()
    }
    
    // Use the same background gradient as the main view
    private var backgroundGradient: some View {
        ZStack {
            Color(uiColor: .systemBackground)
            
            LinearGradient(
                colors: [
                    accentColor.opacity(0.1),
                    Color.blue.opacity(0.15), 
                    Color.purple.opacity(0.1)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }
        .ignoresSafeArea()
    }
}

#Preview {
    ContentView()
}

// MARK: - Liquid Glass Design System
extension View {
    func glassEffect<S: Shape>(in shape: S) -> some View {
        self
            .background(.ultraThinMaterial)
            .clipShape(shape)
            .overlay(
                shape
                    .stroke(.primary.opacity(0.08), lineWidth: 0.5)
            )
            .shadow(color: .black.opacity(0.1), radius: 10, x: 0, y: 5)
    }
}

struct GlassEffectContainer<Content: View>: View {
    let content: Content
    
    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }
    
    var body: some View {
        content
            .padding()
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(.primary.opacity(0.08), lineWidth: 0.5)
            )
            .shadow(color: .black.opacity(Color.primary == .white ? 0.3 : 0.1), radius: 20, x: 0, y: 10)
    }
}
