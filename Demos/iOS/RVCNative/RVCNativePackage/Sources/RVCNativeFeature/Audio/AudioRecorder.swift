#if os(iOS)
import Foundation
import AVFoundation

@MainActor
public class AudioRecorder: ObservableObject {
    @Published public var isRecording = false
    @Published public var recordingURL: URL?
    @Published public var permissionGranted = false
    
    private var audioRecorder: AVAudioRecorder?
    
    public init() {
        checkPermission()
    }
    
    private func checkPermission() {
        switch AVAudioApplication.shared.recordPermission {
        case .granted:
            permissionGranted = true
        case .denied:
            permissionGranted = false
        case .undetermined:
            AVAudioApplication.requestRecordPermission { allowed in
                Task { @MainActor in
                    self.permissionGranted = allowed
                }
            }
        @unknown default:
            permissionGranted = false
        }
    }
    
    public func startRecording() {
        let recordingSession = AVAudioSession.sharedInstance()
        
        do {
            try recordingSession.setCategory(.playAndRecord, mode: .default)
            try recordingSession.setActive(true)
            
            let docPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let audioFilename = docPath.appendingPathComponent("recorded_audio.wav")
            
            let settings: [String: Any] = [
                AVFormatIDKey: Int(kAudioFormatLinearPCM),
                AVSampleRateKey: 44100,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ]
            
            audioRecorder = try AVAudioRecorder(url: audioFilename, settings: settings)
            audioRecorder?.record()
            
            isRecording = true
            recordingURL = nil // Reset previous recording
            
        } catch {
            print("Failed to start recording: \(error)")
        }
    }
    
    public func stopRecording() {
        audioRecorder?.stop()
        isRecording = false
        if let url = audioRecorder?.url {
            recordingURL = url
        }
        audioRecorder = nil
    }
}
#endif
