#if os(iOS)
import Foundation
import AVFoundation
import SwiftUI

@MainActor
public class AudioPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published public var isPlaying: Bool = false
    @Published public var currentTime: Double = 0
    @Published public var duration: Double = 0
    
    private var audioPlayer: AVAudioPlayer?
    private var timer: Timer?
    
    public override init() {
        super.init()
    }
    
    public func play(url: URL) {
        do {
            // Ensure audio session is active and set to playback
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.delegate = self
            audioPlayer?.prepareToPlay()
            audioPlayer?.play()
            
            duration = audioPlayer?.duration ?? 0
            isPlaying = true
            startTimer()
        } catch {
            print("AudioPlayer error: \(error)")
        }
    }
    
    public func stop() {
        audioPlayer?.stop()
        isPlaying = false
        stopTimer()
    }
    
    public func seek(to time: Double) {
        audioPlayer?.currentTime = time
        currentTime = time
    }
    
    private func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self = self, let player = self.audioPlayer, player.isPlaying else { return }
            Task { @MainActor in
                self.currentTime = player.currentTime
            }
        }
    }
    
    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
    
    nonisolated public func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.isPlaying = false
            self.currentTime = 0
            self.stopTimer()
        }
    }
}
#endif
