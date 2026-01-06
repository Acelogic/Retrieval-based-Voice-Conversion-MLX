#if os(iOS)
import Foundation
import AVFoundation
import SwiftUI

@MainActor
public class AudioPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published public var isPlaying: Bool = false
    
    private var audioPlayer: AVAudioPlayer?
    
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
            
            isPlaying = true
        } catch {
            print("AudioPlayer error: \(error)")
        }
    }
    
    public func stop() {
        audioPlayer?.stop()
        isPlaying = false
    }
    
    nonisolated public func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.isPlaying = false
        }
    }
}
#endif
