import Foundation
import SwiftUI

/// Represents a completed voice conversion result with associated audio files and waveform data.
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

/// Navigation tabs for the main interface.
enum Tab {
    case gallery
    case lab
    case results
    case system
}

/// Accent color theme options for the app.
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
