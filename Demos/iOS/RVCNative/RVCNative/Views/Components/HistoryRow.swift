import SwiftUI

/// A row displaying a conversion history entry in the results list.
struct HistoryRow: View {
    let result: ConversionResult
    let isSelected: Bool
    let action: () -> Void
    let onDelete: () -> Void

    @AppStorage("accentTheme") private var accentTheme: AccentTheme = .system
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
