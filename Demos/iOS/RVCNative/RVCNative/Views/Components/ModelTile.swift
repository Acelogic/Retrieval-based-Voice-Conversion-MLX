import SwiftUI

/// A tile displaying a voice model in the gallery grid.
struct ModelTile: View {
    let name: String
    let isSelected: Bool
    let isImported: Bool
    let isEditMode: Bool
    let action: () -> Void
    let onDelete: () -> Void

    @AppStorage("accentTheme") private var accentTheme: AccentTheme = .system
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
