import SwiftUI

// MARK: - Liquid Glass Design System

extension View {
    /// Applies a glass-like effect with material blur and subtle stroke.
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

/// Container view with glass effect styling.
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
