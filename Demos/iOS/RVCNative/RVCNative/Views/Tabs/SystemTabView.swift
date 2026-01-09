import SwiftUI
import UIKit

/// System tab displaying console/terminal output logs.
struct SystemTabView: View {
    @ObservedObject var logRedirector: ConsoleLogRedirector
    let accentColor: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header with copy/clear buttons
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

            // Scrollable log content
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
}
