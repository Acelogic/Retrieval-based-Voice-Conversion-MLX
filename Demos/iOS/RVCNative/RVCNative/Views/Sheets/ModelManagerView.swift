import SwiftUI

/// Sheet for managing imported voice models.
struct ModelManagerView: View {
    @Binding var isPresented: Bool
    @State private var models: [String] = []
    var onModelDeleted: (String) -> Void

    @AppStorage("accentTheme") private var accentTheme: AccentTheme = .system
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
                        .scrollContentBackground(.hidden)
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
