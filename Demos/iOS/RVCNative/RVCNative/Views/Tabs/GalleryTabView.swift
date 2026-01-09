import SwiftUI
import RVCNativeFeature

/// Gallery tab displaying voice model selection grid.
struct GalleryTabView: View {
    // State bindings from parent
    @Binding var selectedModel: String
    @Binding var isModelLoaded: Bool
    @Binding var importedModels: [String]
    @Binding var isEditMode: Bool
    @Binding var modelToDelete: String?
    @Binding var showDeleteDialog: Bool
    @Binding var activeImport: ContentView.ImportType?

    // Callbacks
    var onLoadModel: (String, Bool) -> Void
    var onRefreshModels: () -> Void

    // Theme
    @AppStorage("accentTheme") private var accentTheme: AccentTheme = .system
    @Environment(\.colorScheme) private var colorScheme

    private var accentColor: Color {
        accentTheme.color(for: colorScheme)
    }

    var body: some View {
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
            onRefreshModels()
        }
    }

    // MARK: - Header

    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Model Gallery")
                    .font(.title2)
                    .bold()
                Text("Select a voice model")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button(action: { withAnimation { isEditMode.toggle() } }) {
                Text(isEditMode ? "Done" : "Edit")
                    .font(.subheadline.bold())
                    .foregroundStyle(accentColor)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .glassEffect(in: Capsule())
            }
        }
    }

    // MARK: - Model Selection Status

    private var modelSelectionView: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(isModelLoaded ? Color.green.opacity(0.15) : Color.primary.opacity(0.05))
                    .frame(width: 50, height: 50)

                Image(systemName: isModelLoaded ? "checkmark.circle.fill" : "person.crop.circle")
                    .font(.system(size: 24))
                    .foregroundStyle(isModelLoaded ? .green : .primary.opacity(0.6))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(selectedModel)
                    .font(.headline)
                    .foregroundStyle(.primary)

                Text(isModelLoaded ? "Model loaded and ready" : "Tap a model below to load it")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding()
        .glassEffect(in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Model Gallery Grid

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
                    isEditMode: isEditMode,
                    action: { onLoadModel("Coder", false) },
                    onDelete: { }
                )

                ModelTile(
                    name: "Slim Shady",
                    isSelected: selectedModel == "Slim Shady",
                    isImported: false,
                    isEditMode: isEditMode,
                    action: { onLoadModel("Slim Shady", false) },
                    onDelete: { }
                )

                // Imported Models
                ForEach(importedModels, id: \.self) { modelName in
                    ModelTile(
                        name: modelName,
                        isSelected: selectedModel == modelName,
                        isImported: true,
                        isEditMode: isEditMode,
                        action: { onLoadModel(modelName, true) },
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

    // MARK: - Background

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
