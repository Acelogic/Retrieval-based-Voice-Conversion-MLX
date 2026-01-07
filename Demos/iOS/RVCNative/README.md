# RVCNative - iOS App

A native iOS implementation of **Retrieval-based Voice Conversion (RVC)** powered by **MLX Swift**. This app demonstrates real-time voice conversion using machine learning models running entirely on-device using Apple Silicon GPU acceleration.

Built with a **workspace + SPM package** architecture for clean separation between app shell and feature code.

## Features

- **On-Device Voice Conversion**: Real-time RVC inference using MLX Swift with Metal GPU acceleration
- **Multiple Models**: Support for multiple voice conversion models (Coder, Slim Shady, etc.)
- **Audio Input Options**:
  - Import audio files from Files app
  - Record audio directly in-app using device microphone
- **Custom Model Import**:
  - **Native .pth Conversion**: Convert standard PyTorch RVC models directly on-device
  - **Zip Support**: Import .zip archives containing models
  - **Auto-Conversion**: Automatically converts to MLX-optimized `.safetensors` format
- **Playback Controls**: Play original and converted audio with visual feedback
- **Waveform Visualization**: Real-time comparison of original vs converted audio waveforms
- **Model Components**:
  - **HuBERT**: Feature extraction with transformer encoder
  - **RMVPE**: High-quality pitch detection and extraction
  - **NSF-HiFiGAN Generator**: Neural source-filter vocoder for audio synthesis
  - **Flow-based Synthesizer**: Residual coupling blocks for voice conversion

## Implementation Status

### Completed
- ✅ Full RVC pipeline implementation in Swift
- ✅ HuBERT encoder with GELU activation
- ✅ RMVPE pitch extraction with F0 decoding
- ✅ Generator with native ConvTransposed1d upsampling
- ✅ Weight loading with PyTorch → Swift key remapping
- ✅ Channels-last (MLX) format handling
- ✅ Waveform visualization UI
- ✅ Audio recording and playback
- ✅ Model management and switching
- ✅ Native .pth -> .safetensors converter (Pure Swift)
- ✅ Volume Envelope (RMS mixing) for noise reduction
- ✅ Nested Zip file handling for imports

### Architecture Highlights

**MLX Swift Integration**: The implementation leverages MLX Swift's native operations for optimal performance on Apple Silicon:
- Native `ConvTransposed1d` for upsampling layers (10x, 8x, 2x, 2x)
- Efficient channels-last tensor format (B, T, C)
- Metal GPU acceleration for all operations
- Proper weight format handling: `[outputChannels, kernelSize, inputChannels]`

**Weight Mapping**: Runtime key remapping handles PyTorch → Swift module structure differences:
```swift
// dec.ups.0 → dec.up_0 (array → named properties)
// dec.resblocks.X.convs1.Y → dec.resblock_X.c1_Y
// flow.flows.{0,2,4,6} → flow.flows.{0,1,2,3} (index remapping)
```

**Critical Format Fixes**:
- RMVPE F0 decoding: `440 * 2^((cents-4080)/1200)`
- HuBERT GELU activation: `gelu(x, approximate: .none)` (no approximation)
- Generator input transpose: `(B, C, T) → (B, T, C)` for MLX format
- Flow reverse pass with proper mask handling

## AI Assistant Rules Files

This template includes **opinionated rules files** for popular AI coding assistants. These files establish coding standards, architectural patterns, and best practices for modern iOS development using the latest APIs and Swift features.

### Included Rules Files
- **Claude Code**: `CLAUDE.md` - Claude Code rules
- **Cursor**: `.cursor/*.mdc` - Cursor-specific rules
- **GitHub Copilot**: `.github/copilot-instructions.md` - GitHub Copilot rules

### Customization Options
These rules files are **starting points** - feel free to:
- ✅ **Edit them** to match your team's coding standards
- ✅ **Delete them** if you prefer different approaches
- ✅ **Add your own** rules for other AI tools
- ✅ **Update them** as new iOS APIs become available

### What Makes These Rules Opinionated
- **No ViewModels**: Embraces pure SwiftUI state management patterns
- **Swift 6+ Concurrency**: Enforces modern async/await over legacy patterns
- **Latest APIs**: Recommends iOS 18+ features with optional iOS 26 guidelines
- **Testing First**: Promotes Swift Testing framework over XCTest
- **Performance Focus**: Emphasizes @Observable over @Published for better performance

**Note for AI assistants**: You MUST read the relevant rules files before making changes to ensure consistency with project standards.

## Project Architecture

```
RVCNative/
├── RVCNative.xcworkspace/              # Open this file in Xcode
├── RVCNative.xcodeproj/                # App shell project
├── RVCNative/                          # App target (minimal)
│   ├── Assets.xcassets/                # App-level assets (icons, colors)
│   ├── RVCNativeApp.swift              # App entry point
│   └── RVCNative.xctestplan            # Test configuration
├── RVCNativePackage/                   # 🚀 Primary development area
│   ├── Package.swift                   # Package configuration
│   ├── Sources/RVCNativeFeature/       # Your feature code
│   └── Tests/RVCNativeFeatureTests/    # Unit tests
└── RVCNativeUITests/                   # UI automation tests
```

## Key Architecture Points

### Workspace + SPM Structure
- **App Shell**: `RVCNative/` contains minimal app lifecycle code
- **Feature Code**: `RVCNativePackage/Sources/RVCNativeFeature/` is where most development happens
- **Separation**: Business logic lives in the SPM package, app target just imports and displays it

### Buildable Folders (Xcode 16)
- Files added to the filesystem automatically appear in Xcode
- No need to manually add files to project targets
- Reduces project file conflicts in teams

## Development Notes

### Code Organization
Most development happens in `RVCNativePackage/Sources/RVCNativeFeature/` - organize your code as you prefer.

### Public API Requirements
Types exposed to the app target need `public` access:
```swift
public struct NewView: View {
    public init() {}
    
    public var body: some View {
        // Your view code
    }
}
```

### Adding Dependencies
Edit `RVCNativePackage/Package.swift` to add SPM dependencies:
```swift
dependencies: [
    .package(url: "https://github.com/example/SomePackage", from: "1.0.0")
],
targets: [
    .target(
        name: "RVCNativeFeature",
        dependencies: ["SomePackage"]
    ),
]
```

### Test Structure
- **Unit Tests**: `RVCNativePackage/Tests/RVCNativeFeatureTests/` (Swift Testing framework)
- **UI Tests**: `RVCNativeUITests/` (XCUITest framework)
- **Test Plan**: `RVCNative.xctestplan` coordinates all tests

## Configuration

### XCConfig Build Settings
Build settings are managed through **XCConfig files** in `Config/`:
- `Config/Shared.xcconfig` - Common settings (bundle ID, versions, deployment target)
- `Config/Debug.xcconfig` - Debug-specific settings  
- `Config/Release.xcconfig` - Release-specific settings
- `Config/Tests.xcconfig` - Test-specific settings

### Entitlements Management
App capabilities are managed through a **declarative entitlements file**:
- `Config/RVCNative.entitlements` - All app entitlements and capabilities
- AI agents can safely edit this XML file to add HealthKit, CloudKit, Push Notifications, etc.
- No need to modify complex Xcode project files

### Asset Management
- **App-Level Assets**: `RVCNative/Assets.xcassets/` (app icon, accent color)
- **Feature Assets**: Add `Resources/` folder to SPM package if needed

### SPM Package Resources
To include assets in your feature package:
```swift
.target(
    name: "RVCNativeFeature",
    dependencies: [],
    resources: [.process("Resources")]
)
```

### Generated with XcodeBuildMCP
This project was scaffolded using [XcodeBuildMCP](https://github.com/cameroncooke/XcodeBuildMCP), which provides tools for AI-assisted iOS development workflows.