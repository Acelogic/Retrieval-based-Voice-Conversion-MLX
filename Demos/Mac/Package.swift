// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RVCNativeMac",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../iOS/RVCNative/RVCNativePackage")
    ],
    targets: [
        .executableTarget(
            name: "RVCNativeMac",
            dependencies: [
                .product(name: "RVCNativeFeature", package: "RVCNativePackage")
            ]
        )
    ]
)
