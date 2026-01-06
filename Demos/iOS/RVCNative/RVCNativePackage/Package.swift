// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "RVCNativeFeature",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "RVCNativeFeature",
            type: .static,
            targets: ["RVCNativeFeature"]
        ),
    ],
    dependencies: [
        .package(path: "../Vendor/mlx-swift")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "LinkerFix",
            dependencies: []
        ),
        .target(
            name: "RVCNativeFeature",
            dependencies: [
                "LinkerFix",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
            ],
            resources: [
                .copy("Assets")
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Foundation")
            ]
        ),
        .testTarget(
            name: "RVCNativeFeatureTests",
            dependencies: [
                "RVCNativeFeature"
            ]
        ),
    ]
)
