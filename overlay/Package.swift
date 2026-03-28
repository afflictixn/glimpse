// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "GlimpseOverlay",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "GlimpseOverlay",
            path: "Sources/OverlayChat",
            linkerSettings: [
                .linkedFramework("Carbon"),
                .linkedFramework("Network")
            ]
        )
    ]
)
