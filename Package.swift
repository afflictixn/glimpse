// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OverlayChat",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "OverlayChat",
            path: "Sources/OverlayChat",
            linkerSettings: [
                .linkedFramework("Carbon")
            ]
        )
    ]
)
