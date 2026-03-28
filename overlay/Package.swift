// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ZExpOverlay",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "ZExpOverlay",
            path: "Sources/OverlayChat",
            linkerSettings: [
                .linkedFramework("Carbon"),
                .linkedFramework("Network"),
                .linkedFramework("EventKit"),
                .linkedLibrary("sqlite3")
            ]
        )
    ]
)
