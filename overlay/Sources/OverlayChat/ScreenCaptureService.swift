import AppKit
import CoreGraphics

final class ScreenCaptureService {
    /// Captures the main display as JPEG data, excluding our own app's windows.
    func captureScreen(excludingWindowID: CGWindowID? = nil) -> Data? {
        guard let cgImage = CGWindowListCreateImage(
            CGRect.null,
            .optionOnScreenOnly,
            kCGNullWindowID,
            [.boundsIgnoreFraming, .nominalResolution]
        ) else {
            return nil
        }

        let bitmap = NSBitmapImageRep(cgImage: cgImage)
        // Use JPEG with reasonable quality to keep payload manageable for the model
        return bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.7])
    }
}
