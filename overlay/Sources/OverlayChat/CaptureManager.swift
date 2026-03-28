import Foundation
import AppKit

/// Captures screenshots on trigger events and POSTs them to the Python backend's /capture/ingest endpoint.
/// Replaces Python's CaptureLoop — all macOS permission-gated calls now live in Swift.
final class CaptureManager {
    private let captureService = ScreenCaptureService()
    private let focusedAppService = FocusedAppService()
    private let backendURL: URL
    private let minCaptureIntervalMs: Double
    private let backgroundQueue = DispatchQueue(label: "com.zexp.capture", qos: .utility)

    private var lastCaptureTime: TimeInterval = 0
    private let lock = NSLock()
    private var inFlight = false

    init(
        backendBaseURL: String = "http://localhost:3030",
        minCaptureIntervalMs: Double = 2000
    ) {
        self.backendURL = URL(string: "\(backendBaseURL)/capture/ingest")!
        self.minCaptureIntervalMs = minCaptureIntervalMs
    }

    /// Called by EventMonitor when a trigger fires.
    func handleTrigger(_ trigger: CaptureTrigger) {
        backgroundQueue.async { [weak self] in
            self?.captureAndSend(trigger: trigger)
        }
    }

    private func captureAndSend(trigger: CaptureTrigger) {
        // Enforce minimum capture interval
        lock.lock()
        let now = ProcessInfo.processInfo.systemUptime
        let elapsedMs = (now - lastCaptureTime) * 1000
        guard elapsedMs >= minCaptureIntervalMs else {
            lock.unlock()
            return
        }
        guard !inFlight else {
            lock.unlock()
            return
        }
        inFlight = true
        lastCaptureTime = now
        lock.unlock()

        defer {
            lock.lock()
            inFlight = false
            lock.unlock()
        }

        // Capture screenshot
        guard let imageData = captureService.captureScreen() else {
            print("[zexp] CaptureManager: screenshot failed")
            return
        }

        // Get focused app info
        let (appName, windowTitle) = focusedAppService.getFocusedApp()

        // Build multipart form request
        let boundary = UUID().uuidString
        var request = URLRequest(url: backendURL)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30

        var body = Data()

        // image field
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"frame.jpg\"\r\n")
        body.append("Content-Type: image/jpeg\r\n\r\n")
        body.append(imageData)
        body.append("\r\n")

        // app_name field
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"app_name\"\r\n\r\n")
        body.append(appName)
        body.append("\r\n")

        // window_name field
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"window_name\"\r\n\r\n")
        body.append(windowTitle)
        body.append("\r\n")

        // trigger field
        body.append("--\(boundary)\r\n")
        body.append("Content-Disposition: form-data; name=\"trigger\"\r\n\r\n")
        body.append(trigger.rawValue)
        body.append("\r\n")

        body.append("--\(boundary)--\r\n")

        request.httpBody = body

        // Send synchronously on background queue (simpler, avoids callback complexity)
        let semaphore = DispatchSemaphore(value: 0)
        var responseError: Error?

        let task = URLSession.shared.dataTask(with: request) { _, response, error in
            if let error = error {
                responseError = error
            } else if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                print("[zexp] CaptureManager: ingest returned \(http.statusCode)")
            }
            semaphore.signal()
        }
        task.resume()
        semaphore.wait()

        if let error = responseError {
            print("[zexp] CaptureManager: send failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - Data helpers

private extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}
