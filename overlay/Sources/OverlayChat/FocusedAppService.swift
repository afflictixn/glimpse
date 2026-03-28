import AppKit
import ApplicationServices

/// Retrieves the currently focused app name and window title using NSWorkspace + Accessibility APIs.
final class FocusedAppService {
    func getFocusedApp() -> (appName: String, windowTitle: String) {
        let frontApp = NSWorkspace.shared.frontmostApplication
        let appName = frontApp?.localizedName ?? ""

        var windowTitle = ""
        if let pid = frontApp?.processIdentifier, pid != 0 {
            let appRef = AXUIElementCreateApplication(pid)
            var focusedWindow: AnyObject?
            let err = AXUIElementCopyAttributeValue(appRef, kAXFocusedWindowAttribute as CFString, &focusedWindow)
            if err == .success, let window = focusedWindow {
                var titleValue: AnyObject?
                let titleErr = AXUIElementCopyAttributeValue(window as! AXUIElement, kAXTitleAttribute as CFString, &titleValue)
                if titleErr == .success, let title = titleValue as? String {
                    windowTitle = title
                }
            }
        }

        return (appName, windowTitle)
    }
}
