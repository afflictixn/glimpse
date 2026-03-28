#!/bin/bash
# Build GlimpseOverlay and wrap it as a proper macOS .app bundle
# so it can request TCC permissions (Calendar, Contacts, etc.)

set -e
cd "$(dirname "$0")"

APP_NAME="GlimpseOverlay"
APP_DIR=".build/${APP_NAME}.app"
CONTENTS="${APP_DIR}/Contents"
MACOS="${CONTENTS}/MacOS"

echo "Building ${APP_NAME}..."
swift build

echo "Creating app bundle..."
rm -rf "${APP_DIR}"
mkdir -p "${MACOS}"

# Copy binary
cp ".build/debug/${APP_NAME}" "${MACOS}/${APP_NAME}"

# Create Info.plist with privacy usage descriptions
cat > "${CONTENTS}/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.glimpse.overlay</string>
    <key>CFBundleName</key>
    <string>GlimpseOverlay</string>
    <key>CFBundleExecutable</key>
    <string>GlimpseOverlay</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSCalendarsUsageDescription</key>
    <string>Glimpse needs calendar access to show your upcoming events and detect scheduling conflicts.</string>
    <key>NSContactsUsageDescription</key>
    <string>Glimpse needs contacts access to help you prepare for meetings and remember people.</string>
    <key>NSCalendarsFullAccessUsageDescription</key>
    <string>Glimpse needs full calendar access to show your upcoming events and detect scheduling conflicts.</string>
</dict>
</plist>
PLIST

echo "App bundle created at: ${APP_DIR}"
echo "Run with: open ${APP_DIR}"
