# Edge Glow GPU Optimization — What We Tried

## Current State
The edge glow uses SwiftUI `RoundedRectangle.stroke()` + `AngularGradient` with 4 blur layers.
It looks great but consumes 20-50% GPU during active animation (suggestions/warnings).
In idle mode, gradient updates are skipped so GPU drops to near zero.

## Problem
Full-screen blur is extremely expensive. Each `.blur(radius: N)` samples a ~2N×2N pixel
area for every pixel on a Retina screen (~5.2M pixels). With 4 layers at blur radii
[3, 8, 18, 28] plus an accent at blur 3, that's 5 blur passes per frame at 60fps.

## What We Tried

### 1. Reduce layers (3 instead of 4)
- **Result**: Noticeable visual quality loss. The 4-layer falloff [sharp → slight → medium → wide]
  creates realistic light bloom. 3 layers looks flatter.
- **GPU savings**: ~20% reduction. Not enough.

### 2. CoreAnimation CAGradientLayer blobs (Cindori FluidGradient approach)
- Used `CAGradientLayer` with `.radial` type positioned at screen edges
- `CASpringAnimation` runs on WindowServer — near zero app GPU
- **Problem 1**: Black screen — needed layer-hosting pattern (`self.layer = CALayer()` BEFORE
  `self.wantsLayer = true`). Fixed by following Cindori's exact setup.
- **Problem 2**: Radial gradient blobs look like visible circular spots/blotches, not a smooth
  continuous glow. The SwiftUI `RoundedRectangle.stroke()` creates a perfect continuous line
  around the perimeter. CAGradientLayer blobs are fundamentally different — discrete circles
  vs a continuous stroke. Making blobs bigger just makes bigger spots.
- **Result**: Visually broken. Reverted.

### 3. Skip animation in idle mode (KEPT)
- When mode is `.idle` or `.hidden`, stop updating gradient stops
- **Result**: GPU goes to near zero when just working. Good.
- **Limitation**: GPU still spikes during active suggestions (which is most of the time
  when the assistant is being useful).

## What Could Work (Not Yet Tried)

### A. Render at lower resolution
Render the glow into an offscreen buffer at 1/4 resolution, then scale up.
The blur would be ~16x cheaper. SwiftUI's `.drawingGroup()` partially does this
but doesn't reduce resolution.

### B. CALayer with stroke path (not blobs)
Instead of radial gradient blobs, render a `CGPath` rounded-rect stroke into a
`CAShapeLayer`, apply `CIGaussianBlur` as a `CALayer` filter, and use
`CAGradientLayer` as a fill. This would give the continuous stroke look but
with WindowServer-managed animation.

### C. Metal shader
Write a custom Metal fragment shader that computes the glow per-pixel.
Could be extremely efficient as it's a single draw call with no blur passes.
The shader would compute distance-to-edge and apply color/falloff analytically.

### D. Pre-rendered glow texture
Render the glow frames offline, store as image assets, crossfade between them.
Zero compute cost at runtime. Limited animation variety.
