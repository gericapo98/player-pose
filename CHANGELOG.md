# Changelog

## [Unreleased]

## [v1.0.1-beta.1](https://github.com/gericapo98/player-pose/releases/tag/v1.0.1-beta.1) — 2026-08-24 (pre-release)

Fixes from an independent code review of v1.0.0, each confirmed by an adversarial
verification pass before being accepted. Not yet promoted to stable — pending
further real-world validation on more clips, not because the fixes are in doubt.

### Fixed
- Pixel-buffer leak: every frame's buffer was retained for the process lifetime
  (~3.7 MB/frame); now released via an owned `NSData` copy.
- `player-pose-overlay` could silently overwrite its input video when the output
  path collided; it now defaults to `<video>_poses.mp4` and refuses to write onto
  its input.
- Re-detection could hand one player's pose to a neighbouring track; candidates
  for all missing players are now assigned jointly (Hungarian) instead of
  first-come-first-served.
- A partial skeleton inside an already-tracked player no longer spawns a
  duplicate track.
- Early tentative tracks no longer get confirmed prematurely; the output CSV
  path is validated before processing instead of after; frame counts come from
  frames actually decoded; the overlay palette no longer collides with the
  ball's red or the court's green.

## [v1.0.0](https://github.com/gericapo98/player-pose/releases/tag/v1.0.0) — 2026-08-24

Initial release: Apple Vision body-pose detection + a SORT tracker.

### Added
- Apple Vision `VNDetectHumanBodyPoseRequest` multi-person pose detection
  (Neural Engine, no model download).
- SORT tracker (Kalman filter + Hungarian/IoU matching) with re-detection for
  players who leave and re-enter frame.
- `player-pose-track` — video in, per-frame joint CSV out.
- `player-pose-overlay` — renders the tracked skeletons back onto the video.

### Known issues
Fixed in v1.0.1-beta.1: unbounded memory growth on long clips, an overlay
overwrite risk, wrong-track re-detection under crowding, and duplicate tracks.
