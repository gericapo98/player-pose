# player-pose

[![Latest release](https://img.shields.io/github/v/release/gericapo98/player-pose)](https://github.com/gericapo98/player-pose/releases/latest)
[![Pre-release](https://img.shields.io/github/v/release/gericapo98/player-pose?include_prereleases&label=pre-release)](https://github.com/gericapo98/player-pose/releases)

Track player poses in sports video on Apple Silicon. Uses Apple Vision's built-in
multi-person body pose (runs on the Neural Engine, no model files to download) and a
SORT tracker so each player keeps a stable ID across frames. Writes a per-frame CSV
and renders an overlay video so you can watch the tracking.

Built to sit next to [tracknet-coreml](https://github.com/gericapo98/tracknet-coreml):
same clip in, one CSV per extractor, joined on frame index downstream.

## Quickstart

```bash
uv sync

.venv/bin/player-pose-track rally.mp4                                  # writes rally_poses.csv
.venv/bin/player-pose-overlay rally.mp4 rally_poses.csv                # writes rally_poses.mp4
.venv/bin/player-pose-overlay rally.mp4 rally_poses.csv --ball rally_ball.csv   # + shuttle from tracknet-track
```

```python
from player_pose.infer import track_video
from player_pose.pose import VisionPoseDetector
from player_pose.sort import SORTTracker

poses = track_video("rally.mp4", VisionPoseDetector(), SORTTracker(), min_height=0.1)
# list of TrackedPose(track_id, detection) with detection.joints (19, 2) px and .confidence (19,)
```

## Output

`rally_poses.csv` is long format, one row per (frame, player, joint), 19 joints per player:

```
frame,track_id,joint,x,y,confidence
48,3,nose,486.12,375.40,0.9012
48,3,left_eye,,,0
```

Joints are Apple Vision's 19 body points (`nose, eyes, ears, neck, shoulders, elbows,
wrists, root, hips, knees, ankles`), pixel coordinates with origin top-left. Empty `x,y`
means Vision did not find that joint. Track IDs are stable for as long as a player is
followed; a camera cut starts new IDs.

## How it works

1. **Pose** — every frame goes through `VNDetectHumanBodyPoseRequest` as-is (no re-encode).
2. **Track** — SORT: a constant-velocity Kalman filter per player plus Hungarian matching
   on box overlap. Tracks start tentative and are matched after confirmed ones, so a
   one-frame partial detection can't steal a real player's identity.
3. **Re-detect** — Vision shrinks the whole frame to its network input, so the far
   player on a court can be too small to find. When a tracked player goes missing, the
   frame is cropped square around where they were last seen and Vision runs again on
   the crop; candidates must have enough joints and a plausible height to be accepted.

## Options worth knowing

| flag | default | what it does |
|---|---|---|
| `--min-height` | `0.1` | ignore people shorter than this fraction of the frame height (officials, spectators) |
| `--min-joints` | `8` | drop skeletons with fewer confident joints (partial detections) |
| `--max-age` | `30` | frames a player survives unmatched before their ID is retired |
| `--no-redetect` | | plain Vision + SORT, no crop pass |
| `--no-track` | | raw Vision output, `track_id` is just the index within the frame |

## Limitations

- macOS only (Apple Vision via pyobjc), tested on macOS 26 / Apple Silicon.
- Vision's pose model is general-purpose: expect occasional joint dropouts on deep
  lunges and heavy motion blur. Confidences are in the CSV so you can filter.
- Tracking is motion-only (no appearance model): fine for 2–4 players on a fixed
  camera; broadcast footage with moving cameras or many same-kit players would need
  camera-motion compensation and re-identification on top.

## Releases

[v1.0.0](https://github.com/gericapo98/player-pose/releases/tag/v1.0.0) is the
stable release. [v1.0.1-beta.1](https://github.com/gericapo98/player-pose/releases/tag/v1.0.1-beta.1)
carries bug fixes (a memory leak, an overlay-overwrite risk, and two tracking
correctness fixes — see [CHANGELOG.md](CHANGELOG.md)) found by review after
v1.0.0 shipped; it's marked pre-release pending further validation on more
clips, not because the fixes are in doubt. `master` always has the latest work.

## License

MIT.
