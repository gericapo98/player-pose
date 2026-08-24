"""Pipeline: video → per-frame pose detection → SORT → rows for CSV.

CSV is long format, one row per (frame, track, joint), 19 joint rows per person:
frame,track_id,joint,x,y,confidence  — x,y are empty when the joint was not found.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from player_pose.pose import JOINT_NAMES, PoseDetection, VisionPoseDetector
from player_pose.sort import SORTTracker, TrackedPose
from player_pose.video import iter_frames, video_info


def track_video(
    video_file: str | Path,
    detector: VisionPoseDetector,
    tracker: SORTTracker | None,
    redetect: bool = True,
    min_height: float = 0.0,
    progress: bool = True,
) -> list[TrackedPose]:
    """Run detection (and tracking, unless `tracker` is None) over every frame.

    With tracker=None the track_id is just the detection's index within its
    frame — raw Vision output, useful for inspecting the detector on its own.
    With redetect=True, a track that Vision missed this frame is looked for again
    in a crop around where the tracker last saw it. People whose box is shorter
    than `min_height` × frame height are ignored (spectators, officials).
    """
    info = video_info(video_file)
    num_frames = info["num_frames"]
    min_px = min_height * info["height"]

    def tall_enough(dets: list[PoseDetection]) -> list[PoseDetection]:
        return [d for d in dets if d.box[3] - d.box[1] >= min_px]

    poses: list[TrackedPose] = []
    for idx, frame in iter_frames(video_file):
        detections = tall_enough(detector.detect(frame, idx))
        if tracker is None:
            tracked = [TrackedPose(track_id=i, detection=d) for i, d in enumerate(detections)]
        else:
            redetect_fn = (lambda box: tall_enough(detector.detect_roi(frame, idx, box))) if redetect else None
            tracked = tracker.update(detections, redetect=redetect_fn)
        poses.extend(tracked)
        if progress and (idx % 50 == 0 or idx == num_frames - 1):
            print(f"\r{idx + 1}/{num_frames} frames", end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    return poses


def write_csv(poses: list[TrackedPose], out_path: str | Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "joint", "x", "y", "confidence"])
        for pose in poses:
            d = pose.detection
            for name, (x, y), conf in zip(JOINT_NAMES, d.joints, d.confidence):
                if conf > 0:
                    writer.writerow([d.frame, pose.track_id, name, f"{x:.2f}", f"{y:.2f}", f"{conf:.4f}"])
                else:
                    writer.writerow([d.frame, pose.track_id, name, "", "", "0"])


def read_csv(csv_path: str | Path) -> list[TrackedPose]:
    """Inverse of write_csv; boxes and scores are recomputed from the joints."""
    joint_index = {name: i for i, name in enumerate(JOINT_NAMES)}
    grouped: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = defaultdict(
        lambda: (np.full((len(JOINT_NAMES), 2), np.nan, dtype=np.float32),
                 np.zeros(len(JOINT_NAMES), dtype=np.float32))
    )
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            joints, confidence = grouped[(int(row["frame"]), int(row["track_id"]))]
            i = joint_index[row["joint"]]
            if row["x"]:
                joints[i] = (float(row["x"]), float(row["y"]))
                confidence[i] = float(row["confidence"])

    poses = []
    for (frame, track_id), (joints, confidence) in sorted(grouped.items()):
        found = confidence > 0
        xy = joints[found]
        box = np.array([*xy.min(0), *xy.max(0)], dtype=np.float32) if found.any() else np.zeros(4, np.float32)
        score = float(confidence[found].mean()) if found.any() else 0.0
        poses.append(TrackedPose(track_id, PoseDetection(frame, joints, confidence, box, score)))
    return poses
