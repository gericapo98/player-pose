"""Render tracked skeletons and player IDs onto a video so you can watch the tracking.

    player-pose-overlay rally.mp4 rally_poses.csv                        # writes rally_poses.mp4
    player-pose-overlay rally.mp4 rally_poses.csv --ball rally_ball.csv  # + shuttle from tracknet-track

Each player keeps one colour for the whole clip; joints Vision did not find are
simply not drawn. `--ball` takes the Frame,Visibility,X,Y CSV that tracknet-track
writes and draws the ball with a fading trail, same look as tracknet-overlay.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from player_pose.infer import read_csv
from player_pose.pose import JOINT_NAMES
from player_pose.sort import TrackedPose

SKELETON = [
    ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("nose", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("neck", "root"), ("root", "left_hip"), ("root", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
_EDGES = [(JOINT_NAMES.index(a), JOINT_NAMES.index(b)) for a, b in SKELETON]

# BGR, chosen to stay apart from the green court and the yellow/red ball marker
_PALETTE = [(255, 200, 0), (0, 0, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0), (0, 128, 255)]


def track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def read_ball_csv(path: str | Path) -> dict[int, tuple[int, int]]:
    """tracknet-track output → {frame: (x, y)} for frames where the ball was seen."""
    ball = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["Visibility"]) == 1:
                ball[int(row["Frame"])] = (int(row["X"]), int(row["Y"]))
    return ball


def draw_pose(frame: np.ndarray, pose: TrackedPose, scale: float = 1.0) -> None:
    d = pose.detection
    color = track_color(pose.track_id)
    thick = max(1, int(2 * scale))
    found = d.confidence > 0
    for a, b in _EDGES:
        if found[a] and found[b]:
            cv2.line(frame, tuple(d.joints[a].astype(int)), tuple(d.joints[b].astype(int)),
                     color, thick, lineType=cv2.LINE_AA)
    for (x, y), ok in zip(d.joints, found):
        if ok:
            cv2.circle(frame, (int(x), int(y)), max(2, int(3 * scale)), color, -1, lineType=cv2.LINE_AA)
    x1, y1 = d.box[:2].astype(int)
    cv2.putText(frame, f"#{pose.track_id}", (int(x1), max(int(y1) - int(6 * scale), int(12 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color, thick, lineType=cv2.LINE_AA)


def draw_ball(frame: np.ndarray, trail: list[tuple[int, int] | None], scale: float = 1.0) -> None:
    """Fading trail (oldest first) then the current detection, tracknet-overlay style."""
    for age, point in enumerate(trail[:-1]):
        if point is None:
            continue
        strength = (age + 1) / len(trail)
        radius = max(1, int((2 + 3 * strength) * scale))
        color = (0, int(150 + 105 * strength), int(255 * strength))  # BGR: green->yellow
        cv2.circle(frame, point, radius, color, -1, lineType=cv2.LINE_AA)
    if trail and trail[-1] is not None:
        point = trail[-1]
        cv2.circle(frame, point, max(2, int(5 * scale)), (0, 220, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, point, max(4, int(9 * scale)), (0, 0, 255), max(1, int(2 * scale)),
                   lineType=cv2.LINE_AA)


def render_overlay(
    video_file: str | Path,
    poses: list[TrackedPose],
    out_file: str | Path,
    ball: dict[int, tuple[int, int]] | None = None,
    traj_len: int = 8,
    progress: bool = True,
) -> Path:
    """Write a copy of the video with skeletons, player IDs and (optionally) the ball drawn on it."""
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_file}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_file = Path(out_file)
    writer = None
    for fourcc in ("avc1", "mp4v"):  # H.264 when the OS backend provides it
        writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        raise RuntimeError("could not open a video writer for .mp4 output")

    by_frame: dict[int, list[TrackedPose]] = defaultdict(list)
    for p in poses:
        by_frame[p.detection.frame].append(p)
    scale = max(w / 1280, 1.0)  # sizes tuned for 720p, scaled up for larger videos
    trail: list[tuple[int, int] | None] = []

    idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        for pose in by_frame.get(idx, []):
            draw_pose(frame, pose, scale)
        if ball is not None:
            trail.append(ball.get(idx))
            if len(trail) > traj_len:
                trail.pop(0)
            draw_ball(frame, trail, scale)
        writer.write(frame)
        if progress and (idx % 50 == 0 or idx == num_frames - 1):
            print(f"\r{idx + 1}/{num_frames} frames", end="", file=sys.stderr, flush=True)
        idx += 1
    if progress:
        print(file=sys.stderr)

    cap.release()
    writer.release()
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", help="input video file")
    parser.add_argument("csv", help="pose CSV written by player-pose-track")
    parser.add_argument("--ball", help="optional Frame,Visibility,X,Y CSV from tracknet-track")
    parser.add_argument("--traj-len", type=int, default=8, help="ball trail length in frames")
    parser.add_argument("-o", "--out", help="output video path (default: <csv stem>.mp4)")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(args.csv).with_suffix(".mp4")
    ball = read_ball_csv(args.ball) if args.ball else None
    render_overlay(args.video, read_csv(args.csv), out_path, ball=ball,
                   traj_len=args.traj_len, progress=not args.quiet)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
