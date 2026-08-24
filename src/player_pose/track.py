"""Track player poses through a video and write per-frame joints to CSV.

    player-pose-track rally.mp4
    player-pose-track rally.mp4 --max-age 60 -o rally_poses.csv

Pose comes from Apple Vision's built-in body pose model (Neural Engine); identities
come from SORT. Output CSV columns: frame,track_id,joint,x,y,confidence.
"""

import argparse
import sys
import time
from pathlib import Path

from player_pose.infer import iter_tracked, write_csv
from player_pose.pose import VisionPoseDetector
from player_pose.sort import SORTTracker
from player_pose.video import video_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", help="input video file")
    parser.add_argument("--min-joint-confidence", type=float, default=0.3,
                        help="drop joints Vision is less confident about than this")
    parser.add_argument("--min-joints", type=int, default=8,
                        help="drop people with fewer confident joints than this (partial skeletons)")
    parser.add_argument("--min-height", type=float, default=0.1,
                        help="ignore people shorter than this fraction of the frame height (officials, spectators)")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="SORT match threshold")
    parser.add_argument("--max-age", type=int, default=30, help="frames a track survives unmatched")
    parser.add_argument("--min-hits", type=int, default=3, help="matches before a track is reported")
    parser.add_argument("--no-track", action="store_true",
                        help="skip SORT: raw Vision output, track_id is the detection index within the frame")
    parser.add_argument("--no-redetect", action="store_true",
                        help="don't re-run Vision on a crop when a tracked player goes missing")
    parser.add_argument("-o", "--out", help="output CSV path (default: <video stem>_poses.csv)")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path(args.video).with_name(f"{Path(args.video).stem}_poses.csv")
    if out_path.is_dir() or not out_path.parent.is_dir():
        parser.error(f"cannot write {out_path}: not a writable file path")
    info = video_info(args.video)  # fails early if the video cannot be opened

    detector = VisionPoseDetector(min_joint_confidence=args.min_joint_confidence, min_joints=args.min_joints)
    tracker = None if args.no_track else SORTTracker(
        iou_threshold=args.iou_threshold, max_age=args.max_age, min_hits=args.min_hits)

    start = time.perf_counter()
    poses, frames = [], 0
    for idx, tracked in iter_tracked(args.video, detector, tracker,
                                     redetect=not args.no_redetect, min_height=args.min_height):
        poses.extend(tracked)
        frames = idx + 1
        if not args.quiet and idx % 50 == 0:
            print(f"\r{frames}/{info['num_frames']} frames", end="", file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - start

    write_csv(poses, out_path)
    if not args.quiet:
        print(f"\rwrote {out_path} — {frames} frames in {elapsed:.1f}s "
              f"({frames / max(elapsed, 1e-9):.1f} frames/s, video is {info['fps']:.0f} fps)", file=sys.stderr)


if __name__ == "__main__":
    main()
