"""Video reading helpers. Frames are yielded as BGR uint8 (OpenCV convention)."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def video_info(video_file: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_file}")
    info = {
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_frames(video_file: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame_bgr) for every frame in order."""
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_file}")
    idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        yield idx, frame
        idx += 1
    cap.release()
