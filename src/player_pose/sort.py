"""SORT: constant-velocity Kalman filter per track + Hungarian matching on IoU.

Reference: Bewley et al., "Simple Online and Realtime Tracking" (2016). Boxes are
(x1, y1, x2, y2); the filter state is (cx, cy, area, aspect) plus velocities of the
first three, exactly as in the reference implementation.

Departures from plain SORT, all because pose detections are noisier than
pedestrian boxes (partial skeletons produce sliver boxes that come and go):
- Tracks start tentative: confirmed tracks are matched first so a one-frame
  ghost can never steal a real player's detection, and a tentative track that
  is not matched `min_hits` times within a few frames is dropped.
- `update()` accepts a `redetect` callback. Confirmed tracks that went
  unmatched get a second chance — the callback is handed a search box and may
  return extra detections (e.g. from a crop around it). Candidates are pooled
  and assigned jointly (Hungarian again), and accepted only if they look like
  the same person: enough joints, similar height, overlapping the track.
- A detection that lies mostly inside a box already claimed this frame is the
  same person seen twice (a partial skeleton, or the crop finding what the
  full frame also found) and never spawns a new track.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from player_pose.pose import PoseDetection


@dataclass
class TrackedPose:
    track_id: int
    detection: PoseDetection


def _intersection(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = a[:, None, :]
    b = b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return iw * ih, area_a, area_b


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes → (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    inter, area_a, area_b = _intersection(a, b)
    return (inter / np.maximum(area_a + area_b - inter, 1e-6)).astype(np.float32)


def containment_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection / smaller area → (N, M). 1 when one box lies inside the other."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    inter, area_a, area_b = _intersection(a, b)
    return (inter / np.maximum(np.minimum(area_a, area_b), 1e-6)).astype(np.float32)


def _box_to_z(box: np.ndarray) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-6)], dtype=np.float64)


def _x_to_box(x: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(x[2] * x[3], 1e-6))
    h = x[2] / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2], dtype=np.float32)


def _grow_box(box: np.ndarray, factor: float) -> np.ndarray:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) / 2 * factor, (box[3] - box[1]) / 2 * factor
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=np.float32)


class KalmanBoxTracker:
    """One track: constant-velocity Kalman filter over (cx, cy, area, aspect)."""

    _F = np.eye(7)
    _F[0, 4] = _F[1, 5] = _F[2, 6] = 1.0
    _H = np.eye(4, 7)
    _R = np.diag([1.0, 1.0, 10.0, 10.0])
    _Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])

    def __init__(self, track_id: int, box: np.ndarray):
        self.id = track_id
        self.x = np.zeros(7)
        self.x[:4] = _box_to_z(box)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0])
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.confirmed = False
        self.predicted_box = box.astype(np.float32)
        self.last_box = box.astype(np.float32)  # last observed, unlike the extrapolated prediction

    def predict(self) -> np.ndarray:
        if self.x[6] + self.x[2] <= 0:  # area would go negative: freeze area velocity
            self.x[6] = 0.0
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.predicted_box = _x_to_box(self.x)
        return self.predicted_box

    def update(self, box: np.ndarray) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_box = box.astype(np.float32)
        z = _box_to_z(box)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self._H) @ self.P


RedetectFn = Callable[[np.ndarray], list[PoseDetection]]

SAME_PERSON = 0.7  # containment above this = one person seen twice


def _match(tracks: list[KalmanBoxTracker], boxes: np.ndarray, det_idx: list[int], threshold: float):
    """Hungarian matching of `tracks` against `boxes[det_idx]` → list of (track, det index)."""
    if not tracks or not det_idx:
        return []
    iou = iou_matrix(np.array([t.predicted_box for t in tracks]), boxes[det_idx])
    rows, cols = linear_sum_assignment(-iou)
    return [(tracks[r], det_idx[c]) for r, c in zip(rows, cols) if iou[r, c] >= threshold]


class SORTTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3,
                 tentative_max_age: int = 3, redetect_iou: float = 0.2, redetect_min_joints: int = 12,
                 redetect_height_ratio: tuple[float, float] = (0.5, 2.0)):
        self.iou_threshold = iou_threshold
        self.max_age = max_age                  # frames a confirmed track survives without a match
        self.min_hits = min_hits                # matches before a track is confirmed/reported
        self.tentative_max_age = tentative_max_age  # frames an unconfirmed track survives without a match
        self.redetect_iou = redetect_iou        # looser overlap for detections found by `redetect`
        self.redetect_min_joints = redetect_min_joints
        self.redetect_height_ratio = redetect_height_ratio  # candidate height vs last seen height
        self.tracks: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self._next_id = 1

    def _redetect(self, pending: list[KalmanBoxTracker], redetect: RedetectFn,
                  claimed: np.ndarray) -> list[tuple[KalmanBoxTracker, PoseDetection]]:
        """Search around each pending track, pool the candidates, assign them jointly."""
        candidates: list[PoseDetection] = []
        for track in pending:
            # Search around where the player was last seen, not where the filter
            # extrapolates them to — velocity drifts over a long miss. The search
            # area grows slowly the longer they have been missing.
            search = _grow_box(track.last_box, min(1.0 + 0.1 * (track.time_since_update - 1), 2.0))
            for cand in redetect(search):
                if (cand.confidence > 0).sum() < self.redetect_min_joints:
                    continue  # partial skeleton: its box says nothing about the player's extent
                if len(claimed) and containment_matrix(cand.box[None], claimed).max() > SAME_PERSON:
                    continue  # already matched to another track this frame
                pooled = np.array([c.box for c in candidates]).reshape(-1, 4)
                dup = np.flatnonzero(containment_matrix(cand.box[None], pooled)[0] > SAME_PERSON)
                if len(dup):  # crops overlap, so the same person shows up more than once
                    j = int(dup[0])
                    if (cand.confidence > 0).sum() > (candidates[j].confidence > 0).sum():
                        candidates[j] = cand
                    continue
                candidates.append(cand)
        if not candidates:
            return []

        lo, hi = self.redetect_height_ratio
        score = np.zeros((len(pending), len(candidates)), dtype=np.float32)
        for i, track in enumerate(pending):
            last_h = max(track.last_box[3] - track.last_box[1], 1e-6)
            anchors = np.stack([track.predicted_box, track.last_box])
            for j, cand in enumerate(candidates):
                if lo <= (cand.box[3] - cand.box[1]) / last_h <= hi:
                    score[i, j] = iou_matrix(cand.box[None], anchors).max()
        rows, cols = linear_sum_assignment(-score)
        return [(pending[r], candidates[c]) for r, c in zip(rows, cols) if score[r, c] >= self.redetect_iou]

    def update(self, detections: list[PoseDetection], redetect: RedetectFn | None = None) -> list[TrackedPose]:
        self.frame_count += 1
        detections = list(detections)
        for t in self.tracks:
            t.predict()
        self.tracks = [t for t in self.tracks if np.all(np.isfinite(t.predicted_box))]

        boxes = np.array([d.box for d in detections]).reshape(-1, 4)
        confirmed = [t for t in self.tracks if t.confirmed]
        tentative = [t for t in self.tracks if not t.confirmed]

        # Confirmed tracks pick first; tentative ones only get the leftovers.
        matches = _match(confirmed, boxes, list(range(len(boxes))), self.iou_threshold)
        remaining = [i for i in range(len(boxes)) if i not in {di for _, di in matches}]
        matches += _match(tentative, boxes, remaining, self.iou_threshold)
        matched = {id(t) for t, _ in matches}

        if redetect is not None:
            pending = [t for t in confirmed if id(t) not in matched]
            claimed = boxes[[di for _, di in matches]].reshape(-1, 4)
            for track, cand in self._redetect(pending, redetect, claimed):
                detections.append(cand)
                matches.append((track, len(detections) - 1))

        for track, di in matches:
            track.update(detections[di].box)

        claimed = np.array([detections[di].box for _, di in matches]).reshape(-1, 4)
        for di in range(len(boxes)):
            if di in {j for _, j in matches}:
                continue
            if len(claimed) and containment_matrix(boxes[di][None], claimed).max() > SAME_PERSON:
                continue  # same person as a matched box: a partial or a second sighting, not a new player
            self.tracks.append(KalmanBoxTracker(self._next_id, detections[di].box))
            matches.append((self.tracks[-1], di))
            self._next_id += 1

        out = []
        for track, di in matches:
            if track.hits >= self.min_hits:
                track.confirmed = True
            if track.confirmed or self.frame_count <= self.min_hits:
                out.append(TrackedPose(track.id, detections[di]))

        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= (self.max_age if t.confirmed else self.tentative_max_age)]
        return sorted(out, key=lambda p: p.track_id)
