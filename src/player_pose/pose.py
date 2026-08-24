"""Per-frame multi-person pose detection.

The tracker only depends on the `PoseDetection` shape, so the Apple Vision backend
can be swapped for a Core ML model later without touching the rest of the pipeline.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import Quartz
import Vision

# Apple Vision body pose joints, in the order used for arrays, CSV and overlay.
# "left"/"right" are the subject's own left/right, as Vision reports them.
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "neck",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "root",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]

_VISION_JOINTS = [
    Vision.VNHumanBodyPoseObservationJointNameNose,
    Vision.VNHumanBodyPoseObservationJointNameLeftEye,
    Vision.VNHumanBodyPoseObservationJointNameRightEye,
    Vision.VNHumanBodyPoseObservationJointNameLeftEar,
    Vision.VNHumanBodyPoseObservationJointNameRightEar,
    Vision.VNHumanBodyPoseObservationJointNameNeck,
    Vision.VNHumanBodyPoseObservationJointNameLeftShoulder,
    Vision.VNHumanBodyPoseObservationJointNameRightShoulder,
    Vision.VNHumanBodyPoseObservationJointNameLeftElbow,
    Vision.VNHumanBodyPoseObservationJointNameRightElbow,
    Vision.VNHumanBodyPoseObservationJointNameLeftWrist,
    Vision.VNHumanBodyPoseObservationJointNameRightWrist,
    Vision.VNHumanBodyPoseObservationJointNameRoot,
    Vision.VNHumanBodyPoseObservationJointNameLeftHip,
    Vision.VNHumanBodyPoseObservationJointNameRightHip,
    Vision.VNHumanBodyPoseObservationJointNameLeftKnee,
    Vision.VNHumanBodyPoseObservationJointNameRightKnee,
    Vision.VNHumanBodyPoseObservationJointNameLeftAnkle,
    Vision.VNHumanBodyPoseObservationJointNameRightAnkle,
]


@dataclass
class PoseDetection:
    """One person in one frame, in pixel coordinates (origin top-left, y down)."""

    frame: int
    joints: np.ndarray       # (19, 2) float32 x, y; NaN where the joint was not found
    confidence: np.ndarray   # (19,) float32 per-joint confidence, 0 where not found
    box: np.ndarray          # (4,) float32 x1, y1, x2, y2 around the confident joints
    score: float             # mean confidence of the joints that formed `box`


class VisionPoseDetector:
    """Apple Vision `VNDetectHumanBodyPoseRequest` backend (pyobjc).

    Each frame is wrapped as a CGImage without re-encoding; Vision returns one
    observation per person with 19 normalized joints (origin bottom-left), which
    are mapped to pixel space here.
    """

    def __init__(self, min_joint_confidence: float = 0.3, min_joints: int = 8):
        self.min_joint_confidence = min_joint_confidence
        self.min_joints = min_joints  # fewer confident joints than this = partial skeleton, dropped
        self._request = Vision.VNDetectHumanBodyPoseRequest.alloc().init()
        self._colorspace = Quartz.CGColorSpaceCreateDeviceRGB()

    def _cgimage(self, frame_bgr: np.ndarray):
        height, width = frame_bgr.shape[:2]
        data = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA).tobytes()
        provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
        return Quartz.CGImageCreate(
            width, height, 8, 32, width * 4, self._colorspace,
            Quartz.kCGBitmapByteOrder32Little | Quartz.kCGImageAlphaNoneSkipFirst,
            provider, None, False, Quartz.kCGRenderingIntentDefault,
        )

    def detect(self, frame_bgr: np.ndarray, frame_index: int) -> list[PoseDetection]:
        height, width = frame_bgr.shape[:2]
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(self._cgimage(frame_bgr), {})
        ok, error = handler.performRequests_error_([self._request], None)
        if not ok:
            raise RuntimeError(f"Vision request failed on frame {frame_index}: {error}")

        detections = []
        for observation in self._request.results() or []:
            joints = np.full((len(JOINT_NAMES), 2), np.nan, dtype=np.float32)
            confidence = np.zeros(len(JOINT_NAMES), dtype=np.float32)
            for i, name in enumerate(_VISION_JOINTS):
                point, _ = observation.recognizedPointForJointName_error_(name, None)
                if point is None or point.confidence() < self.min_joint_confidence:
                    continue
                location = point.location()
                joints[i] = (location.x * width, (1.0 - location.y) * height)
                confidence[i] = point.confidence()

            found = confidence > 0
            if found.sum() < max(self.min_joints, 2):
                continue
            xy = joints[found]
            box = np.array([*xy.min(0), *xy.max(0)], dtype=np.float32)
            detections.append(PoseDetection(
                frame=frame_index, joints=joints, confidence=confidence,
                box=box, score=float(confidence[found].mean()),
            ))
        return detections

    def detect_roi(self, frame_bgr: np.ndarray, frame_index: int, box: np.ndarray,
                   margin: float = 0.75, min_size: int = 128) -> list[PoseDetection]:
        """Detect inside a square crop around `box`, results in full-frame pixels.

        Vision shrinks the whole frame to its network input, so a person who is
        small relative to the frame vanishes; a crop makes them large again. The
        crop is square (tall narrow crops fail) with side = longest box side grown
        by `margin` on each side, at least `min_size`. No upscaling: it is the
        person's size relative to the crop that matters, not pixel count.
        """
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        side = max(max(x2 - x1, y2 - y1) * (1 + 2 * margin), min_size)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cx1, cy1 = int(max(0, np.floor(cx - side / 2))), int(max(0, np.floor(cy - side / 2)))
        cx2, cy2 = int(min(width, np.ceil(cx + side / 2))), int(min(height, np.ceil(cy + side / 2)))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return []
        offset = np.array([cx1, cy1], dtype=np.float32)
        out = []
        for d in self.detect(np.ascontiguousarray(frame_bgr[cy1:cy2, cx1:cx2]), frame_index):
            d.joints = d.joints + offset
            d.box = d.box + np.concatenate([offset, offset])
            out.append(d)
        return out
