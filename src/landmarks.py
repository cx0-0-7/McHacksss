from dataclasses import dataclass
import os
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ----------------------------
# DATA STRUCTURES
# ----------------------------
@dataclass
class Point:
    x: float
    y: float
    visibility: float = 1.0


@dataclass
class KeyLandmarks:
    # Right Side
    r_shoulder: Point
    r_elbow: Point
    r_wrist: Point
    r_hip: Point
    r_knee: Point
    r_ankle: Point
    
    # Left Side
    l_shoulder: Point
    l_elbow: Point
    l_wrist: Point
    l_hip: Point
    l_knee: Point
    l_ankle: Point


# ----------------------------
# POSE TRACKER CLASS
# ----------------------------
class PoseTracker:
    def __init__(self):
        # Always load model relative to this file
        model_path = os.path.join(
            os.path.dirname(__file__),
            "pose_landmarker_full.task"
        )

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.5
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # Right-side landmark indices
        self.RIGHT = {
            "shoulder": 12,
            "elbow": 14,
            "wrist": 16,
            "hip": 24,
            "knee": 26,
            "ankle": 28,
        }

        # Left-side landmark indices
        self.LEFT = {
            "shoulder": 11,
            "elbow": 13,
            "wrist": 15,
            "hip": 23,
            "knee": 25,
            "ankle": 27,
        }

    def extract_key_landmarks(self, results) -> KeyLandmarks | None:
        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        PL = self.mp_pose.PoseLandmark

        return KeyLandmarks(
            # RIGHT
            r_shoulder=Point(lm[PL.RIGHT_SHOULDER].x, lm[PL.RIGHT_SHOULDER].y, lm[PL.RIGHT_SHOULDER].visibility),
            r_elbow=Point(lm[PL.RIGHT_ELBOW].x, lm[PL.RIGHT_ELBOW].y, lm[PL.RIGHT_ELBOW].visibility),
            r_wrist=Point(lm[PL.RIGHT_WRIST].x, lm[PL.RIGHT_WRIST].y, lm[PL.RIGHT_WRIST].visibility),
            r_hip=Point(lm[PL.RIGHT_HIP].x, lm[PL.RIGHT_HIP].y, lm[PL.RIGHT_HIP].visibility),
            r_knee=Point(lm[PL.RIGHT_KNEE].x, lm[PL.RIGHT_KNEE].y, lm[PL.RIGHT_KNEE].visibility),
            r_ankle=Point(lm[PL.RIGHT_ANKLE].x, lm[PL.RIGHT_ANKLE].y, lm[PL.RIGHT_ANKLE].visibility),
            
            # LEFT
            l_shoulder=Point(lm[PL.LEFT_SHOULDER].x, lm[PL.LEFT_SHOULDER].y, lm[PL.LEFT_SHOULDER].visibility),
            l_elbow=Point(lm[PL.LEFT_ELBOW].x, lm[PL.LEFT_ELBOW].y, lm[PL.LEFT_ELBOW].visibility),
            l_wrist=Point(lm[PL.LEFT_WRIST].x, lm[PL.LEFT_WRIST].y, lm[PL.LEFT_WRIST].visibility),
            l_hip=Point(lm[PL.LEFT_HIP].x, lm[PL.LEFT_HIP].y, lm[PL.LEFT_HIP].visibility),
            l_knee=Point(lm[PL.LEFT_KNEE].x, lm[PL.LEFT_KNEE].y, lm[PL.LEFT_KNEE].visibility),
            l_ankle=Point(lm[PL.LEFT_ANKLE].x, lm[PL.LEFT_ANKLE].y, lm[PL.LEFT_ANKLE].visibility)
        )


def print_visible_landmarks(key: KeyLandmarks, side=""):
    for name, p in key.__dict__.items():
        if p.visibility >= 0.5:
            print(f"{side} {name}: x={p.x:.2f}, y={p.y:.2f}, vis={p.visibility:.2f}")
