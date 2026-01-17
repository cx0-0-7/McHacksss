import mediapipe as mp
from dataclasses import dataclass

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
    shoulder: Point
    elbow: Point
    wrist: Point
    hip: Point
    knee: Point
    ankle: Point

# ----------------------------
# POSE TRACKER CLASS
# ----------------------------
class PoseTracker:
    def __init__(self, complexity=2, detection_con=0.6, tracking_con=0.5,smooth_landmarks=True):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            enable_segmentation=False,
            min_detection_confidence=detection_con,
            min_tracking_confidence=tracking_con,
            smooth_landmarks=smooth_landmarks
        )

        # Define custom connections (Shoulder -> Elbow, Elbow -> Wrist, etc.)
        # These correspond to the Right side of the body
        PL = self.mp_pose.PoseLandmark
        self.CUSTOM_CONNECTIONS = [
            (PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW),
            (PL.RIGHT_ELBOW, PL.RIGHT_WRIST),
            (PL.RIGHT_SHOULDER, PL.RIGHT_HIP),
            (PL.RIGHT_HIP, PL.RIGHT_KNEE),
            (PL.RIGHT_KNEE, PL.RIGHT_ANKLE)
        ]

    def extract_key_landmarks(self, results) -> KeyLandmarks | None:
        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        PL = self.mp_pose.PoseLandmark

        return KeyLandmarks(
            shoulder=Point(lm[PL.RIGHT_SHOULDER].x, lm[PL.RIGHT_SHOULDER].y, lm[PL.RIGHT_SHOULDER].visibility),
            elbow=Point(lm[PL.RIGHT_ELBOW].x, lm[PL.RIGHT_ELBOW].y, lm[PL.RIGHT_ELBOW].visibility),
            wrist=Point(lm[PL.RIGHT_WRIST].x, lm[PL.RIGHT_WRIST].y, lm[PL.RIGHT_WRIST].visibility),
            hip=Point(lm[PL.RIGHT_HIP].x, lm[PL.RIGHT_HIP].y, lm[PL.RIGHT_HIP].visibility),
            knee=Point(lm[PL.RIGHT_KNEE].x, lm[PL.RIGHT_KNEE].y, lm[PL.RIGHT_KNEE].visibility),
            ankle=Point(lm[PL.RIGHT_ANKLE].x, lm[PL.RIGHT_ANKLE].y, lm[PL.RIGHT_ANKLE].visibility),
        )

def print_visible_landmarks(key: KeyLandmarks):
    for name, p in key.__dict__.items():
        if p.visibility >= 0.5:
            print(f"{name}: x={p.x:.2f}, y={p.y:.2f}, vis={p.visibility:.2f}")