import mediapipe as mp
from dataclasses import dataclass

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


mp_pose = mp.solutions.pose

def init_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def extract_key_landmarks(results) -> KeyLandmarks | None:
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark
    PL = mp_pose.PoseLandmark

    return KeyLandmarks(
        shoulder=Point(lm[PL.RIGHT_SHOULDER].x, lm[PL.RIGHT_SHOULDER].y, lm[PL.RIGHT_SHOULDER].visibility),
        elbow=Point(lm[PL.RIGHT_ELBOW].x, lm[PL.RIGHT_ELBOW].y, lm[PL.RIGHT_ELBOW].visibility),
        wrist=Point(lm[PL.RIGHT_WRIST].x, lm[PL.RIGHT_WRIST].y, lm[PL.RIGHT_WRIST].visibility),
        hip=Point(lm[PL.RIGHT_HIP].x, lm[PL.RIGHT_HIP].y, lm[PL.RIGHT_HIP].visibility),
        knee=Point(lm[PL.RIGHT_KNEE].x, lm[PL.RIGHT_KNEE].y, lm[PL.RIGHT_KNEE].visibility),
        ankle=Point(lm[PL.RIGHT_ANKLE].x, lm[PL.RIGHT_ANKLE].y, lm[PL.RIGHT_ANKLE].visibility),
    )


def normalize_landmarks(lm: KeyLandmarks) -> KeyLandmarks:
    scale = abs(lm.shoulder.y - lm.ankle.y)
    if scale == 0:
        return lm

    def norm(p: Point):
        return Point(p.x, p.y / scale, p.visibility)

    return KeyLandmarks(
        shoulder=norm(lm.shoulder),
        elbow=norm(lm.elbow),
        wrist=norm(lm.wrist),
        hip=norm(lm.hip),
        knee=norm(lm.knee),
        ankle=norm(lm.ankle),
    )