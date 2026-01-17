import cv2
import mediapipe as mp
import time  # NEW: Added for accurate timing
from landmarks import PoseTracker, print_visible_landmarks

def draw_skeleton(frame, key, side_color=(255, 0, 0)):
    h, w, _ = frame.shape
    pts = {}
    for name, p in key.__dict__.items():
        if p.visibility >= 0.5:
            cx, cy = int(p.x * w), int(p.y * h)
            pts[name] = (cx, cy)
            cv2.circle(frame, (cx, cy), 6, side_color, -1)

    connections = [
        ("shoulder", "elbow"), ("elbow", "wrist"),
        ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
    ]

    for a, b in connections:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], side_color, 3)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    tracker = PoseTracker()
    
    # NEW: Use real system time for timestamps
    start_time = time.time()

    print("System started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate accurate timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection
        result = tracker.landmarker.detect_for_video(mp_image, timestamp_ms)

        # Check if anything was detected
        if result.pose_landmarks:
            right_key = tracker.extract_key_landmarks(result, side="right")
            if right_key:
                draw_skeleton(frame, right_key, side_color=(255, 0, 0))

            left_key = tracker.extract_key_landmarks(result, side="left")
            if left_key:
                draw_skeleton(frame, left_key, side_color=(0, 0, 255))
        else:
            # Helpful debug hint
            cv2.putText(frame, "No Pose Detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()