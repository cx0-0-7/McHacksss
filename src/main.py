import cv2
from landmarks import PoseTracker, print_visible_landmarks
import mediapipe

def main():
    cap = cv2.VideoCapture(0)
    # Initialize our tracker from the other file
    tracker = PoseTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.pose.process(frame_rgb)

        # Draw skeleton on frame
        if results.pose_landmarks:
            tracker.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                tracker.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec= mediapipe.solutions.drawing_utils.DrawingSpec(
                                        color=(0, 255, 0),       # Green dots
                                        thickness=4,             # Thickness of the outline
                                        circle_radius=6          # Size of the dot
                                    ),  # Apply dot style
                connection_drawing_spec= mediapipe.solutions.drawing_utils.DrawingSpec(
                                            color=(255, 0, 0),       # Blue lines
                                            thickness=3              # Line weight
                                        ) # Apply line style
            )

        # Extract landmarks using our class method
        key = tracker.extract_key_landmarks(results)

        if key:
            print_visible_landmarks(key)
            print("-----")

        cv2.imshow("Pose Detection", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()