import cv2
from landmarks import PoseTracker, print_visible_landmarks

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
                tracker.mp_pose.POSE_CONNECTIONS
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