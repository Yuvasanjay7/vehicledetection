import cv2
from src.detector import VehicleDetector
import config

def main():
    # Initialize the vehicle detector
    detector = VehicleDetector(model_path=config.MODEL_PATH, device=config.DEVICE)

    # Capture video from the camera or a video file
    cap = cv2.VideoCapture(0)  # Change to a file path for video file input

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles in the frame
        detections = detector.detect_vehicles(frame)

        # Draw detections on the frame
        annotated_frame = detector.draw_detections(frame, detections)

        # Display the annotated frame
        cv2.imshow("Vehicle Detection", annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()