import os
import sys
import argparse
import cv2
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

from detector import VehicleDetector
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vehicle Recognition System")

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=config.INPUT_VIDEO_PATH,
        help=f"Path to input video (default: {config.INPUT_VIDEO_PATH})",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=config.OUTPUT_VIDEO_PATH,
        help=f"Path to output video (default: {config.OUTPUT_VIDEO_PATH})",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})",
    )

    parser.add_argument(
        "--display",
        "-d",
        action="store_true",
        default=config.DISPLAY_OUTPUT,
        help="Display output while processing",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=config.MODEL_PATH,
        help=f"Path to model (default: {config.MODEL_PATH})",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help=f"Device to run inference on ('cuda' or 'cpu', default: {config.DEVICE})",
    )

    parser.add_argument(
        "--save-frames",
        action="store_true",
        default=config.SAVE_FRAMES,
        help="Save individual frames",
    )

    return parser.parse_args()


def ensure_directory(path):
    """Ensure directory exists."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def process_video(args):
    """Process the input video and detect vehicles."""
    # Initialize the detector
    detector = VehicleDetector(args.model, args.device)

    # Open video capture
    try:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.input}")
            return False
    except Exception as e:
        print(f"Error opening video file: {e}")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Create output directories if needed
    ensure_directory(args.output)
    if args.save_frames:
        ensure_directory(config.FRAME_OUTPUT_DIR)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    processing_times = []
    vehicle_counts = {}

    print("Starting vehicle detection...")

    # Use tqdm for progress display
    progress_bar = tqdm(total=total_frames, desc="Processing", unit="frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every N frames for speed
        if frame_count % config.PROCESS_EVERY_N_FRAME == 0:
            # Detect vehicles
            start_time = time.time()
            detections = detector.detect_vehicles(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Draw detections on the frame
            annotated_frame = detector.draw_detections(frame, detections)

            # Count vehicles
            counts = detector.count_vehicles(detections)
            for vehicle_type, count in counts.items():
                if count > 0:
                    vehicle_counts[vehicle_type] = (
                        vehicle_counts.get(vehicle_type, 0) + count
                    )

            # Display count of each vehicle type
            y_pos = 30  # Starting position adjusted since we're not showing FPS
            for vehicle_type, count in counts.items():
                if count > 0:
                    cv2.putText(
                        annotated_frame,
                        f"{vehicle_type}: {count}",
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    y_pos += 40

            # Save frame if enabled
            if args.save_frames:
                frame_path = os.path.join(
                    config.FRAME_OUTPUT_DIR, f"frame_{frame_count:06d}.jpg"
                )
                cv2.imwrite(frame_path, annotated_frame)

            # Write to output video
            out.write(annotated_frame)

            # Display if enabled
            if args.display:
                cv2.imshow("Vehicle Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # For skipped frames, just write the original frame to output
            out.write(frame)

        # Update progress bar
        progress_bar.update(1)

    # Clean up
    progress_bar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print stats
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    print(f"\nProcessed {frame_count} frames")
    print(f"Average processing time per frame: {avg_time:.4f} seconds")
    print(f"Average FPS: {(1/avg_time):.2f}" if avg_time > 0 else "Average FPS: N/A")

    print("\nVehicle counts:")
    for vehicle_type, count in vehicle_counts.items():
        print(f"  {vehicle_type}: {count}")

    print(f"\nOutput video saved to {args.output}")

    return True


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Check if input video exists
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

    # Process video
    success = process_video(args)

    if success:
        print("Vehicle recognition completed successfully.")
    else:
        print("Vehicle recognition failed.")
        sys.exit(1)
