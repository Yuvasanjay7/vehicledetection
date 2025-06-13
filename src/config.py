# Input/Output settings
INPUT_VIDEO_PATH = (
    "data/sample_video.mp4"  # Default input path (can be overridden by CLI)
)
OUTPUT_VIDEO_PATH = "data/output_video.mp4"  # Default output path
SAVE_FRAMES = False  # Whether to save individual frames
FRAME_OUTPUT_DIR = "data/frames/"  # Directory to save frames if enabled

# Model settings
MODEL_TYPE = "yolov8"  # Model to use: "yolov8"
MODEL_PATH = "models/yolov8n.pt"  # Path to pretrained model
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detection
IOU_THRESHOLD = 0.45  # IoU threshold for NMS

# Classes we're interested in (from COCO dataset)
# Only detect these classes from COCO: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO class IDs for vehicles
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Processing settings
DEVICE = "mps"  # Use "cuda" if GPU is available
RESIZE_WIDTH = 640  # Width to resize input frames
RESIZE_HEIGHT = 640  # Height to resize input frames
PROCESS_EVERY_N_FRAME = 1  # Process every n-th frame for speed

# Display settings
DISPLAY_OUTPUT = True  # Whether to display output while processing
FONT_SIZE = 0.5
FONT_THICKNESS = 2
BOX_THICKNESS = 2
