MODEL_PATH = "path/to/your/yolov8/model.pt"
DEVICE = "cuda"  # or "cpu" depending on your setup

VEHICLE_CLASSES = [2, 3, 5, 7]  # Class IDs for Car, Motorcycle, Bus, Truck
CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

BOX_THICKNESS = 2
FONT_SIZE = 0.5
FONT_THICKNESS = 1