# Real-Time Vehicle Detection

This project implements a real-time vehicle detection system using the YOLOv8 model. It is designed to detect various types of vehicles in video streams and annotate them with bounding boxes and labels.

## Project Structure

```
real_time_vehicle_detection
├── src
│   ├── detector.py       # Contains the VehicleDetector class for vehicle detection
│   ├── config.py         # Configuration constants for the project
│   ├── utils.py          # Utility functions for video processing
│   └── main.py           # Entry point for the application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd real_time_vehicle_detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the vehicle detection application, execute the following command:

```
python src/main.py
```

Make sure your camera is connected or specify a video file in the `main.py` script.

## Configuration

You can adjust the detection parameters in the `src/config.py` file. The following constants can be modified:

- `MODEL_PATH`: Path to the YOLOv8 model weights.
- `DEVICE`: Device to run the model on (e.g., 'cpu' or 'cuda').
- `VEHICLE_CLASSES`: List of vehicle class IDs to detect.
- `CLASS_NAMES`: Dictionary mapping class IDs to class names.
- `CONFIDENCE_THRESHOLD`: Minimum confidence score for detections.
- `IOU_THRESHOLD`: Intersection over Union threshold for non-max suppression.
- `BOX_THICKNESS`: Thickness of the bounding box drawn around detected vehicles.
- `FONT_SIZE`: Font size for the labels.
- `FONT_THICKNESS`: Thickness of the font for the labels.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.