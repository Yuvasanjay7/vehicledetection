# Vehicle Recognition System

A computer vision system that detects and recognizes vehicles in videos using YOLOv8.

## Features

- Vehicle detection and classification (cars, motorcycles, buses, trucks)
- Real-time video processing with visual output
- Vehicle counting by type
- Support for GPU acceleration on Mac (MPS) and CUDA devices
- Command-line interface for flexible usage
- Performance metrics reporting

## Requirements

- Python 3.8+
- OpenCV
- PyTorch (with MPS/CUDA support recommended)
- Ultralytics YOLOv8

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/meankitdas/vehicle-detection.git
   cd vehicle-detection
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python src/main.py --input path/to/your/video.mp4
```

### Command Line Options

- `--input`, `-i`: Path to input video (default: data/sample_video.mp4)
- `--output`, `-o`: Path to output video (default: data/output_video.mp4)
- `--conf`: Confidence threshold (default: 0.25)
- `--display`, `-d`: Display output while processing
- `--model`, `-m`: Path to model (default: models/yolov8n.pt)
- `--save-frames`: Save individual frames

### Examples

Run detection on a video and display the output:

```bash
python src/main.py --input myVideo.mp4 --display
```

Run with custom confidence threshold:

```bash
python src/main.py --input myVideo.mp4 --conf 0.4
```

Save processed video to a specific location:

```bash
python src/main.py --input myVideo.mp4 --output results/processed.mp4
```

## Project Structure

```
vehicle-recognition-system/
├── src/
│   ├── main.py           # Main script for running the system
│   ├── detector.py       # Vehicle detection implementation
│   └── config.py         # Configuration parameters
├── data/                 # Directory for input/output videos
├── models/               # Directory for ML model files
│   └── yolov8n.pt        # YOLOv8 nano model (downloaded on first run)
└── requirements.txt      # Project dependencies
```

## Note About Large Files

Video files are not included in this repository due to GitHub file size limitations. The system will work with any compatible video file you provide.

## Performance Notes

- For optimal performance on Mac, the system uses Metal Performance Shaders (MPS)
- Processing time varies based on video resolution and hardware specifications

## License

MIT License
