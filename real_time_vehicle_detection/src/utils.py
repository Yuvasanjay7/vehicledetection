def load_video_stream(source: Union[int, str]) -> cv2.VideoCapture:
    """
    Load a video stream from a camera or video file.
    
    Args:
        source: Camera index (int) or video file path (str).
        
    Returns:
        cv2.VideoCapture object for the video stream.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video source {source}")
    return cap

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess the frame for vehicle detection.
    
    Args:
        frame: Original image frame as a numpy array.
        
    Returns:
        Preprocessed frame.
    """
    # Resize frame to the expected input size for the model
    resized_frame = cv2.resize(frame, (640, 640))  # Example size, adjust as needed
    return resized_frame

def release_video_stream(cap: cv2.VideoCapture) -> None:
    """
    Release the video capture object.
    
    Args:
        cap: cv2.VideoCapture object to be released.
    """
    cap.release()
    cv2.destroyAllWindows()