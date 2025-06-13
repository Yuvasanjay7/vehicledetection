class VehicleDetector:
    """Vehicle detection and recognition using YOLOv8."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the vehicle detector with the specified model.
        
        Args:
            model_path: Path to the model weights file. If None, uses config default.
            device: Device to run inference on ('cuda' or 'cpu'). If None, uses config default.
        """
        self.model_path = model_path or config.MODEL_PATH
        self.device = device or config.DEVICE
        
        # Check if model exists, if not download it
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"Model not found at {self.model_path}, downloading YOLOv8 model...")
            self.model = YOLO('yolov8n.pt')  # Will download if not present
            # Save model to specified path
            self.model.save(self.model_path)
        else:
            self.model = YOLO(self.model_path)
            
        print(f"Model loaded from {self.model_path}")
        print(f"Running on device: {self.device}")
        
        # Vehicle classes we're interested in
        self.vehicle_classes = config.VEHICLE_CLASSES
        self.class_names = config.CLASS_NAMES
        
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a single frame.
        
        Args:
            frame: Image frame as numpy array (BGR format from OpenCV)
            
        Returns:
            List of dictionaries containing detection information
        """
        # Run inference
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, iou=config.IOU_THRESHOLD)
        
        # Process results
        detections = []
        if len(results) > 0:
            for result in results[0].boxes:
                class_id = int(result.cls.item())
                
                # Skip if not a vehicle class we're interested in
                if class_id not in self.vehicle_classes:
                    continue
                    
                confidence = float(result.conf.item())
                box = result.xyxy[0].tolist()  # Convert to x1, y1, x2, y2 format
                
                detections.append({
                    'class_id': class_id, 
                    'class_name': self.class_names.get(class_id, 'unknown'),
                    'confidence': confidence,
                    'box': box  # [x1, y1, x2, y2]
                })
                
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Original image frame
            detections: List of detections from detect_vehicles()
            
        Returns:
            Frame with annotations
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            # Extract box coordinates
            box = detection['box']
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Get class and confidence information
            class_name = detection['class_name']
            confidence = detection['confidence']
            label = f"{class_name} {confidence:.2f}"
            
            # Create a unique color for each class
            class_id = detection['class_id']
            color = (0, 255, 0)  # Default green
            
            if class_id == 2:  # Car
                color = (0, 255, 0)  # Green
            elif class_id == 3:  # Motorcycle
                color = (255, 0, 0)  # Blue
            elif class_id == 5:  # Bus
                color = (0, 0, 255)  # Red
            elif class_id == 7:  # Truck
                color = (255, 255, 0)  # Cyan
                
            # Draw box
            cv2.rectangle(
                annotated_frame, 
                (x1, y1), 
                (x2, y2), 
                color, 
                config.BOX_THICKNESS
            )
            
            # Draw label background
            text_size = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                config.FONT_SIZE, 
                config.FONT_THICKNESS
            )[0]
            
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - text_size[1] - 5), 
                (x1 + text_size[0], y1),
                color, 
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                annotated_frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                config.FONT_SIZE, 
                (0, 0, 0),  # Black text
                config.FONT_THICKNESS
            )
            
        return annotated_frame
    
    def count_vehicles(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count detected vehicles by type.
        
        Args:
            detections: List of detections from detect_vehicles()
            
        Returns:
            Dictionary with counts of each vehicle type
        """
        counts = {}
        for class_name in self.class_names.values():
            counts[class_name] = 0
            
        for detection in detections:
            class_name = detection['class_name']
            counts[class_name] += 1
            
        return counts