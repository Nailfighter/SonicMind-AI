import yaml
import os
from ultralytics import YOLO
import cv2

# Configuration for model-based detection
CUSTOM_MODEL_PATH = "/home/harsh/hackathon/git/artist-tracking/models/speaker_detection_best.pt"
USE_MODEL_DETECTION = os.path.exists(CUSTOM_MODEL_PATH)

if USE_MODEL_DETECTION:
    print(f"Loading custom speaker detection model: {CUSTOM_MODEL_PATH}")
    speaker_model = YOLO(CUSTOM_MODEL_PATH)
else:
    speaker_model = None

def detect_speakers(frame=None, config_path="configs/stage_layout.yaml", use_model=True, confidence_threshold=0.3):
    """
    Detects speakers using either a trained model or configuration file.
    Falls back to top camera edges if no speakers are found.

    Args:
        frame: The video frame (optional, required for model-based detection)
        config_path (str): The path to the stage layout configuration file.
        use_model (bool): Whether to use model-based detection if available
        confidence_threshold (float): Minimum confidence for speaker detection (default: 0.3)

    Returns:
        A list of bounding boxes for the speakers.
        Each bounding box is a tuple (x1, y1, x2, y2).
    """
    speaker_boxes = []
    
    # Try model-based detection first if available and frame is provided
    if use_model and speaker_model is not None and frame is not None:
        try:
            results = speaker_model(frame)
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf[0])
                    class_name = speaker_model.names[class_id]
                    
                    # Only include speaker detections with sufficient confidence
                    if class_name == 'speaker' and confidence >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        speaker_boxes.append((x1, y1, x2, y2))
            
            print(f"Model detected {len(speaker_boxes)} speakers with confidence >= {confidence_threshold}")
            
            # If we found speakers with good confidence, return them
            if speaker_boxes:
                return speaker_boxes
            
        except Exception as e:
            print(f"Model detection failed, falling back to config: {e}")
    
    # Fallback to configuration-based detection
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return []

    if 'speakers' in config:
        for speaker in config['speakers']:
            if 'box' in speaker:
                speaker_boxes.append(tuple(speaker['box']))
    
    print(f"Config detected {len(speaker_boxes)} speakers")
    
    # If no speakers found through model or config, default to top edges of camera
    if not speaker_boxes and frame is not None:
        height, width = frame.shape[:2]
        
        # Create speaker zones at top edges (left and right)
        # Left top edge
        left_speaker = (0, 0, width // 4, height // 6)
        # Right top edge  
        right_speaker = (3 * width // 4, 0, width, height // 6)
        
        speaker_boxes = [left_speaker, right_speaker]
        print(f"No speakers detected, defaulting to top camera edges: {len(speaker_boxes)} zones")
    
    return speaker_boxes
