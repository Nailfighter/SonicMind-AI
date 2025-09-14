import cv2
import os
from ultralytics import YOLO
from pathlib import Path
import warnings

# Fix PyTorch 2.8 + ultralytics compatibility issue
import torch
# Set environment variables before any torch operations
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
os.environ['ULTRALYTICS_TRACEBACK'] = '0'
warnings.filterwarnings('ignore', category=FutureWarning)

# For PyTorch 2.8 compatibility, we need to force legacy loading
# This tells ultralytics to use the old loading behavior
import ultralytics.utils.patches
ultralytics.utils.patches.TORCH_WEIGHTS_ONLY = False

# Configuration for model selection
CUSTOM_MODEL_PATH = "models/speaker_detection_best.pt"
DEFAULT_MODEL = "yolov8n.pt"

# Load custom model if available, otherwise use pre-trained model
if os.path.exists(CUSTOM_MODEL_PATH):
    print(f"Loading custom trained model: {CUSTOM_MODEL_PATH}")
    model = YOLO(CUSTOM_MODEL_PATH)
else:
    print(f"Custom model not found, using default: {DEFAULT_MODEL}")
    model = YOLO(DEFAULT_MODEL)

def detect_artists(frame):
    """
    Detects artists (people) in a given frame using a YOLO model.

    Args:
        frame: The video frame (as a NumPy array) to process.

    Returns:
        A list of bounding boxes for detected people.
        Each bounding box is a tuple (x1, y1, x2, y2).
    """
    # Perform inference on the frame
    results = model(frame)

    artist_boxes = []
    # The results object contains the detections
    for result in results:
        # Each result has a 'boxes' attribute with the bounding box info
        for box in result.boxes:
            # Get the class ID of the detected object
            class_id = int(box.cls)
            # The 'person' class in COCO dataset is ID 0
            if model.names[class_id] == 'person':
                # Get the coordinates of the bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                artist_boxes.append((x1, y1, x2, y2))

    return artist_boxes