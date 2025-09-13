import cv2
import os
import shutil
from datetime import datetime


def capture_all_cameras(save_folder="captured_images", camera_indices=[0, 1, 2]):
    """
    Captures one frame from each camera index, saves it with timestamp,
    and clears previous images in the save folder.
    """

    # Remove old folder and create new
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    for i, cam_idx in enumerate(camera_indices):
        cap = cv2.VideoCapture(cam_idx)

        if not cap.isOpened():
            print(f"Warning: Could not open camera {cam_idx}")
            continue

        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to grab frame from camera {cam_idx}")
            cap.release()
            continue

        # Generate filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"camera{i+1}_{timestamp}.jpg"
        filepath = os.path.join(save_folder, filename)

        # Save image
        cv2.imwrite(filepath, frame)
        print(f"Saved: {filepath}")

        cap.release()

    print("Capture complete. All cameras processed.")
    
capture_all_cameras()
