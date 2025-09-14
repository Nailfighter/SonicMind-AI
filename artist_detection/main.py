import argparse
import cv2
from src.detection.artist_detector import detect_artists
from src.detection.speaker_detector import detect_speakers
from src.utils.geometry import calculate_distance, get_center
from src.core.system_controller import SystemController

def main(video_source):
    """
    Main loop for the artist tracking application.
    """
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    # Define the minimum distance threshold (in pixels)
    MIN_DISTANCE_THRESHOLD = 150

    # Initialize the system controller for the mute API (5-second dual mute)
    controller = SystemController(mute_duration=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect artists in the current frame
        artist_locations = detect_artists(frame)
        
        # Detect speakers in the current frame (with fallback to top edges)
        speaker_locations = detect_speakers(frame)

        # Draw bounding boxes for each detected artist
        for (x1, y1, x2, y2) in artist_locations:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for artists

        # Draw bounding boxes for each speaker
        for (x1, y1, x2, y2) in speaker_locations:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red for speakers

        # Check for proximity between artists and speakers
        for artist_box in artist_locations:
            for speaker_box in speaker_locations:
                distance = calculate_distance(artist_box, speaker_box)

                # If distance is below the threshold, draw a line and show a warning
                if distance <= MIN_DISTANCE_THRESHOLD:
                    # Get centers to draw a line
                    artist_center = get_center(artist_box)
                    speaker_center = get_center(speaker_box)

                    # Draw a red line between artist and speaker
                    cv2.line(frame, artist_center, speaker_center, (0, 0, 255), 2)

                    # Put a warning text on the screen
                    cv2.putText(frame, "WARNING: Artist too close to speaker!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(f"WARNING: Artist at {artist_center} is too close to speaker at {speaker_center} (Distance: {distance:.2f})")

                    # Trigger the mute API
                    controller.trigger_mute()

        # Get the current system status from the controller
        status = controller.get_status()

        # Display the system status on the top-left of the screen
        status_text = f"STATUS: {status['status_text']}"
        color = (0, 0, 255) if status['muted'] else (0, 255, 0)
        cv2.putText(frame, status_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # If muted, show the remaining cooldown time
        if status['muted']:
            cooldown_text = f"Cooldown: {status['cooldown_remaining']}s"
            cv2.putText(frame, cooldown_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Artist Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artist Tracking Application")
    parser.add_argument("--video_source", type=str, default="0",
                        help="Path to video file or camera ID (default: 0)")
    args = parser.parse_args()

    # Convert video_source to an integer if it's a digit, otherwise it's a file path
    video_source = int(args.video_source) if args.video_source.isdigit() else args.video_source
    
    main(video_source)
