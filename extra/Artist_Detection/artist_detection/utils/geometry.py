import math

def get_center(box):
    """Calculates the center of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def calculate_distance(box1, box2):
    """
    Calculates the Euclidean distance between the centers of two bounding boxes.
    """
    center1 = get_center(box1)
    center2 = get_center(box2)
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
