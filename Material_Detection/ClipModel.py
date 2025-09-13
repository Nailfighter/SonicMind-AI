import os
import torch
import clip
from PIL import Image

# Pick device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (ViT-B/32 is light; you can also try ViT-L/14)
model, preprocess = clip.load("ViT-B/32", device=device)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model + preprocess
model, preprocess = clip.load("ViT-B/32", device=device)

# Folder where OpenCV saved images
image_folder = "captured_images"

# Define possible categories
text_labels = ["a smooth surface", "a rough surface", "a cracked surface"]

# Tokenize text once
text_tokens = clip.tokenize(text_labels).to(device)

# Loop through each saved image in the folder
for img_name in os.listdir(image_folder):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue  # skip non-image files

    img_path = os.path.join(image_folder, img_name)

    # Preprocess image
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    # Encode image + text
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Best match
    best_idx = similarities.argmax().item()
    confidence = similarities[0][best_idx].item()

    print(f"[{img_name}] → Prediction: {text_labels[best_idx]} (confidence: {confidence:.2f})")
