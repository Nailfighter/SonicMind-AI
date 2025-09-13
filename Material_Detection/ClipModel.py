import os
import torch
import clip
from PIL import Image

# Pick device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model + preprocess
# ViT-L/14 is a larger, more powerful model.
model, preprocess = clip.load("ViT-L/14", device=device)

# Folder where OpenCV saved images

image_folder = "captured_images"

# Use prompt ensembling for materials. Each material has multiple descriptive prompts.
material_templates = {
    "Wood": [
        "a photo of wood texture", "a close-up of a wooden surface", "natural wood grain"
    ],
    "Metal": [
        "a photo of brushed metal", "a shiny metallic surface", "a sheet of metal"
    ],
    "Glass": [
        "a photo of clear glass", "a reflective glass pane", "a glass window"
    ],
    "Concrete": [
        "a photo of rough concrete", "a solid concrete wall", "a concrete floor"
    ],
    "Fabric": [
        "a photo of soft fabric", "a woven textile material", "a cloth texture"
    ],
    "Plastic": [
        "a photo of smooth plastic", "a molded plastic object", "a sheet of plastic"
    ],
    "Person": [
        "a photo of a person", "a silhouette of a person", "a person standing"
    ]
}

# Define acoustic properties with descriptive prompts
acoustic_labels = [
    "a photo of a hard, sound-reflective surface",
    "a photo of a soft, sound-absorbent surface",
    "a photo of a textured, sound-diffusing surface"
]


def classify_image_ensembled(image_features, templates):
    """
    Classifies an image using prompt ensembling.
    Calculates the average similarity across multiple prompts for each class.
    """
    class_predictions = {}

    for class_name, prompts in templates.items():
        text_tokens = clip.tokenize(prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity for all prompts in the class
            similarities = (100.0 * image_features @
                            text_features.T).softmax(dim=-1)

            # Average the similarities for the class
            mean_similarity = similarities.mean().item()
            class_predictions[class_name] = mean_similarity

    # Sort predictions by confidence
    sorted_predictions = sorted(
        class_predictions.items(), key=lambda item: item[1], reverse=True
    )

    return sorted_predictions


def classify_image(image_features, text_labels):
    """Classifies an image given its features and a list of text labels."""
    text_tokens = clip.tokenize(text_labels).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    # Normalize features
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Get top predictions
    k = min(5, len(text_labels))
    values, indices = torch.topk(similarities[0], k=k)

    return [(text_labels[i], v.item()) for i, v in zip(indices, values)]


# Loop through each saved image in the folder
for img_name in os.listdir(image_folder):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue  # skip non-image files

    img_path = os.path.join(image_folder, img_name)

    # Preprocess image
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    # Encode image
    with torch.no_grad():
        image_features = model.encode_image(image)

    # Normalize image features
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # Classify material and acoustic properties
    material_predictions = classify_image_ensembled(
        image_features, material_templates)
    acoustic_predictions = classify_image(image_features, acoustic_labels)

    # --- Analysis ---
    print(f"--- Analysis for {img_name} ---")

    # Filter out "Person" from material predictions
    material_predictions = [
        pred for pred in material_predictions if pred[0] != "Person"]

    # Best material prediction (after filtering)
    if material_predictions:
        best_material, material_confidence = material_predictions[0]
        print(
            f"Material: {best_material} (confidence: {material_confidence:.2f})")
    else:
        print("Material: Could not determine a material other than a person.")

    # Best acoustic prediction
    best_acoustic, acoustic_confidence = acoustic_predictions[0]
    print(
        f"Acoustic Property: {best_acoustic} (confidence: {acoustic_confidence:.2f})\n")
