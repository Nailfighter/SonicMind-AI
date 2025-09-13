import argparse
import torch
from torch.optim import AdamW
from transformers import CLIPProcessor, CLIPModel, get_scheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
import json
import os
from tqdm.auto import tqdm
import ast
from torch.cuda.amp import autocast, GradScaler


class MaterialDataset(Dataset):
    def __init__(self, annotations_path, img_dir, dataset_fraction=1.0):
        with open(annotations_path, 'r') as f:
            # The file is a single JSON object with an "annotations" key
            data = json.load(f)
            self.annotations = data['annotations']

        # Use only a fraction of the dataset if specified
        if dataset_fraction < 1.0:
            num_samples = int(len(self.annotations) * dataset_fraction)
            self.annotations = self.annotations[:num_samples]
            print(
                f"Using {num_samples} samples ({dataset_fraction*100:.0f}% of the dataset).")

        self.img_dir = img_dir

        # Create a set of unique material labels
        self.labels = sorted(
            list(set(anno['class_label'] for anno in self.annotations)))
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        img_path = os.path.join(self.img_dir, annotation['image'])
        image = PILImage.open(img_path).convert("RGB")

        label_text = annotation['class_label']
        label_id = self.label_to_id[label_text]

        return {"image": image, "label_id": label_id, "label_text": label_text}


def main(args):
    # --- Data Loading and Preprocessing ---

    # Load custom dataset
    full_dataset = MaterialDataset(
        annotations_path=os.path.join(
            args.dataset_path, "..", "annotations.json"),
        img_dir=args.dataset_path,
        dataset_fraction=args.dataset_fraction
    )

    # Split dataset into training and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    # --- Model Loading ---
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name)

    # --- Collate Function for DataLoaders ---
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        texts = [f"a photo of {item['label_text']}" for item in batch]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs['labels'] = torch.tensor(
            [item['label_id'] for item in batch], dtype=torch.long)
        return inputs

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)

    # --- Training ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-6)
    scaler = GradScaler()

    num_training_steps = args.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # --- Checkpoint Loading ---
    start_epoch = 0
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
    if args.resume_from_checkpoint and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Load the scaler state
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    progress_bar = tqdm(
        range(start_epoch * len(train_dataloader), num_training_steps))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            # Use autocast for mixed-precision
            with autocast():
                outputs = model(**batch)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                ground_truth = torch.arange(
                    len(logits_per_image), device=device)
                loss = (torch.nn.functional.cross_entropy(logits_per_image, ground_truth) +
                        torch.nn.functional.cross_entropy(logits_per_text, ground_truth)) / 2

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # --- Evaluation ---
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                with autocast():
                    outputs = model(**batch)
                    logits_per_image = outputs.logits_per_image
                    logits_per_text = outputs.logits_per_text
                    ground_truth = torch.arange(
                        len(logits_per_image), device=device)
                    loss = (torch.nn.functional.cross_entropy(logits_per_image, ground_truth) +
                            torch.nn.functional.cross_entropy(logits_per_text, ground_truth)) / 2
                total_loss += loss.item()

        avg_val_loss = total_loss / len(val_dataloader)
        print(
            f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {avg_val_loss:.4f}")

        # --- Save Checkpoint ---
        print(f"Saving checkpoint for epoch {epoch}...")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),  # Save the scaler state
        }, checkpoint_path)

    # Save the fine-tuned model
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Save the labels
    with open(os.path.join(args.output_dir, 'labels.json'), 'w') as f:
        json.dump(full_dataset.labels, f)

    print(f"Fine-tuned model and processor saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a CLIP model.")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Material_dataset/JPEGImages",
        help="Path to the root of your image dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Name of the pretrained CLIP model to use."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fine-tuned-clip",
        help="Directory to save the fine-tuned model."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size. Adjust based on GPU memory."
    )
    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use for training (e.g., 0.5 for 50%)."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Whether to resume training from the last checkpoint."
    )

    args = parser.parse_args()
    main(args)
