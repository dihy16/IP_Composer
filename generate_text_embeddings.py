import torch
import numpy as np
import csv
import argparse
import open_clip
from typing import List, Callable
from torch import nn

def load_descriptions(file_path: str) -> List[str]:
    """Load descriptions from a CSV file."""
    descriptions = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            descriptions.append(row[0])
    return descriptions

def generate_embeddings(
    descriptions: List[str],
    model: nn.Module,
    tokenizer: Callable[[List[str]], torch.Tensor],
    device: str,
    batch_size: int
) -> np.ndarray:
    """Generate text embeddings in batches."""
    final_embeddings = []
    for i in range(0, len(descriptions), batch_size):
        batch_desc = descriptions[i:i + batch_size]
        texts = tokenizer(batch_desc).to(device)
        batch_embeddings = model.encode_text(texts)
        batch_embeddings = batch_embeddings.detach().cpu().numpy()
        final_embeddings.append(batch_embeddings)
        del texts, batch_embeddings
        torch.cuda.empty_cache()
    return np.vstack(final_embeddings)

def save_embeddings(output_file: str, embeddings: np.ndarray) -> None:
    """Save embeddings to a .npy file."""
    np.save(output_file, embeddings)

def main():
    parser = argparse.ArgumentParser(description="Generate text embeddings using CLIP.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file containing text descriptions.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output .npy file.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing embeddings.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (e.g., 'cuda:0' or 'cpu').")
    
    args = parser.parse_args()

    # Load the CLIP model and tokenizer
    model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    model.to(args.device)
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    # Load descriptions from CSV
    descriptions = load_descriptions(args.input_csv)

    # Generate embeddings
    embeddings = generate_embeddings(descriptions, model, tokenizer, args.device, args.batch_size)

    # Save embeddings to output file
    save_embeddings(args.output_file, embeddings)

if __name__ == "__main__":
    main()
