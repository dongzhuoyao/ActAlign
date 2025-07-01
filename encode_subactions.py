import numpy as np
import json
import torch
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import argparse
import os

def encode_subactions(subaction_data, output_file, model, processor, device='cuda:0'):
    """
    Encodes subaction descriptions for each sample using a SIGLIP text encoder.

    Each sample in subaction_data must contain:
      - "cleaned_options": list of action names (e.g., ['kickflip', 'heelflip'])
      - "subactions": dict mapping "option in domain" -> list of subaction phrases
      - "domain": the activity/sport (e.g., 'skateboarding')

    For each action that exists in the subactions dictionary, the subaction phrases are augmented
    with context and encoded into text embeddings.

    Parameters:
      subaction_data (list): List of dicts, each representing a sample with subaction info.
      output_file (str): Path to save the resulting NumPy object array.
      model: SIGLIP model with a text encoder.
      processor: Corresponding processor for tokenizing text.
      device (str): Torch device to run encoding on.

    Returns:
      all_embeddings_np (np.array): Object array; each element is a list of NumPy arrays
                                    (one per action option with subactions).
    """
    all_embeddings = []

    for sample in tqdm(subaction_data, desc="Encoding Subactions"):
        cleaned_options = sample.get("cleaned_options", [])
        subactions_dict = sample.get("subactions", {})
        domain = sample.get("domain", "")
        sample_id = sample.get("id", "unknown")

        sample_embeddings = []

        for opt in cleaned_options:
            key = f"{opt} in {domain}"
            if key not in subactions_dict:
                print(f"Missing key '{key}' in sample ID {sample_id}")
                continue

            subaction_list = subactions_dict[key]
            # Augment each subaction with context
            subaction_texts = [f"This is a video of doing {opt} in {domain} with {sa}" for sa in subaction_list]

            try:
                inputs = processor(text=subaction_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.text_model(**inputs)
                    text_embeds = outputs[1]  # Expected shape: (num_subactions, feature_dim)
                emb_np = text_embeds.cpu().numpy()
                sample_embeddings.append(emb_np)
            except Exception as e:
                print(f"Error encoding sample ID {sample_id}, option '{opt}': {e}")
                continue

        all_embeddings.append(sample_embeddings)

    all_embeddings_np = np.array(all_embeddings, dtype=object)
    np.save(output_file, all_embeddings_np, allow_pickle=True)
    print(f"Saved subaction embeddings for {len(all_embeddings)} samples to {output_file}.")
    return all_embeddings_np

def load_subactions(path):
    print(f"Loading subactions from {path}...")
    with open(path, "r") as f:
        return json.load(f)

def load_model_and_processor(model_name, device):
    print(f"Loading model '{model_name}' on {device}...")
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def main(args):
    subactions_dataset = load_subactions(args.subactions)
    model, processor = load_model_and_processor("google/siglip-so400m-patch14-384", args.device)
    encode_subactions(subactions_dataset, args.output, model, processor, device=args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode generated subactions using SIGLIP text encoder.")
    parser.add_argument("--subactions", required=True, help="Path to generated subactions JSON file.")
    parser.add_argument("--output", required=True, help="Path to save the encoded embeddings (.npy).")
    parser.add_argument("--device", default="cuda:0", help="Torch device to use (e.g., cuda:0, cpu).")
    args = parser.parse_args()

    main(args)
