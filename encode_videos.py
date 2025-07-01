import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
from tqdm import tqdm
import logging
import argparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Column mapping (adjust if needed)
column_index = {
    "youtube_id": 0,
    "id": 1,
    "action": 2,
    "domain": 3,
    "question": 4,
    "answer": 5,
    "choices_str": 6,
    "choice_descriptions": 7,
    "processed_video_path": 8,
    "video_frames": 9
}


def encode_videos(dataset, model, processor, output_file, device='cuda:0', batch_size=16):
    """
    Encodes each video's frames independently using the SIGLIP image encoder in batches.

    Parameters:
      dataset: Iterable of entries (e.g. a NumPy array of rows) with video frames stored in column index 9.
      model: The SIGLIP model (loaded on GPU).
      processor: The corresponding processor.
      output_file (str): Path to save the final NumPy file (object array of embeddings).
      device (str): The GPU device to use.
      batch_size (int): Number of frames to process at once.

    Returns:
      all_embeddings_np: A NumPy object array, where each element is a NumPy array of shape
                         (num_frames, feature_dim) for that video.
    """
    col_idx = {"video_frames": 9}
    all_embeddings = []  # List to hold per-video embeddings

    for i, row in tqdm(enumerate(dataset), total=len(dataset), desc="Encoding Videos"):
        try:
            # Get video frames (assumed shape: (num_frames, channels, height, width))
            video_frames = row[col_idx["video_frames"]]
            # Convert each frame to a PIL image: from (C, H, W) to (H, W, C)
            frames_list = [Image.fromarray(frame.transpose(1, 2, 0).astype(np.uint8))
                           for frame in video_frames]

            # Process frames in batches to avoid OOM errors
            features_list = []
            for j in range(0, len(frames_list), batch_size):
                batch = frames_list[j : j + batch_size]
                inputs = processor(images=batch, padding="max_length", return_tensors="pt")
                # Ensure inputs are on the correct device and dtype
                dtype = next(model.parameters()).dtype
                pixel_values = inputs["pixel_values"].to(device, dtype=dtype)
                with torch.no_grad():
                    batch_features = model.get_image_features(pixel_values)
                features_list.append(batch_features.cpu())
            # Concatenate all batches
            features = torch.cat(features_list, dim=0)
            features_np = features.numpy()
            all_embeddings.append(features_np)
            logging.info(f"Processed video {i+1}/{len(dataset)}")
        except Exception as e:
            logging.error(f"Error processing video at index {i}: {e}")
            all_embeddings.append(None)

    # Convert list to NumPy object array and save
    all_embeddings_np = np.array(all_embeddings, dtype=object)
    np.save(output_file, all_embeddings_np, allow_pickle=True)
    logging.info(f"Saved embeddings for {len(dataset)} videos to {output_file}")
    return all_embeddings_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode videos with SIGLIP and save embeddings.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the output embeddings (.npy file)."
    )
    parser.add_argument(
        "--device",
        default="cuda:4",
        help="Torch device to use (e.g., cuda:0 or cpu). Default is cuda:4."
    )
    args = parser.parse_args()

    # Load SIGLIP model and processor
    model_name = "google/siglip-so400m-patch14-384"
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    # Load dataset from default input file
    dataset = np.load("processed_dataset.npy", allow_pickle=True)

    # Move model to the specified device
    model = model.to(args.device)

    # Run encoding
    embeddings = encode_videos(
        dataset,
        model,
        processor,
        output_file=args.output,
        device=args.device,
        batch_size=32
    )

    print(f"Embeddings saved to {args.output}")
