import numpy as np
import torch
import re
import argparse
from transformers import AutoProcessor, AutoModel

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

def load_dataset(path):
    print(f"Loading dataset from {path}...")
    dataset = np.load(path, allow_pickle=True)
    dataset = [row[:9] for row in dataset if row is not None]
    return np.array(dataset, dtype=object)

def load_video_embeddings(path):
    print(f"Loading video embeddings from {path}...")
    return np.load(path, allow_pickle=True)

def pool_and_normalize_features(encoded_videos):
    pooled_list = []
    for i, video_features in enumerate(encoded_videos):
        if video_features is None:
            continue
        video_tensor = torch.tensor(video_features, dtype=torch.float32)
        pooled = video_tensor.mean(dim=0)
        norm = pooled.norm(p=2)
        if norm.item() > 0:
            pooled = pooled / norm
        pooled_list.append(pooled)
    return torch.stack(pooled_list, dim=0) if pooled_list else torch.empty(0)

def extract_options_from_metas(metas):
    pattern = re.compile(r'^\s*\d+\.?\s*')
    options_list = []
    answer_indices = []
    for meta in metas:
        raw_options = meta[6].split('\n')
        cleaned_options = [pattern.sub("", option).strip().lower()
                           for option in raw_options if option.strip()]
        cleaned_options = [f"This is a video of {meta[3]} executing {op}" for op in cleaned_options]
        options_list.append(cleaned_options)
        try:
            ans_idx = int(meta[5]) - 1
        except Exception:
            ans_idx = -1
        answer_indices.append(ans_idx)
    return options_list, answer_indices

def encode_options(options, model, processor, device='cuda:0'):
    all_embeddings = []
    for opts in options:
        inputs = processor(text=opts, return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            video_embeds = model.get_text_features(**inputs)
        video_embeds = video_embeds / video_embeds.norm(p=2)
        all_embeddings.append(video_embeds.cpu())
    return all_embeddings

def compute_similarity_for_dataset(video_embeddings, text_embeddings_list, logit_scale=None, logit_bias=None, apply_sigmoid=False):
    similarities = []
    for v_emb, t_emb in zip(video_embeddings, text_embeddings_list):
        sim = v_emb @ t_emb.T
        if logit_scale is not None and logit_bias is not None:
            scale = torch.tensor(logit_scale, device=sim.device).exp()
            bias = torch.tensor(logit_bias, device=sim.device)
            sim = sim * scale + bias
        if apply_sigmoid:
            sim = torch.sigmoid(sim)
        similarities.append(sim)
    return similarities

def compute_accuracy(sim_list, answer_indices, k_list=[1]):
    correct_top1 = 0
    predicted_indices = []
    topk_counts = {k: 0 for k in k_list}
    num_videos = len(sim_list)
    if isinstance(answer_indices, torch.Tensor):
        answer_indices = answer_indices.tolist()

    for sim, true_idx in zip(sim_list, answer_indices):
        pred_idx = torch.argmax(sim).item()
        predicted_indices.append(pred_idx)
        if pred_idx == true_idx:
            correct_top1 += 1
        for k in k_list:
            topk_inds = torch.topk(sim, k=k).indices.tolist()
            if true_idx in topk_inds:
                topk_counts[k] += 1

    top1_accuracy = correct_top1 / num_videos if num_videos > 0 else 0.0
    topk_accuracies = {k: (topk_counts[k] / num_videos if num_videos > 0 else 0.0) for k in k_list}
    return top1_accuracy, predicted_indices, topk_accuracies

def main(args):
    dataset = load_dataset(args.input_dataset)
    video_embeddings_raw = load_video_embeddings(args.video_embeddings)
    video_embeddings = pool_and_normalize_features(video_embeddings_raw)

    options, answer_indices = extract_options_from_metas(dataset)

    print("Loading SIGLIP model and processor...")
    model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    model = model.to(args.device)

    print("Encoding text options...")
    options_embeddings = encode_options(options, model, processor, device=args.device)

    print("Computing similarity scores...")
    logit_scale = model.logit_scale.detach().cpu().item()
    logit_bias = model.logit_bias.detach().cpu().item()
    sims = compute_similarity_for_dataset(video_embeddings, options_embeddings, logit_scale, logit_bias, apply_sigmoid=True)

    print("Evaluating accuracy...")
    accuracy, preds, top_k = compute_accuracy(sims, answer_indices, k_list=[1, 2, 3])
    for k, acc in top_k.items():
        print(f"Top-{k} Accuracy: {acc:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline accuracy using mean-pooled video embeddings and text encoding.")
    parser.add_argument("--input_dataset", required=True, help="Path to the processed dataset (.npy)")
    parser.add_argument("--video_embeddings", required=True, help="Path to the video embeddings (.npy)")
    parser.add_argument("--device", default="cuda:0", help="Torch device to use (e.g., cuda:0 or cpu)")
    args = parser.parse_args()
    main(args)
