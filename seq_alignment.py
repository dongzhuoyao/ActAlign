import numpy as np
import json
import torch
import torch.nn.functional as F
from transformers import AutoModel
from tqdm import tqdm
import argparse

def to_tensor(x, device, dtype=torch.float32):
    """
    Converts input to a torch tensor on the specified device and dtype.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=dtype)
    else:
        return torch.tensor(x, dtype=dtype).to(device)

def get_sim_matrix(
    video_seq, candidate_seq, logit_scale, logit_bias,
    device='cuda:0', apply_sigmoid=False, smooth=False, smooth_kernel_size=30
):
    """
    Computes the similarity matrix between video frame embeddings and subaction embeddings.
    Applies optional temporal smoothing over the video frames.
    
    Returns:
      sim_matrix: (T_frames, T_actions) similarity matrix
    """
    video = to_tensor(video_seq, device)
    cand  = to_tensor(candidate_seq, device)

    # Optional temporal smoothing
    if smooth and smooth_kernel_size > 1:
        T, D = video.shape
        pad = (smooth_kernel_size - 1) // 2
        v = video.t()[None]  # shape: [1, D, T]
        weight = torch.ones(D, 1, smooth_kernel_size, device=video.device) / smooth_kernel_size
        video = F.conv1d(v, weight=weight, padding=pad, groups=D)[0].t()  # shape: (T, D)

    # Normalize
    video = video / video.norm(p=2, dim=1, keepdim=True)
    cand  = cand  / cand.norm(p=2, dim=1, keepdim=True)

    sim = video @ cand.T  # cosine similarity

    if logit_scale is not None and logit_bias is not None:
        scale = torch.tensor(logit_scale, device=sim.device).exp()
        bias  = torch.tensor(logit_bias, device=sim.device)
        sim = sim * scale + bias

    if apply_sigmoid:
        sim = torch.sigmoid(sim)

    return sim.cpu()

def dtw_max_similarity(similarity_matrix):
    """
    Computes DTW alignment score to maximize total similarity.
    Returns average similarity score and alignment path.
    """
    T, F = similarity_matrix.shape
    dp = np.full((T, F), -np.inf)
    dp[0, 0] = similarity_matrix[0, 0]

    for i in range(T):
        for j in range(F):
            if i == 0 and j == 0:
                continue
            best_prev = max(
                dp[i - 1, j] if i > 0 else -np.inf,
                dp[i, j - 1] if j > 0 else -np.inf,
                dp[i - 1, j - 1] if i > 0 and j > 0 else -np.inf
            )
            dp[i, j] = best_prev + similarity_matrix[i, j]

    # Backtrack for alignment path
    i, j = T - 1, F - 1
    path = [(i, j)]
    while (i, j) != (0, 0):
        choices = []
        if i > 0 and j > 0:
            choices.append(((i - 1, j - 1), dp[i - 1, j - 1]))
        if i > 0:
            choices.append(((i - 1, j), dp[i - 1, j]))
        if j > 0:
            choices.append(((i, j - 1), dp[i, j - 1]))
        (i, j), _ = max(choices, key=lambda x: x[1])
        path.append((i, j))
    path.reverse()

    total_score = dp[T - 1, F - 1]
    avg_score = total_score / len(path)
    return avg_score, path

def compute_alignment_scores(encoded_videos, subactions_embeddings, logit_scale, logit_bias, device='cuda:0'):
    """
    For each sample, computes DTW-based alignment scores between the video and each option's subaction sequence.

    Returns:
      List of lists: each inner list contains alignment scores for the candidate options.
    """
    scores = []
    for i in tqdm(range(len(encoded_videos)), desc="Computing Alignment Scores"):
        video_seq = encoded_videos[i]
        candidate_seqs = subactions_embeddings[i]
        sample_scores = []
        for candidate_seq in candidate_seqs:
            sim_matrix = get_sim_matrix(
                video_seq, candidate_seq, logit_scale, logit_bias,
                device=device, apply_sigmoid=True, smooth=False
            )
            avg_score, _ = dtw_max_similarity(sim_matrix.numpy())
            sample_scores.append(avg_score)
        scores.append(sample_scores)
    return scores

def compute_accuracy(score_list, answer_indices, k_list=[1]):
    """
    Computes top-k accuracy from alignment scores and ground truth indices.
    """
    topk_counts = {k: 0 for k in k_list}
    num_samples = len(score_list)

    for scores, true_idx in zip(score_list, answer_indices):
        sims = torch.tensor(scores)
        for k in k_list:
            topk = torch.topk(sims, k).indices.tolist()
            if true_idx in topk:
                topk_counts[k] += 1

    return {k: (topk_counts[k] / num_samples if num_samples > 0 else 0.0) for k in k_list}

def main(args):
    print(f"Loading subactions from {args.subactions}...")
    with open(args.subactions, "r") as f:
        subactions_dataset = json.load(f)

    print(f"Loading video embeddings from {args.video_embeddings}...")
    encoded_videos = np.load(args.video_embeddings, allow_pickle=True)

    print(f"Loading subaction embeddings from {args.subaction_embeddings}...")
    subaction_embeddings = np.load(args.subaction_embeddings, allow_pickle=True)

    print("Loading model for logit scale/bias...")
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384").to(args.device)
    logit_scale = model.logit_scale.detach().cpu().item()
    logit_bias = model.logit_bias.detach().cpu().item()

    alignment_scores = compute_alignment_scores(
        encoded_videos,
        subaction_embeddings,
        logit_scale,
        logit_bias,
        device=args.device
    )

    ground_truth = [sample['correct_answer_index'] for sample in subactions_dataset]
    acc = compute_accuracy(alignment_scores, ground_truth, k_list=[1, 2, 3])

    print("\n Accuracy Results:")
    for k, a in acc.items():
        print(f"Top-{k} Accuracy: {a:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute DTW-based alignment accuracy for subactions.")
    parser.add_argument("--subactions", required=True, help="Path to generated subactions JSON.")
    parser.add_argument("--video_embeddings", required=True, help="Path to video embeddings (.npy).")
    parser.add_argument("--subaction_embeddings", required=True, help="Path to encoded subactions (.npy).")
    parser.add_argument("--device", default="cuda:0", help="Torch device to use (e.g., cuda:0 or cpu).")
    args = parser.parse_args()

    main(args)
