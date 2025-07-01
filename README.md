# ActAlign: Zero-Shot Fine-Grained Video Classification via Language-Guided Sequence Alignment
**[📄 Paper](https://arxiv.org/abs/2506.22967) | [🌐 Project Page](https://amir-aghdam.github.io/act-align/)**  

This repository contains code and instructions for reproducing results from our paper:  
**"ActAlign: Zero-Shot Fine-Grained Video Classification via Language-Guided Sequence Alignment"**
<details open>
<summary><strong>🧠 Abstract</strong></summary>

<br>

<p align="justify">
We address the task of zero-shot fine-grained video classification, where no video examples or temporal annotations are available for unseen action classes. While contrastive vision-language models such as SigLIP demonstrate strong open-set recognition via mean-pooled image-text similarity, they fail to capture the temporal structure critical for distinguishing fine-grained activities. We introduce ActAlign, a zero-shot framework that formulates video classification as sequence alignment. For each class, a large language model generates an ordered sub-action sequence, which is aligned with video frames using Dynamic Time Warping (DTW) in a shared embedding space. Without any video-text supervision or fine-tuning, ActAlign achieves 30.5% accuracy on the extremely challenging ActionAtlas benchmark, where human accuracy is only 61.6%. ActAlign outperforms billion-parameter video-language models while using approximately 8x less parameters. These results demonstrate that structured language priors, combined with classical alignment techniques, offer a scalable and general approach to unlocking the open-set recognition potential of vision-language models for fine-grained video understanding.
</p>

</details>

## 🔧 Setup
All scripts assume Python ≥ 3.8 and require the following Python packages:

```bash
pip install numpy pandas opencv-python torch transformers datasets tqdm openai yt_dlp backoff
```

---

## 📁 Overview of Files

| Script                   | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `process_data.py`        | Downloads and processes videos from ActionAtlas, extracts frames, stores metadata |
| `encode_videos.py`       | Encodes video frames into frame-level SIGLIP embeddings                     |
| `generate_subactions.py` | Generates fine-grained subactions for each action label using GPT           |
| `encode_subactions.py`   | Encodes generated subactions using SIGLIP's text encoder                    |
| `seq_alignment.py`       | Performs DTW-based alignment between video frames and subaction embeddings |
| `baseline.py`            | Computes a mean-pooling baseline using cosine similarity                   |

---

## 🚀 Reproduction Instructions

Below is the ordered pipeline to reproduce the results:

### 1. Preprocess Dataset

```bash
python process_data.py --output processed_dataset.npy
```

Downloads and processes the ActionAtlas and saves metadata and frames as a numpy array into `processed_dataset.npy`.

---

### 2. Encode Videos with SIGLIP

```bash
python encode_videos.py --output video_embeddings.npy --device cuda:0
```

Encodes all video frames using SigLIP vision encoder and stores the embeddings.

---

### 3. Generate Subactions Using GPT

```bash
python generate_subactions.py \
    --input processed_dataset.npy \
    --output generated_subactions.json \
    --apikey <your_openai_api_key> \
    --temp 0.2
```

Generates fine-grained subaction descriptions for each action label.

> 📌 **Note:** To modify the prompt, edit the `generate_prompt()` function in `generate_subactions.py`.

---

### 4. Encode Subactions

```bash
python encode_subactions.py \
    --subactions generated_subactions.json \
    --output encoded_subactions.npy \
    --device cuda:0
```

Encodes the generated subaction scripts using SigLIP's text encoder.

---
### 5. Run Baseline (Mean-Pooling + Cosine Similarity)

```bash
python baseline.py \
    --input_dataset processed_dataset.npy \
    --video_embeddings video_embeddings.npy \
    --device cuda:0
```

Evaluates a simple baseline using mean-pooled frame embeddings compared with label text embeddings from the original ActionAtlas classes.

---

### 6. Compute Subaction Alignment Scores (DTW)

```bash
python seq_alignment.py \
    --subactions generated_subactions.json \
    --video_embeddings video_embeddings.npy \
    --subaction_embeddings encoded_subactions.npy \
    --device cuda:0
```

Performs video classification using DTW alignment and evaluates Top-1, Top-2, and Top-3 accuracy.

---



## 📊 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{aghdam2025actalignzeroshotfinegrainedvideo,
    title={ActAlign: Zero-Shot Fine-Grained Video Classification via Language-Guided Sequence Alignment}, 
    author={Amir Aghdam and Vincent Tao Hu},
    year={2025},
    eprint={2506.22967},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.22967},
    }
```

---

## 📬 Contact

Please feel free to reach out to me for any questions or collaboration opportunities using the information on my Github profile.
