import argparse
import os
import tempfile
import numpy as np
import pandas as pd
import cv2
import yt_dlp
import logging
from yt_dlp.utils import DownloadError
from datasets import load_dataset

# --- Set up logging ---
log_filename = "processing_errors.log"
logging.basicConfig(filename=log_filename,
                    filemode="a",  # Append mode
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# --- Function to download video using yt_dlp ---
def download_youtube_video(youtube_id, output_dir, cookies_file=None):
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    ydl_opts = {
        'outtmpl': os.path.join(output_dir, f'{youtube_id}.%(ext)s'),
        'format': 'best[ext=mp4]',
        'quiet': True,
    }
    if cookies_file:
        ydl_opts['cookies'] = cookies_file
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return os.path.join(output_dir, f"{youtube_id}.mp4")
    except DownloadError as e:
        error_msg = f"DownloadError for video {youtube_id}: {e}"
        logging.error(error_msg)
        print(error_msg)
        return None

# --- Function to extract frames using OpenCV ---
def extract_frames_cv2(video_path, start_ts=0.0, end_ts=None, obfuscation_dict=None):
    """
    Reads the video from video_path using OpenCV, extracts frames between start_ts and end_ts,
    applies region obfuscation if obfuscation_dict is provided, and returns a NumPy array of frames.
    The returned array has shape (num_frames, channels, height, width) in RGB.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_ts is None:
        end_ts = total_frames / fps
    start_frame = int(start_ts * fps)
    end_frame = int(end_ts * fps)
    
    # If obfuscation_dict is provided, ensure its keys are integers
    if obfuscation_dict and isinstance(obfuscation_dict, dict):
        obfuscation_dict = {int(k): v for k, v in obfuscation_dict.items() if v is not None}
    
    frames = []
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame >= start_frame and current_frame < end_frame:
            # Convert frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if obfuscation_dict and current_frame in obfuscation_dict:
                frame = obfuscate_frame(frame, obfuscation_dict[current_frame])
            frames.append(frame)
        current_frame += 1
        if current_frame >= end_frame:
            break
    cap.release()
    if not frames:
        raise ValueError(f"No frames were read from {video_path}")
    video_np = np.stack(frames, axis=0)  # (num_frames, height, width, channels)
    video_np = np.transpose(video_np, (0, 3, 1, 2))  # (num_frames, channels, height, width)
    return video_np, fps

# --- Function to write video from frames using OpenCV ---
def write_video_from_frames(frames_np, fps, output_path):
    """
    Writes a video file from a NumPy array of frames.
    frames_np should have shape (num_frames, channels, height, width).
    """
    frames = [cv2.cvtColor(np.transpose(frame, (1, 2, 0)), cv2.COLOR_RGB2BGR) for frame in frames_np]
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return output_path

# --- Dummy obfuscation functions ---
def obfuscate_frame(frame, instructions):
    """
    Applies a simple Gaussian blur to the region specified by each instruction.
    Each instruction should have a 'bounding_poly' with 'vertices' (list of [x, y] pairs).
    """
    for instr in instructions:
        vertices = instr.get("bounding_poly", {}).get("vertices", [])
        if not vertices:
            continue
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
            frame[y_min:y_max, x_min:x_max] = blurred_roi
    return frame

# --- Modified process_dataset_entry using OpenCV exclusively ---
def process_dataset_entry(entry, output_dir=None, cookies_file=None):
    """
    Processes a dataset entry:
      - Downloads the YouTube video using yt_dlp (using cookies if provided).
      - Extracts frames between start_timestamp and end_timestamp using OpenCV.
      - If 'frames_to_obfuscate' is provided as a dict, applies region obfuscation.
      - Writes the processed video to disk.
      - Returns a dictionary with metadata, the processed video path, and a NumPy array of video frames.
    
    If the video cannot be downloaded or processed, returns a dict with an 'error' key.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="processed_video_")
    
    youtube_id = entry.get('youtube_id')
    video_path = download_youtube_video(youtube_id, output_dir, cookies_file=cookies_file)
    if video_path is None or not os.path.exists(video_path):
        error_msg = f"Video {youtube_id} could not be downloaded (possibly private)."
        logging.error(error_msg)
        return {
            'youtube_id': youtube_id,
            'id': entry.get('id'),
            'error': error_msg
        }
    
    start_ts = entry.get('start_timestamp', 0)
    try:
        video_frames, fps = extract_frames_cv2(video_path, start_ts=start_ts, end_ts=entry.get('end_timestamp'), obfuscation_dict=entry.get('frames_to_obfuscate'))
    except Exception as e:
        error_msg = f"Error reading frames for video {youtube_id}: {e}"
        logging.error(error_msg)
        return {
            'youtube_id': youtube_id,
            'id': entry.get('id'),
            'error': error_msg
        }
    
    processed_video_path = os.path.join(output_dir, f"{youtube_id}_processed.mp4")
    try:
        write_video_from_frames(video_frames, fps, processed_video_path)
    except Exception as e:
        error_msg = f"Error writing video file for video {youtube_id}: {e}"
        logging.error(error_msg)
        return {
            'youtube_id': youtube_id,
            'id': entry.get('id'),
            'error': error_msg
        }
    
    result = {
        'youtube_id': youtube_id,
        'id': entry.get('id'),
        'action': entry.get('action'),
        'domain': entry.get('domain'),
        'question': entry.get('question'),
        'answer': entry.get('answer'),
        'choices_str': entry.get('choices_str'),
        'choice_descriptions': entry.get('choice_descriptions'),
        'processed_video_path': processed_video_path,
        'video_frames': video_frames  # NumPy array representation
    }
    
    return result

# --- New function: Process dataset and save DataFrame to .npy files ---
def process_dataset_and_save_df(dataset, output_npy_file, missing_npy_file, output_dir=None, cookies_file=None):
    """
    Processes a list of dataset entries sequentially.
    For each entry, if processing is successful, the result dictionary (including the NumPy video frames)
    is appended to a list for successful entries.
    If processing fails (e.g., video cannot be downloaded or read), the entry is appended to a separate list.
    
    The function converts each list into a Pandas DataFrame and saves each DataFrame as a .npy file.
    
    Parameters:
      dataset (HuggingFace dataset): The dataset entries.
      output_npy_file (str): Path to the .npy file for successful entries.
      missing_npy_file (str): Path to the .npy file for missing/failed entries.
      output_dir (str, optional): Directory for intermediate video files.
      cookies_file (str, optional): Path to a cookies file for yt_dlp if needed.
    
    Returns:
      (df_success, df_missing): Tuple of DataFrames.
    """
    results_success = []
    results_missing = []
    
    for entry in dataset:
        processed = process_dataset_entry(entry, output_dir=output_dir, cookies_file=cookies_file)
        if 'error' in processed:
            results_missing.append(processed)
        else:
            results_success.append(processed)
    
    df_success = pd.DataFrame(results_success)
    df_missing = pd.DataFrame(results_missing)
    
    np.save(output_npy_file, df_success, allow_pickle=True)
    np.save(missing_npy_file, df_missing, allow_pickle=True)
    
    return df_success, df_missing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ActionAtlas and save results to a .npy file.")
    parser.add_argument("--output", required=True, help="Path to the output .npy file for successful entries.")
    args = parser.parse_args()

    # Derive missing file name based on output name
    base, ext = os.path.splitext(args.output)
    missing_npy_file = base + "_missing" + ext

    # Load the Hugging Face dataset (using the test split of ActionAtlas-v1.0)
    ds = load_dataset("mrsalehi/ActionAtlas-v1.0")['test']

    df_success, df_missing = process_dataset_and_save_df(
        ds,
        output_npy_file=args.output,
        missing_npy_file=missing_npy_file
    )

    print("DataFrame for successful entries saved to", args.output)
    print("DataFrame for missing entries saved to", missing_npy_file)