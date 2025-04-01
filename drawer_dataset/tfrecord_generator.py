import tensorflow as tf
import cv2
import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
TRAIN_JSON = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/labels/train.json"
EVAL_JSON = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/labels/evaluation.json"
TEST_JSON = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/labels/test.json"  # New test set
LABELS_JSON = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/labels/labels.json"

TRAIN_VIDEO_DIR = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/train/video/"
EVAL_VIDEO_DIR = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/eval/video/"
TEST_VIDEO_DIR = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/test/video/"  # Test videos repository
OUTPUT_DIR = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/tfrecords/"

IMAGE_SIZE = (224, 224)  # Standard frame size
FRAME_SKIP = 2  # Extract every Nth frame
FILTER_LABEL = 0  # Set to integer (e.g., 0,1,2) or None for all

# Load label mapping
with open(LABELS_JSON, "r") as f:
    label_map = json.load(f)
    label_map = {k: int(v) for k, v in label_map.items()}  # Convert to integer labels
    label_map = {v: k for k, v in label_map.items()}  # Invert mapping for TFRecord

# --- Function to Process a Single Video ---
def video_to_tfrecord(video_path, label, num_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # If num_frames is not provided, set it based on the video length
    if num_frames is None:
        num_frames = min(16, total_frames)  # Set to 16 or total_frames, whichever is smaller

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frame_indices = frame_indices[::FRAME_SKIP]  # Skip frames based on the FRAME_SKIP value

    frames = []
    timestamps = []

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMAGE_SIZE)  # Resize
        _, buffer = cv2.imencode(".jpg", frame)  # Encode as JPEG
        frames.append(buffer.tobytes())  # Store as bytes
        timestamps.append(int(i))  # Save timestamp

    cap.release()

    if len(frames) < num_frames // 2:
        print(f"Warning: Not enough frames in {video_path}, skipping...")
        return None  # Skip if too few frames

    # Create TFRecord Example
    feature = {
        'data_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_path.encode()])),
        'image/frame_rate': tf.train.Feature(float_list=tf.train.FloatList(value=[frame_rate])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=frames)),
        'image/timestamp': tf.train.Feature(int64_list=tf.train.Int64List(value=timestamps)),
        'task_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()  # Return serialized TFRecord


# --- Function to Process Dataset ---
def process_dataset(json_path, video_dir, output_tfrecord, is_test=False, num_frames=None):
    with open(json_path, "r") as f:
        data = json.load(f)

    video_paths = []
    labels = []

    for item in data:
        video_id = item["id"]
        label_name = item.get("label", None)  # No label for test set
        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            continue

        if not is_test:
            if label_name not in label_map:
                print(f"Skipping unknown label: {label_name}")
                continue

            label = label_map[label_name]
            if FILTER_LABEL is not None and label != FILTER_LABEL:
                continue
        else:
            label = -1  # Dummy label for test videos

        video_paths.append(video_path)
        labels.append(label)

    # --- Parallel Processing ---
    serialized_examples = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_video = {executor.submit(video_to_tfrecord, v, l, num_frames): v for v, l in zip(video_paths, labels)}
        
        for future in as_completed(future_to_video):
            result = future.result()
            if result is not None:
                serialized_examples.append(result)

    # --- Write TFRecords Sequentially ---
    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        for example in serialized_examples:
            writer.write(example)

# --- Process Datasets ---
process_dataset(TRAIN_JSON, TRAIN_VIDEO_DIR, os.path.join(OUTPUT_DIR, "train.tfrecord"))
process_dataset(EVAL_JSON, EVAL_VIDEO_DIR, os.path.join(OUTPUT_DIR, "eval.tfrecord"))
process_dataset(TEST_JSON, TEST_VIDEO_DIR, os.path.join(OUTPUT_DIR, "test.tfrecord"), is_test=True)
