import tensorflow as tf
import cv2
import os
import json

# Paths
TRAIN_JSON = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/labels/train.json"
EVAL_JSON = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/labels/evaluation.json"
TEST_JSON = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/labels/test.json"  # New test set
LABELS_JSON = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/labels/labels.json"

TRAIN_VIDEO_DIR = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/train/video/"
EVAL_VIDEO_DIR = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/eval/video/"
TEST_VIDEO_DIR = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/test/video/"  # Test videos repository
OUTPUT_DIR = "/home/lianniello/Hold_thesis/HOLD/drawer_dataset/tfrecords/"

# Filtering setting (set to None to disable filtering, or specify a label index to filter)
FILTER_LABEL = 0  # Set to an integer like 0, 1, 2, ... or None to include all classes

# Load label mapping
with open(LABELS_JSON, "r") as f:
    label_map = json.load(f)  # {"opening a drawer":"0", "closing a drawer":"1", ...}
    label_map = {k: int(v) for k, v in label_map.items()}  # Convert to integer labels
    label_map_inv = {v: k for k, v in label_map.items()}  # Reverse mapping {0: "opening a drawer", ...}

# Function to process a single video
def video_to_tfrecord(video_path, label, writer):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode(".jpg", frame)  # Encode frame as JPEG
        frames.append(buffer.tobytes())  # Convert to bytes

    cap.release()

    if len(frames) == 0:
        print(f"Warning: No frames extracted from {video_path}")
        return

    # Create TF Example
    example = tf.train.Example(features=tf.train.Features(feature={
        'frames': tf.train.Feature(bytes_list=tf.train.BytesList(value=frames)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'num_frames': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(frames)]))
    }))

    writer.write(example.SerializeToString())

# Function to process dataset (train, eval, or test)
def process_dataset(json_path, video_dir, output_tfrecord, is_test=False):
    with open(json_path, "r") as f:
        data = json.load(f)  # [{"id":"0001","label":"closing a drawer"}, ...] (train/eval) or [{"id":"0001"}] (test)

    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        for item in data:
            video_id = item["id"]
            label_name = item.get("label", None)  # No label for test set

            if not is_test:
                if label_name not in label_map:
                    print(f"Skipping unknown label: {label_name}")
                    continue
                label = label_map[label_name]  # Convert label name to integer
                
                # Apply label filtering (if enabled)
                if FILTER_LABEL is not None and label != FILTER_LABEL:
                    continue
            else:
                label = -1  # Assign a dummy label for test videos

            video_path = os.path.join(video_dir, f"{video_id}.mp4")

            if os.path.exists(video_path):
                video_to_tfrecord(video_path, label, writer)
            else:
                print(f"Missing video: {video_path}")

# Process training, evaluation, and test sets
process_dataset(TRAIN_JSON, TRAIN_VIDEO_DIR, os.path.join(OUTPUT_DIR, "train.tfrecord"))
process_dataset(EVAL_JSON, EVAL_VIDEO_DIR, os.path.join(OUTPUT_DIR, "eval.tfrecord"))
process_dataset(TEST_JSON, TEST_VIDEO_DIR, os.path.join(OUTPUT_DIR, "test.tfrecord"), is_test=True)
