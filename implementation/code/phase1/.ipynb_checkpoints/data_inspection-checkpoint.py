import os
import numpy as np
import json

# Paths
EEG_DIR = "/home/poorna/data/SEED-DV/EEG"
CAPTIONS_FILE = "/home/poorna/data/SEED-DV/data/captions.txt"
METADATA_DIR = "/home/poorna/data/SEED-DV/data/metadata"

# 1. Load file lists
eeg_files = sorted([f for f in os.listdir(EEG_DIR) if f.endswith(".npy")])
metadata_files = sorted([f for f in os.listdir(METADATA_DIR) if f.endswith(".json")])

print(f"EEG files: {len(eeg_files)}")
print(f"Metadata files: {len(metadata_files)}")

# 2. Load captions
with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    captions = [line.strip() for line in f.readlines()]

print(f"Captions: {len(captions)}")

# 3. Verify alignment
if len(eeg_files) == len(metadata_files) == len(captions):
    print("✅ Data lengths match!")
else:
    print("❌ Data length mismatch!")
    exit()

# 4. Inspect a random sample
sample_idx = 0  # change to any index for debugging
eeg_sample = np.load(os.path.join(EEG_DIR, eeg_files[sample_idx]))
with open(os.path.join(METADATA_DIR, metadata_files[sample_idx]), "r") as f:
    metadata_sample = json.load(f)

print("\nSample Inspection:")
print(f"EEG shape: {eeg_sample.shape}")          # Should be [62, 2601]
print(f"Caption: {captions[sample_idx]}")
print(f"Metadata: {metadata_sample}")

