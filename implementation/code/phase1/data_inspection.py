import os
import glob
import json
import numpy as np

# Paths
EEG_DIR = "/home/poorna/data/preprocessed_eeg"
CAPTIONS_FILE = "/home/poorna/data/SEED-DV/Video/BLIP-caption/combined_captions.txt"
METADATA_DIR = "/home/poorna/data/final_text"

# 1. Load combined captions
with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    captions = [line.strip() for line in f if line.strip()]
print(f"Captions loaded: {len(captions)}")

# 2. Load all metadata JSON files (nested folders)
metadata_files = sorted(
    glob.glob(os.path.join(METADATA_DIR, "annotations*", "*.json")),
    key=lambda x: (
        int(''.join(filter(str.isdigit, os.path.basename(os.path.dirname(x)))) or '0'),
        int(''.join(filter(str.isdigit, os.path.basename(x))) or '0')
    )
)
print(f"Metadata JSON files: {len(metadata_files)}")

# 3. Get all EEG files (include session2)
eeg_files = sorted(glob.glob(os.path.join(EEG_DIR, "sub*", "*_preprocessed.npy")))
print(f"Found {len(eeg_files)} EEG files for subjects.")

# 4. Check alignment for each EEG file
for eeg_file in eeg_files:
    eeg_data = np.load(eeg_file)
    num_epochs = eeg_data.shape[0]
    file_name = os.path.basename(eeg_file)

    aligned = (num_epochs == len(captions) == len(metadata_files))
    status = "✅ Aligned" if aligned else "❌ Mismatch"

    print(f"{file_name:<30} | EEG epochs: {num_epochs:<4} | Captions: {len(captions):<4} | Metadata: {len(metadata_files):<4} | {status}")

# 5. Optional: Inspect the first sample of the first EEG file
if eeg_files:
    eeg_data = np.load(eeg_files[0])
    sample_idx = 0
    with open(metadata_files[sample_idx], "r") as f:
        metadata_sample = json.load(f)

    print("\nSample Inspection:")
    print("EEG shape:", eeg_data[sample_idx].shape)  # (69, 400)
    print("Caption:", captions[sample_idx])
    print("Metadata:", metadata_sample)

