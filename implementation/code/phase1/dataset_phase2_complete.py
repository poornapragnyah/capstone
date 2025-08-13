import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class EEGTextDataset(Dataset):
    def __init__(self, eeg_dir, metadata_dir, tokenizer, max_length=64, use_emotional_tone=True, subjects=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_emotional_tone = use_emotional_tone

        # 1. Load EEG files (recursive)
        eeg_files = []
        for root, dirs, files in os.walk(eeg_dir):
            for f in files:
                if f.endswith(".npy") and "_preprocessed" in f:
                    eeg_files.append(os.path.join(root, f))
        eeg_files = sorted(eeg_files)

        if subjects:
            eeg_files = [f for f in eeg_files if any(sub in f for sub in subjects)]

        self.eeg_file_paths = eeg_files
        self.eeg_data_list = [np.load(path, mmap_mode='r') for path in self.eeg_file_paths]

        # Build global index map
        self.index_map = []
        for subj_idx, eeg_data in enumerate(self.eeg_data_list):
            n_samples = eeg_data.shape[0]  # e.g., 1400
            self.index_map.extend([(subj_idx, i) for i in range(n_samples)])

        print(f"Found {len(self.eeg_file_paths)} EEG files")
        print(f"Total EEG samples across subjects: {len(self.index_map)}")

        # 2. Load metadata JSONs once
        metadata_files = sorted(
            [os.path.join(dp, f)
             for dp, dn, filenames in os.walk(metadata_dir)
             for f in filenames if f.endswith(".json")]
        )
        print(f"Metadata JSONs loaded: {len(metadata_files)}")

        # 3. Extract captions from metadata
        base_captions = []
        for path in metadata_files:
            with open(path, "r") as f:
                meta = json.load(f)
                caption_text = meta["caption"]["text"]
                if self.use_emotional_tone and "emotional_tone" in meta["caption"]:
                    caption_text += f". Tone: {meta['caption']['emotional_tone']}"
                base_captions.append(caption_text)

        # Repeat captions for each subject
        num_subjects = len(self.eeg_file_paths)
        self.captions = base_captions * num_subjects

        # 4. Sanity check
        if len(self.index_map) != len(self.captions):
            raise ValueError(
                f"EEG samples ({len(self.index_map)}) and captions ({len(self.captions)}) mismatch!"
            )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        subj_idx, local_idx = self.index_map[idx]

        eeg_tensor = torch.tensor(self.eeg_data_list[subj_idx][local_idx], dtype=torch.float32)  # (62, 400)

        caption = self.captions[idx]
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        return eeg_tensor, tokenized


# -------------------------------
# Testing
# -------------------------------
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("/home/poorna/models/bert-base-uncased")

    EEG_DIR = "/home/poorna/data/preprocessed_eeg"
    METADATA_DIR = "/home/poorna/data/final_text"

    dataset = EEGTextDataset(
        eeg_dir=EEG_DIR,
        metadata_dir=METADATA_DIR,
        tokenizer=tokenizer,
        max_length=64,
        use_emotional_tone=True
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for eeg_batch, tokenized_batch in loader:
        print("EEG batch shape:", eeg_batch.shape)
        print("Input IDs shape:", tokenized_batch["input_ids"].shape)
        print("Sample caption:", dataset.captions[0])
        break

    print(f"\nTotal samples loaded: {len(dataset)}")

