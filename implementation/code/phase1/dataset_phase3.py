import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class EEGTextMetaDataset(Dataset):
    def __init__(self, eeg_dir, metadata_dir, tokenizer, max_length=64, use_emotional_tone=True, sample_limit=800):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_emotional_tone = use_emotional_tone
        self.sample_limit = sample_limit

        # -------------------------
        # 1. Load EEG files (recursive)
        # -------------------------
        eeg_files = []
        for root, dirs, files in os.walk(eeg_dir):
            for f in files:
                if f.endswith(".npy") and "_preprocessed" in f:
                    eeg_files.append(os.path.join(root, f))
        eeg_files = sorted(eeg_files)

        # Only first subject for 800 samples
        self.eeg_file_paths = [eeg_files[0]]
        self.eeg_data = np.load(self.eeg_file_paths[0], mmap_mode='r')

        # Apply sample limit
        self.n_samples = min(self.eeg_data.shape[0], sample_limit)

        # -------------------------
        # 2. Load Metadata JSONs
        # -------------------------
        metadata_files = sorted(
            [os.path.join(dp, f)
             for dp, dn, filenames in os.walk(metadata_dir)
             for f in filenames if f.endswith(".json")]
        )
        metadata_files = metadata_files[:self.n_samples]  # restrict to first 800
        self.metadata_list = [json.load(open(f)) for f in metadata_files]

        print(f"EEG samples: {self.n_samples}, Metadata files: {len(self.metadata_list)}")

        # -------------------------
        # 3. Build captions
        # -------------------------
        self.captions = []
        for meta in self.metadata_list:
            caption_text = meta["caption"]["text"]
            if self.use_emotional_tone and "emotional_tone" in meta["caption"]:
                caption_text += f". Tone: {meta['caption']['emotional_tone']}"
            self.captions.append(caption_text)

        # -------------------------
        # 4. Build metadata encoders
        # -------------------------
        scene_categories = sorted(list({m["semantic_features"]["scene_category"] for m in self.metadata_list}))
        colors = sorted(list({m["visual_attributes"]["major_colors"][0]["color"].split()[0] for m in self.metadata_list}))

        self.scene_to_id = {scene: i for i, scene in enumerate(scene_categories)}
        self.color_to_id = {c: i for i, c in enumerate(colors)}

        print("Scene categories:", self.scene_to_id)
        print("Colors:", self.color_to_id)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 1. EEG tensor
        eeg_tensor = torch.tensor(self.eeg_data[idx], dtype=torch.float32)  # (62, 400)

        # 2. Tokenized caption
        caption = self.captions[idx]
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        # 3. Metadata tensor
        meta = self.metadata_list[idx]
        scene_id = self.scene_to_id[meta["semantic_features"]["scene_category"]]
        color_id = self.color_to_id[meta["visual_attributes"]["major_colors"][0]["color"].split()[0]]
        motion_score = float(meta["visual_attributes"]["optical_flow_score"]["value"])
        metadata_tensor = torch.tensor([scene_id, color_id, motion_score], dtype=torch.float32)

        return eeg_tensor, tokenized, metadata_tensor


# -------------------------------
# Test Phase 3 with 800 samples
# -------------------------------
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("/home/poorna/models/bert-base-uncased")

    EEG_DIR = "/home/poorna/data/preprocessed_eeg"
    METADATA_DIR = "/home/poorna/data/final_text"

    dataset = EEGTextMetaDataset(
        eeg_dir=EEG_DIR,
        metadata_dir=METADATA_DIR,
        tokenizer=tokenizer,
        max_length=64,
        sample_limit=800
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for eeg_batch, tokenized_batch, meta_batch in loader:
        print("EEG batch:", eeg_batch.shape)           # [32, 62, 400]
        print("Input IDs:", tokenized_batch["input_ids"].shape)  # [32, 64]
        print("Metadata:", meta_batch.shape)           # [32, 3]
        break

    print(f"\nTotal samples loaded: {len(dataset)}")

# Inspect one sample from the batch
sample_idx = 0
print("\nSample Inspection:")
print("EEG shape:", eeg_batch[sample_idx].shape)                 # (62, 400)
print("Token IDs:", tokenized_batch["input_ids"][sample_idx][:20])  # first 20 IDs
print("Decoded caption:", tokenizer.decode(tokenized_batch["input_ids"][sample_idx], skip_special_tokens=True))
print("Metadata tensor:", meta_batch[sample_idx])                # [scene_id, color_id, motion_score]

# Also map IDs back to scene/color for readability
scene_id = int(meta_batch[sample_idx][0].item())
color_id = int(meta_batch[sample_idx][1].item())
motion_score = float(meta_batch[sample_idx][2].item())

# Show mappings
scene_name = [k for k, v in dataset.scene_to_id.items() if v == scene_id][0]
color_name = [k for k, v in dataset.color_to_id.items() if v == color_id][0]

print(f"Scene: {scene_name}, Color: {color_name}, Motion: {motion_score:.3f}")

