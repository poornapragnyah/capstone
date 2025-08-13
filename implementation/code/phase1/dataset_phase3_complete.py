import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class EEGTextMetaDataset(Dataset):
    def __init__(self, eeg_dir, metadata_dir, tokenizer, max_length=64, use_emotional_tone=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_emotional_tone = use_emotional_tone

        # -------------------------
        # 1. Load EEG files (all subjects)
        # -------------------------
        eeg_files = []
        for root, dirs, files in os.walk(eeg_dir):
            for f in files:
                if f.endswith(".npy") and "_preprocessed" in f:
                    eeg_files.append(os.path.join(root, f))
        eeg_files = sorted(eeg_files)

        if not eeg_files:
            raise FileNotFoundError(f"No EEG .npy files found in {eeg_dir}")

        self.eeg_file_paths = eeg_files
        self.eeg_data_list = []
        self.index_map = []

        for subj_idx, path in enumerate(self.eeg_file_paths):
            eeg = np.load(path, mmap_mode='r')

            # --- Assertion 1: EEG shape ---
            assert eeg.ndim == 3 and eeg.shape[1:] == (62, 400), \
                f"EEG file {path} has shape {eeg.shape}, expected (*, 62, 400)"

            self.eeg_data_list.append(eeg)

            # build index map
            n_samples = eeg.shape[0]
            self.index_map.extend([(subj_idx, i) for i in range(n_samples)])

        total_samples = len(self.index_map)
        print(f"Found {len(self.eeg_file_paths)} EEG files → Total samples: {total_samples}")

        # -------------------------
        # 2. Load Metadata JSONs
        # -------------------------
        metadata_files = sorted(
            [os.path.join(dp, f)
             for dp, dn, filenames in os.walk(metadata_dir)
             for f in filenames if f.endswith(".json")]
        )
        if not metadata_files:
            raise FileNotFoundError(f"No metadata JSON files found in {metadata_dir}")

        self.metadata_list = []
        for fpath in metadata_files:
            with open(fpath, 'r', encoding='utf-8') as f:
                meta = json.load(f)

                # --- Assertion 2: Metadata content validity ---
                assert "semantic_features" in meta and "scene_category" in meta["semantic_features"], \
                    f"Missing scene_category in {fpath}"
                assert "visual_attributes" in meta and "major_colors" in meta["visual_attributes"], \
                    f"Missing major_colors in {fpath}"
                assert "optical_flow_score" in meta["visual_attributes"], \
                    f"Missing optical_flow_score in {fpath}"

                self.metadata_list.append(meta)

        base_count = len(self.metadata_list)
        print(f"Loaded {base_count} metadata JSON files")

        # -------------------------
        # 3. Build base captions
        # -------------------------
        base_captions = []
        for meta in self.metadata_list:
            caption_text = meta["caption"]["text"]
            if self.use_emotional_tone and "emotional_tone" in meta["caption"]:
                caption_text += f". Tone: {meta['caption']['emotional_tone']}"
            base_captions.append(caption_text)

        # --- Assertion 3: captions vs metadata ---
        assert len(base_captions) == len(self.metadata_list), \
            "Mismatch between metadata and base captions count"

        # Repeat for each subject
        num_subjects = len(self.eeg_file_paths)
        self.captions = base_captions * num_subjects
        self.metadata_repeated = self.metadata_list * num_subjects

        # --- Assertion 4: Repeated counts ---
        assert len(self.captions) == len(self.metadata_repeated) == len(self.index_map), \
            "Mismatch after repeating captions and metadata for subjects"

        # -------------------------
        # 4. Encode metadata categories
        # -------------------------
        scene_categories = sorted(list({m["semantic_features"]["scene_category"] for m in self.metadata_list}))
        colors = sorted(list({m["visual_attributes"]["major_colors"][0]["color"].split()[0]
                              for m in self.metadata_list}))

        self.scene_to_id = {scene: i for i, scene in enumerate(scene_categories)}
        self.color_to_id = {c: i for i, c in enumerate(colors)}

        print(f"Scene categories: {len(self.scene_to_id)} | Colors: {len(self.color_to_id)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        subj_idx, local_idx = self.index_map[idx]

        # 1. EEG tensor
        eeg_tensor = torch.tensor(self.eeg_data_list[subj_idx][local_idx], dtype=torch.float32)  # (62, 400)

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
        meta = self.metadata_repeated[idx]
        scene_id = self.scene_to_id[meta["semantic_features"]["scene_category"]]
        color_id = self.color_to_id[meta["visual_attributes"]["major_colors"][0]["color"].split()[0]]
        motion_score = float(meta["visual_attributes"]["optical_flow_score"]["value"])
        metadata_tensor = torch.tensor([scene_id, color_id, motion_score], dtype=torch.float32)

        return eeg_tensor, tokenized, metadata_tensor


# -------------------------------
# Test the dataset
# -------------------------------
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("/home/poorna/models/bert-base-uncased")

    EEG_DIR = "/home/poorna/data/preprocessed_eeg"
    METADATA_DIR = "/home/poorna/data/final_text"

    dataset = EEGTextMetaDataset(
        eeg_dir=EEG_DIR,
        metadata_dir=METADATA_DIR,
        tokenizer=tokenizer,
        max_length=64
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for eeg_batch, tokenized_batch, meta_batch in loader:
        print("\nEEG batch:", eeg_batch.shape)
        print("Input IDs:", tokenized_batch["input_ids"].shape)
        print("Metadata batch:", meta_batch.shape)

        # Sample inspection
        sample_idx = 0
        print("\nSample Inspection:")
        print("EEG shape:", eeg_batch[sample_idx].shape)
        print("Token IDs:", tokenized_batch["input_ids"][sample_idx][:20])
        print("Decoded caption:", tokenizer.decode(
            tokenized_batch["input_ids"][sample_idx], skip_special_tokens=True
        ))
        print("Metadata tensor:", meta_batch[sample_idx])

        scene_id = int(meta_batch[sample_idx][0].item())
        color_id = int(meta_batch[sample_idx][1].item())
        motion_score = float(meta_batch[sample_idx][2].item())
        scene_name = [k for k, v in dataset.scene_to_id.items() if v == scene_id][0]
        color_name = [k for k, v in dataset.color_to_id.items() if v == color_id][0]

        print(f"Scene: {scene_name}, Color: {color_name}, Motion: {motion_score:.3f}")
        break

    print(f"\nTotal samples loaded: {len(dataset)}")

