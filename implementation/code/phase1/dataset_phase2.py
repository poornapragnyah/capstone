import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np

# -------------------------------
# Dataset Class
# -------------------------------
class EEGTextDataset(Dataset):
    def __init__(self, eeg_file, captions, tokenizer, max_length=32):
        """
        EEG + Text Dataset
        eeg_file: Path to a single .npy file (shape: [1400, 62, 400])
        captions: List of captions (length: 1400)
        tokenizer: Hugging Face tokenizer
        max_length: Max tokens per caption
        """
        self.eeg_data = np.load(eeg_file)  # Load EEG data into memory
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Check alignment
        assert len(self.eeg_data) == len(self.captions), \
            f"EEG ({len(self.eeg_data)}) and captions ({len(self.captions)}) mismatch"

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        # 1. EEG tensor
        eeg_tensor = torch.tensor(self.eeg_data[idx], dtype=torch.float32)  # Shape: (62, 400)

        # 2. Tokenized caption
        caption = self.captions[idx]
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}  # Remove batch dimension

        return eeg_tensor, tokenized

# -------------------------------
# Testing the Dataset + DataLoader
# -------------------------------
if __name__ == "__main__":
    # 1. Load tokenizer (from local cache if available)
    tokenizer = BertTokenizer.from_pretrained("/home/poorna/models/bert-base-uncased")

    # 2. Load captions
    CAPTIONS_FILE = "/home/poorna/data/SEED-DV/Video/BLIP-caption/combined_captions.txt"
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f.readlines() if line.strip()]

    # 3. EEG file
    EEG_FILE = "/home/poorna/data/preprocessed_eeg/sub1/sub1_preprocessed.npy"

    # 4. Create Dataset and DataLoader
    dataset = EEGTextDataset(
        eeg_file=EEG_FILE,
        captions=captions,
        tokenizer=tokenizer,
        max_length=32
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 5. Pull one batch to verify
    for eeg_batch, tokenized_batch in loader:
        print("EEG batch shape:", eeg_batch.shape)  # [32, 62, 400]
        print("Input IDs shape:", tokenized_batch["input_ids"].shape)  # [32, 32]
        print("Attention mask shape:", tokenized_batch["attention_mask"].shape)  # [32, 32]
        print("Sample caption:", captions[0])
        print("Sample token IDs:", tokenized_batch["input_ids"][0][:10])  # first 10 tokens
        break

