import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EEGOnlyDataset(Dataset):
    def __init__(self, eeg_file):
        self.eeg_data = np.load(eeg_file)  # shape: (1400, 69, 400)

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return torch.tensor(self.eeg_data[idx], dtype=torch.float32)  # (69, 400)

# Test
dataset = EEGOnlyDataset("/home/poorna/data/preprocessed_eeg/sub1/sub1_preprocessed.npy")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for eeg_batch in loader:
    print("EEG batch shape:", eeg_batch.shape)  # [32, 69, 400]
    break

