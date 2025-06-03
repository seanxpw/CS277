from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np


class CS277Dataset(Dataset):
    def __init__(self, root_dir, train, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the data.
            train (bool): If True, use training data; if False, use validation data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        root_dir = Path(root_dir)
        if (root_dir / 'input.npy').exists():
            assert (root_dir / 'label.npy').exists()
            self.input = np.expand_dims(np.load(root_dir / 'input.npy'), axis=1).astype(np.float32)
            self.label = np.load(root_dir / 'label.npy').astype(np.float32)
            assert self.input.shape[0] == self.label.shape[0]
            if train:
                self.input = self.input[:int(0.8 * len(self.input))]
                self.label = self.label[:int(0.8 * len(self.label))]
            else:
                self.input = self.input[int(0.8 * len(self.input)):]
                self.label = self.label[int(0.8 * len(self.label)):]
        else:
            raise RuntimeError(
                f"Dataset files not found. Please ensure 'input.npy' and 'label.npy' exist in the specified directory ({root_dir}).")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.input[idx], self.label[idx]

        if self.transform:
            sample = self.transform(self.input[idx]), self.transform(self.label[idx])

        return sample
