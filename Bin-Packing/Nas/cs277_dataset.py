import re
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
        match = re.search(r'bin_packing_optimal_size=(\d+)', root_dir)
        if match:
            self.size = int(match.group(1))
        self.transform = transform
        root_dir = Path(root_dir)
        if (root_dir / 'input.npy').exists():
            assert (root_dir / 'label.npy').exists()
            self.input = np.load(root_dir / 'input.npy')
            self.label = np.load(root_dir / 'label.npy').astype(np.float32)
            assert self.input.shape[0] == self.label.shape[0]
            k = self.input.shape[1]
            input_dup = np.repeat(self.input[:, np.newaxis, :], k, axis=1)  # [batch_size, k, k]
            for i in range(k):
                input_dup[:, i, :] = np.roll(self.input, -i, axis=1)
            self.input = np.expand_dims(input_dup, axis=1).astype(np.float32)
            self.input = (self.input - 1) / (100 - 1)
            self.label /= self.size
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

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.input[idx], self.label[idx]

        if self.transform:
            sample = self.transform(self.input[idx]), self.transform(self.label[idx])

        return sample
