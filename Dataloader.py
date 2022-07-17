from torch.utils.data import Dataset
import numpy as np
import glob
import os
import torch

class TrainDataset(Dataset):
    def __init__(self, path="train_img_np", transform=None):
        train_path = os.path.join(path, "input")
        target_path = os.path.join(path, 'target')
        self.train_files = glob.glob(os.path.join(train_path, "*.npy"))
        self.target_files = glob.glob(os.path.join(target_path, "*.npy"))
        
        self.transform = transform

    def __getitem__(self, index):
        image = np.load(self.train_files[index]).astype(np.float32)
        target = np.load(self.target_files[index]).astype(np.float32)
        return torch.tensor(image), torch.tensor(target)

    def __len__(self):
        return len(self.train_files)