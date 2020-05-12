import numpy as np

import os

import torch
from torch.utils.data import Dataset

from PIL import Image

def load_image_PIL(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DatasetRGB(Dataset):

    def __init__(self, root_dir, json_dataset, spatial_transform=None, seqLen=20, minLen=1):
        
        self.root_dir = root_dir
        self.spatial_transform = spatial_transform
        self.seqLen = seqLen
        self.minLen = minLen

        self.scenes = []
        self.labels = []

        self.label_ids = {}

        for subject in json_dataset:
            for label_name in json_dataset[subject]:
                for scene in json_dataset[subject][label_name]:
                    if len(scene) >= self.minLen:
                        scene_PILs = []
                        for i in np.linspace(0, len(scene), self.seqLen, endpoint=False):
                            scene_PILs.append(load_image_PIL(os.path.join(root_dir, scene[int(i)])))
                        self.scenes.append(scene_PILs)
                        label_id = None
                        try:
                            label_id = self.label_ids[label_name]
                        except KeyError:
                            label_id = len(self.label_ids)
                            self.label_ids[label_name] = label_id
                        self.labels.append(label_id)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        label = self.labels[idx]
        self.spatial_transform.randomize_parameters()
        frames = torch.stack([self.spatial_transform(frame) for frame in self.scenes[idx]], 0)
        return frames, label