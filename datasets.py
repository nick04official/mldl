import os

import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image

from spatial_transforms import Compose, Scale

def load_image_PIL(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DatasetRGB(Dataset):

    def __init__(self, root_dir, json_dataset, spatial_transform=None, seqLen=20, minLen=1, device='cpu', scale_to=256):
        
        if device not in {'cpu', 'cuda'}:
            raise ValueError('Wrong device, only cpu or cuda are allowed')

        scale = Compose([Scale(scale_to)])

        self.root_dir = root_dir
        self.spatial_transform = spatial_transform
        self.seqLen = seqLen
        self.minLen = minLen
        self.device = device

        self.scenes = []
        self.labels = []

        self.label_ids = {}

        for subject in json_dataset:
            for label_name in json_dataset[subject]:
                for scene in json_dataset[subject][label_name]:
                    
                    if len(scene) >= self.minLen:
                        frames = []
                        for i in np.linspace(0, len(scene), self.seqLen, endpoint=False):
                            pil_image = load_image_PIL(os.path.join(root_dir, scene[int(i)]))
                            pil_image = scale(pil_image)
                            if self.device == 'cpu':
                                # Do not apply transformations, save as PIL on the CPU RAM
                                frames.append(pil_image)
                            elif self.device == 'cuda':
                                # Apply transformations and convert to tensor
                                # to be later transferred to GPU VRAM
                                frames.append(self.spatial_transform(pil_image))
                        
                        if self.device == 'cpu':
                            self.scenes.append(frames)
                        elif self.device == 'cuda':
                            self.scenes.append(torch.stack(frames, 0).to('cuda:0'))

                        label_id = None
                        try:
                            label_id = self.label_ids[label_name]
                        except KeyError:
                            label_id = len(self.label_ids)
                            self.label_ids[label_name] = label_id
                        self.labels.append(label_id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]

        if self.device == 'cpu':
            self.spatial_transform.randomize_parameters()
            frames = torch.stack([self.spatial_transform(frame) for frame in self.scenes[idx]], 0)
        elif self.device == 'cuda':
            frames = self.scenes[idx]
        
        return frames, label