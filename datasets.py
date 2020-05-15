import os

import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image

from spatial_transforms import Compose, Scale

def load_image_PIL(path, mode='RGB'):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)

class DatasetRGB(Dataset):

    def __init__(self, root_dir, json_dataset, spatial_transform=None, seqLen=20, minLen=1, device='cpu', scale_to=256, label_ids=None):
        
        if device not in {'cpu', 'cuda'}:
            raise ValueError('Wrong device, only cpu or cuda are allowed')

        self.root_dir = root_dir
        self.spatial_transform = spatial_transform
        self.seqLen = seqLen
        self.minLen = minLen
        self.device = device

        self.scenes = []
        self.labels = []

        if label_ids == None:
            self.label_ids = {}
        else:
            self.label_ids = dict(label_ids)

        scale = Compose([Scale(scale_to)])

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


class DatasetFlow(Dataset):

    def __init__(self, root_dir, json_dataset, spatial_transform=None, stack_size=5, sequence_mode='single_random', n_sequences=1, device='cpu', scale_to=256, label_ids=None):
        
        if sequence_mode not in {'single_random', 'single_midtime', 'multiple'}:
            raise ValueError('Wrong sequence_mode')
        if device not in {'cpu', 'cuda'}:
            raise ValueError('Wrong device, only cpu or cuda are allowed')

        self.spatial_transform = spatial_transform
        self.stack_size = stack_size
        self.sequence_mode = sequence_mode
        self.n_sequences = n_sequences

        self.scenes = []
        self.labels = []

        if label_ids == None:
            self.label_ids = {}
        else:
            self.label_ids = dict(label_ids)

        scale = Compose([Scale(scale_to)])

        for subject in json_dataset:
            for label_name in json_dataset[subject]:
                for scene in json_dataset[subject][label_name]:    
                    scene_len = len(scene['x']) 
                    if scene_len >= self.stack_size:
                        frames = {'x': [], 'y': []}
                        for i in range(scene_len):
                            frames['x'].append(scale(load_image_PIL(os.path.join(root_dir, scene['x'][i]), mode='L')))
                            frames['y'].append(scale(load_image_PIL(os.path.join(root_dir, scene['y'][i]), mode='L')))
                        
                        self.scenes.append(frames)

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
        scene = self.scenes[idx]

        self.spatial_transform.randomize_parameters()

        scene_len = len(scene['x'])
        
        first_frames = None
        if self.sequence_mode == 'single_random':
            first_frames = [ np.random.choice(range(0, scene_len - self.stack_size + 1)) ]
        elif self.sequence_mode == 'single_midtime':
            first_frames = [ (scene_len - self.stack_size) // 2 ]
        elif self.sequence_mode == 'multiple':
            first_frames = np.linspace(0, scene_len - self.stack_size, self.n_sequences)
            
        sequences = []
        for first_frame in first_frames:
            sequence = []
            for k in range(first_frame, first_frame + self.stack_size):
                sequence.append(self.spatial_transform(scene['x'][k], inv=True))
                sequence.append(self.spatial_transform(scene['y'][k], inv=False))
            sequences.append(torch.stack(sequence, 0).squeeze(1))
        # TODO: prendo solo la prima sequenza
        return sequences[0], label