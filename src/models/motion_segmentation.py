import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

class MotionSegmentationBlock(nn.Module):

    def __init__(self, input_layers=512, output_layers=100, resolution=(7, 7)):
        super(MotionSegmentationBlock, self).__init__()

        self.relu = nn.ReLU()
        self.convolution = conv1x1(input_layers, output_layers)
        self.fc = nn.Linear(resolution[0] * resolution[1] * output_layers, 2 * resolution[0] * resolution[1])


    def get_training_parameters(self):
        
        train_params = []

        if self.training:
            for params in self.parameters():
                params.requires_grad = True
                train_params += [params]

        return train_params

    def forward(self, x):
        
        x = self.relu(x)
        x = self.convolution(x)
                
        # flatten for fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
