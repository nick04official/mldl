import torch
import torch.nn as nn

from torch.autograd import Variable

from models.resnet.resnet import resnet34
from models.conv_lstm import ConvLSTM

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class MotionSegmentationBlock(nn.Module):

    def __init__(self, input_layers=512, output_layers=100, resolution=(7, 7)):
        super(MotionSegmentationBlock, self).__init__()

        self.convolution = conv3x3(input_layers, output_layers)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(resolution[0] * resolution[1] * output_layers, resolution[0] * resolution[1])
        self.classifier = nn.Sequential(self.dropout, self.fc)

        #torch.nn.init.xavier_normal_(self.convolution.weight)
        #torch.nn.init.xavier_normal_(self.convolution.bias)
        #torch.nn.init.xavier_normal_(self.fc.weight)
        #torch.nn.init.xavier_normal_(self.fc.bias)

    def get_training_parameters(self):
        
        train_params = []

        if self.training:
            for params in self.parameters():
                params.requires_grad = True
                train_params += [params]

        return train_params

    def forward(self, x):

        x = self.convolution(x)
        x = self.relu(x)
        
        # flatten for fully connected layer
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
