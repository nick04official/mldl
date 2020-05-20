import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.autograd import Variable

from models.conv_lstm import ConvLSTM
from models.resnet.resnet import resnet34
from models.motion_segmentation import MotionSegmentationBlock

class AttentionModel(nn.Module):

    def __init__(self, num_classes=61, mem_size=512, noCam=False, enable_motion_segmentation=False):
        super(AttentionModel, self).__init__()

        self.num_classes = num_classes
        self.noCam = noCam
        self.mem_size = mem_size
        self.enable_motion_segmentation = enable_motion_segmentation

        self.resnet = resnet34(pretrained=True, noBN=True)
        self.weight_softmax = self.resnet.fc.weight
        self.lstm_cell = ConvLSTM(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

        self.motion_segmentation = MotionSegmentationBlock()

        self._custom_train_mode = True
    

    def train(self, mode=True):
        correct_values = {True, 'stage2', 'stage1', False}
        
        if mode not in correct_values:
            raise ValueError('Invalid modes, correct values are: ' + ' '.join(correct_values))

        self._custom_train_mode = mode
        
        # Fai fare il training completo solo se mode == True
        super().train(mode == True)

        self.resnet.train(mode)
        self.lstm_cell.train(mode)
        #if mode == 'stage2' or mode == True:
        if mode != False:
           self.motion_segmentation.train(True) 
        if mode != False:
            self.classifier.train(True)

    def get_training_parameters(self):
        train_params = []

        # Prima levo i gradienti a tutti, e poi li aggiungo solo a quelli
        # su cui faccio il training
        for params in self.parameters():
            params.requires_grad = False

        # è responsabilità della funzione negli oggetti aggiungere i gradienti
        train_params += self.resnet.get_training_parameters()
        train_params += self.lstm_cell.get_training_parameters()
        # trainiamo l'ultimo layer a tutti gli stagi, eccetto se non sono in training
        if self._custom_train_mode != False:
            for params in self.classifier.parameters():
                params.requires_grad = True
                train_params += [params]
            if self.enable_motion_segmentation:
                for params in self.motion_segmentation.parameters():
                    params.requires_grad = True
                    train_params += [params]

        return train_params

    def forward(self, inputVariable):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        
        ms_feats = None
        if self.enable_motion_segmentation:
            ms_feats = Variable(torch.zeros(inputVariable.size(0), inputVariable.size(1), 49).cuda())

        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resnet(inputVariable[t])

            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)

            if self.enable_motion_segmentation:
                ms_feats[t] = self.motion_segmentation(feature_convNBN)

            if self.noCam:
                state = self.lstm_cell(feature_convNBN, state)
            else:
                state = self.lstm_cell(attentionFeat, state)
            
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)

        return feats, ms_feats