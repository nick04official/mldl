import torch
import torch.nn as nn

import numpy as np
import cv2

from torch.nn import functional as F
from torch.autograd import Variable

from models.conv_lstm import ConvLSTM
from models.resnet.resnet import resnet34
from models.motion_segmentation import MotionSegmentationBlock

from spatial_transforms import Compose, ToTensor, CenterCrop, Scale, Normalize
from PIL import Image

class AttentionModel(nn.Module):

    def __init__(self, num_classes=61, mem_size=512, no_cam=False, enable_motion_segmentation=False):
        super(AttentionModel, self).__init__()

        self.num_classes = num_classes
        self.noCam = no_cam
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
        if mode == 'stage2' or mode == True:
           self.motion_segmentation.train(True) 
        if mode != False:
            self.classifier.train(True)

    def get_training_parameters(self, name='all'):
        train_params = []
        train_params_ms = []

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

        train_params_ms = self.motion_segmentation.get_training_parameters()

        if name == 'all':
            return train_params + train_params_ms
        elif name == 'main':
            return train_params
        elif name == 'ms':
            return train_params_ms

    def load_weights(self, file_path):
        model_dict = torch.load(file_path)
        if 'model_state_dict' in model_dict:
            self.load_state_dict(model_dict['model_state_dict'])
        else:
            self.load_state_dict(model_dict)

    def forward(self, inputVariable):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        
        ms_feats = None
        if self.enable_motion_segmentation:
            ms_feats = Variable(torch.zeros(inputVariable.size(0), inputVariable.size(1), 49 * 2).cuda())

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

        return {'classifications': feats, 'ms_feats': ms_feats, 'lstm_feats': feats1}

    def get_class_activation_id(self, inputVariable):
        logit, _, _ = self.resnet(inputVariable)
        return logit

    def get_cam_visualisation(self, input_pil_image, preprocess_for_viz=None, preprocess_for_model=None):
        if preprocess_for_viz == None:
            preprocess_for_viz = Compose([
                Scale(256),
                CenterCrop(224),
            ])
        if preprocess_for_model == None:
            normalize = Normalize(
                mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

            preprocess_for_model = Compose([
                Scale(256),
                CenterCrop(224),
                ToTensor(),
                normalize
            ])

        tensor_image = preprocess_for_model(input_pil_image)
        pil_image = preprocess_for_viz(input_pil_image)

        logit, feature_conv, _ = self.resnet(tensor_image.unsqueeze(0).cuda())

        bz, nc, h, w = feature_conv.size()
        feature_conv = feature_conv.view(bz, nc, h*w)

        h_x = F.softmax(logit, dim=1).data
        probs, idx = h_x.sort(1, True)

        cam_img = torch.bmm(self.weight_softmax[idx[:, 0]].unsqueeze(1), feature_conv).squeeze(1)
        cam_img = F.softmax(cam_img, 1).data
        cam_img = cam_img.cpu()
        cam_img = cam_img.reshape(h, w)
        cam_img = cam_img - torch.min(cam_img)
        cam_img = cam_img / torch.max(cam_img)

        cam_img = np.uint8(255 * cam_img)
        img = np.uint8(pil_image)

        output_cam = cv2.resize(cam_img, pil_image.size)
        heatmap = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET)
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)
        
        result = heatmap * 0.4 + img * 0.6
        result = cv2.cvtColor(np.uint8(result), cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(result)


class AttentionModelTwoHeads(nn.Module):

    def __init__(self, num_classes=61, mem_size=512, no_cam=False):
        super(AttentionModelTwoHeads, self).__init__()

        self.num_classes = num_classes
        self.noCam = no_cam
        self.mem_size = mem_size

        self.resnet = resnet34(pretrained=True, noBN=True)
        self.weight_softmax = self.resnet.fc.weight

        self.lstm_cell_1 = ConvLSTM(512, mem_size)
        self.lstm_cell_2 = ConvLSTM(512, mem_size)

        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)

        self.fc = nn.Linear(2*mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

        self._custom_train_mode = True
    

    def train(self, mode=True):
        correct_values = {True, 'stage2', 'stage1', False}
        
        if mode not in correct_values:
            raise ValueError('Invalid modes, correct values are: ' + ' '.join(correct_values))

        self._custom_train_mode = mode
        
        # Fai fare il training completo solo se mode == True
        super().train(mode == True)

        self.resnet.train(mode)
        self.lstm_cell_1.train(mode)
        self.lstm_cell_2.train(mode)
        if mode != False:
            self.classifier.train(True)

    def get_training_parameters(self):
        train_params = []

        for params in self.parameters():
            params.requires_grad = False

        train_params += self.resnet.get_training_parameters()

        train_params += self.lstm_cell_1.get_training_parameters()
        train_params += self.lstm_cell_2.get_training_parameters()

        if self._custom_train_mode != False:
            for params in self.classifier.parameters():
                params.requires_grad = True
                train_params += [params]

        return train_params

    def load_weights(self, file_path):
        model_dict = torch.load(file_path)
        if 'model_state_dict' in model_dict:
            self.load_state_dict(model_dict['model_state_dict'])
        else:
            self.load_state_dict(model_dict)

    def get_resnet_output_feats(self, input_frames):
        logit, feature_conv, feature_convNBN = self.resnet(input_frames)

        if self.noCam:
            return feature_convNBN

        bz, nc, h, w = feature_conv.size()
        feature_conv1 = feature_conv.view(bz, nc, h*w)
        probs, idxs = logit.sort(1, True)
        class_idx = idxs[:, 0]
        cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
        attentionMAP = F.softmax(cam.squeeze(1), dim=1)
        attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
        attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)

        return attentionFeat

    def forward(self, rgb_frames, flow_frames):

        state_1 = (Variable(torch.zeros((rgb_frames.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((rgb_frames.size(1), self.mem_size, 7, 7)).cuda()))
        
        state_2 = (Variable(torch.zeros((rgb_frames.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((rgb_frames.size(1), self.mem_size, 7, 7)).cuda()))
        
        for t in range(rgb_frames.size(0)):
            rgb_feats = self.get_resnet_output_feats(rgb_frames[t])
            
            flow_feats = self.get_resnet_output_feats(flow_frames[t])
            
            state_1 = self.lstm_cell_1(rgb_feats, state_1)
            state_2 = self.lstm_cell_2(flow_feats, state_2)

        feats1 = self.avgpool(torch.cat((state_1[1], state_2[1]), dim=1)).view(state_1[1].size(0), -1)
        feats = self.classifier(feats1)

        return {'classifications': feats, 'lstm_feats': feats1}

    def get_cam_visualisation(self, input_pil_image, preprocess_for_viz=None, preprocess_for_model=None):
        if preprocess_for_viz == None:
            preprocess_for_viz = Compose([
                Scale(256),
                CenterCrop(224),
            ])
        if preprocess_for_model == None:
            normalize = Normalize(
                mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

            preprocess_for_model = Compose([
                Scale(256),
                CenterCrop(224),
                ToTensor(),
                normalize
            ])

        tensor_image = preprocess_for_model(input_pil_image)
        pil_image = preprocess_for_viz(input_pil_image)

        logit, feature_conv, _ = self.resnet_rgb(tensor_image.unsqueeze(0).cuda())

        bz, nc, h, w = feature_conv.size()
        feature_conv = feature_conv.view(bz, nc, h*w)

        h_x = F.softmax(logit, dim=1).data
        probs, idx = h_x.sort(1, True)

        cam_img = torch.bmm(self.weight_softmax[idx[:, 0]].unsqueeze(1), feature_conv).squeeze(1)
        cam_img = F.softmax(cam_img, 1).data
        cam_img = cam_img.cpu()
        cam_img = cam_img.reshape(h, w)
        cam_img = cam_img - torch.min(cam_img)
        cam_img = cam_img / torch.max(cam_img)

        cam_img = np.uint8(255 * cam_img)
        img = np.uint8(pil_image)

        output_cam = cv2.resize(cam_img, pil_image.size)
        heatmap = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET)
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)
        
        result = heatmap * 0.4 + img * 0.6
        result = cv2.cvtColor(np.uint8(result), cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(result)