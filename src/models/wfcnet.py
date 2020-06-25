# pylint: disable=not-callable

import torch

from torch import nn
from torch.autograd import Variable

from models.resnet.resnet import resnet34
from models.attention_model import AttentionModel, NewAttentionModel, NewAttentionModelBi

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
 
class BasicBlockConvDeconv(nn.Module):
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockConvDeconv, self).__init__()
        if stride > 0:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                       conv3x3(inplanes, planes))
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out = out + residual
        out = self.relu(out)
 
        return out
 
class WFCOutputNormalizer(nn.Module):
 
    def __init__(self):
        super(WFCOutputNormalizer, self).__init__()
 
        self.sigmoid = nn.Sigmoid()
 
        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1).unsqueeze(0).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1).unsqueeze(0).cuda()
 
    def forward(self, x):
        x = self.sigmoid(x)
        x = (x - self.mean) / self.std
        return x
 
class WFCNet(nn.Module):
    def __init__(self, block=BasicBlockConvDeconv, in_channels=2, out_channels=3):
        self.inplanes = 8
        super(WFCNet, self).__init__()
 
        self.conv_in = nn.Conv2d(in_channels, 8, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
 
        self.layer1 = self._make_layer(block, 8, blocks=2)
        self.layer2 = self._make_layer(block, 16, blocks=3, stride=2)
 
        self.layer1b = self._make_layer(block, 16, blocks=3, stride=-2)
        self.layer2b = self._make_layer(block, 8, blocks=2, stride=-2)
        
        self.conv_out = nn.Conv2d(8, out_channels, kernel_size=1, stride=1,
                                  bias=False)
        
        self.normalize = WFCOutputNormalizer()
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 and stride > 0:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif stride != 1 and stride < 0:
            downsample = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode='nearest'),
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.conv_in(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
    
        x = self.layer1b(x)
        x = self.layer2b(x)
 
        x = self.conv_out(x)
 
        x = self.normalize(x)
 
        return x

    def train(self, mode=True):
        correct_values = {True, False}
        
        if mode not in correct_values:
            raise ValueError('Invalid modes, correct values are: ' + ' '.join(correct_values))
 
        self._custom_train_mode = mode
 
        super().train(mode == True)
 
    def get_training_parameters(self):
        train_params = []
 
        for params in self.parameters():
            params.requires_grad = False

        if self._custom_train_mode == True:
            for params in self.parameters():
                params.requires_grad = True
                train_params += [params]
        
        return train_params

class WFCNetResnet(nn.Module):
 
    def __init__(self, in_channels=10, num_classes=61):
        super(WFCNetResnet, self).__init__()
        self.wfcnet = WFCNet(in_channels=in_channels)
        self.resnet = resnet34(pretrained=True, noBN=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
 
    def forward(self, x):
        
        logits_t = Variable(torch.cuda.FloatTensor(x.size(0), x.size(1), 61))
        for t in range(x.size(0)):
            images = self.wfcnet(x[t])
            _, _, features = self.resnet(images)
            features_avgd = self.avgpool(features)
            features_flattened = features_avgd.view(features_avgd.size(0), -1)
            logits_t[t] = self.fc(features_flattened)

        x = logits_t.mean(dim=0)
 
        return {'classifications': x}
        
 
    def train(self, mode=True):
        correct_values = {True, 'wfcnet', 'wfcnet+layer4resnet', False}
        
        if mode not in correct_values:
            raise ValueError('Invalid modes, correct values are: ' + ' '.join(correct_values))
 
        self._custom_train_mode = mode
 
        super().train(mode == True)

        if self._custom_train_mode != False:
            self.wfcnet.train(True)
 
    def get_training_parameters(self):
        train_params = []
 
        for params in self.parameters():
            params.requires_grad = False
 
        if 'wfcnet' in self._custom_train_mode:
            train_params += self.wfcnet.get_training_parameters()

        if 'layer4resnet' in self._custom_train_mode:
            for params in self.resnet.layer4.parameters():
                params.requires_grad = True
                train_params += [params]

        if self._custom_train_mode != False:
            for params in self.fc.parameters():
                params.requires_grad = True
                train_params += [params]
 
        return train_params

    def load_weights(self, file_path):
        model_dict = torch.load(file_path)
        if 'model_state_dict' in model_dict:
            self.load_state_dict(model_dict['model_state_dict'])
        else:
            self.load_state_dict(model_dict)


class WFCNetAttentionModel(nn.Module):

    def __init__(self, wfcnet_in_channels=10):
        super(WFCNetAttentionModel, self).__init__()
        self.wfcnet = WFCNet(in_channels=wfcnet_in_channels)
        self.attention_model = NewAttentionModel()

    def forward(self, x):
        wfc = Variable(torch.zeros(x.size(0), x.size(1), 3, 224, 224).cuda())
        for t in range(x.size(0)):
            wfc[t] = self.wfcnet(x[t])
        x = self.attention_model(wfc)
        return {'classifications': x['classifications']}

    def train(self, mode=True):
        correct_values = {True, 'stage1', 'stage2', False}
        
        if mode not in correct_values:
            raise ValueError('Invalid modes, correct values are: ' + ' '.join(map(str, correct_values)) + ', received ' + str(mode))

        self._custom_train_mode = mode

        super().train(mode == True)

        if self._custom_train_mode == True:
            self.wfcnet.train(True)
        self.attention_model.train(mode)

    def get_training_parameters(self):
        train_params = []

        for params in self.parameters():
            params.requires_grad = False

        train_params += self.wfcnet.get_training_parameters()        
        train_params += self.attention_model.get_training_parameters()

        return train_params

    def _load_weights_path(self, model, file_path, strict=False):
        model_dict = torch.load(file_path)
        if 'model_state_dict' in model_dict:
            model.load_state_dict(model_dict['model_state_dict'], strict)
        else:
            model.load_state_dict(model_dict, strict)

    def load_weights(self, file_path):
        if type(file_path) == dict:
            self._load_weights_path(self, file_path['wfcnet'], strict=False)
        else:
            self._load_weights_path(self, file_path)

class WFCNetAttentionModelBi(nn.Module):

    def __init__(self, wfcnet_in_channels=10):
        super(WFCNetAttentionModelBi, self).__init__()
        self.wfcnet = WFCNet(in_channels=wfcnet_in_channels)
        self.attention_model = NewAttentionModelBi()

    def forward(self, rgb, flow):
        wfc = Variable(torch.zeros(flow.size(0), flow.size(1), 3, 224, 224).cuda())
        for t in range(flow.size(0)):
            wfc[t] = self.wfcnet(flow[t])
        x = self.attention_model(rgb, wfc)
        return {'classifications': x['classifications']}

    def train(self, mode=True):
        correct_values = {True, 'stage1', 'stage2', False}
        
        if mode not in correct_values:
            raise ValueError('Invalid modes, correct values are: ' + ' '.join(map(str, correct_values)) + ', received ' + str(mode))

        self._custom_train_mode = mode

        super().train(mode == True)

        if self._custom_train_mode == True:
            self.wfcnet.train(True)
        self.attention_model.train(mode)

    def get_training_parameters(self):
        train_params = []

        for params in self.parameters():
            params.requires_grad = False

        
        train_params += self.wfcnet.get_training_parameters()
        train_params += self.attention_model.get_training_parameters()

        return train_params

    def _load_weights_path(self, model, file_path, strict=False):
        model_dict = torch.load(file_path)
        if 'model_state_dict' in model_dict:
            model.load_state_dict(model_dict['model_state_dict'], strict)
        else:
            model.load_state_dict(model_dict, strict)

    def load_weights(self, file_path):
        if type(file_path) == dict:
            self._load_weights_path(self, file_path['wfcnet'], strict=False)
        else:
            self._load_weights_path(self, file_path)