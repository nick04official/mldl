import torch
import torch.nn as nn

from models.resnet.flow_resnet import flow_resnet34
from models.attention_model import AttentionModel

class TwoStreamAttentionModel(nn.Module):
    def __init__(self, flowModel='', frameModel='', stackSize=5, memSize=512, num_classes=61):
        super(TwoStreamAttentionModel, self).__init__()

        self.flowModel = flow_resnet34(False, channels=2*stackSize, num_classes=num_classes)
        if flowModel != '':
            self.flowModel.load_state_dict(torch.load(flowModel))
        
        self.frameModel = AttentionModel(num_classes, memSize)
        if frameModel != '':
            self.frameModel.load_state_dict(torch.load(frameModel))
        
        self.fc2 = nn.Linear(512 * 2, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(self.dropout, self.fc2)

    def train(self, mode=True):
        correct_values = {True, 'layer4', False}
        
        if mode not in correct_values:
            raise ValueError('Invalid modes, correct values are: ' + ' '.join(correct_values))

        self._custom_train_mode = mode
        
        # Fai fare il training completo solo se mode == True
        super().train(mode == True)

        self.flowModel.train(mode)
        # TODO: loro sembrano non farlo questo
        self.frameModel.train('stage2' if mode == 'layer4' else mode)
        if mode != False:
            self.classifier.train(True)

    def get_training_parameters(self, name = 'all'):
        rgb_train_params = []
        flow_train_params = []
        train_params = []

        # Prima levo i gradienti a tutti, e poi li aggiungo solo a quelli
        # su cui faccio il training
        for params in self.parameters():
            params.requires_grad = False

        # è responsabilità della funzione negli oggetti aggiungere i gradienti
        flow_train_params += self.flowModel.get_training_parameters()
        rgb_train_params += self.frameModel.get_training_parameters()
        if self._custom_train_mode != False:
            for params in self.classifier.parameters():
                params.requires_grad = True
                train_params += [params]
        if name == 'all':
            return train_params + rgb_train_params + flow_train_params
        elif name == 'rgb':
            return rgb_train_params
        elif name == 'flow':
            return flow_train_params
        elif name == 'fc':
            return train_params

    def _load_weights_path(self, model, file_path):
        model_dict = torch.load(file_path)
        if 'model_state_dict' in model_dict:
            model.load_state_dict(model_dict['model_state_dict'])
        else:
            model.load_state_dict(model_dict)

    def load_weights(self, file_path):
        if type(file_path) == dict:
            self._load_weights_path(self.frameModel, file_path['rgb'])
            self._load_weights_path(self.flowModel, file_path['flow'])
        else:
            self._load_weights_path(self, file_path)
        

    def forward(self, inputVariable):
        inputVariableFrame, inputVariableFlow = inputVariable
        out_flow = self.flowModel(inputVariableFlow)
        out_rgb = self.frameModel(inputVariableFrame)

        flow_feats = out_flow['conv_feats']
        rgb_feats = out_rgb['lstm_feats']

        twoStreamFeats = torch.cat((flow_feats, rgb_feats), 1)
        return {'classifications': self.classifier(twoStreamFeats)}
