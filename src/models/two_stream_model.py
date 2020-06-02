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

    def get_training_parameters(self):
        train_params = []
        flow_train_params = []

        # Prima levo i gradienti a tutti, e poi li aggiungo solo a quelli
        # su cui faccio il training
        for params in self.parameters():
            params.requires_grad = False

        # è responsabilità della funzione negli oggetti aggiungere i gradienti
        flow_train_params += self.flowModel.get_training_parameters()
        train_params += self.frameModel.get_training_parameters()
        if self._custom_train_mode != False:
            for params in self.classifier.parameters():
                params.requires_grad = True
                train_params += [params]

        return train_params, flow_train_params

    def forward(self, inputVariable):
        inputVariableFrame, inputVariableFlow = inputVariable
        _, flowFeats = self.flowModel(inputVariableFlow)
        _, rgbFeats = self.frameModel(inputVariableFrame)
        twoStreamFeats = torch.cat((flowFeats, rgbFeats), 1)
        return self.classifier(twoStreamFeats), None
