# pylint: disable=eval-used,unused-import

from configparser import ConfigParser 

from sklearn.model_selection import ParameterGrid

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from models.attention_model import AttentionModel
from models.two_stream_model import TwoStreamAttentionModel
from models.resnet.flow_resnet import flow_resnet34
from models.wfcnet import WFCNetResnet, WFCNetAttentionModel, WFCNetAttentionModelBi

from torch.nn import CrossEntropyLoss

from loss import ClassificationLoss, MSClassificationLoss, MSRegressionClassificationLoss


class ParameterParser():
    """Class which allows reading tuning parameters from a .ini file"""

    def __init__(self, path, assert_sections=("GLOBAL", "TRAIN_DATA_LOADER", "VALID_DATA_LOADER", "MODEL", "TRAINING", "LOSS", "OPTIMIZER", "SCHEDULER")):
        """
        Default constructor
            Input:  - path .ini file
                    - sections to check to be present in the .ini file
            Output: //
        """
    
        self.__config = ConfigParser()
        self.__config.read(path)
        self.__assert_sections = assert_sections

        self._params = {}

        for section in self.__assert_sections:
            try:
                self.__config[section]
            except KeyError:
                print("Section {} is not present in .ini file".format(section))

        for section in [s for s in self.__config if s != 'DEFAULT' and s != 'GLOBAL']:
            for param in self.__config[section]:
                # We know this might be a security hazard
                el_value = eval(self.__config[section][param])
                if type(el_value) != list:
                    el_value = [el_value]
                self._params['{}____{}'.format(section, param)] = el_value
    
    def __iter__(self):
        grid = list(ParameterGrid(self._params))
        grid_new = list()
        for config in grid:
            sections = {k.split('____')[0] for k in config}
            new_config = {k: {} for k in sections}
            for section_param in config:
                section, param = section_param.split('____')
                new_config[section][param] = config[section_param]
            grid_new.append(new_config)
        return iter(grid_new)

    def __getattr__(self, name):
        return eval(self.__config['GLOBAL'][name])