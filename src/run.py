import argparse, json, os, pprint, sys

import hashlib

import numpy as np

import torch, torchvision

from utils.configuration import ParameterParser

from datasets import DatasetRGB, DatasetFlow, DatasetRGBFlow, DatasetMMAPS, DatasetRGBMMAPS

from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip, Binarize)

from models.attention_model import AttentionModel
from models.two_stream_model import TwoStreamAttentionModel
from models.resnet.flow_resnet import flow_resnet34

from utils.trainer import train_model, forward_rgb, forward_rgbmmaps, forward_flow, forward_rgbflow

def main(configs):

    from utils.generate_dataset_jsons import generate_dataset_jsons
    generate_dataset_jsons(configs.dataset_folder)
    with open('dataset_rgb_train.json') as json_file:
        dataset_rgb_train_json = json.load(json_file) 
    with open('dataset_rgb_valid.json') as json_file:
        dataset_rgb_valid_json = json.load(json_file) 
    with open('dataset_mmaps_train.json') as json_file:
        dataset_mmaps_train_json = json.load(json_file) 
    with open('dataset_mmaps_valid.json') as json_file:
        dataset_mmaps_valid_json = json.load(json_file) 
    with open('dataset_flow_train.json') as json_file:
        dataset_flow_train_json = json.load(json_file) 
    with open('dataset_flow_valid.json') as json_file:
        dataset_flow_valid_json = json.load(json_file) 

    torch.backends.cudnn.benchmark = True

    if os.path.exists(configs.output_folder):
        print('Warning: output folder {} already exists!'.format(configs.output_folder))
    try:
        os.makedirs(configs.output_folder)
    except FileExistsError:
        pass

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    randomHorizontalFlip = RandomHorizontalFlip()
    randomMultiScaleCornerCrop = MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224)

    train_transforms = Compose([randomHorizontalFlip, 
                                randomMultiScaleCornerCrop,
                                ToTensor(), 
                                normalize])

    train_transforms.randomize_parameters()

    valid_transforms = Compose([CenterCrop(224),
                                ToTensor(),
                                normalize])


    train_transforms_mmaps = Compose([randomHorizontalFlip, 
                                randomMultiScaleCornerCrop,
                                Scale(7),
                                ToTensor(), Binarize()])

    valid_transforms_mmaps = Compose([CenterCrop(224),
                                Scale(7),
                                ToTensor(), Binarize()])

    print('Loading dataset')

    if configs.dataset == 'DatasetRGB':
        dataset_rgb_train = DatasetRGB(configs.dataset_folder, dataset_rgb_train_json, spatial_transform=train_transforms, seqLen=configs.dataset_rgb_n_frames, minLen=5)
        dataset_rgb_valid = DatasetRGB(configs.dataset_folder, dataset_rgb_valid_json, spatial_transform=valid_transforms, seqLen=configs.dataset_rgb_n_frames, minLen=5, device='cuda')
        
        dataset_train = dataset_rgb_train
        dataset_valid = dataset_rgb_valid

        forward_fn = forward_rgb
    elif configs.dataset == 'DatasetRGBMMAPS':
        dataset_rgb_train = DatasetRGB(configs.dataset_folder, dataset_rgb_train_json, spatial_transform=train_transforms, seqLen=configs.dataset_rgb_n_frames, minLen=5)
        dataset_rgb_valid = DatasetRGB(configs.dataset_folder, dataset_rgb_valid_json, spatial_transform=valid_transforms, seqLen=configs.dataset_rgb_n_frames, minLen=5, device='cuda')
        dataset_mmaps_train = DatasetMMAPS(configs.dataset_folder, dataset_mmaps_train_json, spatial_transform=train_transforms_mmaps, seqLen=configs.dataset_rgb_n_frames, minLen=1, enable_randomize_transform=False)
        dataset_mmaps_valid = DatasetMMAPS(configs.dataset_folder, dataset_mmaps_valid_json, spatial_transform=valid_transforms_mmaps, seqLen=configs.dataset_rgb_n_frames, minLen=1, enable_randomize_transform=False, device='cuda')
        
        dataset_rgbmmaps_train = DatasetRGBMMAPS(dataset_rgb_train, dataset_mmaps_train)
        dataset_rgbmmaps_valid = DatasetRGBMMAPS(dataset_rgb_valid, dataset_mmaps_valid)

        dataset_train = dataset_rgbmmaps_train
        dataset_valid = dataset_rgbmmaps_valid

        forward_fn = forward_rgbmmaps
    elif configs.dataset == 'DatasetFlow':
        dataset_flow_train = DatasetFlow(configs.dataset_folder, dataset_flow_train_json, spatial_transform=train_transforms, stack_size=configs.dataset_flow_stack_size, sequence_mode='single_random')
        dataset_flow_valid = DatasetFlow(configs.dataset_folder, dataset_flow_valid_json, spatial_transform=valid_transforms, stack_size=configs.dataset_flow_stack_size, sequence_mode='single_midtime')

        dataset_train = dataset_flow_train
        dataset_valid = dataset_flow_valid

        forward_fn = forward_flow
    elif configs.dataset == 'DatasetFlowMultiple':
        dataset_flow_train = DatasetFlow(configs.dataset_folder, dataset_flow_train_json, spatial_transform=train_transforms, stack_size=configs.dataset_flow_stack_size, sequence_mode='multiple_jittered')
        dataset_flow_valid = DatasetFlow(configs.dataset_folder, dataset_flow_valid_json, spatial_transform=valid_transforms, stack_size=configs.dataset_flow_stack_size, sequence_mode='multiple')

        dataset_train = dataset_flow_train
        dataset_valid = dataset_flow_valid

        forward_fn = forward_flow
    elif configs.dataset == 'DatasetRGBFlow':
        dataset_rgb_train = DatasetRGB(configs.dataset_folder, dataset_rgb_train_json, spatial_transform=train_transforms, seqLen=configs.dataset_rgb_n_frames, minLen=5)
        dataset_rgb_valid = DatasetRGB(configs.dataset_folder, dataset_rgb_valid_json, spatial_transform=valid_transforms, seqLen=configs.dataset_rgb_n_frames, minLen=5)
        dataset_flow_train = DatasetFlow(configs.dataset_folder, dataset_flow_train_json, spatial_transform=train_transforms, stack_size=configs.dataset_flow_stack_size, sequence_mode='single_random', enable_randomize_transform=False)
        dataset_flow_valid = DatasetFlow(configs.dataset_folder, dataset_flow_valid_json, spatial_transform=valid_transforms, stack_size=configs.dataset_flow_stack_size, sequence_mode='single_midtime', enable_randomize_transform=False)

        dataset_rgbflow_train = DatasetRGBFlow(dataset_rgb_train, dataset_flow_train)
        dataset_rgbflow_valid = DatasetRGBFlow(dataset_rgb_valid, dataset_flow_valid)

        dataset_train = dataset_rgbflow_train
        dataset_valid = dataset_rgbflow_valid

        forward_fn = forward_rgbflow
    else:
        raise ValueError('Unknown dataset type: {}'.format(configs.dataset))
    
    report_file = open(os.path.join(configs.output_folder, 'report.txt'), 'w')

    for config in configs:
        config_name = hashlib.md5(str(config).encode('utf-8')).hexdigest()
        print('Running', config_name)

        with open(os.path.join(configs.output_folder, config_name + '.params.txt'), 'w') as f:
            f.write(pprint.pformat(config))

        train_loader = torch.utils.data.DataLoader(dataset_train, **config['TRAIN_DATA_LOADER'])
        valid_loader = torch.utils.data.DataLoader(dataset_valid, **config['VALID_DATA_LOADER'])

        model_class = config['MODEL']['model']
        model_params = {k:v for (k,v) in config['MODEL'].items() if k != 'model'}
        model = model_class(**model_params)
        if config['TRAINING'].get('_model_state_dict', None) is not None:
            model.load_weights(config['TRAINING']['_model_state_dict'])
        model.train(config['TRAINING']['train_mode'])
        model.cuda()

        loss_class = config['LOSS']['loss']
        loss_params = {k:v for (k,v) in config['LOSS'].items() if k != 'loss'}
        loss_fn = loss_class(**loss_params)
        
        model_weights = []
        for i in range(10):
            group_name = '_group_' + str(i)
            if group_name + '_params' in config['OPTIMIZER']:
                model_weights_group = {'params' : model.get_training_parameters(name=config['OPTIMIZER'][group_name + '_params'])}
                for k in config['OPTIMIZER']:
                    if k.startswith(group_name) and k != group_name + '_params':
                        model_weights_group[k[9:]] = config['OPTIMIZER'][k]
                model_weights.append(model_weights_group)
        if len(model_weights) == 0:
            model_weights = model.get_training_parameters()

        optimizer_class = config['OPTIMIZER']['optimizer']
        optimizer_params = {k:v for (k,v) in config['OPTIMIZER'].items() if k != 'optimizer' and not k.startswith('_')}
        optimizer = optimizer_class(model_weights, **optimizer_params)

        scheduler_class = config['SCHEDULER']['scheduler']
        scheduler_params = {k:v for (k,v) in config['SCHEDULER'].items() if k != 'scheduler' and not k.startswith('_')}
        scheduler_lr = scheduler_class(optimizer, **scheduler_params)

        model_state_dict_path = os.path.join(configs.output_folder, config_name + '.model')
        logfile = open(os.path.join(configs.output_folder, config_name + '.log.txt'), 'w')

        result = train_model(model=model, train_loader=train_loader, valid_loader=valid_loader, 
                    forward_fn=forward_fn, loss_fn=loss_fn, optimizer=optimizer, scheduler_lr=scheduler_lr,
                    model_state_dict_path=model_state_dict_path, logfile=logfile,
                    **{k:v for (k,v) in config['TRAINING'].items() if not k.startswith('_')})

        max_valid_accuracy_idx = np.argmax(result['accuracies_valid'])

        print('{} | Train Loss {:04.2f} | Train Accuracy = {:05.2f}% | Valid Loss {:04.2f} | Valid Accuracy = {:05.2f}%'
        .format(config_name, 
                result['losses_train'][max_valid_accuracy_idx], result['accuracies_train'][max_valid_accuracy_idx] * 100,
                result['losses_valid'][max_valid_accuracy_idx], result['accuracies_valid'][max_valid_accuracy_idx] * 100
           ), file=report_file)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: need to specify configuration file")
        exit()
    main(ParameterParser(sys.argv[1]))