import sys

import numpy as np

import torch

from torch.autograd import Variable

from tqdm.auto import tqdm

def forward_rgbmmaps(model, batch, loss_fn):
    rgb_input, mmaps_input, label = batch

    inputVariable = Variable(rgb_input.permute(1, 0, 2, 3, 4).cuda())
    mmapVariable = Variable(mmaps_input.cuda())
    labelVariable = Variable(label.cuda())

    output = model(inputVariable)

    output_label = output['classifications']
    output_mmaps = output['ms_feats']

    loss_values = loss_fn(output_label, labelVariable, output_mmaps, mmapVariable)
    
    n_elements = int(len(labelVariable))

    _, predicted = torch.max(output_label.data, 1)
    n_corrects = int((predicted == labelVariable).sum())

    return loss_values, n_elements, n_corrects

def forward_rgb(model, batch, loss_fn):
    rgb_input, label = batch

    inputVariable = Variable(rgb_input.permute(1, 0, 2, 3, 4).cuda())
    labelVariable = Variable(label.cuda())

    output = model(inputVariable)

    output_label = output['classifications']

    loss_values = loss_fn(output_label, labelVariable)
    
    n_elements = int(len(labelVariable))

    _, predicted = torch.max(output_label.data, 1)
    n_corrects = int((predicted == labelVariable).sum())

    return loss_values, n_elements, n_corrects


@torch.no_grad()
def get_loss_accuracy(model, set_loader, forward_fn, loss_fn):

    model.train(False)

    running_corrects = 0
    losses = np.zeros(len(set_loader))

    n_samples = 0
    
    for i, batch in enumerate(set_loader):
        loss_values, n_elements, n_corrects = forward_fn(model, batch, loss_fn)

        #loss_values = loss_fn(output_label, labelVariable)
        
        n_samples += n_elements
        losses[i] = float(loss_values.sum())
        running_corrects += n_corrects

    accuracy = (running_corrects / n_samples)
    loss = losses.sum() / n_samples

    return loss, accuracy

def train_model(model, train_loader, valid_loader, forward_fn, loss_fn, optimizer, scheduler_lr, model_state_dict_path, n_epochs, train_mode=True, logfile=sys.stdout):
    
    current_train_loader = None
    next_train_loader = None

    min_valid_accuracy = 0

    accuracies_train = []
    accuracies_valid = []
    losses_valid = []
    losses_train = []

    
    for epoch in tqdm(range(n_epochs)):

        model.train(train_mode)

        train_running_corrects = 0
        train_losses = np.zeros(len(train_loader))
        train_n_samples = 0

        if current_train_loader == None:
            current_train_loader = iter(train_loader)

        next_valid_loader = None

        for iteration, batch in enumerate(current_train_loader):
            optimizer.zero_grad()

            loss_values, n_elements, n_corrects = forward_fn(model, batch, loss_fn)

            train_n_samples += n_elements
            train_losses[iteration] = float(loss_values.sum())
            train_running_corrects += n_corrects
            
            loss_values.mean().backward()

            optimizer.step()

            if next_train_loader == None and (iteration + 1) >= int(len(current_train_loader) / 2):
                next_train_loader = iter(train_loader)

            if next_valid_loader == None and (iteration + 1) >= int(len(current_train_loader) / 2):
                next_valid_loader = iter(valid_loader)

        train_accuracy = (train_running_corrects / train_n_samples)
        train_loss = train_losses.sum() / train_n_samples

        if next_valid_loader == None:
            next_valid_loader = iter(valid_loader)

        valid_loss, valid_accuracy = get_loss_accuracy(model, next_valid_loader, forward_fn, loss_fn)

        print('Epoch: {:03d} | Train Loss {:04.2f} | Train Accuracy = {:05.2f}% | Valid Loss {:04.2f} | Valid Accuracy = {:05.2f}%'
            .format(epoch + 1, train_loss, train_accuracy * 100, valid_loss, valid_accuracy * 100), file=logfile)

        accuracies_train.append(train_accuracy)
        accuracies_valid.append(valid_accuracy)
        losses_train.append(train_loss)
        losses_valid.append(valid_loss)
        
        scheduler_lr.step()

        if valid_accuracy > min_valid_accuracy:
            torch.save(model.state_dict(), model_state_dict_path)
            min_valid_accuracy = valid_accuracy

        current_train_loader = next_train_loader
        next_train_loader = None

        del next_valid_loader

    del current_train_loader

    state = {
        'model_state_dict': torch.load(model_state_dict_path),
        'accuracies_train': accuracies_train,
        'accuracies_valid': accuracies_valid,
        'losses_valid': losses_valid,
        'losses_train': losses_train
    }
    torch.save(state, model_state_dict_path)

    print('Max valid accuracy = {:05.2f}%'.format(min_valid_accuracy * 100), file=logfile)

    return state