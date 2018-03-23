# -*- coding: utf-8 -*-

from __future__ import print_function, division

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms

from model import PCBModel
from utils import *


# ---------------------- Settings ----------------------
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--dataset', type=str, default='market1501',
                    choices=['market1501', 'cuhk03', 'duke'])
parser.add_argument('--train_all', action='store_true',
                    help='Use all training data. Set true when training the final model.')

# Hyperparameters
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--learning_rate', default=0.1, type=float,
                    help='FC params learning rate')
parser.add_argument('--epochs', default=60, type=int,
                    help='The number of epochs to train')
arg = parser.parse_args()

torch.manual_seed(arg.seed)
torch.cuda.manual_seed_all(arg.seed)

USE_GPU = torch.cuda.is_available()
if not os.path.isdir('./model'):
    os.mkdir('./model')
    os.mkdir('./model/' + arg.dataset)


# ---------------------- Train function ----------------------
def train(model, criterion, optimizer, scheduler, dataloaders, num_epochs):

    start_time = time.time()

    # Logger instance
    logger = Logger(arg.dataset)

    best_model_state = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):

        logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
        logger.x_epoch.append(epoch + 1)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            batch_num = 0
            for data in dataloaders[phase]:
                batch_num += 1

                input, label = data

                if USE_GPU:
                    input = Variable(input.cuda())
                    label = Variable(label.cuda())
                else:
                    input, label = Variable(input), Variable(label)

                optimizer.zero_grad()

                output = model(input)

                # Compute PCB loss
                loss = torch.sum(
                    torch.cat([criterion(feat, label) for feat in output]))
                # loss = criterion(output, label)

                # Backward only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]

            epoch_loss = running_loss / batch_num

            logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # Save result to logger
            logger.y_loss[phase].append(epoch_loss)

        last_model_state = model.state_dict()
        if epoch % 20 == 19:
            save_network(model, arg.dataset, epoch)

        logger.info('-' * 10)

    # Save the loss curve
    logger.save_curve()

    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save final model weights
    model.load_state_dict(last_model_state)
    save_network(model, arg.dataset, 'final')
    return model


# ---------------------- Set dataloaders ----------------------
transform_train_list = [
    transforms.Resize(size=(384, 128), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(384, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}


image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(DATASET_PATH[arg.dataset], 'train' + ('_all' if arg.train_all else '')),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(DATASET_PATH[arg.dataset], 'val'),
                                             data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=arg.batch_size,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}


# For debugging
# inputs, classes = next(iter(dataloaders['train']))


# ---------------------- Training settings ----------------------
model = PCBModel(len(image_datasets['train'].classes))

# Use multiple GPUs
if torch.cuda.device_count() > 1:
    model_wraped = nn.DataParallel(model)
else:
    model_wraped = model

if USE_GPU:
    model_wraped = model_wraped.cuda()

criterion = nn.CrossEntropyLoss()

# Finetune the net
optimizer = optim.SGD([
    {'params': model.backbone.parameters(), 'lr': arg.learning_rate / 10},
    {'params': model.local_conv.parameters(), 'lr': arg.learning_rate},
    {'params': model.fc_list.parameters(), 'lr': arg.learning_rate}
], momentum=0.9, weight_decay=5e-4, nesterov=True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# ---------------------- Start training ----------------------
model = train(model_wraped, criterion, optimizer, scheduler, dataloaders,
              arg.epochs)
