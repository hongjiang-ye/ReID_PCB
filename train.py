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

from model import ft_net
from utils import save_network, Logger


######################################################################
# Settings
# --------

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--name', default='ft_ResNet50',
                    type=str, help='model name')
parser.add_argument('--data_dir', default='/home/share/hongjiang/Market-1501-v15.09.15/pytorch',
                    type=str, help='Training dataset path')
parser.add_argument('--train_all', action='store_true',
                    help='Use all training data. Set true when training the final model.')

# Hyperparameters
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--learning_rate', default=0.1, type=float,
                    help='FC params learning rate')
parser.add_argument('--epochs', default=60, type=int,
                    help='The number of epochs to train')
arg = parser.parse_args()


torch.manual_seed(arg.seed)
torch.cuda.manual_seed_all(arg.seed)

######################################################################
# Training and logging functions
# --------

USE_GPU = torch.cuda.is_available()


def train(model, criterion, optimizer, scheduler, dataloaders, num_epochs):

    start_time = time.time()

    # Logger instance
    logger = Logger(arg.name)

    best_model_state = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        logger.x_epoch.append(epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:

                inputs, labels = data

                if USE_GPU:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # Backward only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]

                num_corrects = torch.sum(preds == labels.data)
                running_corrects += num_corrects
                # print('Epoch {} {}/Batch {}: Loss: {:.4f} Acc: {:.4f}'.format(
                #     epoch + 1, phase, i + 1, loss.data[0] / arg.batch_size, num_corrects / arg.batch_size))

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects / len(image_datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Save result to logger
            logger.y_loss[phase].append(epoch_loss)
            logger.y_err[phase].append(1.0 - epoch_acc)

        # Save the best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_state = model.state_dict()

        last_model_state = model.state_dict()
        if epoch % 10 == 9:
            save_network(model, arg.name, epoch)

        # Save the loss curve
        logger.save_curve()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Save best model weights
    model.load_state_dict(best_model_state)
    save_network(model, arg.name, 'best')

    # Save last model weights
    model.load_state_dict(last_model_state)
    save_network(model, arg.name, 'last')
    return model


######################################################################
# Make model directory
# ---------

if not os.path.isdir('./model'):
    os.mkdir('./model')
    os.mkdir('./model/' + arg.name)


######################################################################
# Set dataloaders
# ---------


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
image_datasets['train'] = datasets.ImageFolder(os.path.join(arg.data_dir, 'train' + ('_all' if arg.train_all else '')),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(arg.data_dir, 'val'),
                                             data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=arg.batch_size,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}


# For debugging
# inputs, classes = next(iter(dataloaders['train']))


######################################################################
# Training settings
# ----------------------

model = ft_net(len(image_datasets['train'].classes))
# print(model)

# Use multiple GPUs
if torch.cuda.device_count() > 1:
    model_wraped = nn.DataParallel(model)
else:
    model_wraped = model

if USE_GPU:
    model_wraped = model_wraped.cuda()

criterion = nn.CrossEntropyLoss()

# Finetune the net
ignored_params = list(map(id, model.model.fc.parameters())) + \
    list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

optimizer = optim.SGD([
    {'params': base_params, 'lr': arg.learning_rate / 10},
    {'params': model.model.fc.parameters(), 'lr': arg.learning_rate},
    {'params': model.classifier.parameters(), 'lr': arg.learning_rate}
], momentum=0.9, weight_decay=5e-4, nesterov=True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


######################################################################
# Start training
# ----------------------

model = train(model_wraped, criterion, optimizer, scheduler, dataloaders,
              arg.epochs)
