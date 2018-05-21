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
from test import test
import utils


# ---------------------- Settings ----------------------
parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--save_path', type=str, default='./model')
parser.add_argument('--dataset', type=str, default='market1501',
                    choices=['market1501', 'cuhk03', 'duke'])
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--learning_rate', default=0.1, type=float,
                    help='FC params learning rate')
parser.add_argument('--epochs', default=60, type=int,
                    help='The number of epochs to train')
parser.add_argument('--share_conv', action='store_true')
parser.add_argument('--stripes', type=int, default=6)
arg = parser.parse_args()

# Fix random seed
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# Make saving directory
save_dir_path = os.path.join(arg.save_path, arg.dataset)
os.makedirs(save_dir_path, exist_ok=True)


# ---------------------- Train function ----------------------
def train(model, criterion, optimizer, scheduler, dataloader, num_epochs, device):

    start_time = time.time()

    # Logger instance
    logger = utils.Logger(save_dir_path)
    logger.info('-' * 10)
    logger.info(vars(arg))

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

        model.train()
        scheduler.step()

        # Training
        running_loss = 0.0
        batch_num = 0
        for inputs, labels in dataloader:
            batch_num += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # with torch.set_grad_enabled(True):
            outputs = model(inputs)

            # Sum up the stripe softmax loss
            loss = 0
            for logits in outputs:
                stripe_loss = criterion(logits, labels)
                loss += stripe_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset.imgs)
        logger.info('Training Loss: {:.4f}'.format(epoch_loss))

        # Save result to logger
        logger.x_epoch_loss.append(epoch + 1)
        logger.y_train_loss.append(epoch_loss)

        if (epoch + 1) % 10 == 0 or epoch + 1 == num_epochs:
            # Testing / Validating
            torch.cuda.empty_cache()
            model.set_return_features(True)
            CMC, mAP, _ = test(model, arg.dataset, 512)
            model.set_return_features(False)
            logger.info('Testing: top1:%.2f top5:%.2f top10:%.2f mAP:%.2f' %
                        (CMC[0], CMC[4], CMC[9], mAP))

            logger.x_epoch_test.append(epoch + 1)
            logger.y_test['top1'].append(CMC[0])
            logger.y_test['mAP'].append(mAP)
            if epoch + 1 != num_epochs:
                utils.save_network(model, save_dir_path, str(epoch + 1))

        logger.info('-' * 10)

    # Save the loss curve
    logger.save_curve()

    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save final model weights
    utils.save_network(model, save_dir_path, 'final')


# For debugging
# inputs, classes = next(iter(dataloaders['train']))

# ---------------------- Training settings ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = utils.getDataLoader(
    arg.dataset, arg.batch_size, 'train', shuffle=True, augment=True)
model = PCBModel(num_classes=len(train_dataloader.dataset.classes),
                 num_stripes=arg.stripes, share_conv=arg.share_conv, return_features=False)

criterion = nn.CrossEntropyLoss()

# Finetune the net
optimizer = optim.SGD([
    {'params': model.backbone.parameters(), 'lr': arg.learning_rate / 10},
    {'params': model.local_conv.parameters() if arg.share_conv else model.local_conv_list.parameters(),
     'lr': arg.learning_rate},
    {'params': model.fc_list.parameters(), 'lr': arg.learning_rate}
], momentum=0.9, weight_decay=5e-4, nesterov=True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Use multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# ---------------------- Start training ----------------------
train(model, criterion, optimizer, scheduler, train_dataloader,
      arg.epochs, device)

torch.cuda.empty_cache()
