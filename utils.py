import os
import logging
import torch
from collections import OrderedDict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------- Global settings ----------------------
DATASET_PATH = {
    'market1501': '/home/share/hongjiang/Market-1501-v15.09.15/pytorch',
    'duke': '/home/share/hongjiang/DukeMTMC-reID/pytorch',
    'cuhk03': '/home/share/hongjiang/cuhk03_release/pytorch'}
TRAINING_IDS = {
    'market1501': 751,
    'duke': 702,
    'cuhk03': 767}


# ---------------------- Helper functions ----------------------
def save_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.state_dict(), save_path)


def load_network(network, name, epoch_label):
    save_path = os.path.join(
        './model', name, 'net_%s.pth' % epoch_label)

    # Original saved file with DataParallel
    state_dict = torch.load(save_path, map_location='cpu')

    # If the model saved with DataParallel, the keys in state_dict contains 'module'
    if list(state_dict.keys())[0][:6] == 'module':

        # Create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key_name = k[7:]  # remove `module.`
            new_state_dict[key_name] = v

        network.load_state_dict(new_state_dict)
    else:
        network.load_state_dict(state_dict)

    return network


# ---------------------- Logger ----------------------
class Logger(logging.Logger):
    '''Inherit from logging.Logger.
    Print logs to console and file.
    Add functions to draw the training log curve.'''

    def __init__(self, model_name):
        self.dir_path = os.path.join('./model', model_name)
        if not os.path.isdir(self.dir_path):
            os.mkdir(self.dir_path)

        super(Logger, self).__init__('Training logger')

        # Print logs to console and file
        self.model_name = model_name
        self.file_handler = logging.FileHandler(
            os.path.join(self.dir_path, 'train_log.txt'))
        self.console_handler = logging.StreamHandler()
        self.log_format = logging.Formatter(
            "%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        self.file_handler.setFormatter(self.log_format)
        self.console_handler.setFormatter(self.log_format)
        self.addHandler(self.file_handler)
        self.addHandler(self.console_handler)

        # Draw curve
        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(111)
        self.x_epoch = []
        self.y_loss = {}
        self.y_loss['train'] = []
        self.y_loss['val'] = []

    def save_curve(self):

        self.ax0.plot(
            self.x_epoch, self.y_loss['train'], 'bs-', markersize='2', label='train')
        self.ax0.plot(
            self.x_epoch, self.y_loss['val'], 'rs-', markersize='2', label='val')
        self.ax0.set_title('PCBModel')
        self.ax0.set_ylabel('Loss')
        self.ax0.set_xlabel('Epoch')
        self.ax0.legend()

        save_path = os.path.join(self.dir_path, 'train_log.jpg')
        self.fig.savefig(save_path)
