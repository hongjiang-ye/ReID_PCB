import os
import torch
from collections import OrderedDict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


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


class Logger(object):
    '''Draw the training log curve'''

    def __init__(self, model_name):
        self.model_name = model_name
        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(121, title="loss")
        self.ax1 = self.fig.add_subplot(122, title="top1err")
        self.x_epoch = []
        self.y_loss = {}
        self.y_loss['train'] = []
        self.y_loss['val'] = []
        self.y_err = {}
        self.y_err['train'] = []
        self.y_err['val'] = []

    def save_curve(self):
        # Draw the loss cruve
        self.ax0.plot(self.x_epoch, self.y_loss['train'], 'bo-', label='train')
        self.ax0.plot(self.x_epoch, self.y_loss['val'], 'ro-', label='val')
        self.ax1.plot(self.x_epoch, self.y_err['train'], 'bo-', label='train')
        self.ax1.plot(self.x_epoch, self.y_err['val'], 'ro-', label='val')

        if len(self.x_epoch) == 1:
            self.ax0.legend()
            self.ax1.legend()

        save_path = os.path.join('./model', self.model_name, 'train_log.jpg')
        self.fig.savefig(save_path)
