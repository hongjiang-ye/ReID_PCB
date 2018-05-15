import os
import logging
import torch
from torchvision import datasets, transforms
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
# DATASET_PATH = {
#     'market1501': '/Users/ruby/Desktop/Lab/others/Market-1501-v15.09.15/pytorch',
#     'duke': '/Users/ruby/Desktop/Lab/others/DukeMTMC-reID/pytorch',
#     'cuhk03': '/Users/ruby/Desktop/Lab/others/cuhk03_release/pytorch'}


# ---------------------- Helper functions ----------------------
def save_network(network, path, epoch_label):
    file_path = os.path.join(path, 'net_%s.pth' % epoch_label)
    torch.save(network.state_dict(), file_path)


def load_network(network, path, epoch_label):
    file_path = os.path.join(path, 'net_%s.pth' % epoch_label)

    # Original saved file with DataParallel
    state_dict = torch.load(
        file_path, map_location=lambda storage, loc: storage)

    # If the model saved with DataParallel, the keys in state_dict contains 'module'
    if list(state_dict.keys())[0][:6] == 'module':
        # Create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key_name = k[7:]  # remove `module.`
            new_state_dict[key_name] = v

        state_dict = new_state_dict

    # # ------------- PCB specific -------------
    # # Load PCB from another dataset, change the fc_list parameters' shape
    # for name in state_dict.keys():
    #     if name[0:7] == 'fc_list':
    #         desired_shape = network.state_dict()[name].shape
    #         if desired_shape != state_dict[name].shape:
    #             state_dict[name] = torch.randn(desired_shape)
    # # ------------------------------------------------

    network.load_state_dict(state_dict)

    return network


def getDataLoader(dataset, batch_size, part, shuffle=True, augment=True):
    """Return the dataloader and imageset of the given dataset

    Arguments:
        dataset {string} -- name of the dataset: [market1501, duke, cuhk03]
        batch_size {int} -- the batch size to load
        part {string} -- which part of the dataset: [train, query, gallery]

    Returns:
        (torch.utils.data.DataLoader, torchvision.datasets.ImageFolder) -- the data loader and the image set
    """

    transform_list = [
        transforms.Resize(size=(384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if augment:
        transform_list.insert(1, transforms.RandomHorizontalFlip())

    data_transform = transforms.Compose(transform_list)

    image_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH[dataset], part),
                                         data_transform)

    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=4)

    return dataloader


def save_rank_list_img(query_dataloader, gallery_dataloader, sorted_index_list, sorted_true_list, junk_index_list):

    rank_lists_imgs = []

    # randomly select 10 query images to show the rank list
    query_index_list = list(range(len(sorted_index_list)))
    random.shuffle(query_index_list)
    selected_query_index = query_index_list[:10]

    for i in selected_query_index:

        cur_rank_list = []

        query_img = query_dataloader.dataset[i][0]
        query_img_with_boundary = torch.nn.functional.pad(
            query_img, (3, 3, 3, 3), "constant", value=0)
        cur_rank_list.append(query_img_with_boundary)

        sorted_index = sorted_index_list[i]
        sorted_true = sorted_true_list[i]
        junk_index = junk_index_list[i]

        # show the top 10(not junk) gallery images of the rank list
        num = 0
        idx = 0
        while num < 10:
            if sorted_index[idx] in junk_index:
                idx += 1
                continue

            gallery_img = gallery_dataloader.dataset[sorted_index[idx]][0]

            gallery_img_with_boundary = torch.nn.functional.pad(
                gallery_img, (3, 3, 3, 3), "constant", value=0)

            if sorted_true[idx]:
                # True, with green boundary
                gallery_img_with_boundary[1, :, :] = torch.nn.functional.pad(
                    gallery_img[1, :, :], (3, 3, 3, 3), "constant", value=5)
            else:
                # False, with red boundary
                gallery_img_with_boundary[0, :, :] = torch.nn.functional.pad(
                    gallery_img[0, :, :], (3, 3, 3, 3), "constant", value=5)

            cur_rank_list.append(gallery_img_with_boundary)
            idx += 1
            num += 1

        cur_rank_list_img = torch.cat(cur_rank_list, dim=2)
        cur_rank_list_img = torch.nn.functional.pad(
            cur_rank_list_img, (1, 1, 0, 0), "constant", value=0)

        rank_lists_imgs.append(cur_rank_list_img.numpy().transpose((1, 2, 0)))

    fig = np.concatenate(rank_lists_imgs, axis=0)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    fig = fig * std + mean
    fig = np.clip(fig, 0, 1)

    return fig


# ---------------------- Logger ----------------------
class Logger(logging.Logger):
    '''Inherit from logging.Logger.
    Print logs to console and file.
    Add functions to draw the training log curve.'''

    def __init__(self, dir_path):
        self.dir_path = dir_path
        os.makedirs(self.dir_path, exist_ok=True)

        super(Logger, self).__init__('Training logger')

        # Print logs to console and file
        file_handler = logging.FileHandler(
            os.path.join(self.dir_path, 'train_log.txt'))
        console_handler = logging.StreamHandler()
        log_format = logging.Formatter(
            "%(asctime)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        self.addHandler(file_handler)
        self.addHandler(console_handler)

        # Draw curve
        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(121, title="Training loss")
        self.ax1 = self.fig.add_subplot(122, title="Testing CMC/mAP")
        self.x_epoch_loss = []
        self.x_epoch_test = []
        self.y_train_loss = []
        self.y_test = {}
        self.y_test['top1'] = []
        self.y_test['mAP'] = []

    def save_curve(self):
        self.ax0.plot(self.x_epoch_loss, self.y_train_loss,
                      'bs-', markersize='2', label='test')
        self.ax0.set_ylabel('Training')
        self.ax0.set_xlabel('Epoch')
        self.ax0.legend()

        self.ax1.plot(self.x_epoch_test, self.y_test['top1'],
                      'rs-', markersize='2', label='top1')
        self.ax1.plot(self.x_epoch_test, self.y_test['mAP'],
                      'bs-', markersize='2', label='mAP')
        self.ax1.set_ylabel('%')
        self.ax1.set_xlabel('Epoch')
        self.ax1.legend()

        save_path = os.path.join(self.dir_path, 'train_log.jpg')
        self.fig.savefig(save_path)

    def save_img(self, fig):
        plt.imsave(os.path.join(self.dir_path, 'rank_list.jpg'), fig)
