# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.metrics import average_precision_score

from model import PCBModel
import utils


# ---------------------- Extract features ----------------------
def get_cam_label(img_path):
    camera_ids = []
    labels = []
    for path, _ in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_ids.append(int(camera[0]))
    return np.array(camera_ids), np.array(labels)


# Filp lr and average the features. Use in the future?
def fliplr(img):
    # Create inverse index at dim 3(width)
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)  # Inverse the width dim
    return img_flip


def extract_feature(model, inputs, requires_norm, vectorize, requires_grad=False):

    # Move to model's device
    inputs = inputs.to(next(model.parameters()).device)

    with torch.set_grad_enabled(requires_grad):
        features = model(inputs)

    size = features.shape

    if requires_norm:
        # [N, C*H]
        features = features.view(size[0], -1)

        # norm feature
        fnorm = features.norm(p=2, dim=1)
        features = features.div(fnorm.unsqueeze(dim=1))

    if vectorize:
        features = features.view(size[0], -1)
    else:
        # Back to [N, C, H=S]
        features = features.view(size)

    return features


# ---------------------- Evaluation ----------------------
def evaluate(query_features, query_labels, query_cams, gallery_features, gallery_labels, gallery_cams):
    """Evaluate the CMC and mAP

    Arguments:
        query_features {np.ndarray of size NxC} -- Features of probe images
        query_labels {np.ndarray of query size N} -- Labels of probe images
        query_cams {np.ndarray of query size N} -- Cameras of probe images
        gallery_features {np.ndarray of size N'xC} -- Features of gallery images
        gallery_labels {np.ndarray of gallery size N'} -- Lables of gallery images
        gallery_cams {np.ndarray of gallery size N'} -- Cameras of gallery images

    Returns:
        (torch.IntTensor, float) -- CMC list, mAP
    """

    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    AP = 0
    sorted_index_list, sorted_y_true_list, junk_index_list = [], [], []

    for i in range(len(query_labels)):
        query_feature = query_features[i]
        query_label = query_labels[i]
        query_cam = query_cams[i]

        # Prediction score
        score = np.dot(gallery_features, query_feature)

        match_query_index = np.argwhere(gallery_labels == query_label)
        same_camera_index = np.argwhere(gallery_cams == query_cam)

        # Positive index is the matched indexs at different camera i.e. the desired result
        positive_index = np.setdiff1d(
            match_query_index, same_camera_index, assume_unique=True)

        # Junk index is the indexs at the same camera or the unlabeled image
        junk_index = np.append(
            np.argwhere(gallery_labels == -1),
            np.intersect1d(match_query_index, same_camera_index))  # .flatten()

        index = np.arange(len(gallery_labels))
        # Remove all the junk indexs
        sufficient_index = np.setdiff1d(index, junk_index)

        # compute AP
        y_true = np.in1d(sufficient_index, positive_index)
        y_score = score[sufficient_index]
        AP += average_precision_score(y_true, y_score)

        # Compute CMC
        # Sort the sufficient index by their scores, from large to small
        sorted_index = np.argsort(y_score)[::-1]
        sorted_y_true = y_true[sorted_index]
        match_index = np.argwhere(sorted_y_true == True)

        if match_index.size > 0:
            first_match_index = match_index.flatten()[0]
            CMC[first_match_index:] += 1

        # keep with junk index, for using the index to show the img from dataloader
        all_sorted_index = np.argsort(score)[::-1]
        all_y_true = np.in1d(index, match_query_index)
        all_sorted_y_true = all_y_true[all_sorted_index]

        sorted_index_list.append(all_sorted_index)
        sorted_y_true_list.append(all_sorted_y_true)
        junk_index_list.append(junk_index)

    CMC = CMC.float()
    CMC = CMC / len(query_labels) * 100  # average CMC
    mAP = AP / len(query_labels) * 100

    return CMC, mAP, (sorted_index_list, sorted_y_true_list, junk_index_list)


# ---------------------- Start testing ----------------------
def test(model, dataset, batch_size):
    model.eval()

    gallery_dataloader = utils.getDataLoader(
        dataset, batch_size, 'gallery', shuffle=False, augment=False)
    query_dataloader = utils.getDataLoader(
        dataset, batch_size, 'query', shuffle=False, augment=False)

    gallery_cams, gallery_labels = get_cam_label(
        gallery_dataloader.dataset.imgs)
    query_cams, query_labels = get_cam_label(query_dataloader.dataset.imgs)

    # Extract feature
    gallery_features = []
    query_features = []

    for inputs, _ in gallery_dataloader:
        gallery_features.append(extract_feature(
            model, inputs, requires_norm=True, vectorize=True).cpu().data)
    gallery_features = torch.cat(gallery_features, dim=0).numpy()

    for inputs, _ in query_dataloader:
        query_features.append(extract_feature(
            model, inputs, requires_norm=True, vectorize=True).cpu().data)
    query_features = torch.cat(query_features, dim=0).numpy()

    CMC, mAP, (sorted_index_list, sorted_y_true_list, junk_index_list) = evaluate(
        query_features, query_labels, query_cams, gallery_features, gallery_labels, gallery_cams)

    rank_list_fig = utils.save_rank_list_img(
        query_dataloader, gallery_dataloader, sorted_index_list, sorted_y_true_list, junk_index_list)

    return CMC, mAP, rank_list_fig


# ---------------------- Testing settings ----------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing arguments')
    parser.add_argument('--save_path', type=str, default='./model')
    parser.add_argument('--which_epoch', default='final',
                        type=str, help='0,1,2,3...or final')
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke'])
    parser.add_argument('--batch_size', default=512,
                        type=int, help='batchsize')
    parser.add_argument('--share_conv', action='store_true')
    arg = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir_path = os.path.join(arg.save_path, arg.dataset)
    logger = utils.Logger(save_dir_path)
    logger.info(vars(arg))

    # Just to get the correct num_classes of classifier fc
    train_dataloader = utils.getDataLoader(
        arg.dataset, arg.batch_size, 'train')

    model = utils.load_network(PCBModel(num_classes=len(train_dataloader.dataset.classes),
                                        share_conv=arg.share_conv,
                                        return_features=True),
                               save_dir_path, arg.which_epoch)
    model = model.to(device)
    CMC, mAP, rank_list_fig = test(model, arg.dataset, arg.batch_size)

    logger.info('Testing: top1:%.2f top5:%.2f top10:%.2f mAP:%.2f' %
                (CMC[0], CMC[4], CMC[9], mAP))
    logger.save_img(rank_list_fig)

    torch.cuda.empty_cache()
