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
from utils import load_network


######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Testing arguments')
parser.add_argument('--which_epoch', default='final',
                    type=str, help='0,1,2,3...or final')
parser.add_argument('--test_dir', default='/home/share/hongjiang/Market-1501-v15.09.15/pytorch',
                    type=str, help='./test_data')
parser.add_argument('--batch_size', default=32, type=int, help='batchsize')

arg = parser.parse_args()

MODEL_NAME = 'PCB'


######################################################################
# Functions
# --------

def get_id(img_path):
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


def fliplr(img):
    # Create inverse index at dim 3(width)
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)  # Inverse the width dim
    return img_flip


def extract_feature(model, dataloaders):

    model.eval()

    # features = torch.FloatTensor()
    features = []

    for data in dataloaders:
        img, _ = data

        if USE_GPU:
            input_img = Variable(img.cuda())
        else:
            input_img = Variable(img)

        output = model(input_img)
        # [N, C, H=S, W=1]
        feature = model.features_H.data.cpu().numpy()

        # [N, C*H]
        # feature = feature.view(feature.size(0), -1)
        feature = feature.reshape(len(feature), -1)

        # norm feature
        # fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
        # feature = feature.div(fnorm.expand_as(feature))
        # features = feature
        fnorm = np.linalg.norm(feature, axis=1)
        feature = np.divide(feature, fnorm.reshape(len(fnorm), 1))
        features.append(feature)

    features = np.vstack(features)

    return features


#######################################################################
# Evaluation
# --------

def evaluate(query_features, query_labels, query_cams, gallery_features, gallery_labels, gallery_cams):

    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    AP = 0

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
        lexsort_index = np.argsort(y_score)
        sorted_y_true = y_true[lexsort_index[::-1]]
        match_index = np.argwhere(sorted_y_true == True)

        if match_index.size > 0:
            first_match_index = match_index.flatten()[0]
            CMC[first_match_index:] += 1

    CMC = CMC.float()
    CMC = CMC / len(query_labels)  # average CMC
    mAP = AP / len(query_labels)

    return CMC, mAP


######################################################################
# Main
# ---------

data_transforms = transforms.Compose([
    transforms.Resize((384, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_datasets = {x: datasets.ImageFolder(os.path.join(arg.test_dir, x), data_transforms)
                  for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=arg.batch_size, shuffle=False, num_workers=4) for x in ['gallery', 'query']}

USE_GPU = torch.cuda.is_available()


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cams, gallery_labels = get_id(gallery_path)
query_cams, query_labels = get_id(query_path)

model_structure = PCBModel(751)
model = load_network(model_structure, MODEL_NAME, arg.which_epoch)

# Remove the final fc layer and classifier layer
for i in range(len(model.fc_list)):
    model.fc_list[i] = nn.Sequential()
# model.classifier = nn.Sequential()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Change to test mode
model = model.eval()
if USE_GPU:
    model = model.cuda()

# Extract feature
gallery_features = extract_feature(model, dataloaders['gallery'])
query_features = extract_feature(model, dataloaders['query'])


CMC, mAP = evaluate(query_features, query_labels, query_cams,
                    gallery_features, gallery_labels, gallery_cams)
print('top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], mAP))
