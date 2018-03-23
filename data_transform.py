import os
import argparse
import h5py
import scipy.io
import threading
from PIL import Image
from shutil import copyfile

from utils import *

parser = argparse.ArgumentParser(description='Transforming arguments')
parser.add_argument('--dataset', type=str, default='market1501',
                    choices=['market1501', 'cuhk03', 'duke'])
arg = parser.parse_args()


def makeDir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def transform_market_duke(src_root_path, dst_root_path):
    '''
    Download market1501 dataset from "http://www.liangzheng.org/Project/project_reid.html".
    Download duke dataset from "https://github.com/layumi/DukeMTMC-reID_evaluation"
    Please unzip the file and not change any of the directory names.
    Change the paths in utils.py to the unzipped directory path.
    '''

    def transform_to_path(images_src_path, images_dst_path, make_val=False):
        makeDir(images_dst_path)
        if make_val:
            images_val_dst_path = os.path.join(dst_root_path, 'val')
            makeDir((images_val_dst_path))

        for _, _, files in os.walk(images_src_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg':
                    continue
                label = name.split('_')
                img_src_path = os.path.join(images_src_path, name)
                img_dst_path = os.path.join(images_dst_path, label[0])
                if not os.path.isdir(img_dst_path):
                    os.mkdir(img_dst_path)

                    if make_val:
                        img_dst_path = os.path.join(
                            images_val_dst_path, label[0])
                        os.mkdir(img_dst_path)
                copyfile(img_src_path, img_dst_path + '/' + name)

    transform_to_path(os.path.join(src_root_path, 'bounding_box_test'),
                      os.path.join(dst_root_path, 'gallery'))
    transform_to_path(os.path.join(src_root_path, 'query'),
                      os.path.join(dst_root_path, 'query'))
    transform_to_path(os.path.join(src_root_path, 'bounding_box_train'),
                      os.path.join(dst_root_path, 'train_all'))
    transform_to_path(os.path.join(src_root_path, 'bounding_box_train'),
                      os.path.join(dst_root_path, 'train'), make_val=True)


def transform_cuhk03(src_root_path, dst_root_path):
    '''
    Download cuhk03 dataset from "http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html".
    Please unzip the file and not change any of the directory names.
    Then download "cuhk03_new_protocol_config_detected.mat" from "https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03"
    and put it with cuhk-03.mat. We meed this new protocol to split the dataset.
    '''

    cuhk03 = h5py.File(os.path.join(src_root_path, 'cuhk-03.mat'))
    config = scipy.io.loadmat(os.path.join(
        src_root_path, 'cuhk03_new_protocol_config_detected.mat'))

    train_idx = config['train_idx'].flatten()
    gallery_idx = config['gallery_idx'].flatten()
    query_idx = config['query_idx'].flatten()
    labels = config['labels'].flatten()
    filelist = config['filelist'].flatten()
    cam_id = config['camId'].flatten()

    imgs = cuhk03['detected'][0]
    cam_imgs = []
    for i in range(len(imgs)):
        cam_imgs.append(cuhk03[imgs[i]][:].T)

    def transform_to_path(set_name, idx, make_val=False):
        images_dst_path = os.path.join(dst_root_path, set_name)
        makeDir(images_dst_path)
        if make_val:
            images_val_dst_path = os.path.join(dst_root_path, 'val')
            makeDir((images_val_dst_path))

        for i in idx:
            i -= 1  # Start from 0
            file_name = filelist[i][0]
            cam_pair_id = int(file_name[0])
            cam_label = int(file_name[2: 5])
            cam_image_idx = int(file_name[8: 10])

            np_image = cuhk03[cam_imgs[cam_pair_id - 1]
                              [cam_label - 1][cam_image_idx - 1]][:].T

            unified_cam_id = (cam_pair_id - 1) * 2 + cam_id[i]
            img = Image.fromarray(np_image)

            id_label = str(labels[i]).zfill(4)
            img_dst_path = os.path.join(images_dst_path, id_label)

            # If the dir not exists yet, save this first image to val set
            if not os.path.isdir(img_dst_path):
                if make_val:
                    os.mkdir(img_dst_path)
                    img_dst_path = os.path.join(
                        dst_root_path, "val", id_label)

                os.mkdir(img_dst_path)

            img_name = id_label + '_' + 'c' + \
                str(unified_cam_id) + '_' + str(cam_image_idx).zfill(2)
            img.save(os.path.join(img_dst_path, img_name + '.jpg'))

    transform_to_path('train_all', train_idx)
    transform_to_path('train', train_idx, make_val=True)
    transform_to_path('gallery', gallery_idx)
    transform_to_path('query', query_idx)


if __name__ == '__main__':
    dst_root_path = DATASET_PATH[arg.dataset]
    src_root_path = os.path.split(dst_root_path)[0]
    makeDir(dst_root_path)

    if arg.dataset == 'market1501' or arg.dataset == 'duke':
        transform_market_duke(src_root_path, dst_root_path)
    if arg.dataset == 'cuhk03':
        transform_cuhk03(src_root_path, dst_root_path)
