from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PCBModel(nn.Module):
    def __init__(self, num_classes, num_stripes=6, local_conv_out_channels=256):

        super(PCBModel, self).__init__()
        self.num_stripes = num_stripes
        self.num_classes = num_classes
        self.features_G = []
        self.features_H = []

        resnet = models.resnet50(pretrained=True)
        # Modifiy the stride of last conv layer
        resnet.layer4[0].conv2 = nn.Conv2d(
            512, 512, kernel_size=3, bias=False, stride=1, padding=1)
        resnet.layer4[0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048))

        # Remove avgpool and fc layer of resnet
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # Add new layers
        self.avgpool = nn.AdaptiveAvgPool2d((self.num_stripes, 1))
        self.local_conv = nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True))

        # Classifier for each stripe
        self.fc_list = nn.ModuleList()
        for _ in range(num_stripes):
            fc = nn.Sequential(
                nn.Linear(local_conv_out_channels, num_classes))
            # nn.Softmax())

            # fc initialize
            nn.init.normal(fc[0].weight, std=0.001)
            nn.init.constant(fc[0].bias, 0)

            self.fc_list.append(fc)

    def forward(self, x):
        features = self.backbone(x)

        # [N, C, H, W]
        assert features.size(
            2) % self.num_stripes == 0, 'Image height cannot be divided by num_strides'

        self.features_G = self.avgpool(features)

        # [N, C=256, H=S, W=1]
        self.features_H = self.local_conv(self.features_G)

        # H=S * [N, num_classes]
        predicitions_list = []
        for i in range(self.num_stripes):
            local_feature = self.features_H[:, :, i, :].contiguous()
            local_feature = local_feature.view(local_feature.size(0), -1)
            predicitions_list.append(self.fc_list[i](local_feature))

        return predicitions_list
