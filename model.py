# -*- coding: utf-8 -*-
# @Time : 2024/12/7 22:38
# @Author : nanji
# @Site : 
# @File : model.py
# @Software: PyCharm 
# @Comment :
import torch
from torchvision.models.resnet import resnet50
from torch import nn
from torch.nn import functional as F


class FaceNetModel(nn.Module):
    def __init__(self, emd_size=256, class_nums=1000):
        super().__init__()
        self.emd_size = emd_size
        self.resnet = resnet50()
        self.class_nums = class_nums
        self.faceNet = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,

            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            nn.Flatten()
        )

        self.fc = nn.Linear(32768, self.emd_size)
        self.l2_norm = F.normalize
        self.fc_class = nn.Linear(emd_size, self.class_nums)

    def forward(self, x):
        x = self.faceNet(x)
        x = self.fc(x)
        x = self.l2_norm(x) * 10
        return x

    def forward_class(self, x):
        x = self.forward(x)
        x = self.fc_class(x)
        return x


if __name__ == '__main__':
    model = FaceNetModel()
    model(torch.ones(2, 3, 128, 128))
