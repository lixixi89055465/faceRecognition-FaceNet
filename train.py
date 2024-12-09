# -*- coding: utf-8 -*-
# @Time : 2024/12/7 23:45
# @Author : nanji
# @Site : https://www.bilibili.com/video/BV1gQ4y1X7AK/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=50305204d8a1be81f31d861b12d4d5cf
# @File : train.py
# @Software: PyCharm 
# @Comment :
import torch
from torch import nn
from torch.utils import data as Data
import cv2
from glob import glob
from tqdm import tqdm
import re
from config import Config
import numpy as np
import random
from model import FaceNetModel

from torch.nn import functional as F

config = Config.from_json_file('config.json')


# 读取数据
def read_data():
    paths = glob('./data/CASIA-WebFace/*/*')
    paths_dict = dict()
    for path in paths:
        man_num = re.findall(r"(\d+)\\(\d+.jpg)", path)[0]
        if man_num[0] not in paths_dict:
            paths_dict[man_num[0]] = [path]
        else:
            paths_dict[man_num[0]].append(path)
    keys = list(paths_dict.keys())
    class_num = len(keys)
    new_paths = []
    for i in range(class_num):
        key = keys[i]
        paths = paths_dict[key]
        for path in paths:
            new_path = []
            new_path.append(path)
            rand_int = random.randint(0, len(paths) - 1)
            new_path.append(paths[rand_int])
            rand_num_man = random.randint(0, class_num - 1)
            if rand_num_man == i:
                try:
                    rand_num_man += 1
                    n_keys = keys[rand_num_man]
                except:
                    rand_num_man -= 1
                    n_keys = keys[rand_num_man]
            else:
                n_keys = keys[rand_num_man]
            n_paths = paths_dict[n_keys]
            n_num = random.randint(0, len(n_paths) - 1)
            new_path.append(n_paths[n_num])
            new_paths.append(new_path)
    return new_paths


# 构建自定义数据集
class FaceData(Data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def to_tuple(self, x):
        return x, x

    def read_image(self, path):
        image = cv2.imread(path)
        image = cv2.resize(image, (config.image_size, config.image_size)) / 127.5 - 1.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, item):
        a_path, p_path, n_path = self.paths[item]
        a_img, p_img, n_img = self.read_image(a_path)
        # 提取 label
        s_l = int(re.findall(r"(\d+)\\(\d+).jpg", a_img)[0])
        n_l = int(re.findall(r"(\d+)\\(\d+).jpg", p_img)[0])
        return a_img, p_img, n_img, s_l, n_path
        return self.paths[item]

    def __len__(self):
        return len(self.paths)


def train():
    paths = read_data()
    train_data = FaceData(paths)
    train_data = Data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    # 初始化网络
    model = FaceNetModel(config.image_size, config.class_nums)
    model.train()
    optimizer = torch.optim.optimizer(model.parameters(), lr=config.lr)
    loss_t = TripletLoss(config.alpha)
    loss_c = nn.CrossEntropyLoss()
    nb = len(train_data)
    for epoch in range(1, config.epochs + 1):
        pbar = tqdm(train_data, total=nb)
        for step, (a_x, p_x, n_x, s_y, n_y) in enumerate(pbar):
            a_out, p_out, n_out = model(a_x), model(p_x), model(n_x)


class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.pairwize_distance = nn.PairwiseDistance()

    def forward(self, a_x, p_x, n_x):
        s_d = self.pairwize_distance(a_x, p_x)
        n_d = self.pairwise_distance(a_x, n_x)
        return torch.clamp(s_d - n_d + self.alpha, min=0.02)


if __name__ == '__main__':
    read_data()
