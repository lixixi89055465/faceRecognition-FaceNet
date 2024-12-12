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
    config.class_num = len(keys)
    new_paths = []
    for i in range(config.class_num):
        key = keys[i]
        paths = paths_dict[key]
        for path in paths:
            new_path = []
            new_path.append(path)
            rand_int = random.randint(0, len(paths) - 1)
            new_path.append(paths[rand_int])
            rand_num_man = random.randint(0, config.class_num - 1)
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
        a_img, p_img, n_img = self.read_image(a_path), self.read_image(p_path), self.read_image(n_path)
        # 提取 label
        s_l = int(re.findall(r"\\(\d+)\\", a_path)[0])
        n_l = int(re.findall(r"\\(\d+)\\", n_path)[0])
        return np.float32(a_img), np.float32(p_img), np.float32(n_img), np.int64(s_l), np.int64(n_l)

    def __len__(self):
        return len(self.paths)


def train():
    paths = read_data()
    train_data = FaceData(paths)
    train_data = Data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    # 初始化网络
    model = FaceNetModel(config.image_size, config.class_num)
    model.train()
    # optimizer = torch.optim.optimizer(model.parameters(), lr=config.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_t_fc = TripletLoss(config.alpha)
    loss_c_fc = nn.CrossEntropyLoss()
    nb = len(train_data)
    old_loss_time = None
    for epoch in range(1, config.epochs + 1):
        pbar = tqdm(train_data, total=nb)
        for step, (a_x, p_x, n_x, s_y, n_y) in enumerate(pbar):
            a_out, p_out, n_out = model(a_x), model(p_x), model(n_x)
            s_d = F.pairwise_distance(a_out, p_out)
            n_d = F.pairwise_distance(a_out, n_out)
            thing = (n_d - s_d < config.alpha).flatten()
            mask = np.where(thing.numpy() == 1)[0]
            if not len(mask):
                continue
            # 计算三元损失
            a_out, p_out, n_out = a_out[mask], p_out[mask], n_out[mask]
            loss_t = torch.mean(loss_t_fc(a_out, p_out, n_out))
            # 计算熵损失
            a_x, p_x, n_x = a_x[mask], p_x[mask], n_x[mask]
            input_x = torch.cat([a_x, p_x, n_x])
            s_y, n_y = s_y[mask], n_y[mask]
            output_y = torch.cat([s_y, s_y, n_y])

            out = model.forward_class(input_x)
            loss_c = loss_c_fc(out, output_y.unsqueeze(1))
            loss = loss_t + loss_c
            if old_loss_time is None:
                old_loss_time = loss
                loss_time = loss
            else:
                old_loss_time += loss
                loss_time = old_loss_time / (step + 1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            s = ("train ===> epoch:{}"
                 "----- step:{}"
                 "---------loss_t {:.4f}"
                 "----- loss_c:{:.4f}-------loss:{:.4f}"
                 "------{:.4f}".
                 format(epoch, step, loss_t, loss_c, loss, loss_time))
            pbar.set_description(s)


class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.pairwise_distance = nn.PairwiseDistance()

    def forward(self, a_x, p_x, n_x):
        s_d = self.pairwise_distance(a_x, p_x)
        n_d = self.pairwise_distance(a_x, n_x)
        return torch.clamp(s_d - n_d + self.alpha, min=0.02)


if __name__ == '__main__':
    # read_data()
    train()
