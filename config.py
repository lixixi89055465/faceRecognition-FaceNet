# -*- coding: utf-8 -*-
# @Time : 2024/12/7 23:25
# @Author : nanji
# @Site : 
# @File : config.py
# @Software: PyCharm 
# @Comment :
import json


class Config:
    def __init__(self):
        self.batch_size = None
        self.epochs = None
        self.lr = None
        self.last_lr = None
        self.alpha = None
        self.class_nums = None

    @classmethod
    def from_json_file(cls, file):
        config = Config()
        with open(file, 'r') as f:
            config.__dict__ = json.load(f)
        return config

    def __str__(self):
        return str(self.__dict__)


if __name__ == '__main__':
    config = Config.from_json_file('config.json')
    print(config)
