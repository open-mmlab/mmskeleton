# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

# visualization
import time

# operation
from tools import * 

class Feeder(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 mode,
                 random_choose=False,
                 random_shift=False,
                 window_size=-1,
                 temporal_downsample_step=1,
                 mean_subtraction=0,
                 debug=False):
        self.debug = debug
        self.mode = mode
        self.data_path = data_path
        self.label_path= label_path 
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.window_size = window_size
        self.mean_subtraction = mean_subtraction
        self.temporal_downsample_step = temporal_downsample_step

        self.load_data()

    def load_data(self):
        # data: N C V T M
        with open(self.label_path, 'r') as f:
            self.sample_name, self.label = pickle.load(f)
        self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        data_numpy = self.data[index]
        label = self.label[index]

        # processing
        if self.temporal_downsample_step != 1:
            if self.mode is 'train':
                data_numpy = downsample(data_numpy,
                                         self.temporal_downsample_step)
            else:
                data_numpy = temporal_slice(data_numpy,
                                             self.temporal_downsample_step)
        if self.mode is 'train':
            if self.random_shift:
                data_numpy = random_shift(data_numpy)
            if self.random_choose:
                data_numpy = random_choose(data_numpy, self.window_size)

        # mean subtraction
        if self.mean_subtraction != 0:
            data_numpy = mean_subtractor(data_numpy, self.mean_subtraction)

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
