# sys
import os
import sys
import cvbase
import numpy as np
import random

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

# visualization
import matplotlib.pyplot as plt
import time


class Feeder(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 mode,
                 random_choose=False,
                 random_shift=False,
                 window_size=-1,
                 temporal_downsample_step=1,
                 mean_subtraction=0,
                 num_sample=-1,
                 debug=False):
        self.debug = debug
        self.mode = mode
        self.data_path = data_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.window_size = window_size
        self.mean_subtraction = mean_subtraction
        self.temporal_downsample_step = temporal_downsample_step
        self.num_sample = num_sample

        self.load_data()

    def load_data(self):
        # data: N C V T M
        self.label = np.load('{}/{}_label.npy'.format(
            self.data_path, self.mode))[0:self.num_sample]
        self.data = np.load('{}/{}_data.npy'.format(
            self.data_path, self.mode))[0:self.num_sample]
        try:
            self.sample_name = cvbase.pickle_load(
                '{}/{}_name.pkl'.format(self.data_path, self.mode))
        except:
            self.sample_name = [str(i) for i in range(len(self.label))]

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
                data_numpy = _downsample(data_numpy,
                                         self.temporal_downsample_step)
            else:
                data_numpy = _temporal_slice(data_numpy,
                                             self.temporal_downsample_step)
        if self.mode is 'train':
            if self.random_shift:
                data_numpy = _random_shift(data_numpy)
            if self.random_choose:
                data_numpy = _random_choose(data_numpy, self.window_size)

        # mean subtraction
        if self.mean_subtraction != 0:
            data_numpy = _mean_subtractor(data_numpy, self.mean_subtraction)

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def _downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def _temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def _random_choose(data_numpy, size):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = random.randint(0, T - size)
    return data_numpy[:, begin:begin + size, :, :]


def _mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def _random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def feeder_test(data_path):
    # load data
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, mode='train', random_shift=False),
        batch_size=64,
        shuffle=True,
        num_workers=2)
    # loader = torch.utils.data.DataLoader(
    #     dataset=Data_feeder(data_path, mode='val'),
    #     batch_size=64,
    #     shuffle=False,
    #     num_workers=128)

    for batch_idx, (data, label) in enumerate(loader):
        data = data.numpy()
        N, C, T, V, M = data.shape
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        pose, = ax.plot(np.zeros(V), np.zeros(V), 'g^')
        ax.axis([-1, 1, -1, 1])
        for n in range(N):
            for t in range(T):
                for m in range(M):
                    print(n, t, m)
                    print(data[n, 0, t, :, m])
                    pose.set_xdata(data[n, 0, t, :, m])
                    pose.set_ydata(data[n, 1, t, :, m])
                    fig.canvas.draw()
                    plt.pause(0.01)


if __name__ == '__main__':
    data_path = '../resource/NTU-RGB-D/xview'
    feeder_test(data_path)
