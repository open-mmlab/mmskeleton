# sys
import os
import sys
import numpy as np
import random
import pickle
import json
# torch
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# operation
from .utils import skeleton


class KineticsFeeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """
    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 num_person_in=5,
                 num_person_out=2,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)

        if self.debug:
            self.sample_name = self.sample_name[0:2]

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)

        sample_id = [name.split('.')[0] for name in self.sample_name]
        self.label = np.array(
            [label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array(
            [label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        if self.ignore_empty_sample:
            self.sample_name = [
                s for h, s in zip(has_skeleton, self.sample_name) if h
            ]
            self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  #sample
        self.C = 3  #channel
        self.T = 300  #frame
        self.V = 18  #joint
        self.M = self.num_person_out  #person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break
                pose = skeleton_info['pose']
                score = skeleton_info['score']
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        label = video_info['label_index']
        assert (self.label[index] == label)

        # data augmentation
        if self.random_shift:
            data_numpy = skeleton.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = skeleton.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = skeleton.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = skeleton.random_move(data_numpy)

        # sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose(
                (1, 2, 0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        # match poses between 2 frames
        if self.pose_matching:
            data_numpy = skeleton.openpose_match(data_numpy)

        return data_numpy, label

    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        assert (all(self.label >= 0))
        return skeleton.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        assert (all(self.label >= 0))
        return skeleton.calculate_recall_precision(self.label, score)
