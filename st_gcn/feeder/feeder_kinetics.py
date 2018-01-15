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
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

# visualization
import matplotlib.pyplot as plt
import time

# operation
import tools


class Feeder_kinetics(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 mode,
                 ignore_empty_sample=True,
                 random_choose=False,
                 window_size=-1,
                 random_shift=False,
                 random_move=False,
                 pose_matching=False,
                 num_person=1,
                 num_match_trace=2,
                 temporal_downsample_step=1,
                 num_sample=-1,
                 debug=False):
        self.debug = debug
        self.mode = mode
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.temporal_downsample_step = temporal_downsample_step
        self.num_sample = num_sample
        self.num_person=num_person
        self.num_match_trace = num_match_trace
        self.pose_matching=pose_matching
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
        self.label = np.array([label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array([label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence 
        if self.ignore_empty_sample:
            self.sample_name = [s for h, s in zip(has_skeleton, self.sample_name) if h]
            self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  #sample
        self.C = 3                      #channel
        self.T = 300                    #frame
        self.V = 18                     #joint
        self.M = self.num_person        #person
         
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
        data_numpy = np.zeros((self.C, self.T, self.V, self.M)) 
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m>=self.M:
                    break
                pose = skeleton_info['pose']
                score = skeleton_info['score']
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score
        
        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0]=0
        data_numpy[1][data_numpy[2] == 0]=0

        # get & check label index
        label = video_info['label_index']
        assert(self.label[index] == label)
                
        # processing
        if self.temporal_downsample_step != 1:
            if self.mode is 'train':
                data_numpy = tools.downsample(data_numpy,
                                         self.temporal_downsample_step)
            else:
                data_numpy = tools.temporal_slice(data_numpy,
                                             self.temporal_downsample_step)
        # data augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size>0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # match poses between 2 frames
        if self.pose_matching:
            C, T, V, M = data_numpy.shape
            forward_map = np.zeros((T, M),dtype=int) - 1
            backward_map = np.zeros((T, M),dtype=int) - 1

            # match pose
            for t in range(T-1):
                for m in range(M):
                    s = (data_numpy[2,t,:,m].reshape(1,V,1) != 0 ) * 1
                    if s.sum()==0:
                        continue
                    res = data_numpy[:,t+1,:,:] - data_numpy[:,t,:,m].reshape(C,V,1)
                    n = (res*res*s).sum(axis=1).sum(axis=0).argsort()[0]#next pose
                    forward_map[t, m] = n 
                    backward_map[t+1, n] = m 

            # find start point
            start_point = []
            for t in range(T):
                for m in range(M):
                    if backward_map[t,m] == -1:
                        start_point.append((t,m))

            # generate path
            path_list = []
            c=0
            for i in range(len(start_point)):
                path = [start_point[i]]
                while(1):
                    t,m = path[-1] 
                    n = forward_map[t, m]
                    if n == -1:
                        break
                    else:
                        path.append((t+1, n))
                    #print(c,t)
                    c=c+1
                path_list.append(path)
            
            # generate data
            new_M = self.num_match_trace 
            path_length = [len(p) for p in path_list] 
            sort_index = np.array(path_length).argsort()[::-1][:new_M]
            if self.mode == 'train':
                np.random.shuffle(sort_index)
                sort_index = sort_index[:M]
            	new_data_numpy = np.zeros((C, T, V, M))
            else:
            	new_data_numpy = np.zeros((C, T, V, new_M))
            for i, p in enumerate(sort_index):
                path = path_list[p] 
                for t,m in path:
                    new_data_numpy[:,t,:,i] = data_numpy[:,t,:,m]

            data_numpy = new_data_numpy 

        return data_numpy, label

    def top_k(self, score, top_k):
        assert(all(self.label>=0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def test(data_path, label_path, vid=None):
    loader = torch.utils.data.DataLoader(
        dataset=Feeder_kinetics(data_path, label_path, mode='val',pose_matching=False, max_person=10),
        batch_size=64,
        shuffle=False,
        num_workers=2)
    
    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [ name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label = loader.dataset[index]
        data=data.reshape((1,)+data.shape)

    # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        pose, = ax.plot(np.zeros(V*M), np.zeros(V*M), 'g^')
        ax.axis([-1, 1, -1, 1])

        for n in range(N):
            for t in range(T):
                pose.set_xdata(data[n, 0, t, :, 0])
                pose.set_ydata(data[n, 1, t, :, 0])
                fig.canvas.draw()
                plt.pause(0.01)

if __name__ == '__main__':
    data_path = '../../data/Kinetics/kinetics_val'
    label_path = '/mnt/SSD/Kinetics/dataset/kinetics_val_label.json'

    test(data_path, label_path, vid='iqkx0rrCUCo')
