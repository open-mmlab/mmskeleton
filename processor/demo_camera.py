#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time

import numpy as np
import torch
import skvideo.io

from .io import IO
import tools
import tools.utils as utils

import cv2

class naive_pose_tracker():
    def __init__(self, data_frame=64, num_joint=18):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.trace_info = list()
        self.current_frame = 0

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.current_frame:
            return 

        if len(multi_pose.shape) != 3:
            return
        

        self.current_frame = current_frame
        
        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)

        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)

                if current_frame <= latest_frame:
                    continue

                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis
            
            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]
                new_trace = self.cat_pose(trace, p, pad=current_frame-latest_frame-1)
                self.trace_info[matching_trace] = (new_trace, current_frame)
            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))
    
    def get_skeleton_sequence(self):
        
        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.current_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None
        
        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.current_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace               
    def cat_pose(self, trace, pose, pad=0):
        # trace.shape: (num_frame, num_joint, 3)
        if pad != 0:
            num_joint = trace.shape[1]
            trace = np.concatenate((trace, np.zeros((pad, num_joint, 3))), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    
    # calculate the distance between a existing trace and the input pose
    def get_dis(self, trace, pose, thereshold=100):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * thereshold
        return mean_dis, is_close

        

class DemoCamera(IO):
    """
        Demo for Skeleton-based Action Recognition
    """
    def start(self, fps=30):

        # load openpose python api
        if self.arg.openpose is not None:
            # sys.path.append('{}/python'.format(self.arg.openpose))
            sys.path.append('{}/build/python'.format(self.arg.openpose))
        from openpose import pyopenpose as op 

        video_name = self.arg.video.split('/')[-1].split('.')[0]
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
    
        # load pose model
        opWrapper = op.WrapperPython()
        params = dict(model_folder='./models', model_pose='COCO')
        opWrapper.configure(params)
        opWrapper.start()

        self.model.eval()
        # video_capture = cv2.VideoCapture(0)
        video_capture = cv2.VideoCapture(self.arg.video)
        realtime = False
        pose_tracker = naive_pose_tracker()
        start_time = time.time()
        frame_index = 0
        while(True):
            
            tic = time.time()
            # pose estimation
            ret, oriImg = video_capture.read()
            oriImg = np.rot90(oriImg)
            H, W, _ = oriImg.shape
            oriImg = cv2.resize(oriImg, (256 * W // H , 256))
            H, W, _ = oriImg.shape

            print(H, W)
            datum = op.Datum()
            datum.cvInputData = oriImg
            opWrapper.emplaceAndPop([datum])
            multi_pose = datum.poseKeypoints # shape = (num_person, num_joint, 3)
            if len(multi_pose.shape) != 3:
                continue 

            # normalization
            multi_pose[:, :, 0] = multi_pose[:, :, 0]/W
            multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:,:,0][multi_pose[:,:,2] == 0] = 0
            multi_pose[:,:,1][multi_pose[:,:,2] == 0] = 0

            # pose tracking
            if realtime:
                frame_index = int((time.time() - start_time)*fps)
            else:
                frame_index += 1
            pose_tracker.update(multi_pose, frame_index)
            data_numpy = pose_tracker.get_skeleton_sequence()
            data = torch.from_numpy(data_numpy)
            data = data.unsqueeze(0)
            data = data.float().to(self.dev).detach()

            # forward
            output, feature = self.model.extract_feature(data)
            output = output[0]
            feature = feature[0]
            intensity = (feature*feature).sum(dim=0)**0.5
            intensity = intensity.cpu().detach().numpy()
            label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
            print('Prediction result: {}'.format(label_name[label]))
        
            if data is None:
                print(None)
            else:
                print(data.shape, time.time()-tic)


            # visualization
            print('\nVisualization...')
            label_name_sequence = [label_name[label]]
            edge = self.model.graph.edge
            images = utils.visualization.stgcn_visualize(
                data_numpy[:, [-1]], edge, intensity, [oriImg], label_name[label] , label_name_sequence, self.arg.height)
            for image in images:
                break
            image = image.astype(np.uint8)

            cv2.imshow("ST-GCN", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # cv2.imshow("ST-GCN", datum.cvOutputData)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # # pack openpose ouputs
        # video = utils.video.get_video_frames(self.arg.video)
        # height, width, _ = video[0].shape
        # video_info = utils.openpose.json_pack(
        #     output_snippets_dir, video_name, width, height)
        # if not os.path.exists(output_sequence_dir):
        #     os.makedirs(output_sequence_dir)
        # with open(output_sequence_path, 'w') as outfile:
        #     json.dump(video_info, outfile)
        # if len(video_info['data']) == 0:
        #     print('Can not find pose estimation results.')
        #     return
        # else:
        #     print('Pose estimation complete.')

        # # parse skeleton data
        # pose, _ = utils.video.video_info_parsing(video_info)
        # data = torch.from_numpy(pose)
        # data = data.unsqueeze(0)
        # data = data.float().to(self.dev).detach()

        # # extract feature
        # print('\nNetwork forwad...')
        # self.model.eval()
        # output, feature = self.model.extract_feature(data)
        # output = output[0]
        # feature = feature[0]
        # intensity = (feature*feature).sum(dim=0)**0.5
        # intensity = intensity.cpu().detach().numpy()
        # label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        # print('Prediction result: {}'.format(label_name[label]))
        # print('Done.')



        # pose = data

        # # visualization
        # print('\nVisualization...')
        # label_sequence = output.sum(dim=2).argmax(dim=0)
        # label_name_sequence = [[label_name[p] for p in l ]for l in label_sequence]
        # edge = self.model.graph.edge
        # images = utils.visualization.stgcn_visualize(
        #     pose, edge, intensity, video, label_name[label] , label_name_sequence, self.arg.height)
        # print('Done.')

        # # save video
        # print('\nSaving...')
        # if not os.path.exists(output_result_dir):
        #     os.makedirs(output_result_dir)
        # writer = skvideo.io.FFmpegWriter(output_result_path,
        #                                 outputdict={'-b': '300000000'})
        # for img in images:
        #     writer.writeFrame(img)
        # writer.close()
        # print('The Demo result has been saved in {}.'.format(output_result_path))

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--video',
            default='./resource/media/skateboarding.mp4',
            help='Path to video')
        parser.add_argument('--openpose',
            default=None,
            help='Path to openpose')
        parser.add_argument('--output_dir',
            default='./data/demo_result',
            help='Path to save results')
        parser.add_argument('--height',
            default=1080,
            type=int,
            help='Path to save results')
        parser.set_defaults(config='./config/st_gcn/kinetics-skeleton/camera.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
