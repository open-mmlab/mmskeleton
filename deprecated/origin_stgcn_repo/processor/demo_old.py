#!/usr/bin/env python
import os
import argparse
import json
import shutil

import numpy as np
import torch
import skvideo.io

from .io import IO
import tools
import tools.utils as utils

class Demo(IO):
    """
        Demo for Skeleton-based Action Recgnition
    """
    def start(self):

        openpose = '{}/examples/openpose/openpose.bin'.format(self.arg.openpose)
        video_name = self.arg.video.split('/')[-1].split('.')[0]
        output_snippets_dir = './data/openpose_estimation/snippets/{}'.format(video_name)
        output_sequence_dir = './data/openpose_estimation/data'
        output_sequence_path = '{}/{}.json'.format(output_sequence_dir, video_name)
        output_result_dir = self.arg.output_dir
        output_result_path = '{}/{}.mp4'.format(output_result_dir, video_name)
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
    
        # pose estimation
        openpose_args = dict(
            video=self.arg.video,
            write_json=output_snippets_dir,
            display=0,
            render_pose=0, 
            model_pose='COCO')
        command_line = openpose + ' '
        command_line += ' '.join(['--{} {}'.format(k, v) for k, v in openpose_args.items()])
        shutil.rmtree(output_snippets_dir, ignore_errors=True)
        os.makedirs(output_snippets_dir)
        os.system(command_line)

        # pack openpose ouputs
        video = utils.video.get_video_frames(self.arg.video)
        height, width, _ = video[0].shape
        video_info = utils.openpose.json_pack(
            output_snippets_dir, video_name, width, height)
        if not os.path.exists(output_sequence_dir):
            os.makedirs(output_sequence_dir)
        with open(output_sequence_path, 'w') as outfile:
            json.dump(video_info, outfile)
        if len(video_info['data']) == 0:
            print('Can not find pose estimation results.')
            return
        else:
            print('Pose estimation complete.')

        # parse skeleton data
        pose, _ = utils.video.video_info_parsing(video_info)
        data = torch.from_numpy(pose)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()

        # extract feature
        print('\nNetwork forwad...')
        self.model.eval()
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()
        label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        print('Prediction result: {}'.format(label_name[label]))
        print('Done.')

        # visualization
        print('\nVisualization...')
        label_sequence = output.sum(dim=2).argmax(dim=0)
        label_name_sequence = [[label_name[p] for p in l ]for l in label_sequence]
        edge = self.model.graph.edge
        images = utils.visualization.stgcn_visualize(
            pose, edge, intensity, video,label_name[label] , label_name_sequence, self.arg.height)
        print('Done.')

        # save video
        print('\nSaving...')
        if not os.path.exists(output_result_dir):
            os.makedirs(output_result_dir)
        writer = skvideo.io.FFmpegWriter(output_result_path,
                                        outputdict={'-b': '300000000'})
        for img in images:
            writer.writeFrame(img)
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))

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
            default='3dparty/openpose/build',
            help='Path to openpose')
        parser.add_argument('--output_dir',
            default='./data/demo_result',
            help='Path to save results')
        parser.add_argument('--height',
            default=1080,
            type=int)
        parser.set_defaults(config='./config/st_gcn/kinetics-skeleton/demo_old.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
