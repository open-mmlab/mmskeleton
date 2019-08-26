import argparse
import os

import torch
from mmcv import Config
import mmskeleton
from mmskeleton.utils import call_obj

def parse_args():
    parser = argparse.ArgumentParser(description='Train a action recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    processor = call_obj(cfg.processor, cfg.processor_args)

if __name__ == "__main__":
    main()
    