import argparse
import os

import torch
from mmcv import Config
import mmskeleton
from mmskeleton.utils import call_obj

def parse_args():
    parser = argparse.ArgumentParser(description='Run a processor')
    parser.add_argument('config', help='processor config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    call_obj(cfg.processor, cfg.processor_args)

if __name__ == "__main__":
    main()
    