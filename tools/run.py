import argparse
import os
import sys

import torch
from mmcv import Config
import mmskeleton
from mmskeleton.utils import call_obj, set_attr, get_attr


""" Configuration Structure 

argparse_cfg:
  <shortcut_name 1>:
    bind_to: <full variable path>
    help: <help message>
  <shortcut_name 2>:
    ...

processor_cfg: 
  name: <full processor path>
  ...

"""


def parse_cfg():
    parser = argparse.ArgumentParser(description='Run a processor')
    parser.add_argument('config', help='configuration file path')

    # load argument setting from configuration file
    cfg = Config.fromfile(sys.argv[1])
    for key, info in cfg.argparse_cfg.items():
        if 'bind_to' not in info:
            continue
        default = get_attr(cfg, info['bind_to'])
        kwargs = dict(default=default, type=type(default))
        kwargs.update({k:v for k,v in info.items() if k != 'bind_to'})
        parser.add_argument('--' + key, **kwargs)
    args = parser.parse_args()

    # update config from command line
    for key, info in cfg.argparse_cfg.items():
        if 'bind_to' not in info:
            continue
        value = getattr(args, key)
        set_attr(cfg, info['bind_to'], value)
    
    return cfg

def main():
    cfg = parse_cfg()
    print(cfg)
    call_obj(**cfg.processor_cfg)

if __name__ == "__main__":
    main()
    