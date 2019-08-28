import argparse
import os
import sys

import torch
from mmcv import Config
import mmskeleton
from mmskeleton.utils import call_obj


""" Configuration Formation 

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

    # add argument from configuration file
    cfg = Config.fromfile(sys.argv[1])
    for key, info in cfg.argparse_cfg.items():
        if 'bind_to' not in info:
            continue
        default = eval('cfg.{}'.format(info['bind_to']))
        parser.add_argument('--' + key, default=default, help=info.get('help', None))
    args = parser.parse_args()

    # update config from command line
    for key, info in cfg.argparse_cfg.items():
        if 'bind_to' not in info:
            continue
        exec("cfg.{} = '{}'".format(info['bind_to'], getattr(args, key)))
    
    return cfg

def main():
    cfg = parse_cfg()
    call_obj(**cfg.processor_cfg)

if __name__ == "__main__":
    main()
    