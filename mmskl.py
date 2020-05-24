import argparse
import os
import sys
import logging

import torch
import mmskeleton
from mmcv import Config
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

config_shortcut = dict(
    pose_demo_HD='./configs/pose_estimation/pose_demo_HD.yaml',
    pose_demo='./configs/pose_estimation/pose_demo.yaml')


def parse_cfg():

    parser = argparse.ArgumentParser(description='Run a processor.')
    parser.add_argument('config', help='configuration file path')

    if len(sys.argv) <= 1:
        args = parser.parse_args()
        return
    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        args = parser.parse_args()
        return

    # pop positional args
    config_args = []
    tmp = []
    for i, arg in enumerate(sys.argv):
        if i > 1:
            if arg[0] != '-':
                tmp.append(i)
            else:
                break

    for i in tmp[::-1]:
        config_args.append(sys.argv.pop(i))
    branch = config_args

    # load argument setting from configuration file
    if sys.argv[1] in config_shortcut:
        sys.argv[1] = config_shortcut[sys.argv[1]]

    print('Load configuration information from {}'.format(sys.argv[1]))
    cfg = Config.fromfile(sys.argv[1])
    for b in branch:
        if b in cfg:
            cfg = get_attr(cfg, b)
        else:
            print('The branch "{}" can not be found in {}.'.format(
                '-'.join(branch), sys.argv[1]))
            return dict()
    if 'description' in cfg:
        parser.description = cfg.description
    if 'argparse_cfg' not in cfg:
        cfg.argparse_cfg = dict()
    for key, info in cfg.argparse_cfg.items():
        if 'bind_to' not in info:
            continue
        default = get_attr(cfg, info['bind_to'])
        if 'type' not in info:
            if default is not None:
                info['type'] = type(default)
        else:
            info['type'] = eval(info['type'])
        kwargs = dict(default=default)
        kwargs.update({k: v for k, v in info.items() if k != 'bind_to'})
        parser.add_argument('--' + key, **kwargs)
    args = parser.parse_args()

    # update config from command line
    for key, info in cfg.argparse_cfg.items():
        if 'bind_to' not in info:
            continue
        value = getattr(args, key)
        set_attr(cfg, info['bind_to'], value)

    # replace pre_defined arguments in configuration files
    def replace(cfg, **format_args):
        if isinstance(cfg, str):
            return cfg.format(**format_args)
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                set_attr(cfg, k, replace(v, **format_args))
        elif isinstance(cfg, list):
            for k in range(len(cfg)):
                cfg[k] = replace(cfg[k], **format_args)
        return cfg

    format_args = dict()
    format_args['config_path'] = args.config
    format_args['config_name'] = os.path.basename(format_args['config_path'])
    format_args['config_prefix'] = format_args['config_name'].split('.')[0]
    cfg = replace(cfg, **format_args)
    return cfg


def main():
    cfg = parse_cfg()
    if 'processor_cfg' in cfg:
        call_obj(**cfg.processor_cfg)
    else:
        print('No processor specified.')


if __name__ == "__main__":
    main()
