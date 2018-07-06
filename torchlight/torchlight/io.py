#!/usr/bin/env python
import argparse
import os
import sys
import traceback
import time
import warnings
import pickle
from collections import OrderedDict
import yaml
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

class IO():
    def __init__(self, work_dir, save_log=True, print_log=True):
        self.work_dir = work_dir
        self.save_log = save_log
        self.print_to_screen = print_log
        self.cur_time = time.time()
        self.split_timer = {}
        self.pavi_logger = None
        self.session_file = None
        self.model_text = ''
        
    # PaviLogger is removed in this version
    def log(self, *args, **kwargs):
        pass
    #     try:
    #         if self.pavi_logger is None:
    #             from torchpack.runner.hooks import PaviLogger
    #             url = 'http://pavi.parrotsdnn.org/log'
    #             with open(self.session_file, 'r') as f:
    #                 info = dict(
    #                     session_file=self.session_file,
    #                     session_text=f.read(),
    #                     model_text=self.model_text)
    #             self.pavi_logger = PaviLogger(url)
    #             self.pavi_logger.connect(self.work_dir, info=info)
    #         self.pavi_logger.log(*args, **kwargs)
    #     except:  #pylint: disable=W0702
    #         pass

    def load_model(self, model, **model_args):
        Model = import_class(model)
        model = Model(**model_args)
        self.model_text += '\n\n' + str(model)
        return model

    def load_weights(self, model, weights_path, ignore_weights=None):
        if ignore_weights is None:
            ignore_weights = []
        if isinstance(ignore_weights, str):
            ignore_weights = [ignore_weights]

        self.print_log('Load weights from {}.'.format(weights_path))
        weights = torch.load(weights_path)
        weights = OrderedDict([[k.split('module.')[-1],
                                v.cpu()] for k, v in weights.items()])

        # filter weights
        for i in ignore_weights:
            ignore_name = list()
            for w in weights:
                if w.find(i) == 0:
                    ignore_name.append(w)
            for n in ignore_name:
                weights.pop(n)
                self.print_log('Filter [{}] remove weights [{}].'.format(i,n))

        for w in weights:
            self.print_log('Load weights [{}].'.format(w))

        try:
            model.load_state_dict(weights)
        except (KeyError, RuntimeError):
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            for d in diff:
                self.print_log('Can not find weights [{}].'.format(d))
            state.update(weights)
            model.load_state_dict(state)
        return model

    def save_pkl(self, result, filename):
        with open('{}/{}'.format(self.work_dir, filename), 'wb') as f:
            pickle.dump(result, f)

    def save_h5(self, result, filename):
        with h5py.File('{}/{}'.format(self.work_dir, filename), 'w') as f:
            for k in result.keys():
                f[k] = result[k]

    def save_model(self, model, name):
        model_path = '{}/{}'.format(self.work_dir, name)
        state_dict = model.state_dict()
        weights = OrderedDict([[''.join(k.split('module.')),
                                v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, model_path)
        self.print_log('The model has been saved as {}.'.format(model_path))

    def save_arg(self, arg):

        self.session_file = '{}/config.yaml'.format(self.work_dir)

        # save arg
        arg_dict = vars(arg)
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        with open(self.session_file, 'w') as f:
            f.write('# command line: {}\n\n'.format(' '.join(sys.argv)))
            yaml.dump(arg_dict, f, default_flow_style=False, indent=4)

    def print_log(self, str, print_time=True):
        if print_time:
            # localtime = time.asctime(time.localtime(time.time()))
            str = time.strftime("[%m.%d.%y|%X] ", time.localtime()) + str

        if self.print_to_screen:
            print(str)
        if self.save_log:
            with open('{}/log.txt'.format(self.work_dir), 'a') as f:
                print(str, file=f)

    def init_timer(self, *name):
        self.record_time()
        self.split_timer = {k: 0.0000001 for k in name}

    def check_time(self, name):
        self.split_timer[name] += self.split_time()

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_timer(self):
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(self.split_timer.values()))))
            for k, v in self.split_timer.items()
        }
        self.print_log('Time consumption:')
        for k in proportion:
            self.print_log(
                '\t[{}][{}]: {:.4f}'.format(k, proportion[k],self.split_timer[k])
                )


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2dict(v):
    return eval('dict({})'.format(v))  #pylint: disable=W0123


def _import_class_0(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval('dict({})'.format(values))  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)
