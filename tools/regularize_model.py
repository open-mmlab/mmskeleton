# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from collections import OrderedDict


def main(model_path):
    model = torch.load(model_path)
    model = OrderedDict([[k.split('module.')[-1],
                          v.cpu()] for k, v in model.items()])
    torch.save(model, model_path)


if __name__ == '__main__':
    model = [
        'kinetics-st_gcn.pt', 'kinetics-tcn.pt', 'ntuxsub-st_gcn.pt',
        'ntuxsub-tcn.pt', 'ntuxview-st_gcn.pt', 'ntuxview-tcn.pt'
    ]
    for m in model:
        model_path = './model/{}'.format(m)
        main(model_path)
