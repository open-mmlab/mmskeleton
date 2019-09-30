import os
import torch


def visible_gpu(gpus):
    """
        set visible gpu.

        can be a single id, or a list

        return a list of new gpus ids
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, gpus)))
    return list(range(len(gpus)))


def ngpu(gpus):
    """
        count how many gpus used.
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    return len(gpus)


def occupy_gpu(gpus=None):
    """
        make program appear on nvidia-smi.
    """
    if gpus is None:
        torch.zeros(1).cuda()
    else:
        gpus = [gpus] if isinstance(gpus, int) else list(gpus)
        for g in gpus:
            torch.zeros(1).cuda(g)
