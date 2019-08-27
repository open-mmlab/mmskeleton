from collections import OrderedDict
import torch
from mmskeleton.utils import call_obj, import_obj
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict
from mmcv import Config
from mmcv.parallel import MMDataParallel


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def train(
    work_dir,
    model_cfg,
    dataset_cfg,
    optimizer_cfg,
    batch_size,
    lr_config,
    total_epochs,
    optimizer_config=None,
    gpus=1,
    checkpoint_config=None,
    log_config=None,
    log_level=0,
):

    # prepare data loaders
    dataset = call_obj(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True)
    data_loaders = [data_loader]

    # put model on gpus
    model = call_obj(**model_cfg)
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()

    # build runner
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level)
    runner.register_training_hooks(lr_config, optimizer_config,
                                   checkpoint_config, log_config)

    # run
    runner.run(data_loaders, [('train', 1)], total_epochs)


def batch_processor(model, data, train_mode):
    data, label = data
    losses = model(data)
    print(losses)
    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs
