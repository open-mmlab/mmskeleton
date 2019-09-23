import torch
import numpy as np
from collections import OrderedDict
from mmskeleton.utils import call_obj, import_obj, load_checkpoint, get_mmskeleton_url
from mmcv import ProgressBar
from mmcv.parallel import MMDataParallel
from mmskeleton.datasets.utils.coco_transform import flip_back
from .utils.infernce_utils import get_final_preds
import torchvision.transforms as transforms
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmskeleton.processor.apis import init_twodimestimator, inference_twodimestimator
from mmskeleton.datasets.utils.coco_transform import xywh2cs, get_affine_transform
import cv2


# parse loss
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
    # print(log_vars)
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


# process a batch of data
def batch_processor(models, datas, train_mode):

    losses = models.forward(*datas, return_loss=True)
    loss, log_vars = parse_losses(losses)
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=datas[0].size(0))

    return outputs


# train
def train(
        work_dir,
        model_cfg,
        dataset_cfg,
        batch_size,
        optimizer_cfg,
        total_epochs,
        training_hooks,
        workflow=[('train', 1)],
        gpus=1,
        log_level=0,
        workers=4,
        resume_from=None,
        load_from=None,
):
    # prepare data loaders
    if isinstance(dataset_cfg, dict):
        dataset_cfg = [dataset_cfg]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_loaders = [
        torch.utils.data.DataLoader(
            dataset=call_obj(**d,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
            batch_size=batch_size * gpus,
            shuffle=True,
            num_workers=workers,
            drop_last=True) for d in dataset_cfg
    ]

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    # build runner
    optimizer = call_obj(params=model.parameters(), **optimizer_cfg)
    runner = Runner(model, batch_processor, optimizer, work_dir, log_level)
    runner.register_training_hooks(**training_hooks)

    if resume_from:
        runner.resume(resume_from)
    elif load_from:
        runner.load_checkpoint(load_from)
    # run
    workflow = [tuple(w) for w in workflow]
    runner.run(data_loaders, workflow, total_epochs)


# test
def test(test_cfg,
         model_cfg,
         dataset_cfg,
         checkpoint,
         batch_size,
         work_dir,
         gpus=1,
         workers=4):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = call_obj(**dataset_cfg,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           normalize,
                       ]))

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size * gpus,
                                              shuffle=False,
                                              num_workers=workers * gpus)

    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)

    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    model.eval()
    # prepare for evaluation
    num_samples = len(dataset)
    prog_bar = ProgressBar(num_samples // (batch_size * gpus) + 1)
    all_preds = np.zeros((num_samples, model_cfg.skeleton_head.num_joints, 3),
                         dtype=np.float32)

    all_boxes = np.zeros((num_samples, 6))
    filenames = []
    imgnums = []
    image_path = []
    idx = 0

    # copy from hrnet
    with torch.no_grad():
        for i, (input, meta, target, target_weight) in enumerate(data_loader):
            # get prediction
            outputs = model.forward(input, return_loss=False)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs
            # filp test
            if test_cfg.flip:
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped, return_loss=False)
                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
                # feature is not aligned, shift flipped heatmap for higher accuracy
                if test_cfg.shift_heatmap:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                output = (output + output_flipped) * 0.5

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            num_images = input.size(0)
            preds, maxvals = get_final_preds(test_cfg.post_process,
                                             output.detach().cpu().numpy(), c,
                                             s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images
            prog_bar.update()

        name_values, perf_indicator = dataset.evaluate(test_cfg, all_preds,
                                                       work_dir, all_boxes,
                                                       image_path, filenames,
                                                       imgnums)
    return perf_indicator


def inference_model(
        images,
        detection_model,
        skeleton_model,
):
    batch_size = images.size()[0]
    skeleton_results = dict()
    # process each batch image by image
    for idx, b in enumerate(batch_size):
        # get person bboxes
        image = images[b, :, :, :]
        bbox_result = inference_detector(detection_model, image)
        from IPython import embed
        embed()
    #     person_bboxes = bbox_result_filter(bbox_result)
    #     # get skeleton estimation
    #     if person_bboxes.shape[0] > 0:
    #         image, meta = preprocess_skeleton_inputs(image, person_bboxes)
    #         skeleton_result, maxval= inference_twodimestimator(skeleton_model, image)
    #         skeleton_results[str(idx)] = skeleton_result
    # return skeleton_results


def inference(detection_cfg,
              skeleton_cfg,
              dataset_cfg,
              batch_size,
              gpus=1,
              workers=4):

    dataset = call_obj(**dataset_cfg)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size * gpus,
                                              shuffle=False,
                                              num_workers=workers * gpus)

    # build detection model
    detection_model_file = detection_cfg.model_cfg
    detection_checkpoint_file = detection_cfg.checkpoint_file

    detection_model = init_detector(detection_model_file,
                                    detection_checkpoint_file,
                                    device='cuda:0')
    from IPython import embed
    embed()
    detection_model = MMDataParallel(detection_model,
                                     device_ids=range(gpus)).cuda()

    # skeleton_model_file = skeleton_cfg.model_file
    # skeleton_checkpint_file = skeleton_cfg.checkpoint_file
    # skeleton_model = init_twodimestimator(skeleton_model_file,
    #                                       skeleton_checkpint_file,
    #                                       device='cpu')
    # skeleton_model = MMDataParallel(skeleton_model, device_ids=range(gpus)).cuda()

    for idx, image in enumerate(data_loader):
        skeleton_resluts = inference_model(image, detection_model,
                                           skeleton_model)
    return skeleton_resluts
