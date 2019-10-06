from mmskeleton.utils import call_obj


def train(model_cfg, dataset_cfg, optimizer):
    model = call_obj(**model_cfg)
    dataset = call_obj(**dataset_cfg)
    print('train a pseudo model...')
    print('done.')