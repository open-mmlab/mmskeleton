from mmskeleton.utils import call_obj


def train(model, model_args, dataset, dataset_args, optimizer):
    model = call_obj(model, model_args)
    dataset = call_obj(dataset, dataset_args)