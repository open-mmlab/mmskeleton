processor_cfg = dict(
    name='.processor.pseudo.train',
    optimizer=None,
    model_cfg=dict(
        name='.models.pseudo.model',
        in_channels=3,
        out_channels=60,
        weight=None),
    dataset_cfg=dict(
        name='.datasets.pseudo.dataset',
        data_path=None)
)
