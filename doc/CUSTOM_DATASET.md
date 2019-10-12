## Custom Skeleton-based Dataset
mmskeleton accepts two formats of skeleton data, `.npy` and `.json`.
We recommend users to store their dataset as `.json` files.
Please follow the below example to build a custom dataset.

### Build dataset from videos

We have prepared a mini video set including three 10-seconds clips in the `resource/data_example` with the structure:

    resource/dataset_example
    ├── skateboarding.mp4  
    ├── clean_and_jerk.mp4  
    └── ta_chi.mp4  


Run this command for building skeleton-based dataset for them:
```
mmskl configs/utils/build_dataset_example.yaml [--gpus $GPUS]
```
mmskeleton extracts skeleton sequences for each video via performing **person detection** and **pose estimation** on all frames.
A few `.json` files will be stored under `data/dataset_example` if you did not change default arguments. The directory layout is shown here:

    data/dataset_example
    ├── skateboarding.json  
    ├── clean_and_jerk.json  
    └── ta_chi.json

All annotations share the same basic data structure like below:
```javascript
{
    "info":
        {
            "video_name": "skateboarding.mp4",
            "resolution": [340, 256],
            "num_frame": 300,
            "num_keypoints": 17,
            "keypoint_channels": ["x", "y", "score"],
            "version": "1.0"
        },
    "annotations":
        [
            {
                "frame_index": 0,
                "id": 0,
                "person_id": null,
                "keypoints": [[x, y, score], [x, y, score], ...]
            },
            ...
        ],
    "category_id": 0,
}
```

After that, train the st-gcn model by:
```
mmskl configs/recognition/st_gcn/dataset_example/train.yaml
```
and test the model by:
```
mmskl configs/recognition/st_gcn/dataset_example/test.yaml --checkpoint $CHECKPOINT_PATH
```

### Build your own dataset

If you want to use mmskeleton on your own **skeleton-based data**, the simplest method is reformatting
your data format to `.json` files with the basic structure we mentioned above. 
Or you can design another data feeder to replace [our data feeder](../mmskeleton/datasets/recognition.py),
and specify it by changing `processor_cfg.dataset_cfg.name` in your training configuration file.

If you want to use mmskeleton on your own **video dataset**,
just follow the above tutorial to build the skeleton-based dataset for videos.
Note that, in the above example, the groundtruth of `category_id` is from [another annotation file](../resource/category_annotation_example.json) with the structure:
```javascript
{
    "categories": [
        "skateboarding",
        "clean_and_jerk",
        "ta_chi"
    ],
    "annotations": {
        "clean_and_jerk.mp4": {
            "category_id": 1
        },
        "skateboarding.mp4": {
            "category_id": 0
        },
        "ta_chi.mp4": {
            "category_id": 2
        }
    }
}
```
The `category_id` will be set to `-1` if the category annotations miss.

You can build dataset by:
```
mmskl configs/utils/build_dataset_example.yaml --video_dir $VIDEO_DIR --category_annotation $VIDEO_ANNOTATION --out_dir $OUT_DIR [--gpus $GPUS]
```
To change the person detector, pose estimator or other arguments, modify the `build_dataset_example.yaml`.
