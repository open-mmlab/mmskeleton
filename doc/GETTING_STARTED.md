## Getting Started

[Option] **Create** a conda virtual environment and activate it:

``` shell
pip install conda
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

**Clone** mmskeleton from github:

``` shell
git clone https://github.com/open-mmlab/mmskeleton.git
cd mmskeleton
```

**Install** the mmskeleton:

``` shell
python setup.py develop
```

Sometimes `mmdet` may be not installed successfully. In that case, please install [mmdet](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md) manually.
Then run above command again.

To **verify** that mmskeleton installed correctly, use:
```shell
python mmskl.py pose_demo [--gpus $GPUS]
# or "python mmskl.py pose_demo_HD [--gpus $GPUS]" for a higher accuracy
```
An generated video as below will be saved under the prompted path.

<p align="center">
    <img src="../demo/estimation/pose_demo.gif", width="500">
</p>



### Basic usage:

Any application in mmskeleton is described by a configuration file. That can be started by a uniform command:
``` shell
python mmskl.py $CONFIG_FILE [--options $OPTHION]
```
which is equivalent to
```
mmskl $CONFIG_FILE [--options $OPTHION]
```
Optional arguments `options` is defined in the configuration file.
You can check them via:
``` shell
mmskl $CONFIG_FILE -h
```

### Example:

See [START_RECOGNITION.md](../doc/START_RECOGNITION.md) for learning how to train a model for skeleton-based action recognitoin.

See [CUSTOM_DATASET](../doc/CUSTOM_DATASET.md) for building your own skeleton-based dataset.

See [CREATE_APPLICATION](../doc/CREATE_APPLICATION.md) for creating your own mmskeleton application.



