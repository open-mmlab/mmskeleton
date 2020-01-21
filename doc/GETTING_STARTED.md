## Getting Started

### Installation

a. [Optional] Create a [conda](www.anaconda.com/distribution/) virtual environment and activate it:

``` shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,
``` shell
conda install pytorch torchvision -c pytorch
```

c. Clone mmskeleton from github:

``` shell
git clone https://github.com/open-mmlab/mmskeleton.git
cd mmskeleton
```

d. Install mmskeleton:

``` shell
python setup.py develop
```

e. [Optional] Install mmdetection for person detection:
``` shell
python setup.py develop --mmdet
```
In the event of a failure installation, please install [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md) manually.

f. To verify that mmskeleton and mmdetection installed correctly, use:
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



