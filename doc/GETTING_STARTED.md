## Getting Started

First, install mmsksleton by:
``` shell
python setup.py develop
```
Any application in mmskeleton is described by a configuration file. It can be started by a uniform command:
``` shell
python run.py $CONFIG_FILE [--options $OPTHION]
```
Optional arguments `options` are defined in configuration files,
check them via:
``` shell
python run.py $CONFIG_FILE -h
```

