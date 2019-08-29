## Getting Started

**Install mmsksleton:**
``` shell
python setup.py develop
```

**Usage:**

Any application in mmskeleton is described by a configuration file. That can be started by a uniform command:
``` shell
mmskl $CONFIG_FILE [--options $OPTHION]
# or
python run.py $CONFIG_FILE [--options $OPTHION]
```
Optional arguments `options` are defined in configuration files,
check them via:
``` shell
mmskl $CONFIG_FILE -h
```

**Example:**

See [START_RECOGNITION.md](../doc/START_RECOGNITION.md) for learning how to train a skeleton-based action recognitoin model.