## Create an MMSkeleton Application

MMSkeleton provides various models, datasets, apis, operators for various applications,
such as pose estimation, human detection, action recognition and dataset building.
The workflow of a application is defined by a **processor**, which is usually a python function.

In MMSkeleton, an application is defined in a configuration file.
It is a `.json`, `.yaml` or `.py` file including `processor_cfg` field. 
Here is an example:

```yaml
# yaml

processor_cfg:
  type: <path to processor function>
  dataset:
    type: <path to dataset module>
    data_path: ./data
  #more arguments for processor function...

argparse_cfg:
  data:
    bind_to: processor_cfg.dataset.data_path
    help: the path of data
  #more option arguments for command line...
```

The `processor_cfg` specifies a processor function and its dataset module
In adittion, the `data_path` argument of the dataset is "./data".
The `argparse_cfg` create a option argument `data` which is bound to `data_path`.

Note that, mmskeleton will import processor function or modules according to the given path by the priority of `local directory > system python path > mmskeleton`.



With this configuration, the application can be started by:
```shell
mmskl $CONFIG_FILE [--data $DATA_PATH]
```





