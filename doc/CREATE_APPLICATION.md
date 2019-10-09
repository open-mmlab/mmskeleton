## Create an MMSkeleton Application

MMSkeleton provides various models, datasets, apis, operators for various applications,
such as pose estimation, human detection, action recognition and dataset building.
The workflow of a application is defined by a **processor**, which is usually a python function.

In MMSkeleton, an application is corresponded to a configuration file.
It is a `.json`, `.yaml` or `.py` file including `processor_cfg` field. 
There is an example:

```yaml
# yaml

processor_cfg:
  type: <path to processor module>
  dataset:
    type: <path to dataset module>
    data_path: ./data
  ...

argparse_cfg:
  data:
    bind_to: processor_cfg.dataset.data_path
    help: the path of data
  ...
```

The `processor_cfg` specifies a processor module and its dataset module
In adittion, the `data_path` argument of the dataset is "./data".
The `argparse_cfg` create a option argument `data` which is bound to `data_path`.

With this configuration, the application can be started by:
```shell
mmskl $CONFIG_FILE --data $DATA_PATH
```






