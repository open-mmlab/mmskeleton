import lazy_import

pycocotools = lazy_import.lazy_module("pycocotools")
COCO = lazy_import.lazy_module("pycocotools.COCO")
COCOeval = lazy_import.lazy_module("pycocotools.COCOeval")
mmdet = lazy_import.lazy_module("mmdet")
lazy_import.lazy_module("mmdet.apis")


def is_exist(module_name):
    module = __import__(module_name)
    try:
        lazy_import._load_module(module)
        return True
    except ImportError:
        return False