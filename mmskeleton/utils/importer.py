import sys
import traceback

def import_obj(name):
    mod_str, _sep, class_str = name.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def call_obj(name, **kwargs):
    if isinstance(name, dict):
        return call_obj(**name)

    if name[0] == '.':
        name = 'mmskeleton' + name
    return import_obj(name)(**kwargs)