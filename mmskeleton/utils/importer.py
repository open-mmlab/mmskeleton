import sys
import traceback

def import_obj(entry_str):
    mod_str, _sep, class_str = entry_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def call_obj(entry_str, args={}):
    if entry_str[0] == '.':
        entry_str = 'mmskeleton' + entry_str
    return import_obj(entry_str)(**args)