import sys


def import_obj(name):
    if not isinstance(name, str):
        raise ImportError('Object name should be a string.')

    if name[0] == '.':
        name = 'mmskeleton' + name

    mod_str, _sep, class_str = name.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Object {} cannot be found.'.format(class_str))

def call_obj(name, **kwargs):
    return import_obj(name)(**kwargs)
