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


def set_attr(obj, name, value):
    if not isinstance(name, str):
        raise ImportError('Attribute name should be a string.')

    attr, _sep, others = name.partition('.')
    if others == '':
        setattr(obj, attr, value)
    else:
        set_attr(getattr(obj, attr), others, value)


def get_attr(obj, name):
    if not isinstance(name, str):
        raise ImportError('Attribute name should be a string.')

    attr, _sep, others = name.partition('.')
    if others == '':
        return getattr(obj, attr)
    else:
        return get_attr(getattr(obj, attr), others)