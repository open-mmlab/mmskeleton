import sys


def import_obj(type):
    if not isinstance(type, str):
        raise ImportError('Object type should be string.')

    # if type[0] == '.':
    #     type = 'mmskeleton' + type

    mod_str, _sep, class_str = type.rpartition('.')
    try:
        __import__(mod_str)
        return getattr(sys.modules[mod_str], class_str)
    except ModuleNotFoundError:
        if type[0:11] != 'mmskeleton.':
            return import_obj('mmskeleton.' + type)
        raise ModuleNotFoundError('Object {} cannot be found in {}.'.format(
            class_str, mod_str))


def call_obj(type, **kwargs):
    if isinstance(type, str):
        return import_obj(type)(**kwargs)
    elif callable(type):
        return type(**kwargs)
    else:
        raise ValueError('type should be string all callable.')


def set_attr(obj, type, value):
    if not isinstance(type, str):
        raise ImportError('Attribute type should be string.')

    attr, _sep, others = type.partition('.')
    if others == '':
        attr = int(attr) if attr.isdigit() else attr
        obj[attr] = value
        # setattr(obj, attr, value)
    else:
        attr = int(attr) if attr.isdigit() else attr
        set_attr(obj[attr], others, value)


def get_attr(obj, type):
    if not isinstance(type, str):
        raise ImportError('Attribute type should be string.')

    attr, _sep, others = type.partition('.')

    if attr == '':
        return obj
    else:
        attr = int(attr) if attr.isdigit() else attr
        return get_attr(obj[attr], others)
