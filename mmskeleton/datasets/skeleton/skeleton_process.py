import random
import numpy as np
from mmskeleton.deprecated.datasets.utils import skeleton as skeleton_aaai18

# def stgcn_aaai18_dataprocess(data,
#                              window_size,
#                              random_choose=False,
#                              random_move=False):
#     data = normalize_by_resolution(data)
#     data = mask_by_visibility(data)
#     # processing
#     if random_choose:
#         data['data'] = skeleton_aaai18.random_choose(data['data'], window_size)
#     elif window_size > 0:
#         data['data'] = skeleton_aaai18.auto_pading(data['data'], window_size)
#     if random_move:
#         data['data'] = skeleton_aaai18.random_move(data['data'])
#     data = transpose(data, order=[0, 2, 1, 3])
#     data = to_tuple(data)
#     return data


def normalize_by_resolution(data):

    resolution = data['info']['resolution']
    channel = data['info']['keypoint_channels']
    np_array = data['data']

    for i, c in enumerate(channel):
        if c == 'x':
            np_array[i] = np_array[i] / resolution[0] - 0.5
        if c == 'y':
            np_array[i] = np_array[i] / resolution[1] - 0.5

    data['data'] = np_array
    return data


def get_mask(data, mask_channel, mask_threshold=0):
    data['mask'] = data['data'][[mask_channel]] > mask_threshold
    return data


def mask(data):
    data['data'] = data['data'] * data['mask']
    return data


def normalize(data, mean, std):
    np_array = data['data']
    mean = np.array(mean, dtype=np_array.dtype)
    std = np.array(std, dtype=np_array.dtype)
    mean = mean.reshape(mean.shape + (1, ) * (np_array.ndim - mean.ndim))
    std = std.reshape(std.shape + (1, ) * (np_array.ndim - std.ndim))
    data['data'] = (np_array - mean) / std
    return data


def normalize_with_mask(data, mean, std, mask_channel, mask_threshold=0):
    data = get_mask(data, mask_channel, mask_threshold)
    data = normalize(data, mean, std)
    data = mask(data)
    return data


def mask_by_visibility(data):

    channel = data['info']['keypoint_channels']
    np_array = data['data']

    for i, c in enumerate(channel):
        if c == 'score' or c == 'visibility':
            mask = (np_array[i] == 0)
            for j in range(len(channel)):
                if c != j:
                    np_array[j][mask] = 0

    data['data'] = np_array
    return data


def transpose(data, order, key='data'):
    data[key] = data[key].transpose(order)
    return data


def to_tuple(data, keys=['data', 'category_id']):
    return tuple([data[k] for k in keys])


def temporal_repeat(data, size, random_crop=False):
    """
    repeat on the time axis.
    """

    np_array = data['data']
    T = np_array.shape[2]

    if T >= size:
        if random_crop:
            np_array = np_array[:, :, random.randint(0, T -
                                                     size):][:, :, :size]
        else:
            np_array = np_array[:, :, :size]

    else:
        selected_index = np.arange(T)
        selected_index = np.concatenate(
            (selected_index, selected_index[1:-1][::-1]))
        selected_index = np.tile(selected_index,
                                 size // (2 * T - 2) + 1)[:size]

        np_array = np_array[:, :, selected_index]

    data['data'] = np_array
    return data


def pad_zero(data, size):
    np_array = data['data']
    T = np_array.shape[2]
    if T < size:
        pad_shape = list(np_array.shape)
        pad_shape[2] = size
        np_array_paded = np.zeros(pad_shape, dtype=np_array.dtype)
        np_array_paded[:, :, :T, :] = np_array
        data['data'] = np_array_paded
    return data


def random_crop(data, size):
    np_array = data['data']
    T = np_array.shape[2]
    if T > size:
        begin = random.randint(0, T - size)
        data['data'] = np_array[:, :, begin:begin + size, :]
    return data


def simulate_camera_moving(data,
                           angle_candidate=[-10., -5., 0., 5., 10.],
                           scale_candidate=[0.9, 1.0, 1.1],
                           transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                           move_time_candidate=[1]):

    channel = data['info']['keypoint_channels']
    if channel[0] != 'x' or channel[1] != 'y':
        raise NotImplementedError(
            'The first two channels of keypoints should be ["x", "y"]')

    np_array = data['data']
    T = np_array.shape[2]

    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace for parameters of affine transformation
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = np_array[0:2, :, i_frame]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        np_array[0:2, :, i_frame] = new_xy.reshape(*(
            np_array[0:2, :, i_frame].shape))

    data['data'] = np_array
    return data
