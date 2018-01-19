import numpy as np
import random


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(
        data_numpy,
        angle_candidate=[-10., -5., 0., 5., 10.],
        scale_candidate=[0.9,1.0,1.1],
        transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
        move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape

    move_time = random.choice(move_time_candidate) 
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate,num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
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
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:,:,i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy
            


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert(C==3)
    score = data_numpy[2,:,:,:].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T-1]).argsort(axis = 1).reshape(T-1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T-1, :, :].reshape(2, T-1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T-1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2-xy1)**2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m) 
        forward = distance[choose].argmin(axis=1)         
        for t in range(T-1):
            distance[t, :,forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert(np.all(forward_map >= 0))

    # string data
    for t in range(T-1):
        forward_map[t+1] = forward_map[t+1][forward_map[t]]
    
    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:,t,:,:] = data_numpy[:,t,:,forward_map[t]].transpose(1,2,0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2,:,:,:].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:,:,:,rank]

    return data_numpy

    # # match poses between 2 frames
    # if self.pose_matching:
    #     C, T, V, M = data_numpy.shape
    #     forward_map = np.zeros((T, M), dtype=int) - 1
    #     backward_map = np.zeros((T, M), dtype=int) - 1

    #     # match pose
    #     for t in range(T - 1):
    #         for m in range(M):
    #             s = (data_numpy[2, t, :, m].reshape(1, V, 1) != 0) * 1
    #             if s.sum() == 0:
    #                 continue
    #             res = data_numpy[:, t + 1, :, :] - data_numpy[:, t, :,
    #                                                           m].reshape(
    #                                                               C, V, 1)
    #             n = (res * res * s).sum(axis=1).sum(
    #                 axis=0).argsort()[0]  #next pose
    #             forward_map[t, m] = n
    #             backward_map[t + 1, n] = m

    #     # find start point
    #     start_point = []
    #     for t in range(T):
    #         for m in range(M):
    #             if backward_map[t, m] == -1:
    #                 start_point.append((t, m))

    #     # generate path
    #     path_list = []
    #     c = 0
    #     for i in range(len(start_point)):
    #         path = [start_point[i]]
    #         while (1):
    #             t, m = path[-1]
    #             n = forward_map[t, m]
    #             if n == -1:
    #                 break
    #             else:
    #                 path.append((t + 1, n))
    #             #print(c,t)
    #             c = c + 1
    #         path_list.append(path)

    #     # generate data
    #     new_M = self.num_match_trace
    #     path_length = [len(p) for p in path_list]
    #     sort_index = np.array(path_length).argsort()[::-1][:new_M]
    #     if self.mode == 'train':
    #         np.random.shuffle(sort_index)
    #         sort_index = sort_index[:M]
    #         new_data_numpy = np.zeros((C, T, V, M))
    #     else:
    #         new_data_numpy = np.zeros((C, T, V, new_M))
    #     for i, p in enumerate(sort_index):
    #         path = path_list[p]
    #         for t, m in path:
    #             new_data_numpy[:, t, :, i] = data_numpy[:, t, :, m]

    #     data_numpy = new_data_numpy