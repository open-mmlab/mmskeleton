import skvideo.io
import numpy as np
import cv2

def video_info_parsing(video_info, num_person_in=5, num_person_out=2):
    data_numpy = np.zeros((3, len(video_info['data']), 18, num_person_in))
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            if m >= num_person_in:
                break
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            data_numpy[0, frame_index, :, m] = pose[0::2]
            data_numpy[1, frame_index, :, m] = pose[1::2]
            data_numpy[2, frame_index, :, m] = score

    # centralization
    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sort_index):
        data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                    0))
    data_numpy = data_numpy[:, :, :, :num_person_out]

    label = video_info['label_index']
    return data_numpy, label

def get_video_frames(video_path):
    vread = skvideo.io.vread(video_path)
    video = []
    for frame in vread:
        video.append(frame)
    return video

def video_play(video_path, fps=30):
    cap = cv2.VideoCapture(video_path)

    while(cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1000/fps) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()