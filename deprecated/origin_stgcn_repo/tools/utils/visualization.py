import cv2
import numpy as np


def stgcn_visualize(pose,
                    edge,
                    feature,
                    video,
                    label=None,
                    label_sequence=None,
                    height=1080,
                    fps=None):

    _, T, V, M = pose.shape
    T = len(video)
    pos_track = [None] * M
    for t in range(T):
        frame = video[t]

        # image resize
        H, W, c = frame.shape
        frame = cv2.resize(frame, (height * W // H // 2, height // 2))
        H, W, c = frame.shape
        scale_factor = 2 * height / 1080

        # draw skeleton
        skeleton = frame * 0
        text = frame * 0
        for m in range(M):

            score = pose[2, t, :, m].max()
            if score < 0.3:
                continue

            for i, j in edge:
                xi = pose[0, t, i, m]
                yi = pose[1, t, i, m]
                xj = pose[0, t, j, m]
                yj = pose[1, t, j, m]
                if xi + yi == 0 or xj + yj == 0:
                    continue
                else:
                    xi = int((xi + 0.5) * W)
                    yi = int((yi + 0.5) * H)
                    xj = int((xj + 0.5) * W)
                    yj = int((yj + 0.5) * H)
                cv2.line(skeleton, (xi, yi), (xj, yj), (255, 255, 255),
                         int(np.ceil(2 * scale_factor)))

            if label_sequence is not None:
                body_label = label_sequence[t // 4][m]
            else:
                body_label = ''
            x_nose = int((pose[0, t, 0, m] + 0.5) * W)
            y_nose = int((pose[1, t, 0, m] + 0.5) * H)
            x_neck = int((pose[0, t, 1, m] + 0.5) * W)
            y_neck = int((pose[1, t, 1, m] + 0.5) * H)

            half_head = int(((x_neck - x_nose)**2 + (y_neck - y_nose)**2)**0.5)
            pos = (x_nose + half_head, y_nose - half_head)
            if pos_track[m] is None:
                pos_track[m] = pos
            else:
                new_x = int(pos_track[m][0] + (pos[0] - pos_track[m][0]) * 0.2)
                new_y = int(pos_track[m][1] + (pos[1] - pos_track[m][1]) * 0.2)
                pos_track[m] = (new_x, new_y)
            cv2.putText(text, body_label, pos_track[m],
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5 * scale_factor,
                        (255, 255, 255))

        # generate mask
        mask = frame * 0
        feature = np.abs(feature)
        feature = feature / feature.mean()
        for m in range(M):
            score = pose[2, t, :, m].max()
            if score < 0.3:
                continue

            f = feature[t // 4, :, m]**5
            if f.mean() != 0:
                f = f / f.mean()
            for v in range(V):
                x = pose[0, t, v, m]
                y = pose[1, t, v, m]
                if x + y == 0:
                    continue
                else:
                    x = int((x + 0.5) * W)
                    y = int((y + 0.5) * H)
                cv2.circle(mask, (x, y), 0, (255, 255, 255),
                           int(np.ceil(f[v]**0.5 * 8 * scale_factor)))
        blurred_mask = cv2.blur(mask, (12, 12))

        skeleton_result = blurred_mask.astype(float) * 0.75
        skeleton_result += skeleton.astype(float) * 0.25
        skeleton_result += text.astype(float)
        skeleton_result[skeleton_result > 255] = 255
        skeleton_result.astype(np.uint8)

        rgb_result = blurred_mask.astype(float) * 0.75
        rgb_result += frame.astype(float) * 0.5
        rgb_result += skeleton.astype(float) * 0.25
        rgb_result[rgb_result > 255] = 255
        rgb_result.astype(np.uint8)

        put_text(skeleton, 'inputs of st-gcn', (0.15, 0.5))

        text_1 = cv2.imread('./resource/demo_asset/original_video.png',
                            cv2.IMREAD_UNCHANGED)
        text_2 = cv2.imread('./resource/demo_asset/pose_estimation.png',
                            cv2.IMREAD_UNCHANGED)
        text_3 = cv2.imread('./resource/demo_asset/attention+prediction.png',
                            cv2.IMREAD_UNCHANGED)
        text_4 = cv2.imread('./resource/demo_asset/attention+rgb.png',
                            cv2.IMREAD_UNCHANGED)

        try:
            blend(frame, text_1)
            blend(skeleton, text_2)
            blend(skeleton_result, text_3)
            blend(rgb_result, text_4)
        except:
            pass

        if label is not None:
            label_name = 'voting result: ' + label
            put_text(skeleton_result, label_name, (0.1, 0.5))

        if fps is not None:
            put_text(skeleton, 'fps:{:.2f}'.format(fps), (0.9, 0.5))

        img0 = np.concatenate((frame, skeleton), axis=1)
        img1 = np.concatenate((skeleton_result, rgb_result), axis=1)
        img = np.concatenate((img0, img1), axis=0)

        yield img


def put_text(img, text, position, scale_factor=1):
    t_w, t_h = cv2.getTextSize(text,
                               cv2.FONT_HERSHEY_TRIPLEX,
                               scale_factor,
                               thickness=1)[0]
    H, W, _ = img.shape
    position = (int(W * position[1] - t_w * 0.5),
                int(H * position[0] - t_h * 0.5))
    params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor, (255, 255,
                                                                 255))
    cv2.putText(img, text, *params)


def blend(background, foreground, dx=20, dy=10, fy=0.7):

    foreground = cv2.resize(foreground, (0, 0), fx=fy, fy=fy)
    h, w = foreground.shape[:2]
    b, g, r, a = cv2.split(foreground)
    mask = np.dstack((a, a, a))
    rgb = np.dstack((b, g, r))

    canvas = background[-h - dy:-dy, dx:w + dx]
    imask = mask > 0
    canvas[imask] = rgb[imask]
