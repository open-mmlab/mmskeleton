import cv2
import numpy as np

def stgcn_visualize(pose, edge, feature, video, label=None):

    C, T, V, M = pose.shape
    T = len(video)
    images = []
    for t in range(T):
        frame = video[t]

        # image resize
        H, W, c = frame.shape
        frame = cv2.repsize(frame, (540*W//H, 540))
        H, W, c = frame.shape
        scale_factor = 2
        
        skeleton = frame * 0

        # draw skeleton
        for m in range(M):
            for i, j in edge:
                xi = pose[0, t, i, m]
                yi = pose[1, t, i, m]
                xj = pose[0, t, j, m]
                yj = pose[1, t, j, m]
                if xi+yi==0 or xj + yj==0:
                    continue
                else:
                    xi = int((xi+0.5)*W)
                    yi = int((yi+0.5)*H)
                    xj = int((xj+0.5)*W)
                    yj = int((yj+0.5)*H)
                cv2.line(skeleton, (xi, yi),
                        (xj,yj), (255, 255, 255), int(np.ceil(2*scale_factor)))

        # generate mask
        mask = frame * 0
        feature = np.abs(feature)
        feature = feature/feature.mean()
        for m in range(M):
            f = feature[int(t/4), :, m] ** 5
            if f.mean() != 0:
                f = f / f.mean()
            for v in range(V):
                x = pose[0, t, v, m]
                y = pose[1, t, v, m]
                if x+y==0:
                    continue
                else:
                    x = int((x+0.5)*W)
                    y = int((y+0.5)*H)
                cv2.circle(mask, (x,
                            y), 0, (255, 255, 255), int(np.ceil(f[v]**0.5*6*scale_factor)))
        blurred_mask = cv2.blur(mask, (12,12))

        skeleton_result = (blurred_mask.astype(float) * 0.75 + skeleton.astype(float) * 0.25).astype(np.uint8)
        rgb_result = (blurred_mask.astype(float) * 0.75 + frame.astype(float) * 0.25).astype(np.uint8)
        
        position = (0,int(H*0.98))
        params = (position, cv2.FONT_HERSHEY_COMPLEX, 0.5*scale_factor, (255,255,255))
        cv2.putText(frame, 'original video', *params)
        cv2.putText(skeleton, 'pose esitimation', *params)
        cv2.putText(skeleton_result, 'feature magnitude', *params)
        cv2.putText(rgb_result, 'feature magnitude + rgb', *params)


        img0 = np.concatenate((frame, skeleton), axis=1)
        img1 = np.concatenate((skeleton_result, rgb_result), axis=1)
        img = np.concatenate((img0, img1), axis=0)
        cv2.rectangle(img, (0, int(2*H*0.015)), (2*W, int(2*H*0.065)), (255,255,255), -1)

        if label is not None:
            position = (int(W*0.9),int(H*0.1))
            cv2.putText(img, label, position, cv2.FONT_HERSHEY_COMPLEX,  0.5*scale_factor, (0, 0, 0))

        images.append(img)
    return images