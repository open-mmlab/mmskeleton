import cv2
import numpy as np

def stgcn_visualize(pose, edge, feature, video):

    C, T, V, M = pose.shape
    T = len(video)
    images = []
    for t in range(T):
        frame = video[t]
        H, W, c = frame.shape
        scale_factor = (H*W/10**5)**0.5
        
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
        
        img0 = np.concatenate((frame, skeleton), axis=1)
        img1 = np.concatenate((rgb_result, skeleton_result), axis=1)
        img = np.concatenate((img0, img1), axis=0)

        images.append(img)
    return images