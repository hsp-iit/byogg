import numpy as np

def load_intrinsics(calib_txt):
    calib = np.loadtxt(calib_txt, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    return K
