import os
import sys
import numpy as np

def computeEuclideanDistance(path1, path2):
    estimated = np.loadtxt(path1)
    groundtruth = np.loadtxt(path2)
    err = [np.linalg.norm(point_a-point_b) for point_a, point_b in zip(estimated, groundtruth)]
    return sum(err)/len(err)

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('usage: ', sys.argv[0], ' <dir1> <dir2>')
        sys.exit(1)

    est_labels = [os.path.join(sys.argv[1], fn) for fn in os.listdir(sys.argv[1])]
    gt_labels = [os.path.join(sys.argv[2], fn) for fn in os.listdir(sys.argv[2])]

    list_err = []
    for l1, l2 in zip(est_labels, gt_labels):
        err = computeEuclideanDistance(l1, l2)
        list_err.append(err)
    print("mean error = ", sum(list_err)/len(list_err))
