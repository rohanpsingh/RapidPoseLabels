import numpy as np
import os
import sys
import yaml
import argparse
import gui

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--keypoints", required=True)
    opt = ap.parse_args()

    dataset_path = opt.dataset
    num_keypoints = int(opt.keypoints)

    guiobj = gui.App("Label GUI", dataset_path, num_keypoints)
    kpts_2d = guiobj.dataset_scenes
    kpts_2d = np.asarray(kpts_2d)
    print(kpts_2d)
    sys.exit(0)
