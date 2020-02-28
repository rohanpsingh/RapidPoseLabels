import numpy as np
import os
import sys
import yaml
import argparse
import app.gui
import cv2

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--scenes", required=True)
    opt = ap.parse_args()

    dataset_path = opt.dataset
    num_keypoints = int(opt.keypoints)
    num_scenes = int(opt.scenes)

    guiobj = app.gui.GUI("Label GUI", dataset_path, num_keypoints, num_scenes)
    sys.exit(0)
