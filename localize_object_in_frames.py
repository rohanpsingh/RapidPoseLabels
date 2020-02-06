import numpy as np
import os
import sys
import yaml
import argparse
import gui
import cv2

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--keypoints", required=True)
    opt = ap.parse_args()

    dataset_path = opt.dataset
    num_keypoints = int(opt.keypoints)

    guiobj = gui.App("Label GUI", dataset_path, num_keypoints)
    sys.exit(0)
