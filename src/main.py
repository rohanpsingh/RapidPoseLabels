import sys
import argparse
import app.gui

if __name__ == '__main__':

    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str, help='path to root dir of raw dataset')
    ap.add_argument("--output", required=True, type=str, help='path to output dir')
    ap.add_argument("--keypoints", required=True, type=int, help='number of keypoints to be defined')
    opt = ap.parse_args()

    # run the GUI on input arguments
    guiobj = app.gui.GUI("Label GUI", opt.dataset, opt.output, opt.keypoints)

    # exit
    sys.exit(0)
