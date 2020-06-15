import datetime
import sys
import argparse
import app.gui

if __name__ == '__main__':

    #current date and time
    datetime = 'out_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str, help='path to root dir of raw dataset')
    ap.add_argument("--output", required=False, type=str, default=datetime, help='path to output dir')
    ap.add_argument("--keypoints", required=True, type=int, help='number of keypoints to be defined')
    opt = ap.parse_args()

    # run the GUI on input arguments
    guiobj = app.gui.GUI("Label GUI", opt.dataset, opt.output, opt.keypoints)

    # exit
    sys.exit(0)
