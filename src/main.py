import datetime
import sys
import argparse
import app.gui

if __name__ == '__main__':

    #current date and time
    datetime = 'out_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",
                    required=False,
                    type=str,
                    default=datetime,
                    help='path to output dir')
    ap.add_argument("--keypoints",
                    required=True,
                    type=int,
                    help='number of keypoints to be defined')
    ap.add_argument("--scale",
                    required=False,
                    type=int,
                    default=1000,
                    help='factor to divide by to get actual depth')
    '''
    ap.add_argument("--scenes",
                    required=False,
                    type=lambda s: [i for i in s.split(',')],
                    default=None,
                    help='list of scene dirs to read')
    '''
    opt = ap.parse_args()

    # run the GUI on input arguments
    guiobj = app.gui.GUI("Label GUI", *vars(opt).values())

    # exit
    sys.exit(0)
