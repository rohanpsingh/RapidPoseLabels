import argparse
import os
import cv2
import numpy as np
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
from utils.sparse_model import SparseModel
from utils.annotations import Annotations
from utils.dataset_writer import DatasetWriter

if __name__ == '__main__':

    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help='path to root dir of raw dataset')
    ap.add_argument("--input", required=True, help='path to input .npz zipped archive')
    ap.add_argument("--model", required=True, help='path to input sparse model file')
    ap.add_argument("--output", required=True, help='path to output directory')
    ap.add_argument("--visualize", action='store_true', help='to visualize each label')
    opt = ap.parse_args()

    #set up Annotations
    label_generator = Annotations(opt.dataset, opt.input, opt.model, opt.visualize)
    #extract useful information from input array
    label_generator.process_input()
    #generate labels and writes to output directory
    samples = label_generator.generate_labels()

    #write each sample to disk
    label_writer = DatasetWriter(opt.output)
    for counter, item in enumerate(samples):
        label_writer.write_to_disk(item, counter)
        print("Saved sample: {}".format(repr(counter).zfill(5)), end="\r", flush=True)
    print("Total number of samples generated: {}".format(len(samples)))
