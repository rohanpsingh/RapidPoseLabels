import argparse
import os
from utils.sparse_model import SparseModel
from utils.annotations import Annotations
from utils.dataset_writer import DatasetWriter

if __name__ == '__main__':

    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help='path to root dir of raw dataset')
    ap.add_argument("--experiment", required=True, help='path to experiment directory')
    ap.add_argument("--output", required=True, help='path to output directory')
    ap.add_argument("--truth", action='store_true', help='to generate ground truth labels')
    ap.add_argument("--visualize", action='store_true', help='to visualize each label')
    opt = ap.parse_args()

    if not opt.truth:
        sparse_path = os.path.join(opt.experiment, "sparse_model.txt")
        dense_path = os.path.join(opt.experiment, "dense.ply")
        scene_meta = os.path.join(opt.experiment, "saved_meta_data.npz")
    else:
        sparse_path = os.path.join(opt.experiment, "gt_sparse.txt")
        dense_path = os.path.join(opt.experiment, "gt_cad.ply")
        scene_meta = os.path.join(opt.experiment, "gt_meta_data.npz")
    #set up Annotations
    label_generator = Annotations(opt.dataset, sparse_path, dense_path, scene_meta, opt.visualize)
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
