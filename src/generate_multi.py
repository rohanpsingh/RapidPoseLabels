import argparse
from utils.sparse_model import SparseModel
from utils.multiannotations import MultiAnnotations
from utils.dataset_writer import DatasetWriter

if __name__ == '__main__':

    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help='path to scene directory')
    ap.add_argument("--input", required=True, help='path to input .npz zipped archive')
    ap.add_argument("--model", required=True, help='path to input sparse model file')
    ap.add_argument("--output", required=True, help='path to output directory')
    ap.add_argument("--visualize", action='store_true', help='to visualize each label')
    opt = ap.parse_args()

    #set up Annotations
    objects = [i.strip() for i in opt.input.split(',')]
    models = [i.strip() for i in opt.model.split(',')]
    label_generator = MultiAnnotations(opt.scene, objects, models, opt.visualize)
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
