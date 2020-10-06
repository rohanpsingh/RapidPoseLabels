import argparse
import open3d as o3d
from utils.sparse_model import SparseModel
from utils.partial_model import PartialModel

if __name__ == '__main__':

    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help='path to root dir of raw dataset')
    ap.add_argument("--input", required=True, help='path to input .npz zipped archive')
    ap.add_argument("--model", required=True, help='path to input sparse model file')
    opt = ap.parse_args()

    #set up Annotations
    generator = PartialModel(opt.dataset, opt.input, opt.model)
    #extract useful information from input array
    generator.process_input()
    #generate labels and writes to output directory
    samples = generator.get_regions()

    #interactive visualization
    out_pcd = o3d.geometry.PointCloud()
    for pcd in samples:
        out_pcd += pcd
    o3d.visualization.draw_geometries_with_editing([out_pcd])
