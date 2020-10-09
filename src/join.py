import argparse
import os
import open3d as o3d
from utils.sparse_model import SparseModel
from utils.partial_model import PartialModel

if __name__ == '__main__':

    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help='path to root dir of raw dataset')
    ap.add_argument("--experiment", required=True, help='path to experiment directory')
    opt = ap.parse_args()

    #set up Annotations
    sparse_model_path = os.path.join(opt.experiment, "sparse_model.txt")
    scene_meta_path = os.path.join(opt.experiment, "saved_meta_data.npz")
    generator = PartialModel(opt.dataset, scene_meta_path, sparse_model_path)
    #extract useful information from input array
    generator.process_input()
    #generate labels and writes to output directory
    samples = generator.get_regions()

    #interactive visualization
    out_pcd = o3d.geometry.PointCloud()
    for pcd in samples:
        out_pcd += pcd
    o3d.visualization.draw_geometries_with_editing([out_pcd])
