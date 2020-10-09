import copy
import os
import argparse
import numpy as np
import open3d as o3d
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
from utils.sparse_model import SparseModel

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    vis_mesh_list = []
    vis_mesh_list.append(source_temp)
    vis_mesh_list.append(target_temp)
    vis_mesh_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    o3d.visualization.draw_geometries(vis_mesh_list)

def compute_init_transformation(point_set, point_set_ref):
    corr = np.zeros((len(point_set.points), 2))
    corr[:, 0] = list(range(len(point_set.points)))
    corr[:, 1] = list(range(len(point_set.points)))

    # Estimate rough transformation using correspondences
    p2p = o3d.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(point_set_ref, point_set,
                                            o3d.utility.Vector2iVector(corr))

    return trans_init

def align_using_icp(scene_mesh, object_mesh, initial_guess, visualize=False):
    # Point-to-point ICP for refinement
    reg_p2p = o3d.registration.registration_icp(object_mesh, scene_mesh, 0.01, initial_guess)
    # Visualize if required
    if visualize:
        draw_registration_result(object_mesh, scene_mesh, reg_p2p.transformation)
    return reg_p2p.transformation

if __name__ == '__main__':

    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=lambda s: s.strip('/'), required=True, help='path to experiment directory')
    ap.add_argument("--dataset", type=lambda s: s.strip('/'), required=True, help='path to dataset directory')
    ap.add_argument("--visualize", action='store_true', help='to visualize model fit')
    opt = ap.parse_args()

    #set scale factors
    PLY_SCALE = 1
    PP_SCALE = 1

    path_to_dataset_dir = opt.dataset
    path_to_experiment_dir = opt.experiment

    # Set up paths (hard-coded for now)
    path_to_sparse_model = os.path.join(opt.experiment, "sparse_model.txt")
    path_to_pp_model = os.path.join(opt.dataset, os.path.basename(opt.dataset) + "_picked_points.pp")
    path_to_input_tfs = os.path.join(opt.experiment, "saved_meta_data.npz")
    path_to_output_tfs = os.path.join(opt.experiment, "gt_meta_data.npz")

    # Read sparse model 
    sparse_model_array = SparseModel().reader(path_to_sparse_model)
    sparse_model = o3d.geometry.PointCloud()
    sparse_model.points = o3d.utility.Vector3dVector(sparse_model_array)

    # Read PickedPoints model
    pp_array = SparseModel().reader(path_to_pp_model, PP_SCALE)
    pp_model = o3d.geometry.PointCloud()
    pp_model.points = o3d.utility.Vector3dVector(pp_array)

    # Paths to each of the scene dirs inside root dir
    list_of_scene_dirs = [d for d in os.listdir(path_to_dataset_dir)
                          if os.path.isdir(os.path.join(path_to_dataset_dir, d))]
    list_of_scene_dirs.sort()
    num_scenes = len(list_of_scene_dirs)
    num_keypts = len(pp_model.points)
    print("List of scenes: ", list_of_scene_dirs)
    print("Number of scenes: ", num_scenes)
    print("Number of keypoints: ", num_keypts)

    # Get the relative scene transforamtions from input array
    scene_tf_load = np.load(path_to_input_tfs)['scenes']
    scene_ts  = scene_tf_load[ :(num_scenes-1)*3].reshape((num_scenes-1, 3))
    scene_qs  = scene_tf_load[(num_scenes-1)*3 : (num_scenes-1)*7].reshape((num_scenes-1, 4))
    scene_tfs = np.asarray([tfa.compose(t, tfq.quat2mat(q), np.ones(3)) for t,q in zip(scene_ts, scene_qs)])
    scene_tfs = np.concatenate((np.eye(4)[np.newaxis,:], scene_tfs))

    # Read object mesh
    path_to_object_mesh = os.path.join(opt.dataset, os.path.basename(opt.dataset) + ".ply")
    object_mesh = o3d.io.read_point_cloud(path_to_object_mesh)
    out_tfs = []
    for scene_dir, scene_tf in zip(list_of_scene_dirs, scene_tfs):
        # Get initial guess for ICP using 3D-3D alignment
        sparse_model_copy = copy.deepcopy(sparse_model)
        tf = compute_init_transformation(sparse_model_copy.transform(scene_tf), pp_model)
        # Read scene mesh
        path_to_scene_mesh = os.path.join(opt.dataset, scene_dir, scene_dir+".ply")
        scene_mesh = o3d.io.read_point_cloud(path_to_scene_mesh)
        # Align using ICP
        gt_tf = align_using_icp(scene_mesh, object_mesh, tf, opt.visualize)
        out_tfs.append(gt_tf)
    out_tfs = np.asarray(out_tfs)
    out_rel_tfs = [tf.dot(np.linalg.inv(out_tfs[0])) for tf in out_tfs[1:]]
    out_ts = [i for tf in out_rel_tfs for i in tf[:3,3]]
    out_qs = [i for tf in out_rel_tfs for i in tfq.mat2quat(tf[:3,:3])]
    np.savez(path_to_output_tfs, scenes=(out_ts + out_qs))

