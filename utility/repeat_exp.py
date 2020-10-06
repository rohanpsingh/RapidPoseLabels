import os
import open3d as o3d
import datetime
import argparse
import numpy as np
import app.optimize
from utils.sparse_model import SparseModel

def visualize_points_in_scene(scene_ply_path, scene_obj_kpts):
    """
    Function to visualize a set of 3D points in a .PLY scene.
    """
    vis_mesh_list = []
    scene_cloud = o3d.io.read_point_cloud(scene_ply_path)
    vis_mesh_list.append(scene_cloud)
    for keypt in scene_obj_kpts:
        keypt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        keypt_mesh.translate(keypt)
        keypt_mesh.paint_uniform_color([0.1, 0.1, 0.7])
        vis_mesh_list.append(keypt_mesh)
    o3d.visualization.draw_geometries(vis_mesh_list)
    return

def process(input_array, num_out_scenes, output_dir):
    #process input
    scene_kpts = input_array['ref']
    selection_matrix= input_array['sm']

    #get number of scene and keypoints in original experiment
    num_scenes = scene_kpts.shape[0]
    num_keypts = scene_kpts.shape[2]

    # convert selection matrix to a binary matrix
    col_splits = np.hsplit(selection_matrix, selection_matrix.shape[1]//3)
    row_splits = [np.vsplit(col, col.shape[0]//3) for col in col_splits]
    vis_list = [sum(map(lambda x:(x==np.eye(3)).all(), mat)) for mat in row_splits]
    # binary matrix of shape (num of scenes x num of keypoints)
    vis_mat = np.reshape(vis_list, [num_scenes, num_keypts])

    #slice binary matrix
    vis_mat = vis_mat[:num_out_scenes]

    # convert binary matrix back to selection matrix
    select_vec = vis_mat.flatten()
    total_kpt_count  = len(select_vec)
    found_kpt_count  = len(np.nonzero(select_vec)[0])
    selection_matrix = np.zeros((found_kpt_count*3, total_kpt_count*3))
    for idx, nz_idx in enumerate(np.nonzero(select_vec)[0]):
        selection_matrix[(idx*3):(idx*3)+3, (nz_idx*3):(nz_idx*3)+3] = np.eye(3)

    #slice scene keypoints
    scene_kpts = scene_kpts[:num_out_scenes]

    #initialize quaternions and translations for each scene
    scene_t_ini = np.array([[0, 0, 0]]).repeat(scene_kpts.shape[0], axis=0)
    scene_q_ini = np.array([[1, 0, 0, 0]]).repeat(scene_kpts.shape[0], axis=0)
    scene_P_ini = np.array([[[0, 0, 0]]]).repeat(scene_kpts.shape[2], axis=0)
    #initialize with known keypoints
    scene_P_ini = scene_kpts[0].transpose()[:,np.newaxis,:]

    #main optimization step
    res = app.optimize.predict(scene_kpts, scene_t_ini, scene_q_ini, scene_P_ini, selection_matrix)

    #extract generated sparse object model optimization output
    len_ts = scene_t_ini[1:].size
    len_qs = scene_q_ini[1:].size
    object_model = res.x[len_ts+len_qs:].reshape(scene_P_ini.shape)
    object_model = object_model.squeeze()
    #save the generated sparse object model
    SparseModel().writer(object_model, os.path.join(output_dir, "sparse_model.txt"))
    computed_vector = res.x[:(len_ts+len_qs)]
    success_flag = res.success

    #save the input and the output from optimization step
    out_fn = os.path.join(output_dir, 'saved_meta_data')
    np.savez(out_fn, model=object_model, scenes=computed_vector, ref=scene_kpts, sm=selection_matrix)

    if success_flag:
        print("--------\n--------\n--------")
        print("Computed results saved at {}".format(out_fn))
        print("--------\n--------\n--------")
    return object_model

if __name__ == '__main__':

    #current date and time
    datetime = 'out_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    #get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help='path to input .npz zipped archive')
    ap.add_argument("--ply", required=True, type=str, help='path to input .npz zipped archive')
    ap.add_argument("--num_out_scenes", required=True, type=int, help='number of scenes to use for optimization')
    ap.add_argument("--output", required=False, type=str, default=datetime, help='path to output dir')
    opt = ap.parse_args()

    #copy arguments
    input_array = np.load(opt.input)
    num_out_scenes = opt.num_out_scenes
    output_dir = opt.output
    path_to_scene_ply = opt.ply

    #print info
    print("Number of scenes: ", input_array['ref'].shape[0])
    print("Number of keypoints: ", input_array['ref'].shape[2])
    print("----------")
    print("Repeating experiment for ", num_out_scenes, " scenes")

    # create output dir if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    #perform optimization
    object_model = process(input_array, num_out_scenes, output_dir)
    #visualize result
    visualize_points_in_scene(path_to_scene_ply, object_model)
