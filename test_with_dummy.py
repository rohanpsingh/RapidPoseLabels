import numpy as np
import scipy.optimize
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
import sys
import open3d as o3d
from optimize import *
import dummies

np.set_printoptions(threshold=sys.maxsize, linewidth=700)
np.set_printoptions(precision=4, suppress=True)

def visualize_dummy_object(scene_ply_path, scene_obj_kpts, t_off, q_off):
    vis_mesh_list = []
    input_T = tfa.compose(t_off, tfq.quat2mat(q_off), np.ones(3)) 
    if scene_ply_path is not None:
        scene_cloud = o3d.io.read_triangle_mesh(scene_ply_path)
        scene_cloud.translate(input_T[:-1,-1])
        vis_mesh_list.append(scene_cloud)
    for keypt in scene_obj_kpts.transpose():
        keypt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        keypt_mesh.translate(keypt)
        keypt_mesh.paint_uniform_color([0.1, 0.1, 0.7])
        vis_mesh_list.append(keypt_mesh)
    vis_mesh_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(0.2))
    o3d.visualization.draw_geometries(vis_mesh_list)

def test_with_dummy():

    #f = np.load("saved_scenes.npz")
    #ref_kpts = f['kpts']
    #selection_matrix = f['sm']
    ns = 8
    nk = 21

    camera_ts  = dummies.input_ts[:ns, :] + 1
    camera_qs  = dummies.input_qs[:ns, :]
    obj_model = dummies.yt_obj_model[:nk, :]

    list_of_tfs = []
    for input_t, input_q in zip(camera_ts, camera_qs):
        list_of_tfs.append(tfa.compose(input_t, tfq.quat2mat(input_q), np.ones(3)))

    ref_kpts = np.zeros((ns, 3, nk))
    for (scene_id, t) in enumerate(list_of_tfs):
        for (keypt_id, pt) in enumerate(obj_model):
            pt = np.append(pt, 1)
            ref_kpts[scene_id, :, keypt_id] = t.dot(pt)[:3]

    selection_matrix = np.eye((ns*nk*3))
    scene_t_ini = np.array([[0, 0, 0]]).repeat(ref_kpts.shape[0], axis=0)
    scene_q_ini = np.array([[1, 0, 0, 0]]).repeat(ref_kpts.shape[0], axis=0)
    scene_P_ini = np.array([[0, 0, 0]]).repeat(ref_kpts.shape[2], axis=0)

    optimize = True
    if optimize:
        res = predict(ref_kpts, scene_t_ini, scene_q_ini, scene_P_ini, selection_matrix)
        output_vec = res.x
    else:
        output_vec = np.load("saved_tmp_output.npz")["res"]

    len_ts = scene_t_ini[1:].size
    len_qs = scene_q_ini[1:].size
    len_Ps = scene_P_ini.size
    out_ts = output_vec[:len_ts].reshape(scene_t_ini[1:, :].shape)
    out_qs = output_vec[len_ts:len_ts+len_qs].reshape(scene_q_ini[1:, :].shape)
    out_Ps = output_vec[len_ts+len_qs:].reshape(scene_P_ini.shape)
    object_model = out_Ps.transpose()

    print("--------\n--------\n--------")
    print("Input points:\n", ref_kpts[0])
    visualize_dummy_object("/home/rohan/rohan_m15x/models_3d/model_stl/dewalt_yt.off", object_model, camera_ts[0], camera_qs[0])


if __name__ == '__main__':
    test_with_dummy()
