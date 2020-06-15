import os
import numpy as np
import open3d as o3d
import argparse
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa

def visualize(npz_file, dataset_path):

    #get number of scenes and number of keypoints
    input_array = np.load(npz_file)
    num_scenes = input_array['ref'].shape[0]
    num_keypts = input_array['ref'].shape[2]

    list_of_scene_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    list_of_scene_dirs.sort()

    #get scene transforamtions from input array
    out_ts = input_array['scenes'][:(num_scenes-1)*3].reshape((num_scenes-1, 3))
    out_qs = input_array['scenes'][(num_scenes-1)*3 : (num_scenes-1)*7].reshape((num_scenes-1, 4))
    out_Ts = np.asarray([tfa.compose(t, tfq.quat2mat(q), np.ones(3)) for t,q in zip(out_ts, out_qs)])
    #get object model from input_array
    out_Ps = input_array['model'].reshape((num_keypts, 3))

    #this is the object mode
    object_model = out_Ps
    #these are the relative scene transformations
    scene_tfs    = np.concatenate((np.eye(4)[np.newaxis,:], out_Ts))

    #append sparse object model points to vis_mesh_list
    vis_mesh_list = []
    vis_mesh_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(0.5))
    for pt in object_model:
        keypt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        keypt_mesh.translate(pt)
        keypt_mesh.paint_uniform_color([0.7, 0.1, 0.1]) #red
        vis_mesh_list.append(keypt_mesh)

    colors_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0.8, 0.8, 0.8], [0, 1, 1], [1, 0, 1], [0, 0, 0]]
    #append camera trajectory to vis_mesh_list
    for scene_idx, (cur_scene_dir, sce_T) in enumerate(zip(list_of_scene_dirs, scene_tfs)):
        points = []
        lines = []
        color = [0, 0, 0]
        if scene_idx<len(colors_list):
            color = colors_list[scene_idx]
        #read the camera pose corresponding to each frame
        with open(os.path.join(dataset_path, cur_scene_dir, 'camera.poses'), 'r') as file:
            cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]
        for time_t, cam_pose in enumerate(cam_pose_list[::5]):
            #compose 4x4 camera pose matrix
            cam_T = tfa.compose(np.asarray(cam_pose[:3]), tfq.quat2mat(np.asarray([cam_pose[-1]] + cam_pose[3:-1])), np.ones(3))
            cam_T = np.linalg.inv(np.dot(np.linalg.inv(cam_T), sce_T))

            #add camera position
            traj_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
            #traj_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
            traj_mesh.translate(cam_T[:3,3])
            traj_mesh.rotate(cam_T[:3,:3])
            traj_mesh.paint_uniform_color(color)
            vis_mesh_list.append(traj_mesh)
            points.append(list(cam_T[:3,3]))
            if time_t>0:
                lines.append([time_t-1, time_t])
            line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.asarray(points)),
            lines=o3d.utility.Vector2iVector(lines),)
        line_set.colors = o3d.utility.Vector3dVector([color]*len(lines))
        vis_mesh_list.append(line_set)
    #display
    o3d.visualization.draw_geometries(vis_mesh_list)
    return

if __name__ == '__main__':

    #get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help='path to input .npz zipped archive')
    ap.add_argument("--dataset", required=True, help='path to root dir of raw dataset')
    opt = ap.parse_args()

    visualize(opt.input, opt.dataset)
