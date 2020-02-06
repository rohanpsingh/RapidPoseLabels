import numpy as np
import transforms3d as tf
import cv2
import os
import open3d as o3d

class Pose:
    def __init__(self, dataset_path, scale):
        self.scene_imgs = []
        self.scene_cams = []
        self.scene_plys = []
        self.pts_in_3d    = []
        self.pts_in_3d_tx = []
        self.scale = scale

        self.camera_intrinsics = []
        with open(os.path.join(dataset_path, 'camera.txt'), 'r') as file:
            self.camera_intrinsics = file.readlines()[0].split()
            self.camera_intrinsics = list(map(float, self.camera_intrinsics))

    def convert_2Dto3D(self):
        pts_3d = []
        for rgb, dep, pts in self.scene_imgs:
            w = []
            for pt in pts:
                pt3d_z = (dep[pt[1], pt[0]])*(1.0/self.scale)
                if not pt==[-1, -1] and pt3d_z!=0:
                    pt3d_x = (pt[0] - self.camera_intrinsics[2])*(pt3d_z/self.camera_intrinsics[0])
                    pt3d_y = (pt[1] - self.camera_intrinsics[3])*(pt3d_z/self.camera_intrinsics[1])
                    pt3d = [pt3d_x, pt3d_y, pt3d_z]
                else:
                    pt3d = (0, 0, 0)
                w.append(pt3d)
            pts_3d.append(w)
        pts_3d = np.asarray(pts_3d).transpose(0,2,1)
        self.pts_in_3d = pts_3d

    def transform_points(self):
        scene_tf = []
        for scene_pts, pose in zip(self.pts_in_3d, self.scene_cams):
            pose_t = np.asarray(pose[:3])[:, np.newaxis]
            pose_q = np.asarray([pose[-1]] + pose[3:-1])
            scene_tf.append(tf.quaternions.quat2mat(pose_q).dot(scene_pts) + pose_t.repeat(scene_pts.shape[1], axis=1))
        self.pts_in_3d_tx = np.asarray(scene_tf)

    def visualize(self):
        for ply_path, keypts in zip(self.scene_plys, self.pts_in_3d_tx):
            if (os.path.basename(os.path.dirname(ply_path)) != '4'):
                continue
            vis_mesh_list = []
            scene_cloud = o3d.io.read_point_cloud(ply_path)
            vis_mesh_list.append(scene_cloud)
            for keypt in keypts.transpose():
                keypt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                keypt_mesh.translate(keypt)
                keypt_mesh.paint_uniform_color([0.1, 0.1, 0.7])
                vis_mesh_list.append(keypt_mesh)
            o3d.visualization.draw_geometries(vis_mesh_list)
