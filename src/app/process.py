import os
import numpy as np
import transforms3d.quaternions as tfq
import open3d as o3d
import app.optimize
import app.geo_constrain
from utils.sparse_model import SparseModel

class Process:
    def __init__(self, dataset_path, output_dir, scale):
        """
        Constructor for Process class.
        Input arguments:
        dataset_path   - path to root dataset directory
        output_dir     - path to output directory
        scale          - scale parameter of the RGB-D sensor
                         (1000 for Intel RealSense D435)
        """
        self.scene_imgs = []
        self.scene_cams = []
        self.scene_kpts = []
        self.pts_in_3d  = []
        self.select_vec = []
        self.scale = scale
        self.output_dir = output_dir
        self.sparse_model_file = None

        #get camera intrinsics
        self.camera_intrinsics = []
        with open(os.path.join(dataset_path, 'camera.txt'), 'r') as file:
            self.camera_intrinsics = file.readlines()[0].split()
            self.camera_intrinsics = list(map(float, self.camera_intrinsics))

    def convert_2d_to_3d(self):
        """
        Function to convert 2D keypoint pixels to 3D points in scene.
        """
        self.select_vec = []
        pts_3d = []
        for _, dep, pts in self.scene_imgs:
            w = []
            for pt in pts:
                pt3d_z = (dep[pt[1], pt[0]])*(1.0/self.scale)
                if pt!=[-1, -1] and pt3d_z!=0:
                    pt3d_x = (pt[0] - self.camera_intrinsics[2])*(pt3d_z/self.camera_intrinsics[0])
                    pt3d_y = (pt[1] - self.camera_intrinsics[3])*(pt3d_z/self.camera_intrinsics[1])
                    pt3d = [pt3d_x, pt3d_y, pt3d_z]
                    self.select_vec.append(1)
                else:
                    pt3d = [0, 0, 0]
                    self.select_vec.append(0)
                w.append(pt3d)
            pts_3d.append(w)
        pts_3d = np.asarray(pts_3d).transpose(0,2,1)
        self.pts_in_3d = pts_3d
        return

    def transform_points(self):
        """
        Function to transform 3D points to the origins of respective scenes.
        """
        scene_tf = []
        for scene_pts, pose in zip(self.pts_in_3d, self.scene_cams):
            pose_t = np.asarray(pose[:3])[:, np.newaxis]
            pose_q = np.asarray([pose[-1]] + pose[3:-1])
            scene_tf.append(tfq.quat2mat(pose_q).dot(scene_pts) + pose_t.repeat(scene_pts.shape[1], axis=1))
        self.scene_kpts = np.asarray(scene_tf)
        return

    def visualize_points_in_scene(self, scene_ply_path, scene_obj_kpts):
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

    def compute(self):
        """
        Function to compute the sparse model and the relative scene transformations
        through optimization. Output directory will be created if not exists.
        Returns a success boolean.
        """
        # create output dir if not exists
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        #populate selection matrix from select_vec
        total_kpt_count  = len(self.select_vec)
        found_kpt_count  = len(np.nonzero(self.select_vec)[0])
        selection_matrix = np.zeros((found_kpt_count*3, total_kpt_count*3))
        for idx, nz_idx in enumerate(np.nonzero(self.select_vec)[0]):
            selection_matrix[(idx*3):(idx*3)+3, (nz_idx*3):(nz_idx*3)+3] = np.eye(3)

        computed_vector = []
        success_flag = False
        if self.sparse_model_file is not None:
            object_model = SparseModel().reader(self.sparse_model_file)
            success_flag, res = app.geo_constrain.predict(object_model, self.scene_kpts.transpose(0,2,1), self.select_vec)
            scene_t = np.asarray([np.array(i[:3,3]) for i in res])
            scene_q = np.asarray([tfq.mat2quat(np.array(i[:3,:3])) for i in res])
            computed_vector = np.concatenate((scene_t[1:, :].flatten(), scene_q[1:, :].flatten()))
        else:
            #initialize quaternions and translations for each scene
            scene_t_ini = np.array([[0, 0, 0]]).repeat(self.scene_kpts.shape[0], axis=0)
            scene_q_ini = np.array([[1, 0, 0, 0]]).repeat(self.scene_kpts.shape[0], axis=0)
            scene_P_ini = np.array([[[0, 0, 0]]]).repeat(self.scene_kpts.shape[2], axis=0)

            #main optimization step
            res = app.optimize.predict(self.scene_kpts, scene_t_ini, scene_q_ini, scene_P_ini, selection_matrix)

            #extract generated sparse object model optimization output
            len_ts = scene_t_ini[1:].size
            len_qs = scene_q_ini[1:].size
            object_model = res.x[len_ts+len_qs:].reshape(scene_P_ini.shape)
            object_model = object_model.squeeze()
            #save the generated sparse object model
            SparseModel().writer(object_model, os.path.join(self.output_dir, "sparse_model.txt"))
            computed_vector = res.x[:(len_ts+len_qs)]
            success_flag = res.success

        #save the input and the output from optimization step
        out_fn = os.path.join(self.output_dir, 'saved_meta_data')
        np.savez(out_fn, model=object_model, scenes=computed_vector, ref=self.scene_kpts, sm=selection_matrix)

        if success_flag:
            print("--------\n--------\n--------")
            print("Computed results saved at {}".format(out_fn))
            print("computed_vector  ---> npz.res")
            print("scene_kpts       ---> npz.ref")
            print("selection_matrix ---> npz.sm")
            print("--------\n--------\n--------")

        return success_flag, object_model
