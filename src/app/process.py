import os
import numpy as np
import transforms3d as tf
import open3d as o3d
import app.optimize

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

        # create output dir if not exists
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        self.camera_intrinsics = []
        with open(os.path.join(dataset_path, 'camera.txt'), 'r') as file:
            self.camera_intrinsics = file.readlines()[0].split()
            self.camera_intrinsics = list(map(float, self.camera_intrinsics))

    def convert_2d_to_3d(self):
        """
        Function to convert 2D keypoint pixel to 3D points in scene.
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
            scene_tf.append(tf.quaternions.quat2mat(pose_q).dot(scene_pts) + pose_t.repeat(scene_pts.shape[1], axis=1))
        self.scene_kpts = np.asarray(scene_tf)
        return

    def visualize_points_in_scene(self, scene_ply_path, scene_obj_kpts):
        """
        Function to visualize a set of 3D points in a .PLY scene.
        """
        vis_mesh_list = []
        scene_cloud = o3d.io.read_point_cloud(scene_ply_path)
        vis_mesh_list.append(scene_cloud)
        print("points shape", scene_obj_kpts.shape)
        for keypt in scene_obj_kpts:
            keypt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            keypt_mesh.translate(keypt)
            keypt_mesh.paint_uniform_color([0.1, 0.1, 0.7])
            vis_mesh_list.append(keypt_mesh)
        o3d.visualization.draw_geometries(vis_mesh_list)
        return

    def sparse_model_writer(self, object_model, filename="sparse_model.txt"):
        """
        Function to save the generated sparse model in the following format
        <point x="000000" y="000000" z="000000" name="0"/> in a .txt file.
        Also writes some meta data.
        Input arguments:
        object_model - (Nx3) numpy array holding 3D positions of all keypoints
                       where N is the number of keypoints on the model.
        filename     - name of the output file inside the output directory.
                       (sparse_model.txt by default)
        """
        out_str = ["<SparseObjectPoints>"]
        for idx, point in enumerate(object_model):
            kpt_str = str("\t<point x=\"{}\" y=\"{}\" z=\"{}\"".format(*list(point)))
            kpt_str = kpt_str + str(" name=\"{}\"/>".format(idx))
            out_str.append(kpt_str)
        out_str.append("</SparseObjectPoints>")
        with open(os.path.join(self.output_dir, filename), 'w') as out_file:
            out_file.write("\n".join(out_str))
        return

    def compute(self):
        """
        Function to compute the sparse model and the relative scene transformations
        through optimization.
        Returns a success boolean.
        """
        #populate selection matrix from select_vec
        total_kpt_count  = len(self.select_vec)
        found_kpt_count  = len(np.nonzero(self.select_vec)[0])
        selection_matrix = np.zeros((found_kpt_count*3, total_kpt_count*3))
        for idx, nz_idx in enumerate(np.nonzero(self.select_vec)[0]):
            selection_matrix[(idx*3):(idx*3)+3, (nz_idx*3):(nz_idx*3)+3] = np.eye(3)

        #initialize quaternions and translations for each scene
        scene_t_ini = np.array([[0, 0, 0]]).repeat(self.scene_kpts.shape[0], axis=0)
        scene_q_ini = np.array([[1, 0, 0, 0]]).repeat(self.scene_kpts.shape[0], axis=0)
        scene_P_ini = np.array([[[0, 0, 0]]]).repeat(self.scene_kpts.shape[2], axis=0)

        #main optimization step
        res = app.optimize.predict(self.scene_kpts, scene_t_ini, scene_q_ini, scene_P_ini, selection_matrix)

        #save the input and the output from optimization step
        out_fn = os.path.join(self.output_dir, 'saved_opt_output')
        np.savez(out_fn, res=res.x, ref=self.scene_kpts, sm=selection_matrix)

        #extract generated sparse object model optimization output
        len_ts = scene_t_ini[1:].size
        len_qs = scene_q_ini[1:].size
        object_model = res.x[len_ts+len_qs:].reshape(scene_P_ini.shape)
        object_model = object_model.squeeze()

        # save the generated sparse object model
        self.sparse_model_writer(object_model)

        if res.success:
            print("--------\n--------\n--------")
            print("SUCCESS")
            print("--------\n--------\n--------")

        return res.success, object_model
