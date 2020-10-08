import os
import cv2
import numpy as np
import transforms3d.affines as tfa
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
        self.list_of_scenes = []
        self.scene_cams = []
        self.scene_kpts = []
        self.select_vec = []
        self.scale = scale
        self.output_dir = output_dir
        self.sparse_model_file = None

        #get camera intrinsics
        self.camera_matrix = np.eye(3)
        self.camera_intrinsics = []
        with open(os.path.join(dataset_path, 'camera.txt'), 'r') as file:
            self.camera_intrinsics = file.readlines()[0].split()
            self.camera_intrinsics = list(map(float, self.camera_intrinsics))
        self.camera_matrix[0,0] = self.camera_intrinsics[0]
        self.camera_matrix[1,1] = self.camera_intrinsics[1]
        self.camera_matrix[0,2] = self.camera_intrinsics[2]
        self.camera_matrix[1,2] = self.camera_intrinsics[3]

    def convert_2d_to_3d(self, list_of_scenes):
        """
        Function to convert 2D keypoint pixels to 3D points in scene.
        """
        self.select_vec = []
        pts_3d = []
        for scene in list_of_scenes:
            w = []
            for (pt, dep, _) in scene:
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
        return np.asarray(pts_3d)

    def transform_points(self, points_3d, list_of_scenes):
        """
        Function to transform 3D points to the origins of respective scenes.
        """
        if points_3d.size==0:
            self.scene_kpts = []
            return
        self.scene_kpts = np.empty([0, 3, points_3d.shape[1]])
        for scene_points, scene_meta in zip(points_3d, list_of_scenes):
            scene_poses = [(np.array([pose[-1]] + pose[3:-1]), np.array(pose[:3])) for (_, _, pose) in scene_meta]
            pt_tf = [tfq.quat2mat(quat).dot(pt3d) + trns for pt3d, (quat, trns) in zip(scene_points, scene_poses)]
            self.scene_kpts = np.vstack((self.scene_kpts, np.asarray(pt_tf).transpose()[np.newaxis]))
        return

    def get_projection(self, inputs, tar_cam_pose):
        """
        Function to get corresponding pixel location in an image given pixel location in another image.
        Uses camera_intrinsics and depth images to first get 3D positions of pixels in target frame,
        then projects to another image using cv2.projectPoints().
        Returns: (Nx2) NumPy array of N keypoints' 2D pixel coordinates.
        Input arguments:
        inputs - list of tuples of pixel coordinates, associated depth images and associated camera poses
        """
        if len(inputs)==0:
            return []
        target_frame = tfa.compose(np.array(tar_cam_pose[:3]),
                                   tfq.quat2mat(np.array([tar_cam_pose[-1]] + tar_cam_pose[3:-1])),
                                   np.ones(3))
        point_positions = []
        for (pt, depth, pose) in inputs:
            pt3d_z = (depth[pt[1], pt[0]])*(1.0/self.scale)
            if pt==[-1, -1] or pt3d_z==0:
                continue
            pt3d_x = (pt[0] - self.camera_intrinsics[2])*(pt3d_z/self.camera_intrinsics[0])
            pt3d_y = (pt[1] - self.camera_intrinsics[3])*(pt3d_z/self.camera_intrinsics[1])
            position = [pt3d_x, pt3d_y, pt3d_z]
            source_frame  = tfa.compose(np.array(pose[:3]), tfq.quat2mat(np.array([pose[-1]] + pose[3:-1])), np.ones(3))
            relative_tf = np.linalg.inv(target_frame).dot(source_frame)
            position_tf = relative_tf[:3,:3].dot(position) + relative_tf[:3,3]
            point_positions.append(position_tf)

        #project 3D points to 2D image plane
        rvec = cv2.Rodrigues(np.eye(3))[0]
        tvec = np.zeros(3)
        keypoint_pixels = cv2.projectPoints(np.array(point_positions), rvec, tvec, self.camera_matrix, None)[0]
        return keypoint_pixels.transpose(1,0,2)[0]

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

    def compute(self, sparse_model_flag=False):
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
        if sparse_model_flag:
            object_model = SparseModel().reader(self.sparse_model_file)
            success_flag, res = app.geo_constrain.predict(object_model, self.scene_kpts.transpose(0,2,1), self.select_vec)
            scene_t = np.asarray([np.array(i[:3,3]) for i in res])
            scene_q = np.asarray([tfq.mat2quat(np.array(i[:3,:3])) for i in res])
            computed_vector = np.concatenate((scene_t.flatten(), scene_q.flatten()))
        else:
            #initialize quaternions and translations for each scene
            scene_t_ini = np.array([[0, 0, 0]]).repeat(self.scene_kpts.shape[0], axis=0)
            scene_q_ini = np.array([[1, 0, 0, 0]]).repeat(self.scene_kpts.shape[0], axis=0)
            scene_P_ini = np.array([[[0, 0, 0]]]).repeat(self.scene_kpts.shape[2], axis=0)
            #initialize with known keypoints
            scene_P_ini = self.scene_kpts[0].transpose()[:,np.newaxis,:]

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
            print("--------\n--------\n--------")

        return success_flag, object_model

    def define_grasp_point(self, ply_path):
        """
        Function to define grasp pose.
        """
        # create output dir if not exists
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if (self.scene_kpts.shape)!=(1,3,2):
            raise Exception("2 3D points must exist to define a grasp pose." )
        if not os.path.exists(ply_path):
            raise Exception("%s path does not exist" % ply_path)
        #get 3 user-defined 3D points
        point_1 = self.scene_kpts.transpose(0, 2, 1)[0, 0]
        point_2 = self.scene_kpts.transpose(0, 2, 1)[0, 1]
        #read point cloud and get normals
        scene_cloud = o3d.io.read_point_cloud(ply_path)
        scene_cloud.estimate_normals()
        scene_cloud.normalize_normals()
        scene_points  = scene_cloud.points
        scene_normals = scene_cloud.normals
        scene_tree  = o3d.geometry.KDTreeFlann(scene_cloud)
        #get neightbors and find normal direction
        [_, idx, _] = scene_tree.search_knn_vector_3d(point_1, 200)
        normals = np.asarray(scene_normals)[list(idx)]
        normal_dir = normals.mean(0)
        normal_dir = normal_dir/np.linalg.norm(normal_dir)
        #get intersection of point and plane
        # d = (a.x_0 + b.y_0 + c.z_0)/(a.x_u + b.y_u + c.z_u)
        lamda = np.dot(normal_dir, point_1)/np.dot(normal_dir, point_2)
        point_i  = lamda*point_2
        vector_x = (point_i - point_1)
        vector_y = np.cross(normal_dir, vector_x)
        #normalize
        vector_x = vector_x/np.linalg.norm(vector_x)
        vector_y = vector_y/np.linalg.norm(vector_y)
        #create rotation matrix
        rot_mat = np.array([vector_x, vector_y, normal_dir])
        tf_mat = tfa.compose(point_1, rot_mat, np.ones(3))

        #save the grasp point
        SparseModel().grasp_writer(tf_mat, os.path.join(self.output_dir, "sparse_model.txt"))

        #visualize in open3d
        vis_mesh_list = []
        vis_mesh_list.append(scene_cloud)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
        coordinate_frame.transform(tf_mat)
        vis_mesh_list.append(coordinate_frame)
        o3d.visualization.draw_geometries(vis_mesh_list)
        return True, tf_mat
