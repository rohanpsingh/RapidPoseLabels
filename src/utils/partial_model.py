import os
import numpy as np
import open3d as o3d
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
from utils.sparse_model import SparseModel
from utils.region_growing import RegionGrowing

class PartialModel:
    def __init__(self, dataset_path, input_arr_path, input_model_path):
        """
        Constructor for PartialModel class.
        Input arguments:
        dataset_path   - path to root dataset directory
        input_arr_path - path to input npz zipped archive
        input_model_path - path to sparse model file
        """
        self.dataset_path = dataset_path
        self.input_array  = np.load(input_arr_path)
        self.model_path   = input_model_path

        #get number of scenes and number of keypoints
        self.num_scenes = self.input_array['ref'].shape[0]
        self.num_keypts = self.input_array['ref'].shape[2]

        #paths to each of the scene dirs inside root dir
        self.list_of_scene_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        self.list_of_scene_dirs.sort()
        self.list_of_scene_dirs = self.list_of_scene_dirs[:self.num_scenes]
        print("List of scenes: ", self.list_of_scene_dirs)
        print("Number of scenes: ", self.num_scenes)
        print("Number of keypoints: ", self.num_keypts)

        #this is the object model
        self.object_model = []
        #these are the relative scene transformations
        self.scene_tfs = []

    def process_input(self):
        """
        Function to extract data from the input array.
        Input array is the output of the optimization step
        which holds the generated sparse model of the object
        and the relative scene transformations.
        """
        #get the relative scene transforamtions from input array
        out_ts  = self.input_array['scenes'][ :(self.num_scenes-1)*3].reshape((self.num_scenes-1, 3))
        out_qs  = self.input_array['scenes'][(self.num_scenes-1)*3 : (self.num_scenes-1)*7].reshape((self.num_scenes-1, 4))
        out_tfs = np.asarray([tfa.compose(t, tfq.quat2mat(q), np.ones(3)) for t,q in zip(out_ts, out_qs)])
        self.scene_tfs    = np.concatenate((np.eye(4)[np.newaxis,:], out_tfs))

        #get object model from input_array
        self.object_model = SparseModel().reader(self.model_path)
        return

    def get_fragments_by_radius(self, radius=0.05):
        """
        Function to extract point cloud clusters using a radius vector
        around each keypoint in sparse model, in each scene. Returns the
        list of partial models transformed into world frame.
        """
        point_cloud_list = []
        #iterate through a zip of list of scene dirs and the relative scene tfs
        for data_dir_idx, (cur_scene_dir, sce_t) in enumerate(zip(self.list_of_scene_dirs, self.scene_tfs)):
            #load scene point cloud from .PLY file
            cur_ply_path = os.path.join(self.dataset_path, cur_scene_dir, cur_scene_dir + '.ply')
            pcd = o3d.io.read_point_cloud(cur_ply_path)

            #build KDTree
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)

            #get object keypoints in scene frame
            model_points = np.vstack([self.object_model.transpose(),
                                      np.ones(self.object_model.shape[0])])
            model_points = np.dot(sce_t, model_points).transpose()

            fragment = o3d.geometry.PointCloud()
            for keypt in model_points:
                [_, idx, _] = pcd_tree.search_radius_vector_3d(keypt[:3], radius)
                fragment.points.extend(o3d.utility.Vector3dVector(np.asarray(pcd.points)[idx]))
            fragment.transform(np.linalg.inv(sce_t))
            point_cloud_list.append(fragment)
        return point_cloud_list

    def get_regions(self):
        """
        Function to extract point cloud clusters using region-growing clustering
        with keypoints as initial seed points. Returns the
        list of partial models transformed into world frame.
        """
        point_cloud_list = []

        #initialize region growing class
        reg = RegionGrowing((5/180)*3.14, 1.0)

        #iterate through a zip of list of scene dirs and the relative scene tfs
        for data_dir_idx, (cur_scene_dir, sce_t) in enumerate(zip(self.list_of_scene_dirs, self.scene_tfs)):
            #get object keypoints in scene frame
            model_points = np.vstack([self.object_model.transpose(),
                                      np.ones(self.object_model.shape[0])])
            model_points = np.dot(sce_t, model_points).transpose()

            #load scene point cloud from .PLY file
            cur_ply_path = os.path.join(self.dataset_path, cur_scene_dir, cur_scene_dir + '.ply')
            pcd = o3d.io.read_point_cloud(cur_ply_path)

            #crop point cloud using bbox centered at mean
            crop_pcd = reg.crop_pcd(pcd, model_points[:,:3].mean(0))
            #down sample using voxel grid
            crop_pcd = crop_pcd.voxel_down_sample(voxel_size=0.005)

            #set nearest neighbors of model keypoints as seeds
            crop_pcd.points.extend(o3d.utility.Vector3dVector(model_points[:,:3]))
            seed_indices = list(range(len(crop_pcd.points)-model_points.shape[0], len(crop_pcd.points)))
            reg.set_seeds(seed_indices)
            #extract regions using seeds
            regions = reg.extract(crop_pcd)

            for idx, segment in enumerate(regions):
                reg_pcd = o3d.geometry.PointCloud()
                reg_pcd.points = o3d.utility.Vector3dVector(np.asarray(segment))
                reg_pcd.paint_uniform_color([idx/len(regions), 0.706, 0.5])
                reg_pcd.transform(np.linalg.inv(sce_t))
                point_cloud_list.append(reg_pcd)
        return point_cloud_list
