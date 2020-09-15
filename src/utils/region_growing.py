import math
import open3d as o3d
import numpy as np

class RegionGrowing:
    def __init__(self, angle_thresh, curv_thresh):

        self.threshold_angle = angle_thresh
        self.threshold_curv = curv_thresh
        self.pcd = []
        self.pcd_tree = []

        #set parameters
        self.grow_region_rad = 0.01
        self.curvature_compute_rad = 0.01
        self.normal_compute_rad = 0.01
        self.normal_compute_max_nn = 10
        return

    def set_seeds(self, indices):
        self.ini_seeds = indices
        return

    def compute_cloud_normals(self):
        """
        Compute normals for the entire point cloud.
        """
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.normal_compute_rad, max_nn=self.normal_compute_max_nn))
        self.pcd_normals = self.pcd.normals
        return

    def compute_point_curvature(self, cluster):
        """
        Compute the curvature for a given a cluster of points.
        """
        pcd = o3d.geometry.PointCloud(cluster)
        mean, cov = pcd.compute_mean_and_covariance()
        eig_val, _ = np.linalg.eig(cov)
        return (eig_val[0]/(eig_val.sum()))

    def validate_point(self, point_idx, seed_idx):
        n1 = self.pcd_normals[point_idx]
        n2 = self.pcd_normals[seed_idx]
        dot = np.round(np.dot(n1, n2), 4)
        angle = math.acos(dot)
        return angle<self.threshold_angle

    def validate_seed(self, point_idx):
        [_, indices, _] = self.pcd_tree.search_radius_vector_3d(point_idx, self.curvature_compute_rad)
        neighbors = o3d.utility.Vector3dVector(np.asarray(self.pcd.points)[indices])
        sigma = self.compute_point_curvature(neighbors)
        return sigma<self.threshold_curv

    def extract(self, pcd):
        #set pcd and pcd_tree
        self.pcd = pcd
        self.pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        #compute normals at each point
        self.compute_cloud_normals()

        global_region = []
        for ini_seed_idx in self.ini_seeds:
            ini_seed = self.pcd.points[ini_seed_idx][:3]
            list_of_seeds = []
            list_of_seeds_idxs = []
            current_region = []
            current_region_idxs = []
            list_of_seeds.append(ini_seed)
            list_of_seeds_idxs.append(ini_seed_idx)
            current_region.append(ini_seed)
            current_region_idxs.append(ini_seed_idx)
            while len(list_of_seeds):
                current_seed = list_of_seeds.pop()
                current_seed_idx = list_of_seeds_idxs.pop()
                [_, nghbr_idxs, _] = self.pcd_tree.search_radius_vector_3d(
                    current_seed, self.grow_region_rad)
                for nghbr_idx in nghbr_idxs:
                    nghbr_point = self.pcd.points[nghbr_idx][:3]
                    is_in_region = self.validate_point(nghbr_idx, current_seed_idx)
                    if nghbr_idx not in current_region_idxs and is_in_region:
                        current_region.append(nghbr_point)
                        current_region_idxs.append(nghbr_idx)
                        is_a_seed = self.validate_seed(nghbr_point)
                        if is_a_seed:
                            list_of_seeds.append(nghbr_point)
                            list_of_seeds_idxs.append(nghbr_idx)
            global_region.append(current_region)
        return global_region
