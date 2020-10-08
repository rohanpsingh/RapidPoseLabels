import copy
import os
import argparse
import numpy as np
import open3d as o3d
from utils.sparse_model import SparseModel

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    vis_mesh_list = []
    for pt in source_temp.points:
        keypt = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        keypt.translate(pt)
        keypt.paint_uniform_color([0.7, 0.1, 0.1]) # red
        vis_mesh_list.append(keypt)
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    vis_mesh_list.append(target_temp)
    vis_mesh_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    o3d.visualization.draw_geometries(vis_mesh_list)

class ModelError():
    def __init__(self, sparse_model_path, ply_model_path, pp_file_path, visualize):
        # paths to model files
        self.path_to_sparse_model = sparse_model_path
        self.path_to_ply_model = ply_model_path
        self.path_to_picked_points = pp_file_path
        # visualization flag
        self.visualize = visualize
        
        # icp max-correspondence-distance
        self.threshold = 0.01 # 1cm distance threshold

        # read sparse model 
        sparse_model_array = SparseModel().reader(self.path_to_sparse_model)
        self.sparse_model = o3d.geometry.PointCloud()
        self.sparse_model.points = o3d.utility.Vector3dVector(sparse_model_array)

        # read ply model
        self.ply_model = o3d.io.read_point_cloud(self.path_to_ply_model)
        self.ply_model.scale(PLY_SCALE, np.zeros(3))
        
    def process(self):
        # read picked points
        pp_array = SparseModel().reader(self.path_to_picked_points, PP_SCALE)
        pp_model = o3d.geometry.PointCloud()
        pp_model.points = o3d.utility.Vector3dVector(pp_array)
        corr = np.zeros((len(pp_array), 2))
        corr[:, 0] = list(range(len(pp_array)))
        corr[:, 1] = list(range(len(pp_array)))

        # estimate rough transformation using correspondences
        p2p = o3d.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(self.sparse_model, pp_model,
                                                o3d.utility.Vector2iVector(corr))
        # point-to-point ICP for refinement
        reg_p2p = o3d.registration.registration_icp(
            self.sparse_model, self.ply_model, self.threshold, trans_init)

        # visualize if required
        if self.visualize:
            draw_registration_result(self.sparse_model, self.ply_model, reg_p2p.transformation)
        return reg_p2p.transformation

    def closest_points(self, transformation):
        pcd_tree = o3d.geometry.KDTreeFlann(self.ply_model)
        # Get sparse model points in scene frame
        sparse_points = np.asarray(self.sparse_model.points)
        sparse_points = np.vstack([sparse_points.transpose(),
                                   np.ones(sparse_points.shape[0])])
        sparse_points = np.dot(transformation, sparse_points).transpose()
        # Find the closest point in PLY model to each sparse point
        closest_points = []
        for point in sparse_points:
            [_, nn_idx, _] = pcd_tree.search_knn_vector_3d(point[:3], 1)
            nn_point = np.asarray(self.ply_model.points)[nn_idx][0]
            closest_points.append(nn_point)
        # Create PointCloud of nearest neightbors
        nn_points = np.asarray(closest_points)
        nn_points = np.vstack([nn_points.transpose(),
                               np.ones(nn_points.shape[0])])
        nn_points = np.dot(np.linalg.inv(transformation), nn_points).transpose()
        nn_pcd = o3d.geometry.PointCloud()
        nn_pcd.points = o3d.utility.Vector3dVector(nn_points[:,:3])
        write_path = os.path.join(os.path.dirname(self.path_to_sparse_model), "gt_sparse.txt")
        SparseModel().writer(np.asarray(nn_pcd.points), write_path, True)
        write_path = os.path.join(os.path.dirname(self.path_to_sparse_model), "gt_cad.ply")
        o3d.io.write_point_cloud(write_path, self.ply_model.transform(np.linalg.inv(transformation)))
        return

    def compute_error(self, transformation):
        # compute inliers rmse
        evaluation = o3d.registration.evaluate_registration(self.sparse_model, self.ply_model,
                                                            self.threshold, transformation)
        return evaluation.inlier_rmse

if __name__ == '__main__':

    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help='path to sparse model file')
    ap.add_argument("--ply", required=True, help='path to ground truth model')
    ap.add_argument("--pp", required=True, help='path to MeshLab *.pp file')
    ap.add_argument("--visualize", action='store_true', help='to visualize model fit')
    opt = ap.parse_args()

    #set scale factors
    PLY_SCALE = 1
    PP_SCALE = 1

    # generate annotations and obtain errors
    evaluator = ModelError(*vars(opt).values())
    align_tf = evaluator.process()
    evaluator.closest_points(align_tf)
    err = evaluator.compute_error(align_tf)

    print("Inliers RMSE: ", err)
    print("---")
