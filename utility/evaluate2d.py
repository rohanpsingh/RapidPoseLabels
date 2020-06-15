import os
import sys
import argparse
import numpy as np
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
import cv2
import evaluate3d
from utils.annotations import Annotations

np.set_printoptions(threshold=sys.maxsize, linewidth=700)
np.set_printoptions(precision=4, suppress=True)


class Evaluate(Annotations):
    """
    Class to evaluate accuracy of generated 2D keypoint labels
    if a ground truth sparse model is available, and information
    of relative scene transformations is available.
    """
    def __init__(self, dataset_path, input_arr_path, input_model_path, picked_pts, visualize):
        """
        Constructor for Evaluate class.
        Input arguments:
        dataset_path   - path to root dataset directory
        input_arr_path - path to input npz zipped archive
        input_model_path - path to sparse model file
        picekd_pts     - path to *.pp file for true object keypoints
        visualize      - set 'True' to visualize
        """
        super().__init__(dataset_path, input_arr_path, input_model_path, False)
        self.picked_pts = picked_pts
        self.visualize = visualize

    def get_true_model(self):
        """
        Function to find 3D positions of each defined keypoints in the frame of
        every scene origin. Uses the .npz zipped archive to get relative scene
        transformations, the selection matrix etc.
        Returns a list of (Nx3) 2D numpy arrays where each i-th array in the list
        holds the 3D keypoint pose configuration of the object in the i-th scene.
        """
        list_of_poses = []
        ref_keypts = self.input_array['ref']
        select_mat = self.input_array['sm']
        viz_count = 0
        for sce_id, _ in enumerate(self.list_of_scene_dirs):
            select_mat_block = select_mat[3*viz_count:, sce_id*(3*self.num_keypts):(sce_id+1)*3*self.num_keypts]
            vis_vec = evaluate3d.get_visibility(select_mat_block)
            obj_man = evaluate3d.get_object_manual(ref_keypts[sce_id], vis_vec)
            obj_def = evaluate3d.get_object_definition(self.picked_pts, vis_vec)
            _, _, tform = evaluate3d.procrustes(obj_def, obj_man, False)

            T = tfa.compose(tform['translation'], np.linalg.inv(tform['rotation']), np.ones(3))
            T = np.linalg.inv(T)
            obj_all = evaluate3d.get_object_definition(self.picked_pts, np.ones(self.num_keypts))
            true_object = np.asarray([(T[:3,:3].dot(pt) + T[:3,3]) for pt in obj_all])
            viz_count += len(np.nonzero(vis_vec)[0])
            list_of_poses.append(true_object)
        return list_of_poses

    def visualize_joint(self, input_img, pts1, pts2):
        """
        Function to draw two sets of 2D points (estimated and ground-truth)
        on the input image. Uses opencv draw functions.
        Input arguments:
        input_img - input RGB image (image will be modified)
        pts1      - (Nx2) numpy array of 2D key points (green)
        pts2      - (Nx2) numpy array of 2D key points (red)
        """
        for point in pts1:
            cv2.circle(input_img, tuple(map(int, point)), 3, (255,0,0), -1)
        for point in pts2:
            cv2.circle(input_img, tuple(map(int, point)), 3, (0,255,0), -1)
        cv2.imshow('win', input_img)
        cv2.waitKey(10)
        return

    def get_pixel_errors(self):
        """
        Function to compute 2D distance error between the estimated 2D keypoints and
        projections of ground-truth 3D keypoints.
        Returns a list of scene-wise errors.
        """
        scene_err_list = []
        true_poses = self.get_true_model()
        for idx, (cur_scene_dir, sce_t) in enumerate(zip(self.list_of_scene_dirs, self.scene_tfs)):
            err = []
            #read the names of image frames in this scene
            with open(os.path.join(self.dataset_path, cur_scene_dir, 'associations.txt'), 'r') as file:
                img_name_list = file.readlines()

            #read the camera pose corresponding to each frame
            with open(os.path.join(self.dataset_path, cur_scene_dir, 'camera.poses'), 'r') as file:
                cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]

            for img_name, cam_pose in zip(img_name_list, cam_pose_list):
                #read the RGB images using opencv
                img_name = img_name.split()
                rgb_im_path = os.path.join(self.dataset_path, cur_scene_dir, img_name[3])
                input_rgb_image = cv2.resize(cv2.imread(rgb_im_path), (self.width, self.height))
                #compose 4x4 camera pose matrix
                cam_t = tfa.compose(np.asarray(cam_pose[:3]), tfq.quat2mat(np.asarray([cam_pose[-1]] + cam_pose[3:-1])), np.ones(3))
                #get estimated 2D positions of keypoints
                estpts, _, _ = self.project_points(self.object_model, np.dot(np.linalg.inv(cam_t), sce_t))
                #get the ground truth 2D positions of keypoints
                trupts, _, _ = self.project_points(true_poses[idx], np.linalg.inv(cam_t))

                #calculate the error distance
                err.extend([((x-y)**2).sum()**0.5 for (x,y) in zip(trupts, estpts)])
                #visualize if required
                if self.visualize:
                    self.visualize_joint(input_rgb_image, estpts, trupts)
            scene_err_list.append(err)
        return scene_err_list

if __name__ == '__main__':

    #get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help='path to root dir of raw dataset')
    ap.add_argument("--input", required=True, help='path to input .npz zipped archive')
    ap.add_argument("--model", required=True, help='path to input sparse model file')
    ap.add_argument("--points", required=True, help='path to *.pp file for true sparse model')
    ap.add_argument("--visualize", action='store_true', help='to visualize each label')
    opt = ap.parse_args()

    #generate annotations and obtain errors
    evaluator = Evaluate(opt.dataset, opt.input, opt.model, opt.points, opt.visualize)
    evaluator.process_input()
    error_vec = evaluator.get_pixel_errors()

    mean_error = 0
    for index, e in enumerate(error_vec):
        print("scene {} error: {}".format(index, sum(e)/len(e)))
        mean_error += (sum(e)/len(e))
    print("Mean error: ", mean_error/len(error_vec))
    print("---")

