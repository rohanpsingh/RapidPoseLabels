import numpy as np
import argparse
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
import sys
import cv2
import os
from itertools import combinations
import evaluate3d

np.set_printoptions(threshold=sys.maxsize, linewidth=700)
np.set_printoptions(precision=4, suppress=True)


class Evaluate:
    def __init__(self, dataset_path, input_arr_path, picked_pts, visualize):

        self.dataset_path = dataset_path
        self.input_array  = np.load(input_arr_path)
        self.picked_pts   = picked_pts
        self.visualize    = (visualize.lower()=='true')

        self.cam_mat = np.eye(3)
        with open(os.path.join(dataset_path, 'camera.txt'), 'r') as file:
            camera_intrinsics = file.readlines()[0].split()
            camera_intrinsics = list(map(float, camera_intrinsics))
        self.cam_mat[0,0] = camera_intrinsics[0]
        self.cam_mat[1,1] = camera_intrinsics[1]
        self.cam_mat[0,2] = camera_intrinsics[2]
        self.cam_mat[1,2] = camera_intrinsics[3]

        self.num_scenes = self.input_array['ref'].shape[0]
        self.num_keypts = self.input_array['ref'].shape[2]
        self.list_of_scene_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        self.list_of_scene_dirs.sort()
        self.list_of_scene_dirs = self.list_of_scene_dirs[:self.num_scenes]
        print("List of scenes: ", self.list_of_scene_dirs)
        print("Number of scenes: ", self.num_scenes)
        print("Number of keypoints: ", self.num_keypts)
        self.width = 640
        self.height = 480
        self.bbox_scale = 1.5
        self.viz_count=0

    def get_true_model(self, sce_id):
        ref_keypts = self.input_array['ref']
        select_mat = self.input_array['sm']
        opt_output = self.input_array['res']

        select_mat_block = select_mat[3*self.viz_count:, sce_id*(3*self.num_keypts):(sce_id+1)*3*self.num_keypts]
        vis_vec = evaluate3d.get_visibility(select_mat_block)
        obj_man = evaluate3d.get_object_manual(ref_keypts[sce_id], vis_vec)
        obj_def = evaluate3d.get_object_definition(self.picked_pts, vis_vec)
        d, Z, tform = evaluate3d.procrustes(obj_def, obj_man, False)

        T = tfa.compose(tform['translation'], np.linalg.inv(tform['rotation']), np.ones(3))
        T = np.linalg.inv(T)
        obj_all = evaluate3d.get_object_definition(self.picked_pts, np.ones(self.num_keypts))
        self.true_object = np.asarray([(T[:3,:3].dot(pt) + T[:3,3]) for pt in obj_all])
        self.viz_count += len(np.nonzero(vis_vec)[0])
        return 

    def project3Dto2D(self, input_points, input_T):
        tf = input_T
        rvec,_ = cv2.Rodrigues(tf[:3, :3])
        tvec = tf[:3,3]
        imgpts,_ = cv2.projectPoints(input_points, rvec, tvec, self.cam_mat, None)
        keypts = np.transpose(np.asarray(imgpts), (1,0,2))[0]
        return keypts
        
    def visualize_keypts(self, input_img, pts1, pts2):
        for p in range(pts1.shape[0]):
            cv2.circle(input_img, tuple((int(pts1[p,0]), int(pts1[p,1]))), 3, (255,0,0), -1)
        for p in range(pts2.shape[0]):
            cv2.circle(input_img, tuple((int(pts2[p,0]), int(pts2[p,1]))), 3, (0,255,0), -1)
        cv2.imshow('win', input_img)
        cv2.waitKey(10)
        return

    def process_input(self, dbg=False):
        ref_keypts = self.input_array['ref']
        select_mat = self.input_array['sm']
        opt_output = self.input_array['res']

        out_ts = opt_output[ :(self.num_scenes-1)*3].reshape((self.num_scenes-1, 3))
        out_qs = opt_output[(self.num_scenes-1)*3 : (self.num_scenes-1)*7].reshape((self.num_scenes-1, 4))
        out_Ps = opt_output[(self.num_scenes-1)*7 : ].reshape((self.num_keypts, 3))
        out_Ts = np.asarray([tfa.compose(t, tfq.quat2mat(q), np.ones(3)) for t,q in zip(out_ts, out_qs)])
        self.object_model = out_Ps
        self.scene_tfs    = np.concatenate((np.eye(4)[np.newaxis,:], out_Ts))
        if dbg:
            np.set_printoptions(precision=5, suppress=True)
            print("--------\n--------\n--------")
            print("Output translations:\n", out_ts)
            print("Output quaternions:\n", out_qs)
            print("Output points:\n", out_Ps, out_Ps.shape)
            print("--------\n--------\n--------")
            print("Input points:\n", ref_keypts[0])
        return

    def get_pixel_errors(self):
        scene_err_list = []
        for idx, (cur_scene_dir, sce_T) in enumerate(zip(self.list_of_scene_dirs, self.scene_tfs)):
            err = []
            self.get_true_model(idx)
            with open(os.path.join(self.dataset_path, cur_scene_dir, 'associations.txt'), 'r') as file:
                img_name_list = file.readlines()
            with open(os.path.join(self.dataset_path, cur_scene_dir, 'camera.poses'), 'r') as file:
                cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]
            for img_name, cam_pose in zip(img_name_list[:100], cam_pose_list):
                img_name = img_name.split()
                rgb_im_path = os.path.join(self.dataset_path, cur_scene_dir, img_name[3])
                input_rgb_image = cv2.resize(cv2.imread(rgb_im_path), (self.width, self.height))
                cam_T = tfa.compose(np.asarray(cam_pose[:3]), tfq.quat2mat(np.asarray([cam_pose[-1]] + cam_pose[3:-1])), np.ones(3))
                estpts = self.project3Dto2D(self.object_model, np.dot(np.linalg.inv(cam_T), sce_T))
                trupts = self.project3Dto2D(self.true_object, np.linalg.inv(cam_T))
                if self.visualize:
                    self.visualize_keypts(input_rgb_image, estpts, trupts)
                for (x,y) in zip(trupts, estpts):
                    err.append(((x-y)**2).sum()**0.5)
            scene_err_list.append(err)
        return scene_err_list
                

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--points", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--visualize", required=True, default=True)
    opt = ap.parse_args()

    lab = Evaluate(opt.dataset, opt.input, opt.points, opt.visualize)
    lab.process_input(False)
    error_vec = lab.get_pixel_errors()

    mean_error = 0
    for e in error_vec:
        print("scene error: ", sum(e)/len(e))
        mean_error += (sum(e)/len(e))
    print("Mean error: ", mean_error/len(error_vec))
    print("---")

