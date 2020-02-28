import argparse
import os
import cv2
import numpy as np
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
from itertools import combinations

class Annotations:
    def __init__(self, dataset_path, input_arr_path, visualize):

        self.dataset_path = dataset_path
        self.input_array  = np.load(input_arr_path)
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
        self.width = 640
        self.height = 480
        self.bbox_scale = 1.5

    def project3Dto2D(self, scene_tf, cam_tf, img):
        tf = np.dot(np.linalg.inv(cam_tf), scene_tf)
        rvec,_ = cv2.Rodrigues(tf[:3, :3])
        tvec = tf[:3,3]
        imgpts,_ = cv2.projectPoints(self.object_model, rvec, tvec, self.cam_mat, None)
        keypts = np.transpose(np.asarray(imgpts), (1,0,2))[0]

        def square_distance(x,y): return sum([(xi-yi)**2 for xi, yi in zip(x,y)])
        max_square_distance = 0
        for pair in combinations(keypts,2):
            if square_distance(*pair) > max_square_distance:
                max_square_distance = square_distance(*pair)
                max_pair = pair
        bbox_cn = keypts.mean(0)
        bbox_sd = (max_square_distance**0.5)*self.bbox_scale
        bbox_tl = keypts.mean(0)-(bbox_sd/2)
        bbox_br = keypts.mean(0)+(bbox_sd/2)
        bbox = (bbox_cn, bbox_sd/200.0)

        for p in range(keypts.shape[0]):
            cv2.circle(img, tuple((int(keypts[p,0]), int(keypts[p,1]))), 3, (0,0,255), -1)
        cv2.rectangle(img, (int(bbox_tl[0]), int(bbox_tl[1])), (int(bbox_br[0]), int(bbox_br[1])), (0,255,0), 2)
        if self.visualize:
            cv2.imshow('win', img)
            cv2.waitKey(10)
        return keypts, bbox
        
    def process_input(self, dbg=False):
        ref_keypts = self.input_array['ref']
        select_mat = self.input_array['sm']
        opt_output = self.input_array['res']

        scene_t_ini = np.array([[0, 0, 0]]).repeat(self.num_scenes, axis=0)
        scene_q_ini = np.array([[1, 0, 0, 0]]).repeat(self.num_scenes, axis=0)
        scene_P_ini = np.array([[0, 0, 0]]).repeat(self.num_keypts, axis=0)

        len_ts = scene_t_ini[1:].size
        len_qs = scene_q_ini[1:].size
        len_Ps = scene_P_ini.size
        out_ts = opt_output[:len_ts].reshape(scene_t_ini[1:, :].shape)
        out_qs = opt_output[len_ts:len_ts+len_qs].reshape(scene_q_ini[1:, :].shape)
        out_Ps = opt_output[len_ts+len_qs:].reshape(scene_P_ini.shape)
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

    def generate_labels(self):
        for cur_scene_dir, sce_T in zip(self.list_of_scene_dirs, self.scene_tfs):
            print(cur_scene_dir)
            with open(os.path.join(self.dataset_path, cur_scene_dir, 'associations.txt'), 'r') as file:
                img_name_list = file.readlines()
            with open(os.path.join(self.dataset_path, cur_scene_dir, 'camera.poses'), 'r') as file:
                cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]
            for img_name, cam_pose in zip(img_name_list[:100], cam_pose_list[:100]):
                img_name = img_name.split()
                rgb_im_path = os.path.join(self.dataset_path, cur_scene_dir, img_name[3])
                input_rgb_image = cv2.resize(cv2.imread(rgb_im_path), (self.width, self.height))
                cam_T = tfa.compose(np.asarray(cam_pose[:3]), tfq.quat2mat(np.asarray([cam_pose[-1]] + cam_pose[3:-1])), np.ones(3))
                img = self.project3Dto2D(sce_T, cam_T, input_rgb_image)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--visualize", required=True, default=True)
    opt = ap.parse_args()

    lab = Annotations(opt.dataset, opt.input, opt.visualize)
    lab.process_input(False)
    lab.generate_labels()
