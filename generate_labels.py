import argparse
import os
import cv2
import numpy as np
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
import open3d as o3d
from optimize import *

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

        self.list_of_scene_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        self.width = 640
        self.height = 480

    def project3Dto2D(self, scene_tf, cam_tf, img):
        tf = np.dot(np.linalg.inv(cam_tf), scene_tf)
        rvec,_ = cv2.Rodrigues(tf[:3, :3])
        tvec = tf[:3,3]
        imgpts,_ = cv2.projectPoints(self.object_model, rvec, tvec, self.cam_mat, None)
        imgpts = np.transpose(np.asarray(imgpts), (1,0,2))[0]
        for p in range(imgpts.shape[0]):
            cv2.circle(img, tuple((int(imgpts[p,0]), int(imgpts[p,1]))), 3, (0,0,255), -1)
        if self.visualize:
            cv2.imshow('win', img)
            cv2.waitKey(10)
        return img
        
    def process_input(self, dbg=False):
        ref_keypts = self.input_array['ref']
        select_mat = self.input_array['sm']
        opt_output = self.input_array['res']

        num_scenes = ref_keypts.shape[0]
        num_keypts = ref_keypts.shape[2]
        scene_t_ini = np.array([[0, 0, 0]]).repeat(num_scenes, axis=0)
        scene_q_ini = np.array([[1, 0, 0, 0]]).repeat(num_scenes, axis=0)
        scene_P_ini = np.array([[0, 0, 0]]).repeat(num_keypts, axis=0)

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
            with open(os.path.join(self.dataset_path, cur_scene_dir, 'associations.txt'), 'r') as file:
                img_name_list = file.readlines()
            with open(os.path.join(self.dataset_path, cur_scene_dir, 'camera.poses'), 'r') as file:
                cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]
            for img_name, cam_pose in zip(img_name_list, cam_pose_list):
                img_name = img_name.split()
                rgb_im_path = os.path.join(self.dataset_path, cur_scene_dir, img_name[3])
                input_rgb_image = cv2.resize(cv2.imread(rgb_im_path), (self.width, self.height))
                cam_T = tfa.compose(np.asarray(cam_pose[:3]), tfq.quat2mat(np.asarray([cam_pose[-1]] + cam_pose[3:-1])), np.ones(3))
                img = self.project3Dto2D(sce_T, cam_T, input_rgb_image)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--visualize", required=False, default=True)
    opt = ap.parse_args()

    lab = Annotations(opt.dataset, opt.input, opt.visualize)
    lab.process_input(False)
    lab.generate_labels()
