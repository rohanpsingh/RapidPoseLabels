import argparse
import os
import cv2
import numpy as np
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
from itertools import combinations
import evaluate3d

class Annotations:
    def __init__(self, dataset_path, input_arr_path, output_dir, visualize):

        self.dataset_path = dataset_path
        self.input_array  = np.load(input_arr_path)
        self.output_dir   = output_dir
        self.visualize    = visualize

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

    def writeToDisk(self, img, imgpts, center, scale, i):
        if not os.path.isdir(os.path.join(self.output_dir, 'bboxes')): os.makedirs(os.path.join(self.output_dir, 'bboxes'));
        if not os.path.isdir(os.path.join(self.output_dir, 'center')): os.makedirs(os.path.join(self.output_dir, 'center'));
        if not os.path.isdir(os.path.join(self.output_dir, 'scale')): os.makedirs(os.path.join(self.output_dir, 'scale'));
        if not os.path.isdir(os.path.join(self.output_dir, 'label')): os.makedirs(os.path.join(self.output_dir, 'label'));
        if not os.path.isdir(os.path.join(self.output_dir, 'frames')): os.makedirs(os.path.join(self.output_dir, 'frames'));

        bboxfile = open(os.path.join(self.output_dir, 'bboxes', 'frame_' + repr(i).zfill(5) + '.txt'), 'w')
        bboxfile.write('0\t' + repr(center[0]/self.width) + '\t' + repr(center[1]/self.height) + '\t' +
                       repr(scale*200/self.width) + '\t' + repr(scale*200/self.height) + '\n')
        bboxfile.close()
        centerfile = os.path.join(self.output_dir, 'center', 'center_' + repr(i).zfill(5) + '.txt')
        np.savetxt(centerfile, center)
        scalesfile = os.path.join(self.output_dir, 'scale', 'scales_' + repr(i).zfill(5) + '.txt')
        np.savetxt(scalesfile, np.asarray([scale]))
        labelfile = os.path.join(self.output_dir, 'label', 'label_' + repr(i).zfill(5) + '.txt')
        np.savetxt(labelfile, imgpts)
        cv2.imwrite(os.path.join(self.output_dir, 'frames', 'frame_' + repr(i).zfill(5) + '.jpg'), img)
        return

    def project3Dto2D(self, input_points, input_T, input_img, window, color=(255,0,0)):
        tf = input_T
        rvec,_ = cv2.Rodrigues(tf[:3, :3])
        tvec = tf[:3,3]
        imgpts,_ = cv2.projectPoints(input_points, rvec, tvec, self.cam_mat, None)
        keypts = np.transpose(np.asarray(imgpts), (1,0,2))[0]

        bbox_cn = keypts.mean(0)
        xmin, ymin = keypts.min(0)
        xmax, ymax = keypts.max(0)
        if xmin<0: xmin=0
        if ymin<0: ymin=0
        if xmax>=(self.width-1):  xmax=(self.width-1)
        if ymax>=(self.height-1): ymax=(self.height-1)
        bbox_cn = ((xmax+xmin)/2, (ymax+ymin)/2)
        bbox_sd = max((xmax-xmin), (ymax-ymin))*self.bbox_scale

        if self.visualize:
            for p in keypts:
                cv2.circle(input_img, tuple(map(int, p)), 5, color, -1)
            cv2.rectangle(input_img,
                          (int(bbox_cn[0]-(bbox_sd/2)), int(bbox_cn[1]-(bbox_sd/2))),
                          (int(bbox_cn[0]+(bbox_sd/2)), int(bbox_cn[1]+(bbox_sd/2))),
                          (0,255,0), 2)
            cv2.imshow(window, input_img)
            cv2.waitKey(50)
        return keypts, bbox_cn, bbox_sd/200.0
        
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

    def generate_labels(self):
        counter=0
        for data_dir, (cur_scene_dir, sce_T) in enumerate(zip(self.list_of_scene_dirs, self.scene_tfs)):
            with open(os.path.join(self.dataset_path, cur_scene_dir, 'associations.txt'), 'r') as file:
                img_name_list = file.readlines()
            with open(os.path.join(self.dataset_path, cur_scene_dir, 'camera.poses'), 'r') as file:
                cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]
            zipped_list = list(zip(img_name_list, cam_pose_list))[::10]
            for idx, (img_name, cam_pose) in enumerate(zipped_list):
                img_name = img_name.split()
                rgb_im_path = os.path.join(self.dataset_path, cur_scene_dir, img_name[3])
                input_rgb_image = cv2.resize(cv2.imread(rgb_im_path), (self.width, self.height))
                cam_T = tfa.compose(np.asarray(cam_pose[:3]), tfq.quat2mat(np.asarray([cam_pose[-1]] + cam_pose[3:-1])), np.ones(3))
                estpts, estcen, estsca = self.project3Dto2D(self.object_model, np.dot(np.linalg.inv(cam_T), sce_T), input_rgb_image.copy(), "win1", (0,0,255))
                self.writeToDisk(input_rgb_image, estpts, estcen, estsca, counter)
                counter+=1
            print("Created {} labeled samples from dataset {} (with {} raw samples).".format(len(zipped_list), data_dir, len(img_name_list)))
        return counter

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--visualize", action='store_true')
    opt = ap.parse_args()

    lab = Annotations(opt.dataset, opt.input, opt.output, opt.visualize)
    lab.process_input(False)
    counter = lab.generate_labels()
    print("Total number of samples generated: {}".format(counter))
