import numpy as np
import random
import tkinter as tk
import PIL.Image, PIL.ImageTk
import cv2
import os
from app.process import Process

class GUI:
    def __init__(self, window_title, dataset_path, tot_num_keypoints):

        self.dataset_path = dataset_path
        self.tot_num_keypoints = tot_num_keypoints
        self.pose = Process(dataset_path, 1000)

        list_of_scene_dirs = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        self.scene_dir_itr = iter(list_of_scene_dirs)
        self.cur_scene_dir = next(self.scene_dir_itr)
        self.clicked_pixel = []
        self.scene_kpts_2d = []
        self.image_loaded=False

        self.width = 640
        self.height = 480
        self.tkroot = tk.Tk()
        self.tkroot.title(window_title)
        self.tkroot.geometry('900x500')

        widget_wd = 25
        widget_ht = 3
        # Button definitions and placements
        self.load_btn = tk.Button(self.tkroot, text="Load New Image", 
                                  height=widget_ht, width=widget_wd, 
                                  state=tk.NORMAL,
                                  command=self.btn_func_load)
        self.next_btn = tk.Button(self.tkroot, text="Next KeyPt", 
                                  height=widget_ht, width=widget_wd, 
                                  state=tk.DISABLED,
                                  command=self.btn_func_next)
        self.skip_btn = tk.Button(self.tkroot, text="Skip KeyPt", 
                                  height=widget_ht, width=widget_wd, 
                                  state=tk.DISABLED,
                                  command=self.btn_func_skip)
        self.reset_btn = tk.Button(self.tkroot, text="Reset", 
                                   width=widget_wd, 
                                   state=tk.DISABLED,
                                   command=self.btn_func_reset)
        self.scene_btn = tk.Button(self.tkroot, text="Next Scene", 
                                   width=widget_wd, 
                                   state=tk.DISABLED,
                                   command=self.btn_func_scene)
        self.compute_btn = tk.Button(self.tkroot, text="Compute", 
                                     width=widget_wd, 
                                     state=tk.DISABLED,
                                     command=self.btn_func_compute)
        self.display_btn = tk.Button(self.tkroot, text="Visualize", 
                                     width=widget_wd, 
                                     state=tk.DISABLED,
                                     command=self.btn_func_display)
        self.quit_btn = tk.Button(self.tkroot, text="Quit", 
                                  width=widget_wd, 
                                  state=tk.NORMAL,
                                  command=self.btn_func_quit)
        self.load_btn.grid(column=1, row=0, padx=10)
        self.next_btn.grid(column=1, row=1, padx=10)
        self.skip_btn.grid(column=1, row=2, padx=10)
        self.reset_btn.grid(column=1, row=3, padx=10)
        self.scene_btn.grid(column=1, row=4, padx=10)
        self.compute_btn.grid(column=1, row=5, padx=10)
        self.display_btn.grid(column=1, row=6, padx=10)
        self.quit_btn.grid(column=1, row=7, padx=10)

        #message box
        self.msg_box = tk.Label(self.tkroot, 
                                text="Please load an image",
                                height = 5, width=widget_wd, 
                                bg='blue', fg='white')
        self.dat_box = tk.Label(self.tkroot, 
                                text="Current keypoint list:\n{}".format(self.scene_kpts_2d), 
                                height = 10, width=widget_wd, 
                                bg='blue', fg='white')
        self.msg_box.grid(column=1, row=8, padx=10)
        self.dat_box.grid(column=1, row=9, rowspan=3, padx=10, pady=10)

        # Create a canvas that can fit the image
        self.canvas = tk.Canvas(self.tkroot, width = self.width, height = self.height)
        self.canvas.grid(column=0, row=0, rowspan=10, padx=10, pady=10)
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill='blue')
        
        self.tkroot.mainloop()

    def display_cv_image(self, img):
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def add_kp_to_list(self, kp):
        if len(self.scene_kpts_2d)==self.tot_num_keypoints:
            self.msg_box.configure(text = "all keypoints selected")
            return
        if kp==[]: kp = [-1, -1]
        cv2.circle(self.current_rgb_img, tuple(kp), 5, (0,0,255), -1)
        self.display_cv_image(self.current_rgb_img)
        self.scene_kpts_2d.append(kp)
        self.msg_box.configure(text = "Keypoint added:\n{}".format(kp))
        self.dat_box.configure(text = "Current keypoint list:\n{}".format(np.asarray(self.scene_kpts_2d)))
        self.clicked_pixel = []
        
    def buttonClick(self, event):
        tmp = self.current_rgb_img.copy()
        cv2.circle(tmp, (event.x, event.y), 3, (0,255,0), -1)
        self.display_cv_image(tmp)
        self.clicked_pixel = [event.x, event.y]

    def doubleButtonClick(self, event):
        tmp = self.current_rgb_img.copy()
        cv2.circle(tmp, (event.x, event.y), 3, (0,255,0), -1)
        self.display_cv_image(tmp)
        self.clicked_pixel = [event.x, event.y]
        self.add_kp_to_list(self.clicked_pixel)

    def btn_func_load(self):
        with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'associations.txt'), 'r') as file:
            img_name_list = file.readlines()
        with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'camera.poses'), 'r') as file:
            cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]

        random_indx = random.randrange(len(img_name_list[:-1]))
        random_pair = (img_name_list[random_indx]).split()
        dep_im_path = os.path.join(self.dataset_path, self.cur_scene_dir, random_pair[1])
        rgb_im_path = os.path.join(self.dataset_path, self.cur_scene_dir, random_pair[3])
        self.input_rgb_image = cv2.resize(cv2.cvtColor(cv2.imread(rgb_im_path), cv2.COLOR_BGR2RGB), (self.width, self.height))
        self.input_dep_image = cv2.resize(cv2.imread(dep_im_path, cv2.IMREAD_ANYDEPTH), (self.width, self.height))

        self.current_rgb_img = self.input_rgb_image.copy()
        self.current_img_pos = cam_pose_list[random_indx]
        self.current_mesh    = os.path.join(self.dataset_path, self.cur_scene_dir, self.cur_scene_dir + '.ply')

        self.display_cv_image(self.current_rgb_img)
        self.canvas.bind('<Button-1>', self.buttonClick)
        self.canvas.bind('<Double-Button-1>', self.doubleButtonClick)
        self.msg_box.configure(text = "Loaded image\nfrom scene {}".format(self.cur_scene_dir))
        self.image_loaded=True
        self.next_btn.configure(state=tk.NORMAL)
        self.skip_btn.configure(state=tk.NORMAL)
        self.reset_btn.configure(state=tk.NORMAL)
        self.scene_btn.configure(state=tk.NORMAL)

    def btn_func_next(self):
        self.add_kp_to_list(self.clicked_pixel)

    def btn_func_skip(self):
        self.add_kp_to_list([])

    def btn_func_reset(self):
        self.display_cv_image(self.input_rgb_image)
        self.current_rgb_img = self.input_rgb_image.copy()
        self.clicked_pixel = []
        self.scene_kpts_2d = []
        self.msg_box.configure(text = "Scene reset")
        self.dat_box.configure(text = "Current keypoint list:\n{}".format(np.asarray(self.scene_kpts_2d)))
        
    def btn_func_scene(self):
        while (len(self.scene_kpts_2d) != self.tot_num_keypoints):
            self.add_kp_to_list([])
            
        self.pose.scene_imgs.append((self.input_rgb_image, self.input_dep_image, self.scene_kpts_2d))
        self.pose.scene_cams.append(self.current_img_pos)
        self.pose.scene_plys.append(self.current_mesh)

        self.clicked_pixel = []
        self.scene_kpts_2d = []
        try:
            self.cur_scene_dir = next(self.scene_dir_itr)
            self.msg_box.configure(text = "Moving to scene:\n{}".format(self.cur_scene_dir))
            self.dat_box.configure(text = "Current keypoint list:\n{}".format(np.asarray(self.scene_kpts_2d)))
        except:
            self.msg_box.configure(text = "Done all scenes.\nPlease quit")
            self.dat_box.configure(text = "")
            self.load_btn.configure(state=tk.DISABLED)
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill='blue')
        self.canvas.unbind('<Button-1>')
        self.canvas.unbind('<Double-Button-1>')
        self.image_loaded=False
        self.next_btn.configure(state=tk.DISABLED)
        self.skip_btn.configure(state=tk.DISABLED)
        self.reset_btn.configure(state=tk.DISABLED)
        self.scene_btn.configure(state=tk.DISABLED)
        self.compute_btn.configure(state=tk.NORMAL)
        self.display_btn.configure(state=tk.NORMAL)
        self.pose.convert_2Dto3D()
        print("Flags: ", self.pose.select_vec[-self.tot_num_keypoints:])

    def btn_func_compute(self):
        self.pose.convert_2Dto3D()
        self.pose.transform_points()
        res = self.pose.compute()

    def btn_func_display(self):
        self.pose.convert_2Dto3D()
        self.pose.transform_points()
        self.pose.visualize_scene(self.current_mesh, self.pose.scene_kpts[-1][np.newaxis,:])

    def btn_func_quit(self):
        self.tkroot.destroy()

