import os
import random
import tkinter as tk
from tkinter import filedialog
import numpy as np
import PIL.Image, PIL.ImageTk
import cv2
from app.process import Process
from app.tk_root import TkRoot

class GUI(TkRoot):
    def __init__(self, window_title, dataset_path, output_dir, num_keypoints, scale=1000):
        """
        Constructor for the GUI class.
        Input arguments:
        window_title   - title name for the GUI
        dataset_path   - path to root dataset directory
        output_dir     - path to output directory
        num_keypoints  - total number of keypoints on the object
                         (decided by the user)
        scale          - scale parameter of the RGB-D sensor
                         (1000 for Intel RealSense D435)
        """
        #assumes images are 640x480
        self.width = 640
        self.height = 480

        #get input arguments
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.num_keypoints = num_keypoints

        #get the list of scene directories
        list_of_scene_dirs = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        list_of_scene_dirs.sort()
        print("Number of scenes: ", len(list_of_scene_dirs))
        print("List of scenes: ", list_of_scene_dirs)
        self.scene_dir_itr = iter(list_of_scene_dirs)
        self.cur_scene_dir = next(self.scene_dir_itr)

        #set up the Process object
        self.process = Process(dataset_path, output_dir, scale)

        #member variables
        self.scene_ply_paths = []
        self.scene_kpts_2d = []
        self.clicked_pixel = []
        self.image_loaded=False
        self.current_rgb_img = []
        self.input_rgb_image = []
        self.input_dep_image = []
        self.current_ply_path = []
        self.current_cam_pos  = []

        #run the main loop
        super().__init__(window_title, self.width, self.height)
        super().tkroot_main_loop()

    def display_cv_image(self, img):
        self.display_image = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
        self.canvas.create_image(0, 0, image=self.display_image, anchor=tk.NW)

    def add_kp_to_list(self, kp):
        if len(self.scene_kpts_2d)==self.num_keypoints:
            self.msg_box.configure(text = "all keypoints selected")
            return
        if kp==[]: kp = [-1, -1]
        cv2.circle(self.current_rgb_img, tuple(kp), 5, (0,0,255), -1)
        self.display_cv_image(self.current_rgb_img)
        self.scene_kpts_2d.append(kp)
        self.msg_box.configure(text = "Keypoint added:\n{}".format(kp))
        self.dat_box.configure(text = "Current keypoint list:\n{}".format(np.asarray(self.scene_kpts_2d)))
        self.clicked_pixel = []

    def button_click(self, event):
        tmp = self.current_rgb_img.copy()
        cv2.circle(tmp, (event.x, event.y), 3, (0,255,0), -1)
        self.display_cv_image(tmp)
        self.clicked_pixel = [event.x, event.y]

    def double_button_click(self, event):
        tmp = self.current_rgb_img.copy()
        cv2.circle(tmp, (event.x, event.y), 3, (0,255,0), -1)
        self.display_cv_image(tmp)
        self.clicked_pixel = [event.x, event.y]
        self.add_kp_to_list(self.clicked_pixel)

    def btn_func_skip(self):
        self.add_kp_to_list([])

    def btn_func_reset(self):
        """
        Function to reset the current scene.
        All selected keypoints for the current scene will be cleared.
        """
        self.display_cv_image(self.input_rgb_image)
        self.current_rgb_img = self.input_rgb_image.copy()
        self.clicked_pixel = []
        self.scene_kpts_2d = []
        self.msg_box.configure(text = "Scene reset")
        self.dat_box.configure(text = "Current keypoint list:\n{}".format(np.asarray(self.scene_kpts_2d)))

    def btn_func_load(self):
        """
        Function to load a random image from the current scene dir.
        """
        #read the entire list of image names and camera trajectory for current scene dir
        with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'associations.txt'), 'r') as file:
            img_name_list = file.readlines()
        with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'camera.poses'), 'r') as file:
            cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]

        #read a random RGB and corresponding depth image
        random_indx = random.randrange(len(img_name_list[:-1]))
        random_pair = (img_name_list[random_indx]).split()
        dep_im_path = os.path.join(self.dataset_path, self.cur_scene_dir, random_pair[1])
        rgb_im_path = os.path.join(self.dataset_path, self.cur_scene_dir, random_pair[3])
        self.input_rgb_image = cv2.resize(cv2.cvtColor(cv2.imread(rgb_im_path), cv2.COLOR_BGR2RGB), (self.width, self.height))
        self.input_dep_image = cv2.resize(cv2.imread(dep_im_path, cv2.IMREAD_ANYDEPTH), (self.width, self.height))
        #read the corresponding camera pose from the trajectory
        self.current_cam_pos = cam_pose_list[random_indx]
        #and the ply path of the associated scene (required for visualizations)
        self.current_ply_path = os.path.join(self.dataset_path, self.cur_scene_dir, self.cur_scene_dir + '.ply')

        #create a copy (to reset and redraw at any time)
        self.current_rgb_img = self.input_rgb_image.copy()

        #configure state of buttons and canvas
        self.display_cv_image(self.current_rgb_img)
        self.canvas.bind('<Button-1>', self.button_click)
        self.canvas.bind('<Double-Button-1>', self.double_button_click)
        self.msg_box.configure(text = "Loaded image\nfrom scene {}".format(self.cur_scene_dir))
        self.image_loaded=True
        self.skip_btn.configure(state=tk.NORMAL)
        self.reset_btn.configure(state=tk.NORMAL)
        self.scene_btn.configure(state=tk.NORMAL)

    def btn_func_scene(self):
        """
        Function to lock labeled keypoints in current scene
        and move to next scene.
        """
        while len(self.scene_kpts_2d) != self.num_keypoints:
            self.add_kp_to_list([])

        #rgb image, depth image and list of 2D keypoints for this scene
        self.process.scene_imgs.append((self.input_rgb_image, self.input_dep_image, self.scene_kpts_2d))
        #the camera pose for the frame
        self.process.scene_cams.append(self.current_cam_pos)
        #and scene ply
        self.scene_ply_paths.append(self.current_ply_path)

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
        self.skip_btn.configure(state=tk.DISABLED)
        self.reset_btn.configure(state=tk.DISABLED)
        self.scene_btn.configure(state=tk.DISABLED)
        self.compute_btn.configure(state=tk.NORMAL)
        self.display_btn.configure(state=tk.NORMAL)

    def btn_func_compute(self):
        """
        Function to perform the optimization step.
        """
        #2D-to-3D conversion
        self.process.convert_2d_to_3d()
        #transform points to origins of respective scene
        self.process.transform_points()
        #final computation step
        res, obj = self.process.compute()
        #visualize the generated object model in first scene
        self.process.visualize_points_in_scene(self.scene_ply_paths[0], obj)

    def btn_func_display(self):
        """
        Function to convert the labeled 2D keypoitns into 3D positions
        and visualize them in the scene.
        """
        #2D-to-3D conversion
        self.process.convert_2d_to_3d()
        #transform points to origins of respective scene
        self.process.transform_points()
        #visualize the labeled keypoints in scene
        self.process.visualize_points_in_scene(self.current_ply_path, self.process.scene_kpts[-1].transpose())

    def btn_func_choose(self):
        file_name = filedialog.askopenfilename(initialdir=".", title="Browse sparse model file",
                                               filetypes=(("Text files","*.txt"),("all files","*.*")))
        self.process.sparse_model_file = file_name
        #display main layout
        self.main_layout()

    def btn_func_create(self):
        self.process.sparse_model_file = file_name
        #display main layout
        self.main_layout()
