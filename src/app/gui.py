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
    def __init__(self, window_title, dataset_path, output_dir, num_keypoints, scale=1000, scenes=None):
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
        scenes         - names of scene dirs to read
        """
        #assumes images are 640x480
        self.width = 640
        self.height = 480

        #get input arguments
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.num_keypoints = num_keypoints

        #get the list of scene directories
        list_of_scene_dirs = scenes
        if scenes is None:
            list_of_scene_dirs = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        list_of_scene_dirs.sort()
        print("Number of scenes: ", len(list_of_scene_dirs))
        print("List of scenes: ", list_of_scene_dirs)
        self.scene_dir_itr = iter(list_of_scene_dirs)
        self.cur_scene_dir = next(self.scene_dir_itr)

        #read the entire list of image names and camera trajectory for current scene dir
        with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'associations.txt'), 'r') as file:
            self.img_name_list = file.readlines()
        with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'camera.poses'), 'r') as file:
            self.cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]

        #set up the Process object
        self.process = Process(dataset_path, output_dir, scale)

        #member variables
        self.scene_ply_paths = []
        self.scene_gui_input = []
        self.clicked_pixel = []
        self.image_loaded=False
        self.current_display = []
        self.current_rgb_image = []
        self.current_dep_image = []
        self.current_ply_path = []
        self.current_cam_pos  = []

        #run the main loop
        super().__init__(window_title, self.width, self.height)
        super().tkroot_main_loop()

        #GUI mode flags
        self.build_model_mode  = False
        self.model_exist_mode  = False
        self.define_grasp_mode = False

    def display_cv_image(self, img):
        self.display_image = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
        self.canvas.create_image(0, 0, image=self.display_image, anchor=tk.NW)

    def add_click_to_list(self, keypoint_pixel):
        if len(self.scene_gui_input)==self.num_keypoints:
            self.msg_box.configure(text = "all keypoints selected")
            return
        if keypoint_pixel==[]: keypoint_pixel = [-1, -1]
        cv2.circle(self.current_display, tuple(keypoint_pixel), 5, (0,0,255), -1)
        self.display_cv_image(self.current_display)
        #add the clicked pixel coords, current depth image and current camera pose to list
        self.scene_gui_input.append((keypoint_pixel, self.current_dep_image, self.current_cam_pos))
        list_of_kpt_pixels = [i[0] for i in self.scene_gui_input]
        self.msg_box.configure(text = "Keypoint added:\n{}".format(keypoint_pixel))
        self.dat_box.configure(text = "Current keypoint list:\n{}".format('\n'.join(map(str, list_of_kpt_pixels))))
        self.clicked_pixel = []

    def button_click(self, event):
        tmp = self.current_display.copy()
        cv2.circle(tmp, (event.x, event.y), 3, (0,255,0), -1)
        self.display_cv_image(tmp)
        self.clicked_pixel = [event.x, event.y]

    def double_button_click(self, event):
        tmp = self.current_display.copy()
        cv2.circle(tmp, (event.x, event.y), 3, (0,255,0), -1)
        self.display_cv_image(tmp)
        self.clicked_pixel = [event.x, event.y]
        self.add_click_to_list(self.clicked_pixel)

    def btn_func_skip(self):
        self.add_click_to_list([])

    def btn_func_reset(self):
        """
        Function to reset the current scene.
        All selected keypoints for the current scene will be cleared.
        """
        self.display_cv_image(self.current_rgb_image)
        self.current_display = self.current_rgb_image.copy()
        self.clicked_pixel = []
        self.scene_gui_input = []
        list_of_kpt_pixels = [i[0] for i in self.scene_gui_input]
        self.msg_box.configure(text = "Scene reset")
        self.dat_box.configure(text = "Current keypoint list:\n{}".format('\n'.join(map(str, list_of_kpt_pixels))))

    def btn_func_load(self, index=0):
        """
        Function to load an image from the current scene dir.
        """
        if index:
            read_indx = int(int(index)*(len(self.img_name_list[:-1]))/1000)
        else:
            read_indx = random.randrange(len(self.img_name_list[:-1]))
        #read an RGB and corresponding depth image at the index
        read_pair = (self.img_name_list[read_indx]).split()
        dep_im_path = os.path.join(self.dataset_path, self.cur_scene_dir, read_pair[1])
        rgb_im_path = os.path.join(self.dataset_path, self.cur_scene_dir, read_pair[3])
        self.current_rgb_image = cv2.resize(cv2.cvtColor(cv2.imread(rgb_im_path), cv2.COLOR_BGR2RGB), (self.width, self.height))
        self.current_dep_image = cv2.resize(cv2.imread(dep_im_path, cv2.IMREAD_ANYDEPTH), (self.width, self.height))
        #read the corresponding camera pose from the trajectory
        self.current_cam_pos = self.cam_pose_list[read_indx]
        #and the ply path of the associated scene (required for visualizations)
        self.current_ply_path = os.path.join(self.dataset_path, self.cur_scene_dir, self.cur_scene_dir + '.ply')

        #create a copy (to reset and redraw at any time)
        self.current_display = self.current_rgb_image.copy()
        #get projection of keypoints on current image
        matched = self.process.get_projection(self.scene_gui_input, self.current_cam_pos)
        for keypoint_pixel in matched:
            cv2.circle(self.current_display, tuple(map(int, keypoint_pixel)), 5, (0,0,255), -1)

        #configure state of buttons and canvas
        self.display_cv_image(self.current_display)
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
        while len(self.scene_gui_input) != self.num_keypoints:
            self.add_click_to_list([])

        #keypoint pixel coords, depth images and camera poses for this scene
        self.process.list_of_scenes.append(self.scene_gui_input)
        #and scene ply
        self.scene_ply_paths.append(self.current_ply_path)

        self.clicked_pixel = []
        self.scene_gui_input = []
        try:
            self.cur_scene_dir = next(self.scene_dir_itr)
            list_of_kpt_pixels = [i[0] for i in self.scene_gui_input]
            self.msg_box.configure(text = "Moving to scene:\n{}".format(self.cur_scene_dir))
            self.dat_box.configure(text = "Current keypoint list:\n{}".format('\n'.join(map(str, list_of_kpt_pixels))))
            #read the entire list of image names and camera trajectory for current scene dir
            with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'associations.txt'), 'r') as file:
                self.img_name_list = file.readlines()
            with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'camera.poses'), 'r') as file:
                self.cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]
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
        Function to perform the optimization/procrustes step.
        """
        #2D-to-3D conversion
        keypoint_pos = self.process.convert_2d_to_3d()
        #transform points to origins of respective scene
        self.process.transform_points(keypoint_pos)
        #final computation step
        if self.build_model_mode:
            res, obj = self.process.compute(False)
            #visualize the generated object model in first scene
            self.process.visualize_points_in_scene(self.scene_ply_paths[0], obj)
        elif self.model_exist_mode:
            res, obj = self.process.compute(True)
        elif self.define_grasp_mode:
            res, obj = self.process.define_grasp_point(self.scene_ply_paths[0])

    def btn_func_display(self):
        """
        Function to convert the labeled 2D keypoitns into 3D positions
        and visualize them in the scene.
        """
        #2D-to-3D conversion
        keypoint_pos = self.process.convert_2d_to_3d()
        #transform points to origins of respective scene
        self.process.transform_points(keypoint_pos)
        #visualize the labeled keypoints in scene
        self.process.visualize_points_in_scene(self.current_ply_path, self.process.scene_kpts[-1].transpose())

    def btn_func_choose(self):
        #set GUI mode
        self.build_model_mode  = False
        self.model_exist_mode  = True
        self.define_grasp_mode = False
        if self.num_keypoints<4:
            raise Exception("Number of keypoints is %d (should be >=4)" % self.num_keypoints)
        #browse sparse model file
        file_name = filedialog.askopenfilename(initialdir=".", title="Browse sparse model file",
                                               filetypes=(("Text files","*.txt"),("all files","*.*")))
        self.process.sparse_model_file = file_name
        #display main layout
        if file_name:
            self.main_layout()

    def btn_func_create(self):
        #set GUI mode
        self.build_model_mode  = True
        self.model_exist_mode  = False
        self.define_grasp_mode = False
        if self.num_keypoints<4:
            raise Exception("Number of keypoints is %d (should be >=4)" % self.num_keypoints)
        #display main layout
        self.main_layout()

    def btn_func_grasping(self):
        #set GUI mode
        self.build_model_mode  = False
        self.model_exist_mode  = False
        self.define_grasp_mode = True
        #browse sparse model file
        file_name = filedialog.askopenfilename(initialdir=".", title="Browse sparse model file",
                                               filetypes=(("Text files","*.txt"),("all files","*.*")))
        self.process.sparse_model_file = file_name
        #display main layout
        if file_name:
            self.num_keypoints = 2
            self.main_layout()
            self.skip_btn.configure(state=tk.DISABLED)
            self.scene_btn.configure(state=tk.DISABLED)
