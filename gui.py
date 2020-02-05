import numpy as np
import random
import tkinter as tk
import PIL.Image, PIL.ImageTk
import cv2
import os

class App:
    def __init__(self, window_title, dataset_path, tot_num_keypoints):

        self.dataset_path = dataset_path
        self.tot_num_keypoints = tot_num_keypoints
        self.dataset_scenes = []

        self.scene_dir_itr = iter(os.listdir(self.dataset_path))
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
        self.quit_btn = tk.Button(self.tkroot, text="Quit", 
                                  width=widget_wd, 
                                  state=tk.NORMAL,
                                  command=self.btn_func_quit)
        self.load_btn.grid(column=1, row=0, padx=10, pady=10)
        self.next_btn.grid(column=1, row=1, padx=10)
        self.skip_btn.grid(column=1, row=2, padx=10)
        self.reset_btn.grid(column=1, row=3, padx=10)
        self.scene_btn.grid(column=1, row=4, padx=10)
        self.quit_btn.grid(column=1, row=5, padx=10)

        #message box
        self.msg_box = tk.Label(self.tkroot, 
                                text="Please load an image",
                                height = 5, width=widget_wd, 
                                bg='blue', fg='white')
        self.dat_box = tk.Label(self.tkroot, 
                                text="Current keypoint list:\n{}".format(self.scene_kpts_2d), 
                                height = 10, width=widget_wd, 
                                bg='blue', fg='white')
        self.msg_box.grid(column=1, row=6, padx=10)
        self.dat_box.grid(column=1, row=7, rowspan=3, padx=10, pady=10)

        # Create a canvas that can fit the image
        self.canvas = tk.Canvas(self.tkroot, width = self.width, height = self.height)
        self.canvas.grid(column=0, row=0, rowspan=10, padx=10, pady=10)
        self.canvas.create_rectangle(0, 0, self.width, self.height, fill='blue')
        
        self.tkroot.mainloop()

    def display_cv_image(self, img):
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def check_if_loaded(self):
        if not self.image_loaded:
            self.msg_box.configure(text = "Image not loaded.")
            return False
        else:
            return True

    def add_kp_to_list(self, kp):
        if len(self.scene_kpts_2d)==self.tot_num_keypoints:
            self.msg_box.configure(text = "all keypoints selected")
            return
        if kp==[]: kp = [-1, -1]
        cv2.circle(self.cv_img, tuple(kp), 5, (0,0,255), -1)
        self.display_cv_image(self.cv_img)
        self.scene_kpts_2d.append(kp)
        self.msg_box.configure(text = "Keypoint added:\n{}".format(kp))
        self.dat_box.configure(text = "Current keypoint list:\n{}".format(np.asarray(self.scene_kpts_2d)))
        self.clicked_pixel = []

    def buttonClick(self, event):
        tmp = self.cv_img.copy()
        cv2.circle(tmp, (event.x, event.y), 3, (0,255,0), -1)
        self.display_cv_image(tmp)
        self.clicked_pixel = [event.x, event.y]

    def doubleButtonClick(self, event):
        tmp = self.cv_img.copy()
        cv2.circle(tmp, (event.x, event.y), 3, (0,255,0), -1)
        self.display_cv_image(tmp)
        self.clicked_pixel = [event.x, event.y]
        if not self.check_if_loaded(): return
        self.add_kp_to_list(self.clicked_pixel)

    def btn_func_load(self):
        with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'associations.txt'), 'r') as file:
            img_name_list = file.readlines()

        random_pair = random.choice(img_name_list).split()
        dep_im_path = os.path.join(self.dataset_path, self.cur_scene_dir, random_pair[1])
        rgb_im_path = os.path.join(self.dataset_path, self.cur_scene_dir, random_pair[3])
        input_image = cv2.imread(rgb_im_path)
        input_image = cv2.resize(input_image, (self.width, self.height))
        self.in_img = input_image.copy()
        self.cv_img = input_image.copy()
        self.display_cv_image(self.cv_img)
        self.canvas.bind('<Button-1>', self.buttonClick)
        self.canvas.bind('<Double-Button-1>', self.doubleButtonClick)
        self.msg_box.configure(text = "Loaded image\nfrom scene {}".format(self.cur_scene_dir))
        self.image_loaded=True
        self.next_btn.configure(state=tk.NORMAL)
        self.skip_btn.configure(state=tk.NORMAL)
        self.reset_btn.configure(state=tk.NORMAL)
        self.scene_btn.configure(state=tk.NORMAL)

    def btn_func_next(self):
        if not self.check_if_loaded(): return
        self.add_kp_to_list(self.clicked_pixel)

    def btn_func_skip(self):
        if not self.check_if_loaded(): return
        self.add_kp_to_list([])

    def btn_func_reset(self):
        if not self.check_if_loaded(): return
        self.display_cv_image(self.in_img)
        self.cv_img = self.in_img.copy()
        self.clicked_pixel = []
        self.scene_kpts_2d = []
        self.msg_box.configure(text = "Scene reset")
        self.dat_box.configure(text = "Current keypoint list:\n{}".format(np.asarray(self.scene_kpts_2d)))
        
    def btn_func_scene(self):
        if not self.check_if_loaded(): return
        while (len(self.scene_kpts_2d) != self.tot_num_keypoints):
            self.add_kp_to_list([])
        self.dataset_scenes.append(self.scene_kpts_2d)
        self.clicked_pixel = []
        self.scene_kpts_2d = []
        try:
            self.cur_scene_dir = next(self.scene_dir_itr)
            self.msg_box.configure(text = "moving to scene:\n{}".format(self.cur_scene_dir))
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

    def btn_func_quit(self):
        self.tkroot.destroy()

