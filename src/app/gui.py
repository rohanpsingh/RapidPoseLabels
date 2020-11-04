import os
import random
import cv2
from app.process import Process
from app.qt_root import MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets

class GUI(MainWindow):
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
        self.scene_dir_dict = {idx: item for idx, item in enumerate(list_of_scene_dirs)}

        #set up the Process object
        self.process = Process(dataset_path, output_dir, scale)

        #member variables
        self.cur_scene_id = -1
        self.scene_ply_paths = []
        self.scene_gui_input = []
        self.current_rgb_image = []
        self.current_dep_image = []
        self.current_ply_path = []
        self.current_cam_pos  = []

        #run the main loop
        app = QtWidgets.QApplication([])
        app.setApplicationName("RapidPoseLabelsApplication")
        super().__init__(window_title, self.width, self.height)

        # List the scenes in dock window
        for key, val in self.scene_dir_dict.items():
            item = QtWidgets.QListWidgetItem("scene {}: {}".format(key, os.path.join(self.dataset_path, val)))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)
            self.scene_list.addItem(item)

        super().show()
        app.exec_()

        #GUI mode flags
        self.build_model_mode  = False
        self.model_exist_mode  = False
        self.define_grasp_mode = False

    def read_current_scene(self):
        self.cur_scene_dir = self.scene_dir_dict[self.cur_scene_id]
        #read the entire list of image names and camera trajectory for current scene dir
        with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'associations.txt'), 'r') as file:
            self.img_name_list = file.readlines()
        with open(os.path.join(self.dataset_path, self.cur_scene_dir, 'camera.poses'), 'r') as file:
            self.cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]

    def new_point(self, point):
        if len(self.scene_gui_input)==self.num_keypoints:
            self.statusBar().showMessage("All keypoints selected")
            return
        # Add point to canvas
        self.canvas.current_points.append(point)
        # Add the point, current depth image and current camera pose to scene
        self.scene_gui_input.append(([point.x(), point.y()], self.current_dep_image, self.current_cam_pos))

        # Update status bar
        self.statusBar().showMessage("Keypoint added:{}".format((point.x(), point.y()), 5000))
        # Update keypoint dock widget
        self.keypoint_list.addItem("KP {}: {}".format(len(self.scene_gui_input), (point.x(), point.y())))

    def btn_func_skip(self):
        self.new_point(QtCore.QPoint(-1, -1))

    def btn_func_reset(self):
        """
        Function to reset the current scene.
        All selected keypoints for the current scene will be cleared.
        """
        self.scene_gui_input = []
        self.canvas.last_clicked = None
        self.canvas.current_points = []
        self.canvas.locked_points = []
        self.canvas.update()

        # Update status bar
        self.statusBar().showMessage("Cleared all labels in this scene", 5000)
        # Update keypoint dock widget
        self.keypoint_list.clear()

    def btn_func_load(self, value):
        """
        Function to load an image from the current scene dir.
        """
        if value==-1:
            value = random.randrange(1000)
            self.load_slider.setValue(value)
        read_indx = int(int(value)*(len(self.img_name_list[:-1]))/1000)

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

        # Get projection of keypoints on current image
        matched = self.process.get_projection(self.scene_gui_input, self.current_cam_pos)

        # Update canvas
        pixmap = QtGui.QPixmap(rgb_im_path)
        self.canvas.loadPixmap(pixmap)
        self.canvas.locked_points = [QtCore.QPoint(point[0], point[1]) for point in matched]
        self.canvas.update()

        # Configure state of widgets
        self.skip_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.next_scene_btn.setEnabled(True)
        self.display_btn.setEnabled(True)

        # Update status bar
        self.statusBar().showMessage("Loaded image\nfrom scene {}".format(self.cur_scene_dir), 5000)

    def btn_func_prev_scene(self):
        """
        Function to move to prev scene
        """
        return

    def btn_func_next_scene(self):
        """
        Function to lock labeled keypoints in current scene
        and move to next scene.
        """
        while len(self.scene_gui_input) != self.num_keypoints:
            self.btn_func_skip()

        #keypoint pixel coords, depth images and camera poses for this scene
        self.process.list_of_scenes.append(self.scene_gui_input)
        #and scene ply
        self.scene_ply_paths.append(self.current_ply_path)

        self.scene_gui_input = []
        try:
            self.cur_scene_id +=1
            self.read_current_scene()
            # Configure state of widgets
            self.load_btn.setEnabled(True)
            self.load_slider.setEnabled(True)
            # Update status bar
            self.statusBar().showMessage("Moving to scene:\n{}".format(self.cur_scene_dir), 5000)
            # Update keypoint dock widget
            self.keypoint_list.clear()
            # Update scene dock widget
            item = self.scene_list.item(self.cur_scene_id)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEnabled)
        except KeyError:
            # Confugre state of wigets
            self.load_btn.setEnabled(False)
            self.load_slider.setEnabled(False)
            # Update status bar
            self.statusBar().showMessage("Done all scenes.Please quit")
            # Update keypoint dock widget
            self.keypoint_list.clear()

        # Reset the canvas
        self.canvas.loadPixmap(QtGui.QPixmap())

        # Configure state of widgets
        self.skip_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.next_scene_btn.setEnabled(False)
        self.prev_scene_btn.setEnabled(True)
        self.compute_btn.setEnabled(True)
        self.display_btn.setEnabled(False)

    def btn_func_compute(self):
        """
        Function to perform the optimization/procrustes step.
        """
        #2D-to-3D conversion
        keypoint_pos = self.process.convert_2d_to_3d(self.process.list_of_scenes)
        #transform points to origins of respective scene
        self.process.transform_points(keypoint_pos, self.process.list_of_scenes)
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
        keypoint_pos = self.process.convert_2d_to_3d([self.scene_gui_input])
        #transform points to origins of respective scene
        self.process.transform_points(keypoint_pos, [self.scene_gui_input])
        #visualize the labeled keypoints in scene
        obj = []
        if not self.process.scene_kpts==[]:
            obj = self.process.scene_kpts[0].transpose()
        self.process.visualize_points_in_scene(self.current_ply_path, obj)

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
            self.skip_btn.setEnabled(False)
            self.next_scene_btn.setEnabled(False)
