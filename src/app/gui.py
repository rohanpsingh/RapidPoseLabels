import os
import random
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from app.process import Process
from app.qt_root import MainWindow
from app.dataclasses import Label, Scene

class GUI(MainWindow):
    def __init__(self, window_title, output_dir, num_keypoints, scale=1000):
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

        # Get input arguments
        self.output_dir = output_dir
        self.num_keypoints = num_keypoints
        self.scale = scale

        # Counter for scenes
        self._count = -1

        self.scenes = []
        self.current_rgb_image = []
        self.current_dep_image = []
        self.current_cam_pos  = []

        # Set mode
        self.build_model_mode = True

        #run the main loop
        app = QtWidgets.QApplication([])
        app.setApplicationName("RapidPoseLabelsApplication")
        super().__init__(window_title, self.width, self.height)
        super().show()
        app.exec_()

    def act_func_load_data(self):
        """
        Function to choose the root dataset directory.
        """
        # Open QFileDialog
        dialog = QtWidgets.QFileDialog(caption="Load dataset")
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        if not dialog.exec_():
            return
        self.dataset_path = dialog.selectedFiles()[0]

        # Check if directory contains scenes
        try:
            if not 'camera.txt' in os.listdir(self.dataset_path):
                raise Exception
            for fn in os.listdir(self.dataset_path):
                dn = os.path.join(self.dataset_path, fn)
                if os.path.isdir(dn):
                    if not ('associations.txt' in os.listdir(dn) or
                            'camera.poses' in os.listdir(dn) or
                            'depth' in os.listdir(dn) or
                            'rgb' in os.listdir(dn)):
                        raise Exception
        except:
            print("Check format of root directory.")
            return

        # Get the list of scene directories
        list_of_scene_dirs = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        list_of_scene_dirs.sort()
        print("Number of scenes: ", len(list_of_scene_dirs))
        print("List of scenes: ", list_of_scene_dirs)

        # Initialize Process object
        self.process = Process(self.dataset_path, self.output_dir, self.scale)

        # Initialize Scene object
        self.scenes.clear()
        for idx, item in enumerate(list_of_scene_dirs):
            scene_obj = Scene(idx,
                              item,
                              os.path.join(self.dataset_path, item),
                              os.path.join(self.dataset_path, item, item + '.ply'),
                              [])
            self.scenes.append(scene_obj)

        # List the scenes in dock window
        self.scene_list.clear()
        for scene in self.scenes:
            item = QtWidgets.QListWidgetItem(
                "scene {}: {}".format(scene.index, scene.path)
            )
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)
            self.scene_list.addItem(item)

        # Configure state of widgets
        self.load_model_act.setEnabled(True)
        self.next_scene_btn.setEnabled(True)

    def act_func_load_model(self):
        """
        Function to load existing sparse, keypoint model.
        """
        # Open QFileDialog
        dialog = QtWidgets.QFileDialog(caption="Load model")
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setNameFilter("Sparse model files (*.pp *.txt)")
        if not dialog.exec_():
            return
        filename = dialog.selectedFiles()
        self.process.sparse_model_file = filename

        # Set mode
        self.build_model_mode = False
        self.mode_qlabel.setText("Mode: Use existing model")

    def btn_func_skip(self):
        """
        Function to skip a keypoint.
        (-1, -1) is added to the list of keypoints.
        """
        self.new_point(QtCore.QPoint(-1, -1))

    def btn_func_reset(self):
        """
        Function to reset the current scene.
        All selected keypoints for the current scene will be cleared.
        """
        self.scenes[self._count].labels = []
        self.canvas.last_clicked = None
        self.canvas.current_points = []
        self.canvas.locked_points = []
        self.canvas.update()

        # Update status bar and dock
        self.statusBar().showMessage("Cleared all labels in this scene", 5000)
        super().update_keypoint_dock()

    def btn_func_load(self, value):
        """
        Function to load an image from the current scene dir.
        """
        if value==-1:
            value = random.randrange(1000)
            self.load_slider.setValue(value)
        read_indx = int(int(value)*(len(self.img_name_list[:-1]))/1000)

        # Read the RGB, Depth image and Camera pose at the index
        read_pair = (self.img_name_list[read_indx]).split()
        dep_im_path = os.path.join(self.scenes[self._count].path, read_pair[1])
        rgb_im_path = os.path.join(self.scenes[self._count].path, read_pair[3])
        self.current_rgb_image = cv2.resize(cv2.cvtColor(cv2.imread(rgb_im_path), cv2.COLOR_BGR2RGB), (self.width, self.height))
        self.current_dep_image = cv2.resize(cv2.imread(dep_im_path, cv2.IMREAD_ANYDEPTH), (self.width, self.height))
        self.current_cam_pos = self.cam_pose_list[read_indx]

        # Get projection of keypoints on current image
        matched = self.process.get_projection(
            self.scenes[self._count].labels, self.current_cam_pos
        )

        # Update canvas
        pixmap = QtGui.QPixmap(rgb_im_path)
        self.canvas.loadPixmap(pixmap)
        self.canvas.locked_points = [QtCore.QPoint(point[0], point[1]) for point in matched]
        self.canvas.update()
        # Configure state of widgets
        self.skip_btn.setEnabled(True)
        self.display_btn.setEnabled(True)
        # Update status bar
        self.statusBar().showMessage(
            "Loaded image\nfrom scene {}".format(self.scenes[self._count].name), 5000
        )

    def btn_func_prev_scene(self):
        """
        Function to move to prev scene
        """
        self._count-=1
        success = self.load_scene()
        if not success:
            self.prev_scene_btn.setEnabled(False)
        # Configure the state of widgets
        self.next_scene_btn.setEnabled(True)

    def btn_func_next_scene(self):
        """
        Function to lock labeled keypoints in current scene
        and move to next scene.
        """
        # Fill the scene with dummy labels
        if 0 <= self._count < len(self.scenes):
            while len(self.scenes[self._count].labels) != self.num_keypoints:
                self.btn_func_skip()

        self._count+=1
        success = self.load_scene()
        if not success:
            self.next_scene_btn.setEnabled(False)
        # Configure state of widgets
        self.skip_btn.setEnabled(False)
        self.prev_scene_btn.setEnabled(True)
        self.display_btn.setEnabled(False)

    def btn_func_compute(self):
        """
        Function to perform the optimization/procrustes step.
        """
        #2D-to-3D conversion
        scenes_labels = [scene.labels for scene in self.scenes][:self._count]
        keypoint_pos = self.process.convert_2d_to_3d(scenes_labels)
        # Transform points to origins of respective scene
        self.process.transform_points(keypoint_pos, scenes_labels)
        # Final computation step
        if self.build_model_mode:
            res, obj = self.process.compute(False)
            # Visualize the generated object model in first scene
            self.process.visualize_points_in_scene(self.scenes[0].mesh, obj)
        else:
            res, obj = self.process.compute(True)

    def btn_func_display(self):
        """
        Function to convert the labeled 2D keypoitns into 3D positions
        and visualize them in the scene.
        """
        #2D-to-3D conversion
        keypoint_pos = self.process.convert_2d_to_3d([self.scenes[self._count].labels])
        #transform points to origins of respective scene
        self.process.transform_points(keypoint_pos, [self.scenes[self._count].labels])
        #visualize the labeled keypoints in scene
        obj = []
        if not self.process.scene_kpts==[]:
            obj = self.process.scene_kpts[0].transpose()
        self.process.visualize_points_in_scene(self.scenes[self._count].mesh, obj)


    ########### Utility #############
    ########## functions ############

    def read_current_scene(self):
        try:
            if not (0 <= self._count < len(self.scenes)):
                raise IndexError
            cur_scene_dir = self.scenes[self._count].path
            # Read image names
            with open(os.path.join(cur_scene_dir, 'associations.txt'), 'r') as file:
                self.img_name_list = file.readlines()
            # Read camera trajectory
            with open(os.path.join(cur_scene_dir, 'camera.poses'), 'r') as file:
                self.cam_pose_list = [list(map(float, line.split()[1:])) for line in file.readlines()]
        except:
            return False
        return True

    def new_point(self, point):
        if len(self.scenes[self._count].labels)==self.num_keypoints:
            self.statusBar().showMessage("All keypoints selected")
            return

        # Add point to canvas
        self.canvas.current_points.append(point)
        # Add the point, current depth image and current camera pose to scene
        label = Label([point.x(), point.y()], self.current_dep_image, self.current_cam_pos)
        self.scenes[self._count].labels.append(label)
        # Configure state of widgets
        self.reset_btn.setEnabled(True)
        self.compute_btn.setEnabled(True)
        # Update status bar and dock
        self.statusBar().showMessage("Keypoint added:{}".format((point.x(), point.y()), 5000))
        super().update_keypoint_dock()

    def load_scene(self):
        # Read the current scene
        success = self.read_current_scene()
        if success:
            # Configure state of widgets
            self.load_btn.setEnabled(True)
            self.load_slider.setEnabled(True)
            self.reset_btn.setEnabled(bool(len(self.scenes[self._count].labels)))
            # Update status bar and dock
            self.statusBar().showMessage(
                "Moving to scene:\n{}".format(self.scenes[self._count].name), 5000
            )
            super().update_keypoint_dock()
            item = self.scene_list.item(self._count)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEnabled)
            self.scene_list.setCurrentRow(self._count)
        else:
            # Configure state of wigets
            self.load_btn.setEnabled(False)
            self.load_slider.setEnabled(False)
            self.reset_btn.setEnabled(False)
            # Update status bar and dock
            super().update_keypoint_dock()
            self.scene_list.setCurrentRow(-1)

        # Reset the canvs
        self.canvas.loadPixmap(QtGui.QPixmap())
        return success
