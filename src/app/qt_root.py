from PyQt5 import QtCore, QtGui, QtWidgets
from app.widgets import QCanvas
from app.widgets import QToolMenu

ICONS_DIR = "./app/icons/"

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, window_title, img_width, img_height):
        super(MainWindow, self).__init__()

        self.width = img_width
        self.height = img_height
        
        # Set up the MainWindow object at center of screen
        self.setWindowTitle(window_title)
        self.setGeometry(0,0,1200,600)
        centerPoint = QtWidgets.QDesktopWidget().availableGeometry().center()
        qtRectangle = self.frameGeometry()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

        # Initial layout
        self.main_layout()

    def btn_func_load(self):
        pass

    def btn_func_skip(self):
        pass

    def btn_func_reset(self):
        pass

    def btn_func_next_scene(self):
        pass

    def btn_func_prev_scene(self):
        pass

    def btn_func_compute(self):
        pass

    def btn_func_display(self):
        pass

    def btn_func_create(self):
        pass

    def btn_func_choose(self):
        pass

    def btn_func_grasping(self):
        pass

    def btn_func_quit(self):
        pass

    def act_func_zoom_in(self):
        self.scaleImage(1.1)

    def act_func_zoom_out(self):
        self.scaleImage(0.9)

    def act_func_normal_size(self):
        self.canvas.scale = 1.0
        self.canvas.adjustSize()

    def main_layout(self):
        # Canvas
        self.canvas = QCanvas()
        self.canvas.newPoint.connect(self.new_point)
        self.canvas.zoomRequest.connect(self.zoom_request)
        self.canvas.scrollRequest.connect(self.scroll_request)

        # Scroll area
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setWidget(self.canvas)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setVisible(True)

        # Set central widget
        self.setCentralWidget(self.scrollArea)

        # Action (zoom in)
        self.zoom_in_act = QtWidgets.QAction("Zoom &In (10%)", self, shortcut="Ctrl++")
        self.zoom_in_act.triggered.connect(self.act_func_zoom_in)
        self.zoom_in_act.setEnabled(True)

        # Action (zoom out)
        self.zoom_out_act = QtWidgets.QAction("Zoom &Out (10%)", self, shortcut="Ctrl+-")
        self.zoom_out_act.triggered.connect(self.act_func_zoom_out)
        self.zoom_out_act.setEnabled(True)

        # Action (normal size)
        self.normal_size_act = QtWidgets.QAction("&Normal Size", self, shortcut="Ctrl+0")
        self.normal_size_act.triggered.connect(self.act_func_normal_size)
        self.normal_size_act.setEnabled(True)

        # Button
        self.load_btn = QtWidgets.QAction('Load')
        self.load_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.load_btn.triggered.connect(lambda x : self.btn_func_load(-1))
        self.load_btn.setStatusTip("Click here to load a new image from current scene.")
        self.load_btn.setEnabled(False)

        # Button
        self.skip_btn = QtWidgets.QAction('Skip keypoint')
        self.skip_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.skip_btn.triggered.connect(self.btn_func_skip)
        self.skip_btn.setStatusTip("Click here if keypoint is not visible in current scene.")
        self.skip_btn.setEnabled(False)

        # Button
        self.reset_btn = QtWidgets.QAction('Scene reset')
        self.reset_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.reset_btn.triggered.connect(self.btn_func_reset)
        self.reset_btn.setStatusTip("Click here to clear all labels in current scene.")
        self.reset_btn.setEnabled(False)
        
        # Button
        self.next_scene_btn = QtWidgets.QAction('Next scene')
        self.next_scene_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.next_scene_btn.triggered.connect(self.btn_func_next_scene)
        self.next_scene_btn.setStatusTip("Click here to confirm labels in current scene and move to next.")
        self.next_scene_btn.setEnabled(True)

        # Button
        self.prev_scene_btn = QtWidgets.QAction('Previous scene')
        self.prev_scene_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.prev_scene_btn.triggered.connect(self.btn_func_prev_scene)
        self.prev_scene_btn.setStatusTip("Click here to go to previous scene.")
        self.prev_scene_btn.setEnabled(False)

        # Button
        self.compute_btn = QtWidgets.QAction('Compute')
        self.compute_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.compute_btn.triggered.connect(self.btn_func_compute)
        self.compute_btn.setStatusTip("Click here to solve the optimization problem.")
        self.compute_btn.setEnabled(False)

        # Button
        self.display_btn = QtWidgets.QAction('Visualize')
        self.display_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.display_btn.triggered.connect(self.btn_func_display)
        self.display_btn.setStatusTip("Click here to visualize the labeled points in 3D.")
        self.display_btn.setEnabled(False)

        # Button
        self.quit_btn = QtWidgets.QAction('Quit')
        self.quit_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.quit_btn.triggered.connect(self.btn_func_quit)
        self.quit_btn.setStatusTip("Click here to close the GUI (inputs will not be saved).")
        self.quit_btn.setEnabled(True)

        # Slider
        self.load_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.load_slider.setMinimum(0)
        self.load_slider.setMaximum(999)
        #self.load_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.load_slider.sliderMoved.connect(self.btn_func_load)
        self.load_slider.setStatusTip("Slide to load images from the scene.")
        self.load_slider.setEnabled(False)

        # Toolbar on the right
        right_toolbar = QToolMenu("Tool Menu")
        right_toolbar.setIconSize(QtCore.QSize(32,32))
        self.addToolBar(QtCore.Qt.RightToolBarArea, right_toolbar)
        right_toolbar.addAction(self.load_btn)
        right_toolbar.addWidget(self.load_slider)
        right_toolbar.addAction(self.skip_btn)
        right_toolbar.addAction(self.reset_btn)
        right_toolbar.addAction(self.next_scene_btn)
        right_toolbar.addAction(self.prev_scene_btn)
        right_toolbar.addAction(self.compute_btn)
        right_toolbar.addAction(self.display_btn)
        right_toolbar.addAction(self.quit_btn)
        right_toolbar.addAction(self.zoom_in_act)
        right_toolbar.addAction(self.zoom_out_act)
        right_toolbar.addAction(self.normal_size_act)
        
        # Menus on the left
        self.keypoint_list = QtWidgets.QListWidget()
        self.keypoint_dock = QtWidgets.QDockWidget("Keypoint List")
        self.keypoint_dock.setWidget(self.keypoint_list)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.keypoint_dock)

        self.scene_list = QtWidgets.QListWidget()
        #self.scene_list.setEnabled(False)
        self.scene_dock = QtWidgets.QDockWidget("Scenes")
        self.scene_dock.setWidget(self.scene_list)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.scene_dock)

        # Create a status bar
        self.setStatusBar(QtWidgets.QStatusBar(self))

    def init_layout(self):

        # Button for building a new model
        button1 = QtWidgets.QPushButton('Create a new model')
        button1.toggle()
        button1.clicked.connect(self.btn_func_create)
        button1.setStatusTip("Click here if you do not have a model for you object.")
        button1.setFixedSize(300,50)

        # Button for choosing an existing model file
        button2 = QtWidgets.QPushButton('Use existing model')
        button2.toggle()
        button2.clicked.connect(self.btn_func_choose)
        button2.setStatusTip("Click here if you already have a *.pp model for your object.")
        button2.setFixedSize(300,50)

        # Button for defining a grasping point
        button3 = QtWidgets.QPushButton('Define grasp point')
        button3.toggle()
        button3.clicked.connect(self.btn_func_grasping)
        button3.setStatusTip("Click here to localize a grasp pose for a known object.")
        button3.setFixedSize(300,50)
        
        # Create the layout and place widgets in
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Create a status bar
        self.setStatusBar(QtWidgets.QStatusBar(self))

    def scroll_request(self, delta, orientation):
        # Scroll bars move with mouse wheel
        units = -delta * 0.1
        bar = self.scrollArea.verticalScrollBar()
        value = bar.value() + bar.singleStep() * units
        bar.setValue(value)

    def zoom_request(self, delta, pos):
        # Scale the canvas
        canvas_width_old = self.canvas.width()
        scale = 0.9 if delta < 0 else 1.1
        self.canvas.scale *= scale
        self.canvas.adjustSize()

        # Adjust the scroll bars to follow mouse position
        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), shift=x_shift)
            self.adjustScrollBar(self.scrollArea.verticalScrollBar(), shift=y_shift)

    def scaleImage(self, factor):
        self.canvas.scale *= factor
        self.canvas.adjustSize()
        self.canvas.update()

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor=factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor=factor)

    def adjustScrollBar(self, scrollBar, factor=1, shift=0):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))
        scrollBar.setValue(scrollBar.value() + shift)
