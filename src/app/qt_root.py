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
        self.setGeometry(0,0,880,500)
        centerPoint = QtWidgets.QDesktopWidget().availableGeometry().center()
        qtRectangle = self.frameGeometry()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

        #image on canvas
        self.display_image = []

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
        self.tkroot.destroy()

    def main_layout(self):
        
        # Label (canvas)
        self.canvas = QCanvas(self.width, self.height)
        # Create the layout and place widgets in
        widget = self.canvas
        widget.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(widget)

        # Button
        self.load_btn = QtWidgets.QAction('Load')
        self.load_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.load_btn.triggered.connect(lambda:self.btn_func_load())
        self.load_btn.setStatusTip("Click here to load a new image from current scene.")
        self.load_btn.setEnabled(True)

        # Button
        self.skip_btn = QtWidgets.QAction('Skip keypoint')
        self.skip_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.skip_btn.triggered.connect(lambda:self.btn_func_skip())
        self.skip_btn.setStatusTip("Click here if keypoint is not visible in current scene.")
        self.skip_btn.setEnabled(False)

        # Button
        self.reset_btn = QtWidgets.QAction('Scene reset')
        self.reset_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.reset_btn.triggered.connect(lambda:self.btn_func_reset())
        self.reset_btn.setStatusTip("Click here to clear all labels in current scene.")
        self.reset_btn.setEnabled(False)
        
        # Button
        self.next_scene_btn = QtWidgets.QAction('Next scene')
        self.next_scene_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.next_scene_btn.triggered.connect(lambda:self.btn_func_next_scene())
        self.next_scene_btn.setStatusTip("Click here to confirm labels in current scene and move to next.")
        self.next_scene_btn.setEnabled(False)

        # Button
        self.prev_scene_btn = QtWidgets.QAction('Previous scene')
        self.prev_scene_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.prev_scene_btn.triggered.connect(lambda:self.btn_func_prev_scene())
        self.prev_scene_btn.setStatusTip("Click here to go to previous scene.")
        self.prev_scene_btn.setEnabled(False)

        # Button
        self.compute_btn = QtWidgets.QAction('Compute')
        self.compute_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.compute_btn.triggered.connect(lambda:self.btn_func_compute())
        self.compute_btn.setStatusTip("Click here to solve the optimization problem.")
        self.compute_btn.setEnabled(False)

        # Button
        self.display_btn = QtWidgets.QAction('Visualize')
        self.display_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.display_btn.triggered.connect(lambda:self.btn_func_display())
        self.display_btn.setStatusTip("Click here to visualize the labeled points in 3D.")
        self.display_btn.setEnabled(False)

        # Button
        self.quit_btn = QtWidgets.QAction('Quit')
        self.quit_btn.setIcon(QtGui.QIcon(ICONS_DIR + "new.png"))
        self.quit_btn.triggered.connect(lambda:self.btn_func_quit())
        self.quit_btn.setStatusTip("Click here to close the GUI (inputs will not be saved).")
        self.quit_btn.setEnabled(True)

        # Slider
        self.load_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.load_slider.setMinimum(0)
        self.load_slider.setMaximum(999)
        #self.load_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.load_slider.valueChanged.connect(lambda:self.btn_func_load())
        self.load_slider.setStatusTip("Slide to load images from the scene.")
        self.load_slider.setEnabled(True)

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
        
        # Create a status bar
        self.setStatusBar(QtWidgets.QStatusBar(self))

        # Menus on the left
        self.message_box = QtWidgets.QDockWidget("Keypoint List")
        self.message_box.setWidget(QtWidgets.QListWidget())
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.message_box)

    def init_layout(self):

        # Button for building a new model
        button1 = QtWidgets.QPushButton('Create a new model')
        button1.toggle()
        button1.clicked.connect(lambda:self.btn_func_create())
        button1.setStatusTip("Click here if you do not have a model for you object.")
        button1.setFixedSize(300,50)

        # Button for choosing an existing model file
        button2 = QtWidgets.QPushButton('Use existing model')
        button2.toggle()
        button2.clicked.connect(lambda:self.btn_func_choose())
        button2.setStatusTip("Click here if you already have a *.pp model for your object.")
        button2.setFixedSize(300,50)

        # Button for defining a grasping point
        button3 = QtWidgets.QPushButton('Define grasp point')
        button3.toggle()
        button3.clicked.connect(lambda:self.btn_func_grasping())
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

