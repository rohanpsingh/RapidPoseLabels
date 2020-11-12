from PyQt5 import QtCore, QtGui, QtWidgets
from app.widgets import QCanvas
from app.widgets import QToolMenu
from app.widgets import QKeypointListWidget

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

        # Main layout
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
        self.close()

    def act_func_zoom_in(self):
        self.scaleImage(1.1)

    def act_func_zoom_out(self):
        self.scaleImage(0.9)

    def act_func_normal_size(self):
        self.canvas.scale = 1.0
        self.canvas.adjustSize()

    def act_func_load_data(self):
        pass

    def act_func_load_model(self):
        pass

    def act_func_info(self):
        import webbrowser
        url = "https://github.com/rohanpsingh/rapidposelabels/blob/master/README.md"
        webbrowser.open(url)

    def act_func_icons(self):
        QtWidgets.QMessageBox.information(self, "Icons", "Some icons by Yusuke Kamiyamane <p.yusukekamiyamane.com>")

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
        self.zoom_in_act = QtWidgets.QAction("Zoom &In (10%)", shortcut="Ctrl++")
        self.zoom_in_act.setIcon(QtGui.QIcon.fromTheme("zoom-in"))
        self.zoom_in_act.triggered.connect(self.act_func_zoom_in)
        self.zoom_in_act.setEnabled(True)

        # Action (zoom out)
        self.zoom_out_act = QtWidgets.QAction("Zoom &Out (10%)", shortcut="Ctrl+-")
        self.zoom_out_act.setIcon(QtGui.QIcon.fromTheme("zoom-out"))
        self.zoom_out_act.triggered.connect(self.act_func_zoom_out)
        self.zoom_out_act.setEnabled(True)

        # Action (normal size)
        self.normal_size_act = QtWidgets.QAction("&Normal Size", shortcut="Ctrl+0")
        self.normal_size_act.setIcon(QtGui.QIcon.fromTheme("zoom-original"))
        self.normal_size_act.triggered.connect(self.act_func_normal_size)
        self.normal_size_act.setEnabled(True)

        # Action
        self.load_data_act = QtWidgets.QAction("Load &Dataset", shortcut="Ctrl+O")
        self.load_data_act.setIcon(QtGui.QIcon(ICONS_DIR + "folder-horizontal.png"))
        self.load_data_act.triggered.connect(self.act_func_load_data)
        self.load_data_act.setEnabled(True)

        # Action
        self.load_model_act = QtWidgets.QAction("Load &Model", shortcut="Ctrl+M")
        self.load_model_act.setIcon(QtGui.QIcon(ICONS_DIR + "document-text.png"))
        self.load_model_act.triggered.connect(self.act_func_load_model)
        self.load_model_act.setEnabled(False)

        # Action
        self.info_act = QtWidgets.QAction("&Info", triggered=self.act_func_info)
        self.info_act.setIcon(QtGui.QIcon(ICONS_DIR + "information.png"))

        # Action
        self.icons_act = QtWidgets.QAction("Icons", triggered=self.act_func_icons)

        # Button
        self.load_btn = QtWidgets.QAction('Load New Image', shortcut="Space")
        self.load_btn.setIcon(QtGui.QIcon(ICONS_DIR + "image-sunset.png"))
        self.load_btn.triggered.connect(lambda x : self.btn_func_load(-1))
        self.load_btn.setStatusTip("Click here to load a new image from current scene.")
        self.load_btn.setEnabled(False)

        # Button
        self.skip_btn = QtWidgets.QAction('Skip keypoint', shortcut="Ctrl+Tab")
        self.skip_btn.setIcon(QtGui.QIcon(ICONS_DIR + "minus.png"))
        self.skip_btn.triggered.connect(self.btn_func_skip)
        self.skip_btn.setStatusTip("Click here if keypoint is not visible in current scene.")
        self.skip_btn.setEnabled(False)

        # Button
        self.reset_btn = QtWidgets.QAction('Scene reset', shortcut="Ctrl+R")
        self.reset_btn.setIcon(QtGui.QIcon(ICONS_DIR + "exclamation.png"))
        self.reset_btn.triggered.connect(self.btn_func_reset)
        self.reset_btn.setStatusTip("Click here to clear all labels in current scene.")
        self.reset_btn.setEnabled(False)
        
        # Button
        self.next_scene_btn = QtWidgets.QAction('Next scene', shortcut="Ctrl+N")
        self.next_scene_btn.setIcon(QtGui.QIcon(ICONS_DIR + "arrow.png"))
        self.next_scene_btn.triggered.connect(self.btn_func_next_scene)
        self.next_scene_btn.setStatusTip("Click here to confirm labels in current scene and move to next.")
        self.next_scene_btn.setEnabled(False)

        # Button
        self.prev_scene_btn = QtWidgets.QAction('Previous scene', shortcut="Ctrl+P")
        self.prev_scene_btn.setIcon(QtGui.QIcon(ICONS_DIR + "arrow-180.png"))
        self.prev_scene_btn.triggered.connect(self.btn_func_prev_scene)
        self.prev_scene_btn.setStatusTip("Click here to go to previous scene.")
        self.prev_scene_btn.setEnabled(False)

        # Button
        self.compute_btn = QtWidgets.QAction('Compute')
        self.compute_btn.setIcon(QtGui.QIcon(ICONS_DIR + "tick.png"))
        self.compute_btn.triggered.connect(self.btn_func_compute)
        self.compute_btn.setStatusTip("Click here to solve the optimization problem.")
        self.compute_btn.setEnabled(False)

        # Button
        self.display_btn = QtWidgets.QAction('Visualize')
        self.display_btn.setIcon(QtGui.QIcon(ICONS_DIR + "color.png"))
        self.display_btn.triggered.connect(self.btn_func_display)
        self.display_btn.setStatusTip("Click here to visualize the labeled points in 3D.")
        self.display_btn.setEnabled(False)

        # Button
        self.quit_btn = QtWidgets.QAction('Quit', shortcut="Alt+Q")
        self.quit_btn.setIcon(QtGui.QIcon(ICONS_DIR + "cross-circle.png"))
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
        
        # Docked widgets on the left
        self.keypoint_list = QKeypointListWidget()
        self.keypoint_list.itemClicked.connect(self.keypoint_clicked)
        self.keypoint_dock = QtWidgets.QDockWidget("Keypoint List")
        self.keypoint_dock.setWidget(self.keypoint_list)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.keypoint_dock)

        self.scene_list = QtWidgets.QListWidget()
        self.scene_dock = QtWidgets.QDockWidget("Scenes")
        self.scene_dock.setWidget(self.scene_list)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.scene_dock)

        # Menu toolbar on the top
        self.fileMenu = QtWidgets.QMenu("&File")
        self.fileMenu.addAction(self.load_data_act)
        self.fileMenu.addAction(self.load_model_act)
        self.fileMenu.addAction(self.load_btn)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.quit_btn)

        self.viewMenu = QtWidgets.QMenu("&View")
        self.viewMenu.addAction(self.zoom_in_act)
        self.viewMenu.addAction(self.zoom_out_act)
        self.viewMenu.addAction(self.normal_size_act)

        self.helpMenu = QtWidgets.QMenu("&Help")
        self.helpMenu.addAction(self.info_act)
        self.helpMenu.addAction(self.icons_act)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

        # Create a status bar
        self.setStatusBar(QtWidgets.QStatusBar(self))
        self.mode_qlabel = QtWidgets.QLabel()
        self.mode_qlabel.setText("Mode: Create new model")
        self.statusBar().addPermanentWidget(self.mode_qlabel)

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

    def closeEvent(self, event):
        quit_msg = "All labels will be lost. Are you sure you want to quit?"
        reply = QtWidgets.QMessageBox.question(self, 'Message',
                         quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def update_keypoint_dock(self):
        self.keypoint_list.clear()
        try:
            points = [item.pixel for item in self.scenes[self._count].labels]
            for index, point in enumerate(points):
                self.keypoint_list.addItem(
                    "KP {}: {}".format(index, tuple(point))
                )
        except IndexError:
            return

    def keypoint_clicked(self, item):
        self.canvas.select_id = None
        if item.text() is not "":
            keypoint_id = int(item.text().split(':')[0].split('KP ')[1])
            self.canvas.select_id = keypoint_id
        self.canvas.update()
