from PyQt5 import QtCore, QtGui, QtWidgets

class QKeypointCount(QtWidgets.QWidget):

    numKeypoints = QtCore.pyqtSignal(int)

    def __init__(self, value=8):
        super().__init__()

        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.spinbox.setRange(4,15)
        self.spinbox.setValue(value)
        self.spinbox.setToolTip("Keypoint count")
        self.spinbox.setStatusTip("Set the number of keypoints to be defined on the object")
        self.spinbox.setAlignment(QtCore.Qt.AlignCenter)
        self.spinbox.valueChanged.connect(self.valueChanged)
        
        self.label = QtWidgets.QLabel(self.spinbox.toolTip())
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.spinbox)
        layout.addWidget(self.label)
        self.setLayout(layout)
        
    def minimumSizeHint(self):
        height = super().minimumSizeHint().height()
        fm = QtGui.QFontMetrics(self.font())
        width = fm.width(str(self.spinbox.maximum()))
        return QtCore.QSize(width, height)
            
    def valueChanged(self):
        self.numKeypoints.emit(self.spinbox.value())
