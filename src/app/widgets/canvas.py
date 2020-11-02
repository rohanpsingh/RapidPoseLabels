from PyQt5 import QtGui, QtWidgets, QtCore

class QCanvas(QtWidgets.QLabel):

    def __init__(self, width, height):
        super().__init__()
        self._pixmap = QtGui.QPixmap(width, height)
        self._pixmap.fill()
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setPixmap(self._pixmap.scaled(
            self.width(), self.height(),
            QtCore.Qt.KeepAspectRatio))
        self.setMinimumSize(1, 1)

        self.scale = 1.0

    def mousePressEvent(self, e):
        self.loadPixmap(self._pixmap)
        pos = self.transformPos(e.localPos())
        painter = QtGui.QPainter(self.pixmap())
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0,255,0)))
        painter.drawEllipse(QtCore.QPoint(pos.x(), pos.y()), 6, 6)
        painter.end()
        self.update()

    def mouseDoubleClickEvent(self, e):
        pos = self.transformPos(e.localPos())
        painter = QtGui.QPainter(self.pixmap())
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0,0,255)))
        painter.drawEllipse(QtCore.QPoint(pos.x(), pos.y()), 10, 10)
        painter.end()
        self.update()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(QCanvas, self).size()
        w, h = self.pixmap().width() * s, self.pixmap().height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPoint(x, y)

    def loadPixmap(self, pixmap):
        self._pixmap = pixmap
        self.setPixmap(self._pixmap.scaled(
            self.width(), self.height(),
            QtCore.Qt.KeepAspectRatio))

    def resizeEvent(self, event):
        self.setPixmap(self._pixmap.scaled(
            self.width(), self.height(),
            QtCore.Qt.KeepAspectRatio))
