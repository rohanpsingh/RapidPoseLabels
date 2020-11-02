from PyQt5 import QtGui, QtWidgets, QtCore

class QCanvas(QtWidgets.QWidget):

    def __init__(self, width, height):
        super().__init__()
        self._pixmap = QtGui.QPixmap(width, height)
        self._pixmap.fill()
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        #self.setPixmap(self._pixmap.scaled(self.width(), self.height(),QtCore.Qt.KeepAspectRatio))
        self.setMinimumSize(1, 1)

        self.scale = 1.0
        self.temporary_points = []
        self.permanent_points = []

    def mousePressEvent(self, e):
        pos = self.transformPos(e.localPos())
        self.temporary_points = [QtCore.QPoint(pos.x(), pos.y())]
        self.update()

    def mouseDoubleClickEvent(self, e):
        pos = self.transformPos(e.localPos())
        self.permanent_points.append(QtCore.QPoint(pos.x(), pos.y()))
        self.update()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(QCanvas, self).size()
        w, h = self._pixmap.width() * s, self._pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPoint(x, y)

    def loadPixmap(self, pixmap):
        self._pixmap = pixmap
        self.temporary_points, self.permanent_points = [], []
        self.update()

    def paintEvent(self, event):
        if not self._pixmap:
            return super().paintEvent(event)

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        painter.scale(self.scale, self.scale)
        painter.translate(self.offsetToCenter())
        painter.drawPixmap(0, 0, self._pixmap)

        # Draw green points
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0,255,0)))
        for point in self.temporary_points:
            painter.drawEllipse(point, 6, 6)
        # Draw blue points
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0,0,255)))
        for point in self.permanent_points:
            painter.drawEllipse(point, 6, 6)

        painter.end()
