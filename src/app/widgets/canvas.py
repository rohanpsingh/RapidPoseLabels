from PyQt5 import QtGui, QtWidgets, QtCore

class QCanvas(QtWidgets.QWidget):

    newPoint = QtCore.pyqtSignal(QtCore.QPoint)
    zoomRequest = QtCore.pyqtSignal(int, QtCore.QPoint)
    scrollRequest = QtCore.pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self._pixmap = QtGui.QPixmap()

        self.scale = 1.0
        self.last_clicked = None
        self.select_id = None
        self.current_points = []
        self.locked_points = []

    def mousePressEvent(self, e):
        pos = self.transformPos(e.localPos())
        if not self.outOfPixmap(pos):
            self.last_clicked = QtCore.QPoint(pos.x(), pos.y())
            self.update()

    def mouseDoubleClickEvent(self, e):
        pos = self.transformPos(e.localPos())
        if not self.outOfPixmap(pos):
            self.newPoint.emit(QtCore.QPoint(pos.x(), pos.y()))
            self.update()

    def wheelEvent(self, e):
        mods = e.modifiers()
        delta = e.angleDelta()
        if QtCore.Qt.ControlModifier == int(mods):
            self.zoomRequest.emit(delta.y(), e.pos())
        else:
            self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
            self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        e.accept()

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
        self.last_clicked = None
        self.current_points = []
        self.locked_points = []
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
        if self.last_clicked:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0,255,0)))
            if not self.outOfPixmap(self.last_clicked):
                painter.drawEllipse(self.last_clicked, 6, 6)

        # Draw blue points
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0,0,255)))
        for point in list(self.current_points + self.locked_points):
            if not self.outOfPixmap(point):
                painter.drawEllipse(point, 6, 6)

        # Highlight selected point
        if self.select_id is not None:
            try:
                point = (self.current_points+self.locked_points)[self.select_id]
                if not self.outOfPixmap(point):
                    painter.setBrush(QtGui.QBrush())
                    painter.setPen(QtGui.QPen(QtGui.QColor(0,255,0), 3, QtCore.Qt.DashLine))
                    painter.drawEllipse(point, 9, 9)
                    painter.setPen(QtGui.QPen(QtGui.QColor(255,255,0), 3, QtCore.Qt.DashLine))
                    painter.drawEllipse(point, 12, 12)
            except:
                pass

        painter.end()

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self._pixmap:
            return self.scale * self._pixmap.size()
        return super(QCanvas, self).minimumSizeHint()

    def outOfPixmap(self, p):
        w, h = self._pixmap.width(), self._pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

