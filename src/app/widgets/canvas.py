from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPoint

class QCanvas(QtWidgets.QLabel):

    def __init__(self, width, height):
        super().__init__()
        pixmap = QtGui.QPixmap(width, height)
        pixmap.fill()
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setPixmap(pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))
        self.setMinimumSize(1, 1)

        self.scale = 1.0
        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')        

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)

    def mouseMoveEvent(self, e):
        pos = self.transformPos(e.localPos())
        #pos = e.localPos()
        if self.last_x is None: # First event.
            self.last_x = pos.x()
            self.last_y = pos.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(4)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, pos.x(), pos.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = pos.x()
        self.last_y = pos.y()

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
        return QPoint(x, y)

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def resizeEvent(self, event):
        self.setPixmap(self.pixmap().scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio))
