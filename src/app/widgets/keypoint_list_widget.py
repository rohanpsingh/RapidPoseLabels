from PyQt5 import QtCore, QtGui, QtWidgets

class QKeypointListItem(QtGui.QStandardItem):
    def __init__(self, text):
        super().__init__()
        self.setText(text)
        self.setEditable(False)
        self.setTextAlignment(QtCore.Qt.AlignBottom)

    def clone(self):
        return QKeypointListItem(self.text())

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return '{}("{}")'.format(self.__class__.__name__, self.text())


class StandardItemModel(QtGui.QStandardItemModel):

    itemDropped = QtCore.pyqtSignal()

    def removeRows(self, *args, **kwargs):
        ret = super().removeRows(*args, **kwargs)
        self.itemDropped.emit()
        return ret

class QKeypointListWidget(QtWidgets.QListView):

    itemClicked = QtCore.pyqtSignal(QKeypointListItem)

    def __init__(self):
        super().__init__()

        self.setWindowFlags(QtCore.Qt.Window)
        self.setModel(StandardItemModel())
        self.model().setItemPrototype(QKeypointListItem(None))
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)

        self.clicked.connect(self.itemClickedEvent)

    def __len__(self):
        return self.model().rowCount()

    def __getitem__(self, i):
        return self.model().item(i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def mousePressEvent(self, ev):
        self.selectionModel().clearSelection()
        self.itemClicked.emit(QKeypointListItem(None))
        super().mousePressEvent(ev)

    @property
    def itemDropped(self):
        return self.model().itemDropped

    @property
    def itemChanged(self):
        return self.model().itemChanged

    def itemClickedEvent(self, index):
        self.itemClicked.emit(self.model().itemFromIndex(index))

    def scrollToItem(self, item):
        self.scrollTo(self.model().indexFromItem(item))

    def addItem(self, item):
        if not isinstance(item, str):
            raise TypeError("item must be string")
        self.model().setItem(self.model().rowCount(), QKeypointListItem(item))
        #item.setSizeHint(self.itemDelegate().sizeHint(None))

    def removeItem(self, item):
        index = self.model().indexFromItem(item)
        self.model().removeRows(index.row(), 1)

    def selectItem(self, item):
        index = self.model().indexFromItem(item)
        self.selectionModel().select(index, QtCore.QItemSelectionModel.Select)

    def clear(self):
        self.itemClicked.emit(QKeypointListItem(None))
        self.model().clear()
