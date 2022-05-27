from pycqed.gui import qt_compat as qt


class CheckableComboBox(qt.QtWidgets.QComboBox):

    # Subclass Delegate to increase item height
    class Delegate(qt.QtWidgets.QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = qt.QtWidgets.QApplication.instance().palette()
        palette.setBrush(qt.QtGui.QPalette.ColorRole.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)
        self.default_display_text = ""

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):

        if object == self.lineEdit():
            if event.type() == qt.QtCore.QEvent.Type.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == qt.QtCore.QEvent.Type.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == qt.QtCore.Qt.CheckState.Checked:
                    item.setCheckState(qt.QtCore.Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(qt.QtCore.Qt.CheckState.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        items_selected = False
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == \
                    qt.QtCore.Qt.CheckState.Checked:
                items_selected = True
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)
        if not items_selected:
            text = self.default_display_text

        # Compute elided text (with "...")
        metrics = qt.QtGui.QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(
            text, qt.QtCore.Qt.TextElideMode.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = qt.QtGui.QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(qt.QtCore.Qt.ItemFlag.ItemIsEnabled |
                      qt.QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(qt.QtCore.Qt.CheckState.Unchecked,
                     qt.QtCore.Qt.ItemDataRole.CheckStateRole)
        item.setToolTip(text)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == \
                    qt.QtCore.Qt.CheckState.Checked:
                res.append(self.model().item(i).data())
        return res