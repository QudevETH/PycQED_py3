#!/usr/bin/env python3

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets


class TextToTreeItem:
    """
    Helper class for the search feature of the class DictView.
    Class connects QTreeItemWidgets with their data saved as strings.
    It contains a list of all QTreeItemWidgets from the QTreeWidget
    and their key and value saved on two seperated lists.
    """

    def __init__(self):
        self.key_text_list = []
        self.value_text_list = []
        self.titem_list = []

    def append(self, key_text_list, value_text_list, titem):
        """
        Appends a QTreeWidgetItem and its data to the lists.
        Args:
            key_text_list (str): String of the key, i.e. first column entry of the QTreeWidgetItem
            value_text_list (str): String of the value, i.e. second column entry of the QTreeWidgetItem.
            titem (QTreeItemWidget): TreeItemWidget
        """
        for i, key_text in enumerate(key_text_list):
            self.key_text_list.append(key_text)
            self.value_text_list.append(value_text_list[i])
            self.titem_list.append(titem)

    def find(self, find_str, search_option=0):
        """
        Finds and returns all QTreeWidgetItems which contain the find_str.
        Args:
            find_str (str): String which is looked up in the list of QTreeWidgetItems.
            search_option (int): 0 - search in first and second column (key and value)
                1 - search in first column (key only)
                2 - search in second column (value only)

        Returns: list of QTreeWidgetItem which contain find_str and the respective column set by search_option.

        """
        titem_list = []
        if search_option == 0:
            for i, s in enumerate(self.key_text_list):
                if find_str in s or find_str in self.value_text_list[i]:
                    titem_list.append(self.titem_list[i])
        if search_option == 1:
            for i, s in enumerate(self.key_text_list):
                if find_str in s:
                    titem_list.append(self.titem_list[i])
        if search_option == 2:
            for i, s in enumerate(self.value_text_list):
                if find_str in s:
                    titem_list.append(self.titem_list[i])

        return titem_list


class DictView(QtWidgets.QWidget):
    """
    Class to display a given dictionary with features:
    - search for string in keys and/or values
    - copy value and key to clipboard
    - expand, collapse and hide keys/branches
    """

    def __init__(self, snap: dict, title: str = ''):
        """
        Initialization of the QWidget
        Args:
            snap (dict): Dictionary (usually QCodes/mock station snapshot) which is displayed
            title (str): title which is displayed in the header and in the layout.
        """
        super(DictView, self).__init__()
        self.title = title
        self.find_box = None
        self.find_text = None
        self.tree_widget = None
        self.text_to_titem = TextToTreeItem()
        # Default values
        self.find_str = ""
        self.find_check_box_id = 0
        self.found_titem_list = []
        self.found_idx = 0
        self.find_only_params = False  # set to True if search should only include parameters

        # set up the tree
        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(["Key", "Value"])
        self.tree_widget.header().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.tree_widget.header().resizeSection(0, 400)
        self.tree_widget.setExpandsOnDoubleClick(True)
        self.tree_widget.setSortingEnabled(True)

        root_item = QtWidgets.QTreeWidgetItem(["Root"])
        self.recurse_pdata(snap, root_item)  # build up the tree recursively from a dictionary
        self.tree_widget.addTopLevelItem(root_item)

        root_item.setExpanded(True)  # expand root item
        self.expandParameters(root_item)  # expand all parameter branches by default

        self.create_menus()  # creates context menu and menu bar
        self.make_layout()  # creates layout

    def create_menus(self):
        self._createActions()
        self._connectActions()

        # context menu
        self.tree_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self._setContextMenu)

        # menu bar
        self.menuBar = QtWidgets.QMenuBar(self)
        self._setMenuBar()

    def make_layout(self):
        search_layout = self.make_search_ui()  # Load search layout

        self.tree_dir = QtWidgets.QLabel('Root')  # QLabel with text of tree directory

        # QLabel with text of number of attributes in QTreeItemWidget
        self.attr_nr = QtWidgets.QLabel('Number of attributes:')

        # 1st step: positioning QTreeWidget, QLabel (tree directory) and QLabel (attribute number) vertically
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tree_widget)
        layout.addWidget(self.tree_dir)
        layout.addWidget(self.attr_nr)

        # 2nd step: drawing box around QTreeWidget and QLabels
        gbox = QtWidgets.QGroupBox(self.title)
        gbox.setLayout(layout)

        # 3rd step: Adding search bar, search options and number of entries to the top of the layout
        layout2 = QtWidgets.QVBoxLayout()
        layout2.addLayout(search_layout)
        layout2.addSpacing(10)
        layout2.addWidget(gbox)
        layout2.setMenuBar(self.menuBar)

        self.setLayout(layout2)

    def _createActions(self):
        """
        Function which creates all the actions for the context menu and menu bars
        """
        self.copyKeyAction = QtWidgets.QAction("Copy key")
        self.copyValueAction = QtWidgets.QAction("Copy value")
        self.openContentAction = QtWidgets.QAction("Open in new window")
        self.hideAction = QtWidgets.QAction("Hide Key")
        self.hideAllAction = QtWidgets.QAction("Hide all empty")
        self.showAllAction = QtWidgets.QAction("Show all")
        self.collapseAction = QtWidgets.QAction("Collapse all")
        self.expandBranchAction = QtWidgets.QAction("Expand Branch")
        self.collapseBranchAction = QtWidgets.QAction("Collapse Branch")
        self.closeAction = QtWidgets.QAction("Close Window")
        self.resetWindowAction = QtWidgets.QAction("Reset Window")
        self.expandParametersAction = QtWidgets.QAction("Expand Parameters")

    def _connectActions(self):
        """
        Function which connects the actions defined in _createActions with events (functions).
        lambda function is needed for functions with arguments.
        """
        root_item = self.tree_widget.topLevelItem(0)
        self.copyKeyAction.triggered.connect(lambda: self.copyContent(0))
        self.copyValueAction.triggered.connect(lambda: self.copyContent(1))
        self.hideAction.triggered.connect(self.hideItem)
        self.hideAllAction.triggered.connect(
            lambda: self.hideAllEmpty(root_item))
        self.showAllAction.triggered.connect(
            lambda: self.showAll(root_item))
        self.tree_widget.itemClicked.connect(self.setDirtext)
        self.tree_widget.itemActivated.connect(self.setDirtext)
        self.tree_widget.itemClicked.connect(self.setNrAttributes)
        self.tree_widget.itemActivated.connect(self.setNrAttributes)

        self.collapseAction.triggered.connect(
            lambda: self.expandBranch(root_item, expand=False))
        self.expandBranchAction.triggered.connect(
            lambda: self.expandBranch(self.tree_widget.currentItem(), expand=True, display_vals=False))
        self.collapseBranchAction.triggered.connect(
            lambda: self.expandBranch(self.tree_widget.currentItem(), expand=False))
        self.closeAction.triggered.connect(self.close)
        self.resetWindowAction.triggered.connect(self.resetWindow)
        self.expandParametersAction.triggered.connect(
            lambda: self.expandParameters(root_item))

    def _setMenuBar(self):
        """
        Adds actions to the menu bar and groups them in submenus File, Edit and View
        """
        fileMenu = QtWidgets.QMenu("&File", self)
        editMenu = QtWidgets.QMenu("&Edit", self)
        viewMenu = QtWidgets.QMenu("&View", self)

        fileMenu.addAction(self.closeAction)
        fileMenu.addAction(self.resetWindowAction)

        editMenu.addAction(self.copyKeyAction)
        editMenu.addAction(self.copyValueAction)
        editMenu.addSeparator()  # for better organization of the menu bar
        editMenu.addAction(self.openContentAction)

        viewMenu.addAction(self.hideAction)
        viewMenu.addAction(self.hideAllAction)
        viewMenu.addAction(self.showAllAction)
        viewMenu.addAction(self.collapseAction)
        viewMenu.addAction(self.expandParametersAction)

        self.menuBar.addMenu(fileMenu)
        self.menuBar.addMenu(editMenu)
        self.menuBar.addMenu(viewMenu)

    def _setContextMenu(self, position):
        """
        Adds the actions to the context menu and displays it at a given position.
        Args:
            position (QPoint): Position where the context menu pops up.
        """
        menu = QtWidgets.QMenu()

        menu.addAction(self.copyKeyAction)
        menu.addAction(self.copyValueAction)
        menu.addSeparator()
        menu.addAction(self.openContentAction)
        menu.addSeparator()
        menu.addAction(self.hideAction)
        menu.addAction(self.expandBranchAction)
        menu.addAction(self.collapseBranchAction)
        menu.exec_(self.tree_widget.viewport().mapToGlobal(position))

    def resetWindow(self):
        """
        Resets the QTreeWidget to the default layout:
        - no QTreeWidgetItem is hidden
        - only root item and parameters are expanded
        """
        root_item = self.tree_widget.topLevelItem(0)
        self.showAll(root_item)
        self.expandBranch(root_item, expand=False)
        root_item.setExpanded(True)
        self.expandParameters(root_item)

    def expandParameters(self, tree_item: QtWidgets.QTreeWidgetItem):
        """
        Expands recursively all QTreeWidgetItem with key name 'parameters'
        Args:
            tree_item (QTreeWidgetItem): Item from which all instances with key name 'parameters' are expanded
        """
        if tree_item.data(0, 0) == 'parameters':
            tree_item.setExpanded(True)
        for i in range(tree_item.childCount()):
            self.expandParameters(tree_item.child(i))

    def expandBranch(self, tree_item: QtWidgets.QTreeWidgetItem, expand=True, display_vals=True):
        """
        Expands recursively an entire branch of a QTreeWidget
        Args:
            tree_item (QTreeWidgetItem): item which itself and its children are expanded
            expand (bool): True if items are expanded, False if items are collapsed
            display_vals (bool): True if values should be expanded.
                Caution: expanding all values at once might take too much time and forces the user to force quit the
                program. This might restart the kernel.
        """
        tree_item.setExpanded(expand)
        if not display_vals:
            if tree_item.data(0, 0) == 'parameters':
                return
        for i in range(tree_item.childCount()):
            self.expandBranch(tree_item.child(i), expand=expand, display_vals=display_vals)

    def setDirtext(self):
        """
        Sets the QLabel text self.tree_dir to the directory of the current QTreeWidgetItem
        """
        self.tree_dir.setText(self.getDirtext(self.tree_widget.currentItem()))

    def getDirtext(self, tree_item: QtWidgets.QTreeWidgetItem, separator=' > '):
        """
        Returns string of the directory of a given QWidgetItem in a QWidgetTree
        Args:
            separator (str): Separates
            tree_item (QWidgetItem): Item which directory is returned as a string

        Returns: String of directory of a given QWidgetItem in a QWidgetTree
        """
        dir = str(tree_item.data(0, 0))
        try:
            dir = self.getDirtext(tree_item.parent()) + separator + dir
        except:
            pass
        return dir

    def setNrAttributes(self):
        """
        Sets QLabel text of self.attr_nr as the number of attributes (childs) of the current QWidgetItem.
        """
        self.attr_nr.setText('Number of attributes: %s' % self.tree_widget.currentItem().childCount())

    def showAll(self, titem: QtWidgets.QTreeWidgetItem, hide=False):
        """
        Shows or hides QTreeWidgetItem with all its children starting by a given parent item.
        Args:
            titem (QTreeWidgetItem): Item which itself and its children are recursively showed/hidden
            hide (bool): False to hide item+children, True to show item+children
        """
        for i in range(titem.childCount()):
            self.showAll(titem.child(i), hide=hide)
        titem.setHidden(hide)

    def showTitem(self, titem: QtWidgets.QTreeWidgetItem, expand=False):
        """
        Shows QTreeWidgetItem and its parent recursively until root_item.
        Args:
            titem (QTreeWidgetItem): Item which is displayed
            expand (bool): True to expand of Item which is shown
        """
        titem.setHidden(False)
        titem.setExpanded(expand)
        try:
            self.showTitem(titem.parent(), expand=expand)
        except:
            pass

    def hideAllEmpty(self, titem: QtWidgets.QTreeWidgetItem):
        """
        Hides all QTreeWidgetItems which have no children and no value (data in second column) starting by given
        QTreeWidgetItem
        Args:
            titem (QTreeWidgetItem): Item to start hiding empty keys recursively.
        """
        if titem.childCount() == 0:
            if titem.data(1, 0) == None:
                titem.setHidden(True)
        else:
            for i in range(titem.childCount()):
                self.hideAllEmpty(titem.child(i))

    def hideItem(self):
        """
        Hides currently selected QTreeWidgetItem
        """
        self.tree_widget.currentItem().setHidden(True)

    def copyContent(self, column: int):
        """
        Copies key or value (first or second column) of the currently selected QTreeWidgetItem as a string
        to the clipboard.
        Args:
            column: 0 for first column (key), 1 for second column (value)
        """
        cb = QtWidgets.QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        if column == 0:
            cb.setText(self.tree_widget.currentItem().data(0, 0), mode=cb.Clipboard)
        if column == 1:
            cb.setText(self.tree_widget.currentItem().data(1, 0), mode=cb.Clipboard)

    def make_search_ui(self):
        """
        Creates the UI layout for the search bar and its options.

        Returns: QLayout of search bar
        """
        # Text box
        self.find_box = QtWidgets.QLineEdit()
        self.find_box.returnPressed.connect(self.find_button_clicked)

        # Find Button
        find_button = QtWidgets.QPushButton("Find")
        find_button.clicked.connect(self.find_button_clicked)

        # 'Entries Found' QLabel
        self.find_text = QtWidgets.QLabel('Entries found:')

        # search options
        find_checkboxoptions_list = [QtWidgets.QCheckBox('Key and Value'),
                                     QtWidgets.QCheckBox('Key'),
                                     QtWidgets.QCheckBox('Value')]
        # set default search option
        find_checkboxoptions_list[self.find_check_box_id].setChecked(True)
        # group options in QButtonGroup that only one button can be checked at any given time (exclusive)
        self.find_checkboxoptions = QtWidgets.QButtonGroup()
        for id, w in enumerate(find_checkboxoptions_list):
            self.find_checkboxoptions.addButton(w, id=id)

        # adding additional search options
        self.find_only_params_box = QtWidgets.QCheckBox('Search only in parameters')
        self.find_only_params_box.setChecked(self.find_only_params)

        # adding widgets to layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.find_box)
        layout.addWidget(find_button)

        layout2 = QtWidgets.QHBoxLayout()
        for w in self.find_checkboxoptions.buttons():
            layout2.addWidget(w)
        layout2.addSpacing(30)
        layout2.addWidget(self.find_only_params_box)
        layout2.addStretch()

        layout3 = QtWidgets.QVBoxLayout()
        layout3.addLayout(layout)
        layout3.addLayout(layout2)
        layout3.addWidget(self.find_text)

        return layout3



    def find_button_clicked(self):
        """

        """
        find_str = self.find_box.text()
        find_only_params = self.find_only_params_box.isChecked()

        # Very common for use to click Find on empty string
        if find_str == "":
            return

        # New search string
        if find_str != self.find_str \
                or self.find_check_box_id != self.find_checkboxoptions.checkedId() \
                or self.find_only_params != find_only_params:
            self.find_str = find_str
            self.find_check_box_id = self.find_checkboxoptions.checkedId()
            self.find_only_params = find_only_params
            self.found_titem_list = self.text_to_titem.find(
                find_str=self.find_str, search_option=self.find_checkboxoptions.checkedId())
            if find_only_params:
                titem_list = []
                for titem in self.found_titem_list:
                    try:
                        if titem.parent().data(0, 0) == 'parameters':
                            titem_list.append(titem)
                    except:
                        pass
                self.found_titem_list = titem_list
            self.find_text.setText('Entries found: %s' % (len(self.found_titem_list)))
            if not self.found_titem_list == []:
                self.showAll(self.tree_widget.topLevelItem(0), hide=True)
                for titem in self.found_titem_list:
                    titem.setHidden(False)
                    self.showTitem(titem, expand=True)
            self.found_idx = 0
        elif not self.found_titem_list:
            self.find_text.setText('Entries found: %s' % (len(self.found_titem_list)))
            return
        else:
            item_num = len(self.found_titem_list)
            self.found_idx = (self.found_idx + 1) % item_num
        try:
            self.tree_widget.setCurrentItem(self.found_titem_list[self.found_idx])
            self.tree_dir.setText(self.getDirtext(self.tree_widget.currentItem()))
        except:
            self.find_text.setText('Keys and Values found: %s' % (len(self.found_titem_list)))

    def recurse_pdata(self, pdata, tree_widget):

        if isinstance(pdata, dict):
            for key, val in pdata.items():
                self.tree_add_row(str(key), val, tree_widget)
        elif isinstance(pdata, list):
            for i, val in enumerate(pdata):
                key = str(i)
                self.tree_add_row(key, val, tree_widget)
        else:
            print("This should never be reached!")

    def recurse_parameters(self, pdata, tree_widget):

        if isinstance(pdata, dict):
            for key, val in pdata.items():
                self.tree_add_row_parameter(str(key), val, tree_widget)
        elif isinstance(pdata, list):
            for i, val in enumerate(pdata):
                key = str(i)
                self.tree_add_row(key, val, tree_widget)
        else:
            print("This should never be reached!")

    def tree_add_row_parameter(self, key, val, tree_widget):

        key_text_list = []
        value_text_list = []

        if isinstance(val, dict) or isinstance(val, list):
            key_text_list.append(key)
            value_text_list.append('')
            vals = 'no value'
            try:
                vals = str(val['value'])
            except:
                pass
            row_item = QtWidgets.QTreeWidgetItem([key, vals])
            # tree_widget.itemDoubleClicked(row_item, 0)#.connect(testfct)
            # row_item.itemDoubleClicked.connect(testfct)
            if key == 'parameters':
                self.recurse_parameters(val, row_item)
            else:
                self.recurse_pdata(val, row_item)
        else:
            key_text_list.append(key)
            value_text_list.append(str(val))
            row_item = QtWidgets.QTreeWidgetItem([key, str(val)])

        tree_widget.addChild(row_item)
        self.text_to_titem.append(key_text_list, value_text_list, row_item)

    def tree_add_row(self, key, val, tree_widget):

        key_text_list = []
        value_text_list = []

        if isinstance(val, dict) or isinstance(val, list):
            key_text_list.append(key)
            value_text_list.append('')
            row_item = QtWidgets.QTreeWidgetItem([key])
            if key == 'parameters':
                self.recurse_parameters(val, row_item)
            else:
                self.recurse_pdata(val, row_item)
        else:
            key_text_list.append(key)
            value_text_list.append(str(val))
            row_item = QtWidgets.QTreeWidgetItem([key, str(val)])

        tree_widget.addChild(row_item)
        self.text_to_titem.append(key_text_list, value_text_list, row_item)


class tree(QtWidgets.QWidget):

    def __init__(self, treeitem):
        super(tree, self).__init__()

        self.find_box = None
        self.tree_widget = None
        self.text_to_titem = TextToTreeItem()
        self.find_str = ""
        self.found_titem_list = []
        self.found_idx = 0

        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(["Key", "Value"])
        self.tree_widget.header().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.tree_widget.header().resizeSection(0, 200)

        # set up the tree
        root_item = QtWidgets.QTreeWidgetItem(["Root"])
        self.build_tree(treeitem, root_item)
        for i in range(treeitem.childCount()):
            item = QtWidgets.QTreeWidgetItem(
                [treeitem.child(i).data(0, 0), treeitem.child(i).data(1, 0)])
            self.build_tree(treeitem.child(i), item)
            self.tree_widget.insertTopLevelItem(i, item)

        # creates the context menu
        self._createActions()
        self._connectActions()
        self.tree_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self._openMenu)

        self.tree_widget.setExpandsOnDoubleClick(True)
        self.tree_widget.setSortingEnabled(True)

        # # Add table to layout
        #
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.tree_widget)
        #
        self.setLayout(layout)

    def _createActions(self):
        self.copyKeyAction = QtWidgets.QAction("Copy key")
        self.copyValueAction = QtWidgets.QAction("Copy value")

    def _connectActions(self):
        self.copyKeyAction.triggered.connect(lambda: self._copyContent(0))
        self.copyValueAction.triggered.connect(lambda: self._copyContent(1))

    def _openMenu(self, position):
        menu = QtWidgets.QMenu()
        menu.addAction(self.copyKeyAction)
        menu.addAction(self.copyValueAction)
        menu.exec_(self.tree_widget.viewport().mapToGlobal(position))

    def _copyContent(self, column: int):
        cb = QtWidgets.QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        if column == 0:
            cb.setText(self.tree_widget.currentItem().data(0, 0), mode=cb.Clipboard)
        if column == 1:
            cb.setText(self.tree_widget.currentItem().data(1, 0), mode=cb.Clipboard)

    def build_tree(self, value_tree_widget: QtWidgets.QTreeWidgetItem, tree_widget):
        for i in range(value_tree_widget.childCount()):
            row_item = QtWidgets.QTreeWidgetItem(
                [value_tree_widget.child(i).data(0, 0), value_tree_widget.child(i).data(1, 0)])
            for j in range(value_tree_widget.child(i).childCount()):
                self.build_tree(value_tree_widget.child(i).child(j), row_item)
            tree_widget.addChild(row_item)


class AdditionalWindow(QtWidgets.QMainWindow):
    def __init__(self, dict_view):
        super(AdditionalWindow, self).__init__()
        self.setCentralWidget(tree(dict_view.tree_widget.currentItem()))
        # self.setWindowTitle(dict_view.tree_widget.currentItem().text(0))
        self.setWindowTitle(dict_view.getDirtext(dict_view.tree_widget.currentItem()))
        self.setGeometry(200, 200, 500, 500)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()


class DictViewerWindow(QtWidgets.QMainWindow):

    def __init__(self, dic: dict, title: str = ''):
        super(DictViewerWindow, self).__init__()

        dict_view = DictView(dic, title)

        # open new window with double click
        # dict_view.tree_widget.itemDoubleClicked.connect(lambda: self.openNewWindow(dict_view))
        # open new window via content menu
        dict_view.openContentAction.triggered.connect(lambda: self.openNewWindow(dict_view))

        self.setCentralWidget(dict_view)
        self.setWindowTitle("Dictionary Viewer")
        self.setGeometry(100, 100, 800, 800)

        self.dialogs = list()

        self.show()

    def openNewWindow(self, pickle_view):
        dialog = AdditionalWindow(pickle_view)
        self.dialogs.append(dialog)
        dialog.show()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()
