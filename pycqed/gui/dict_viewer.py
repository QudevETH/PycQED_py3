from pycqed.gui import qt_compat as qt
import sys
from pycqed.utilities import timer


class TextToTreeItem:
    """
    Helper class for the search feature of the class DictView.
    Class connects QTreeWidgetItems with strings that identifies the data
    (these strings are used to search for these QTreeWidgetItems, e.g. their
    data of the columns).
    It contains a list of all QTreeWidgetItems from the QTreeWidget
    and a dictionary of lists of strings, which are used to search for the
    QTreeWidgetItem.
    """

    def __init__(self, search_options):
        self.titem_list = []
        self.find_list = {key: [] for key in search_options}

    def append(self, search_string_dict: dict, titem):
        """
        Appends a QTreeWidgetItem and its data to the lists.
        Args:
            search_string_dict (dict): dictionary of strings which characterize
                the QTreeWidgetItem (e.g. data of the columns). These strings
                are used in self.find() to search for a QTreeWidgetItem with a
                given search string.
            titem (QTreeWidgetItem): TreeItemWidget
        """
        for key, item in search_string_dict.items():
            self.find_list[key].append(item)

        self.titem_list.append(titem)

    def find(self, find_str, search_keys: list):
        """
        Finds and returns all QTreeWidgetItems of which the search_string_dict
        entry contains the find_str. Search_keys need to be specified.
        Args:
            find_str (str): String which is looked up in the list of
                QTreeWidgetItems.
            search_keys (list): Keys of the lists in self.find_list which are
                used for the search

        Returns: list of QTreeWidgetItem which connected entries in
        search_string_dict contain find_str conditioned in the search_keys.

        """
        titem_list = []

        for key in search_keys:
            for i, s in enumerate(self.find_list.get(key)):
                if find_str in s:
                    titem_list.append(self.titem_list[i])

        return titem_list


class DictView(qt.QtWidgets.QWidget):
    """
    Class to display a given dictionary with features:
    - search for string in keys and/or values
    - copy value and key to clipboard
    - expand, collapse and hide keys/branches
    """

    def __init__(self, snap: dict, title: str = '', screen=None):
        """
        Initialization of the QWidget
        Args:
            snap (dict): Dictionary (usually QCodes/mock station snapshot) which
                is displayed
            title (str): title which is displayed in the header and in the
                layout.
            screen (QScreen): Screen properties of the primary screen.
        """
        super(DictView, self).__init__()
        self.title = title
        self.screen = screen
        self.find_box = None
        self.find_text = None
        self.tree_widget = None
        # Default values for the search bar
        self.search_options = ['key', 'value']
        self.current_search_options = ['key']
        self.text_to_titem = TextToTreeItem(self.search_options)
        self.find_str = ""
        self.found_titem_list = []
        self.find_check_box_dict = {}
        self.found_idx = 0
        # set to True if search should only include parameters
        self.find_only_params = False

        # set up the tree
        self.tree_widget = qt.QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(["Key", "Value"])
        self.tree_widget.header().setSectionResizeMode(
            qt.QtWidgets.QHeaderView.ResizeMode.Interactive)

        # initializes the width of the column: 0.4*screen_size.width() is the
        # width of the entire window defined in DictViewerWindow
        screen_size = screen.size()
        self.tree_widget.header().resizeSection(
            0, .5 * .4 * screen_size.width())
        self.tree_widget.setExpandsOnDoubleClick(True)

        root_item = qt.QtWidgets.QTreeWidgetItem(["Root"])
        # build up the tree recursively from a dictionary
        self.dict_to_titem(snap, root_item)
        self.tree_widget.addTopLevelItem(root_item)

        root_item.setExpanded(True)  # expand root item
        # expand all parameter branches by default
        self.expandParameters(root_item)

        # sorting needs to be set after initialization of tree elements
        # lets user choose which column is used to sort
        self.tree_widget.setSortingEnabled(True)
        # default is first column in ascending order
        self.tree_widget.sortByColumn(0, qt.QtCore.Qt.SortOrder.AscendingOrder)
        self.create_menus()  # creates context menu and menu bar
        self.make_layout()  # creates layout

    def create_menus(self):
        self._createActions()
        self._connectActions()

        # context menu
        self.tree_widget.setContextMenuPolicy(
            qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(
            self._setContextMenu)

        # menu bar
        self.menuBar = qt.QtWidgets.QMenuBar(self)
        self._setMenuBar()

    def make_layout(self):
        search_layout = self.make_search_ui()  # Load search layout

        # QLabel with text of tree directory
        self.tree_dir = qt.QtWidgets.QLabel('Root')

        # QLabel with text of number of attributes in QTreeWidgetItem
        self.attr_nr = qt.QtWidgets.QLabel('Number of attributes:')

        # 1st step: positioning QTreeWidget, QLabel (tree directory) and
        # QLabel (attribute number) vertically
        layout = qt.QtWidgets.QVBoxLayout()
        layout.addWidget(self.tree_widget)
        layout.addWidget(self.tree_dir)
        layout.addWidget(self.attr_nr)

        # 2nd step: drawing box around QTreeWidget and QLabels
        gbox = qt.QtWidgets.QGroupBox(self.title)
        gbox.setLayout(layout)

        # 3rd step: Adding search bar, search options and number of entries
        # to the top of the layout
        layout2 = qt.QtWidgets.QVBoxLayout()
        layout2.addLayout(search_layout)
        # adding 10 pixels vertically between two widgets
        layout2.addSpacing(10)
        layout2.addWidget(gbox)
        layout2.setMenuBar(self.menuBar)

        self.setLayout(layout2)

    def _createActions(self):
        """
        Function which creates all the actions for the context menu and
        menu bars
        """
        self.copyKeyAction = qt.QtGui.QAction("Copy key")
        self.copyValueAction = qt.QtGui.QAction("Copy value")
        self.openContentAction = qt.QtGui.QAction("Open in new window")
        self.hideAction = qt.QtGui.QAction("Hide Key")
        self.hideAllAction = qt.QtGui.QAction("Hide all empty")
        self.showAllAction = qt.QtGui.QAction("Show all")
        self.collapseAction = qt.QtGui.QAction("Collapse all")
        self.expandBranchAction = qt.QtGui.QAction("Expand Branch")
        self.collapseBranchAction = qt.QtGui.QAction("Collapse Branch")
        self.closeAction = qt.QtGui.QAction("Close Window")
        self.resetWindowAction = qt.QtGui.QAction("Reset Window")
        self.expandParametersAction = qt.QtGui.QAction("Expand Parameters")

    def _connectActions(self):
        """
        Function which connects the actions defined in _createActions with
        events (functions). lambda function is needed for functions with
        arguments.
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
            lambda: self.expandBranch(
                self.tree_widget.currentItem(), expand=True,
                display_vals=False))
        self.collapseBranchAction.triggered.connect(
            lambda: self.expandBranch(
                self.tree_widget.currentItem(), expand=False))
        self.closeAction.triggered.connect(self.close)
        self.resetWindowAction.triggered.connect(self.resetWindow)
        self.expandParametersAction.triggered.connect(
            lambda: self.expandParameters(root_item))

    def _setMenuBar(self):
        """
        Adds actions to the menu bar and groups them in submenus File, Edit
        and View
        """
        fileMenu = qt.QtWidgets.QMenu("&File", self)
        editMenu = qt.QtWidgets.QMenu("&Edit", self)
        viewMenu = qt.QtWidgets.QMenu("&View", self)

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
        menu = qt.QtWidgets.QMenu()

        menu.addAction(self.copyKeyAction)
        menu.addAction(self.copyValueAction)
        menu.addSeparator()
        menu.addAction(self.openContentAction)
        menu.addSeparator()
        menu.addAction(self.hideAction)
        menu.addAction(self.expandBranchAction)
        menu.addAction(self.collapseBranchAction)
        menu.exec(self.tree_widget.viewport().mapToGlobal(position))

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

    def expandParameters(self, tree_item: qt.QtWidgets.QTreeWidgetItem):
        """
        Expands recursively all QTreeWidgetItem with key name 'parameters'
        Args:
            tree_item (QTreeWidgetItem): Item from which all instances with
            key name 'parameters' are expanded
        """
        if tree_item.data(0, 0) == 'parameters':
            tree_item.setExpanded(True)
        for i in range(tree_item.childCount()):
            self.expandParameters(tree_item.child(i))

    def expandBranch(self, tree_item: qt.QtWidgets.QTreeWidgetItem, expand=True,
                     display_vals=True):
        """
        Expands recursively an entire branch of a QTreeWidget
        Args:
            tree_item (QTreeWidgetItem): item which itself and its children are
                expanded
            expand (bool): True if items are expanded, False if items are
                collapsed
            display_vals (bool): True if values should be expanded.
                Caution: expanding all values at once might take too much time
                and forces the user to force quit the program.
                This might restart the kernel.
        """
        tree_item.setExpanded(expand)
        if not display_vals:
            if tree_item.data(0, 0) == 'parameters':
                return
        for i in range(tree_item.childCount()):
            self.expandBranch(tree_item.child(i), expand=expand,
                              display_vals=display_vals)

    def setDirtext(self):
        """
        Sets the QLabel text self.tree_dir to the directory of the current
        QTreeWidgetItem
        """
        self.tree_dir.setText(self.getDirtext(self.tree_widget.currentItem()))

    def getDirtext(self, tree_item: qt.QtWidgets.QTreeWidgetItem,
                   separator=' > '):
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
        Sets QLabel text of self.attr_nr as the number of attributes (childs)
        of the current QWidgetItem.
        """
        self.attr_nr.setText('Number of attributes: %s'
                             % self.tree_widget.currentItem().childCount())

    def showAll(self, titem: qt.QtWidgets.QTreeWidgetItem, hide=False):
        """
        Shows or hides QTreeWidgetItem with all its children starting by a
        given parent item.
        Args:
            titem (QTreeWidgetItem): Item which itself and its children are
                recursively showed/hidden
            hide (bool): False to hide item+children, True to show item+children
        """
        for i in range(titem.childCount()):
            self.showAll(titem.child(i), hide=hide)
        titem.setHidden(hide)

    def showTitem(self, titem: qt.QtWidgets.QTreeWidgetItem, expand=False):
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

    def hideAllEmpty(self, titem: qt.QtWidgets.QTreeWidgetItem):
        """
        Hides all QTreeWidgetItems which have no children and no value
        (data in second column) starting by given QTreeWidgetItem
        Args:
            titem (QTreeWidgetItem): Item to start hiding empty keys recursively
        """
        if titem.childCount() == 0:
            if titem.data(1, 0) == '':
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
        Copies key or value (first or second column) of the currently selected
        QTreeWidgetItem as a string to the clipboard.
        Args:
            column: 0 for first column (key), 1 for second column (value)
        """
        cb = qt.QtWidgets.QApplication.clipboard()
        cb.clear(mode=cb.Mode.Clipboard)
        if column == 0:
            cb.setText(self.tree_widget.currentItem().data(0, 0),
                       mode=cb.Mode.Clipboard)
        if column == 1:
            cb.setText(self.tree_widget.currentItem().data(1, 0),
                       mode=cb.Mode.Clipboard)

    def make_search_ui(self):
        """
        Creates the UI layout for the search bar and its options.

        Returns: QLayout of search bar
        """
        # Text box
        self.find_box = qt.QtWidgets.QLineEdit()
        self.find_box.returnPressed.connect(self.find_button_clicked)

        # Find Button
        find_button = qt.QtWidgets.QPushButton("Find")
        find_button.clicked.connect(self.find_button_clicked)

        # 'Entries Found' QLabel
        self.find_text = qt.QtWidgets.QLabel('Entries found:')

        self.make_find_checkbox()

        # adding widgets to layout
        layout = qt.QtWidgets.QHBoxLayout()
        layout.addWidget(self.find_box)
        layout.addWidget(find_button)

        layout2 = qt.QtWidgets.QHBoxLayout()
        for key, check_box in self.find_check_box_dict.items():
            layout2.addWidget(check_box)
        # adding 30 pixels horizontally between the buttons to distinguish
        # visually between find_check_box_dict boxes and find_only_params_box
        layout2.addSpacing(30)
        layout2.addWidget(self.find_only_params_box)
        layout2.addStretch()

        layout3 = qt.QtWidgets.QVBoxLayout()
        layout3.addLayout(layout)
        layout3.addLayout(layout2)
        layout3.addWidget(self.find_text)

        return layout3

    def make_find_checkbox(self):
        # search options
        self.find_check_box_dict = \
            {key: qt.QtWidgets.QCheckBox(key.capitalize())
             for key in self.search_options}
        # set default search option
        for key in self.current_search_options:
            self.find_check_box_dict[key].setChecked(True)

        # adding additional search options
        self.find_only_params_box = qt.QtWidgets.QCheckBox(
            'Search only in parameters')
        self.find_only_params_box.setChecked(self.find_only_params)

    def get_current_search_options(self):
        search_options = []
        for key, check_box in self.find_check_box_dict.items():
            if check_box.isChecked():
                search_options.append(key)
        return search_options

    def find_button_clicked(self):
        """
        Action (function) which is started after find_button was clicked.
        Function searches for QTreeWidgetItems and displays only these items
        in the QTreeWidget. Multiple calls of the function with the same
        parameters (search_string, search options) sets the current
        QTreeWidgetItem to the next item which fulfills the search criteria.
        """
        # Sets the current string and search options
        find_str = self.find_box.text()
        find_only_params = self.find_only_params_box.isChecked()

        # tm = timer.Timer("timer", verbose=False)
        # for i in range(10):
        #     self.found_titem_list = self.tree_widget.findItems(
        #         find_str,
        #         qt.QtCore.Qt.MatchFlag.MatchContains | qt.QtCore.Qt.MatchFlag.MatchRecursive,
        #         column=0)
        # tm.checkpoint('current_end')
        # print(tm.duration())
        #
        # tm = timer.Timer("timer2", verbose=False)
        #
        # for i in range(10):
        #     self.found_titem_list = self.text_to_titem.find(
        #             find_str=self.find_str,
        #             search_keys=self.current_search_options)
        # tm.checkpoint('current_end')
        # print(tm.duration())
        # items = self.tree_widget.findItems(
        #     find_str,
        #     qt.QtCore.Qt.MatchFlag(65),
        #                             column=0)
        # print(items)

        # If action fund_button_clicked is called, but the user did not enter
        # a string to search for, the action does nothing
        if find_str == "":
            return

        # only starts a new search if search parameters changed.
        # Otherwise, the next item in self.found_titem_list is activated.
        if find_str != self.find_str \
                or self.current_search_options != \
                self.get_current_search_options() \
                or self.find_only_params != find_only_params:
            self.find_str = find_str
            self.current_search_options = self.get_current_search_options()
            self.find_only_params = find_only_params
            # saves the QTreeWidgetItems which fulfill the search criteria
            # in a list
            # self.found_titem_list = self.text_to_titem.find(
            #     find_str=self.find_str,
            #     search_keys=self.current_search_options)
            self.found_titem_list = self.tree_widget.findItems(
                find_str,
                qt.QtCore.Qt.MatchFlag.MatchContains | qt.QtCore.Qt.MatchFlag.MatchRecursive,
                column=0)
            # If user wants to search only in parameters
            if find_only_params:
                titem_list = []
                for titem in self.found_titem_list:
                    try:
                        if titem.parent().data(0, 0) == 'parameters':
                            titem_list.append(titem)
                    except:
                        pass
                self.found_titem_list = titem_list
            self.find_text.setText('Entries found: %s'
                                   % (len(self.found_titem_list)))
            # If search is not empty, only QTreeWidgetItem which are found
            # are displayed
            if not self.found_titem_list == []:
                self.showAll(self.tree_widget.topLevelItem(0), hide=True)
                for titem in self.found_titem_list:
                    titem.setHidden(False)
                    self.showTitem(titem, expand=True)
            self.found_idx = 0
        # if list of found QTreeWidgetItems is empty and no search parameters
        # are changed, 'Entries found = 0' is displayed,
        # but QTreeWidget stays unchanged
        elif not self.found_titem_list:
            self.find_text.setText('Entries found: %s'
                                   % (len(self.found_titem_list)))
            return
        else:
            # if no search parameters are changed, current item is changed
            # to the next item in the list
            item_num = len(self.found_titem_list)
            self.found_idx = (self.found_idx + 1) % item_num
        try:
            # set current item and display how many entries are found
            # for the current search
            self.tree_widget.setCurrentItem(
                self.found_titem_list[self.found_idx])
            self.setDirtext()
            self.setNrAttributes()
        except:
            # if found index is out of range of found_titem_list
            # (should not happen)
            self.find_text.setText('Entries found: %s'
                                   % (len(self.found_titem_list)))

    def dict_to_titem(self, dictdata: dict,
                      parent: qt.QtWidgets.QTreeWidgetItem,
                      param=False):
        """
        Creates recursively a QTreeWidgetItem tree from a given dictionary.
        If dictionary contains 'parameters', the values of the parameters are
        already displayed in the column of the respective parameter itself, i.e.

        Key                 |   Value   instead of     Key              |  Value
        ...                                            ...
            parameters      |                           parameters      |
                qb1         |   14                       qb1            |
                    ...     |                                ...        |
                    value   |   14                           value      |   14
                    ...     |                                ...        |

        Args:
            dictdata(dict): Dictionary which is turned into a
                QTreeWidgetItem tree
            parent(QTreeWidgetItem): Parent item onto children are added
            param(bool): True if parent is a parameter dictionary,
                i.e. parent.data(0,0) = 'parameters'
        """
        for key, val in dictdata.items():
            self.tree_add_row(str(key), val, parent, param)

    def tree_add_row(self, key: str, val,
                     tree_widget: qt.QtWidgets.QTreeWidgetItem,
                     param=False):
        """
        Adds children to a given tree_widget item from dictionary entries.
        Args:
            key(str): Key of the dictionary as a string.
            val: Item of the dictionary.
            tree_widget (QTreeWidgetItem): Parent tree widget onto key is added
                as a QTreeWidgetItem
            param(bool): True if key is a parameter,
                i.e. tree_widget.data(0,0)= 'parameters'
        """
        if isinstance(val, dict):
            value_text = ''
            # if the key is a parameter, the respective value of the parameter
            # is shown on the row of the parameter
            # for more see docstring of dict_to_titem()
            if param:
                # if parameter has a key called 'value'. the value of this
                # entry is displayed at the level of the parameter. Otherwise,
                # the entire parameter value content is displayed.
                value_text = str(val.get('value', val))
            row_item = qt.QtWidgets.QTreeWidgetItem([key, value_text])
            if key == 'parameters':
                self.dict_to_titem(val, row_item, param=True)
            else:
                self.dict_to_titem(val, row_item, param=False)
        else:
            value_text = str(val)
            row_item = qt.QtWidgets.QTreeWidgetItem([key, str(val)])

        tree_widget.addChild(row_item)
        search_str_dict = {key: '' for key in self.search_options}
        search_str_dict['key'] = key
        search_str_dict['value'] = value_text
        self.text_to_titem.append(search_str_dict, row_item)


class TreeItemViewer(qt.QtWidgets.QWidget):
    """
    Basic viewer for QTreeWidgetItem objects with features:
    - copying key and value (first and second column entry) to the clipboard
    - sorting alphabetically
    """

    def __init__(self, treeitem, screen):
        """
        Initialization of the TreeItemViewer
        Args:
            treeitem (QTreeWidgetItem): Tree item whose children are displayed
            in a QTreeWidget
        """
        super(TreeItemViewer, self).__init__()

        # Initialization of the tree widget
        self.tree_widget = qt.QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(["Key", "Value"])
        self.tree_widget.header().setSectionResizeMode(
            qt.QtWidgets.QHeaderView.ResizeMode.Interactive)
        screen_geom = screen.size()
        # initializes the width of the column: 0.3*screen_size.width() is the
        # width of the entire window defined in AdditionalWindow
        self.tree_widget.header().resizeSection(0,
                                                .5 * .3 * screen_geom.width())
        self.tree_widget.setExpandsOnDoubleClick(True)
        self.tree_widget.setSortingEnabled(True)

        # set up the tree
        for i in range(treeitem.childCount()):
            self.tree_widget.insertTopLevelItem(i, treeitem.child(i).clone())

        # creates the context menu
        self._createActions()
        self._connectActions()
        self.tree_widget.setContextMenuPolicy(
            qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(
            self._setContextMenu)

        # Add tree widget to the layout
        layout = qt.QtWidgets.QHBoxLayout()
        layout.addWidget(self.tree_widget)
        self.setLayout(layout)

    def _createActions(self):
        """
        Function which creates the actions for the context menu and menu bars
        """
        self.copyKeyAction = qt.QtGui.QAction("Copy key")
        self.copyValueAction = qt.QtGui.QAction("Copy value")

    def _connectActions(self):
        """
        Function which connects the actions defined in _createActions with
        events (functions).
        lambda function is needed for functions with arguments.
        """
        self.copyKeyAction.triggered.connect(lambda: self._copyContent(0))
        self.copyValueAction.triggered.connect(lambda: self._copyContent(1))

    def _setContextMenu(self, position):
        """
        Adds the actions to the context menu and displays it at a given position
        Args:
            position (QPoint): Position where the context menu pops up.
        """
        menu = qt.QtWidgets.QMenu()
        menu.addAction(self.copyKeyAction)
        menu.addAction(self.copyValueAction)
        menu.exec(self.tree_widget.viewport().mapToGlobal(position))

    def _copyContent(self, column: int):
        """
        Copies key or value (first or second column) of the currently selected
        QTreeWidgetItem as a string to the clipboard.
        Args:
            column: 0 for first column (key), 1 for second column (value)
        """
        cb = qt.QtWidgets.QApplication.clipboard()
        cb.clear(mode=cb.Mode.Clipboard)
        if column == 0:
            cb.setText(self.tree_widget.currentItem().data(0, 0),
                       mode=cb.Mode.Clipboard)
        if column == 1:
            cb.setText(self.tree_widget.currentItem().data(1, 0),
                       mode=cb.Mode.Clipboard)


class AdditionalWindow(qt.QtWidgets.QMainWindow):
    """
    Class which creates an additional window which contains a QTreeWidget from
    a given QTreeWidgetItem
    """

    def __init__(self, dict_view, screen):
        """
        Initialization of the additional window.
        Args:
            dict_view(QTreeWidgetItem): Tree item which is displayed with its
                children in the additional window
        """
        super(AdditionalWindow, self).__init__()
        self.setCentralWidget(
            TreeItemViewer(dict_view.tree_widget.currentItem(), screen))
        self.setWindowTitle(
            dict_view.getDirtext(dict_view.tree_widget.currentItem()))
        screen_geom = screen.size()
        # layout options (x-coordinate, y-coordinate, width, height) in px
        self.setGeometry(0.2 * screen_geom.width(), 0.2 * screen_geom.height(),
                         0.3 * screen_geom.width(), 0.3 * screen_geom.height())

    def keyPressEvent(self, e):
        """
        Closes window if 'esc' button is pressed
        Args:
            e:
        """
        if e.key() == qt.QtCore.Qt.Key.Key_Escape:
            self.close()


class DictViewerWindow(qt.QtWidgets.QMainWindow):
    """
    Main window to display the dictionary with all the features
    """

    def __init__(self, dic: dict, title: str = '', screen=None):
        """
        Initialization of the main window
        Args:
            dic(dict): Dictionary which is displayed in the main window
            title(str): Title which pops up in the window
        """
        super(DictViewerWindow, self).__init__()
        self.dialogs = list()  # list of additional windows
        self.screen = screen
        dict_view = DictView(dic, title, screen)

        # open new window with double click
        # dict_view.tree_widget.itemDoubleClicked.connect(
        #   lambda: self.openNewWindow(dict_view))
        # open new window via content menu
        dict_view.openContentAction.triggered.connect(
            lambda: self.openNewWindow(dict_view))

        self.setCentralWidget(dict_view)
        self.setWindowTitle("Snapshot Viewer")
        screen_geom = screen.size()
        # layout options (x-coordinate, y-coordinate, width, height) in px
        self.setGeometry(0.1 * screen_geom.width(), 0.1 * screen_geom.height(),
                         0.4 * screen_geom.width(), 0.7 * screen_geom.height())

        self.show()

    def openNewWindow(self, dict_view):
        """
        Opens a new window from a given DictView object
        Args:
            dict_view(DictView): DictView object with a tree_widget attribute
                which current item is displayed in a new window.
        """
        dialog = AdditionalWindow(dict_view, self.screen)
        self.dialogs.append(dialog)
        dialog.show()

    def keyPressEvent(self, e):
        """
        Closes all windows if 'esc' button is pressed
        Args:
            e:
        """
        if e.key() == qt.QtCore.Qt.Key.Key_Escape:
            for d in self.dialogs:
                d.close()
            self.close()


class SnapshotViewer:
    def __init__(self, snapshot: dict, timestamp: str):
        self.snapshot = snapshot
        self.timestamp = timestamp

    def spawn_snapshot_viewer(self):
        qt_app = qt.QtWidgets.QApplication(sys.argv)
        screen = qt_app.primaryScreen()
        snap_viewer = DictViewerWindow(
            dic=self.snapshot,
            title='Snapshot timestamp: %s' % self.timestamp,
            screen=screen)
        qt_app.exec_()


if __name__ == "__main__":
    import argparse
    import pycqed.utilities.settings_manager as sm
    import pycqed.analysis.analysis_toolbox as a_tools

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Directory of the settings.")
    parser.add_argument("--fetch-data-dir",
                        help="Directory of the fetch drive.")
    parser.add_argument("--timestamp", help="Timestamp of the settings.",
                        required=True,
                        type=str)
    parser.add_argument("--file-format",
                        help="File format of the settings: "
                             "hdf5, pickle or msgpack.",
                        required=True,
                        type=str)
    parser.add_argument("--compression",
                        help="Set to True if file is compressed with blosc2",
                        type=bool,
                        default=False)
    parser.add_argument("--extension",
                        help="Extension of the file if not standardized: "
                             "hdf5: .hdf5, pickle: .pickle, msgpack: .msgpack "
                             "and an extra c at the end for compressed files.")
    args = parser.parse_args()

    if args.data_dir is not None:
        a_tools.datadir = args.data_dir

    if args.fetch_data_dir is not None:
        a_tools.fetch_data_dir = args.fetch_data_dir

    if (args.data_dir is None) and (args.fetch_data_dir is None):
        raise NotImplementedError("Data directory or fetch directory needs "
                                  "to be initialized")

    settings_manager = sm.SettingsManager()
    settings_manager.load_from_file(timestamp=args.timestamp,
                                    file_format=args.file_format,
                                    compression=args.compression,
                                    extension=args.extension)

    settings_manager.spawn_snapshot_viewer(timestamp=args.timestamp)
