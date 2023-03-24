from pycqed.gui import qt_compat as qt
import sys
import multiprocessing as mp
import logging
log = logging.getLogger(__name__)


class DictView(qt.QtWidgets.QWidget):
    """
    Class to display a given dictionary with features:
    - search for string in keys and/or values
    - copy value and key to clipboard
    - expand, collapse and hide keys/branches
    """

    def __init__(self, snap: dict, title: str = '', screen=None,
                 timestamps=None):
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
        if timestamps is None:
            self.column_header = ['Key', 'Value']
        else:
            self.column_header = ['Key'] + timestamps

        # Default values for the search bar
        self.current_search_options = ['Key']  # respective name of the column
        self.find_str = ""
        self.found_titem_list = []
        self.find_check_box_dict = {}
        self.found_idx = 0
        # set to True if search should only include parameters
        self.find_only_params = False

        # set up the tree
        self.tree_widget = qt.QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(self.column_header)
        self.tree_widget.header().setSectionResizeMode(
            qt.QtWidgets.QHeaderView.ResizeMode.Interactive)

        # initializes the width of the column: 0.4*screen_size.width() is the
        # width of the entire window defined in DictViewerWindow
        screen_size = screen.size()
        self.tree_widget.header().resizeSection(
            0, int(.5 * .4 * screen_size.width()))
        self.tree_widget.setExpandsOnDoubleClick(True)

        self.root_item = self.tree_widget.invisibleRootItem()
        # build up the tree recursively from a dictionary
        self.dict_to_titem(snap, self.root_item)
        self.tree_widget.addTopLevelItem(self.root_item)

        # expand all parameter branches by default
        self.expand_parameters(self.root_item)

        # sorting needs to be set after initialization of tree elements
        # lets user choose which column is used to sort
        self.tree_widget.setSortingEnabled(True)
        # default is first column in ascending order
        self.tree_widget.sortByColumn(0, qt.QtCore.Qt.SortOrder.AscendingOrder)
        self.create_menus()  # creates context menu and menu bar
        self.make_layout()  # creates layout

    def create_menus(self):
        self._create_actions()
        self._connect_actions()

        # context menu
        self.tree_widget.setContextMenuPolicy(
            qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(
            self._set_context_menu)

        # menu bar
        self.menuBar = qt.QtWidgets.QMenuBar(self)
        self._set_menu_bar()

    def make_layout(self):
        search_layout = self.make_search_ui()  # Load search layout

        # QLabel with text of tree directory
        self.tree_dir = qt.QtWidgets.QLabel('Root')
        # long text should not prevent the window from being resized
        self.tree_dir.setMinimumSize(1, 1)

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
        # long title should not prevent the window from being resized
        gbox.setMinimumSize(1, 1)
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

    def _create_actions(self):
        """
        Function which creates all the actions for the context menu and
        menu bars
        """
        self.copyKeyAction = qt.QAction("Copy Key")
        self.copyValueAction = qt.QAction("Copy Value")
        self.openContentAction = qt.QAction("Open in New Window")
        self.hideAction = qt.QAction("Hide Key")
        self.hideAllAction = qt.QAction("Hide all Empty")
        self.showAllAction = qt.QAction("Show all")
        self.collapseAction = qt.QAction("Collapse all")
        self.expandBranchAction = qt.QAction("Expand Branch")
        self.collapseBranchAction = qt.QAction("Collapse Branch")
        self.closeAction = qt.QAction("Close Window")
        self.resetWindowAction = qt.QAction("Reset Window")
        self.expandParametersAction = qt.QAction("Expand Parameters")
        self.copyStationPathAction = qt.QAction("Copy Station Path")

    def _connect_actions(self):
        """
        Function which connects the actions defined in _createActions with
        events (functions). lambda function is needed for functions with
        arguments.
        """
        self.copyKeyAction.triggered.connect(lambda: self.copy_content(0))
        self.copyValueAction.triggered.connect(lambda: self.copy_content(1))
        self.hideAction.triggered.connect(self.hide_item)
        self.hideAllAction.triggered.connect(
            lambda: self.hide_all_empty(self.root_item))
        self.showAllAction.triggered.connect(
            lambda: self.show_all(self.root_item))
        self.tree_widget.itemClicked.connect(self.set_dirtext)
        self.tree_widget.itemActivated.connect(self.set_dirtext)
        self.tree_widget.itemClicked.connect(self.set_nr_attributes)
        self.tree_widget.itemActivated.connect(self.set_nr_attributes)

        self.collapseAction.triggered.connect(
            lambda: self.expand_branch(self.root_item, expand=False))
        self.expandBranchAction.triggered.connect(
            lambda: self.expand_branch(self.tree_widget.currentItem(),
                                       expand=True, display_vals=False))
        self.collapseBranchAction.triggered.connect(
            lambda: self.expand_branch(self.tree_widget.currentItem(),
                                       expand=False))
        self.closeAction.triggered.connect(self.close)
        self.resetWindowAction.triggered.connect(self.reset_window)
        self.expandParametersAction.triggered.connect(
            lambda: self.expand_parameters(self.root_item))
        self.copyStationPathAction.triggered.connect(self.copy_station_path)

    def _set_menu_bar(self):
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
        editMenu.addAction(self.copyStationPathAction)
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

    def _set_context_menu(self, position):
        """
        Adds the actions to the context menu and displays it at a given position.
        Args:
            position (QPoint): Position where the context menu pops up.
        """
        menu = qt.QtWidgets.QMenu()

        menu.addAction(self.copyKeyAction)
        menu.addAction(self.copyValueAction)
        menu.addAction(self.copyStationPathAction)
        menu.addSeparator()
        menu.addAction(self.openContentAction)
        menu.addSeparator()
        menu.addAction(self.hideAction)
        menu.addAction(self.expandBranchAction)
        menu.addAction(self.collapseBranchAction)
        menu.exec_(self.tree_widget.viewport().mapToGlobal(position))

    def reset_window(self):
        """
        Resets the QTreeWidget to the default layout:
        - no QTreeWidgetItem is hidden
        - only root item and parameters are expanded
        """
        self.show_all(self.root_item)
        self.expand_branch(self.root_item, expand=False)
        self.expand_parameters(self.root_item)

    def expand_parameters(self, tree_item: qt.QtWidgets.QTreeWidgetItem):
        """
        Expands recursively all QTreeWidgetItem with key name 'parameters'
        Args:
            tree_item (QTreeWidgetItem): Item from which all instances with
            key name 'parameters' are expanded
        """
        if tree_item.data(0, 0) == 'parameters':
            tree_item.setExpanded(True)
        for i in range(tree_item.childCount()):
            self.expand_parameters(tree_item.child(i))

    def expand_branch(self, tree_item: qt.QtWidgets.QTreeWidgetItem,
                      expand=True, display_vals=True):
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
            self.expand_branch(tree_item.child(i), expand=expand,
                               display_vals=display_vals)

    def set_dirtext(self):
        """
        Sets the QLabel text self.tree_dir to the directory of the current
        QTreeWidgetItem
        """
        self.tree_dir.setText(self.get_dirtext(self.tree_widget.currentItem()))

    def get_dirtext(self, tree_item: qt.QtWidgets.QTreeWidgetItem,
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
            dir = self.get_dirtext(tree_item.parent()) + separator + dir
        except:
            pass
        return dir

    def set_nr_attributes(self):
        """
        Sets QLabel text of self.attr_nr as the number of attributes (childs)
        of the current QWidgetItem.
        """
        self.attr_nr.setText('Number of attributes: %s'
                             % self.tree_widget.currentItem().childCount())

    def show_all(self, titem: qt.QtWidgets.QTreeWidgetItem, hide=False):
        """
        Shows or hides QTreeWidgetItem with all its children starting by a
        given parent item.
        Args:
            titem (QTreeWidgetItem): Item which itself and its children are
                recursively showed/hidden
            hide (bool): False to hide item+children, True to show item+children
        """
        for i in range(titem.childCount()):
            self.show_all(titem.child(i), hide=hide)
        titem.setHidden(hide)

    def show_titem(self, titem: qt.QtWidgets.QTreeWidgetItem, expand=False):
        """
        Shows QTreeWidgetItem and its parent recursively until root_item.
        Args:
            titem (QTreeWidgetItem): Item which is displayed
            expand (bool): True to expand of Item which is shown
        """
        titem.setHidden(False)
        titem.setExpanded(expand)
        try:
            self.show_titem(titem.parent(), expand=expand)
        except:
            pass

    def hide_all_empty(self, titem: qt.QtWidgets.QTreeWidgetItem):
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
                self.hide_all_empty(titem.child(i))

    def hide_item(self):
        """
        Hides currently selected QTreeWidgetItem
        """
        self.tree_widget.currentItem().setHidden(True)

    def copy_content(self, column: int):
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

    def get_station_path(self, titem: qt.QtWidgets.QTreeWidgetItem):
        key_list = []
        while titem.parent() is not None:
            key_list.append(str(titem.data(0, 0)))
            titem = titem.parent()
        key_list.reverse()

        if len(key_list) > 2:
            return key_list[0]+'.'+key_list[2]
        elif len(key_list) > 1:
            return key_list[0]
        else:
            return ''

    def copy_station_path(self):
        cb = qt.QtWidgets.QApplication.clipboard()
        cb.clear(mode=cb.Mode.Clipboard)
        station_path = 'sm.stations[''].' + \
                       self.get_station_path(self.tree_widget.currentItem()) + \
                       '()'

        cb.setText(station_path, mode=cb.Mode.Clipboard)

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
        if len(self.column_header) < 4:
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
            {key: qt.QtWidgets.QCheckBox(key)
             for key in self.column_header}
        # set default search option
        for key in self.current_search_options:
            self.find_check_box_dict[key].setChecked(True)

        # adding additional search options
        self.find_only_params_box = qt.QtWidgets.QCheckBox(
            'Search only in parameters')
        self.find_only_params_box.setChecked(self.find_only_params)

    def get_current_search_options(self, as_int=False):
        """
        Returns the current search options set by the checkboxes.
        Args:
            as_int (bool): If true, returns the current search options as a
                integer. The integer corresponds to the index of the string
                in the list self.search_options

        Returns: Current search options set by check boxes of the gui as an
            integer or as a string (column name).

        """
        search_options = []
        for key, check_box in self.find_check_box_dict.items():
            if check_box.isChecked():
                if as_int:
                    search_options.append(self.column_header.index(key))
                else:
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

            self.found_titem_list = []
            for column in self.get_current_search_options(as_int=True):
                self.found_titem_list = \
                    self.found_titem_list + \
                    self.tree_widget.findItems(
                        find_str,
                        qt.QtCore.Qt.MatchFlag.MatchContains |
                        qt.QtCore.Qt.MatchFlag.MatchRecursive,
                        column=column)

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
                self.show_all(self.tree_widget.topLevelItem(0), hide=True)
                for titem in self.found_titem_list:
                    titem.setHidden(False)
                    self.show_titem(titem, expand=True)
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
            self.set_dirtext()
            self.set_nr_attributes()
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
            row_item = qt.QtWidgets.QTreeWidgetItem([key, str(val)])
        if '\n' in row_item.data(1, 0):
            row_item.setSizeHint(1, qt.QtCore.QSize(100, 50))
        tree_widget.addChild(row_item)


class ComparisonDictView(DictView):
    """
    QWidget class to display a given dictionary from a comparison of different
     stations in multiple columns. Some functions of DictView need to be
     overwritten.
    """
    def tree_add_row(self, key: str, val: dict,
                        tree_widget: qt.QtWidgets.QTreeWidgetItem, param=False):
        from pycqed.utilities.settings_manager import Timestamp as Timestamp
        values = [''] * len(self.column_header[1:])
        if all((tsp in self.column_header[1:] and
               (isinstance(tsp, Timestamp)))
               for tsp in val.keys()):
            for param in val.keys():
                if isinstance(val[param], dict):
                    values[self.column_header[1:].index(param)] =\
                        str(val[param].get('value', val[param]))
                else:
                    values[self.column_header[1:].index(param)] = \
                        str(val[param])
            row_item = qt.QtWidgets.QTreeWidgetItem([key] + values)
            for i in range(len(self.column_header) - 1):
                if not self.column_header[i+1] in val.keys():
                    row_item.setBackground(i+1, qt.QtGui.QBrush(
                        qt.QtGui.QColor('darkGrey')))
        else:
            row_item = qt.QtWidgets.QTreeWidgetItem([key] + values)
            self.dict_to_titem(val, row_item)

        if any('\n' in row_item.data(i, 0)
               for i in range(len(self.column_header))):
            for i in range(len(self.column_header)):
                row_item.setSizeHint(i, qt.QtCore.QSize(100, 50))
        tree_widget.addChild(row_item)

    def copy_content(self, column: int):
        cb = qt.QtWidgets.QApplication.clipboard()
        cb.clear(mode=cb.Mode.Clipboard)
        if column == 0:
            cb.setText(self.tree_widget.currentItem().data(0, 0),
                       mode=cb.Mode.Clipboard)
        if column == 1:
            data = {
                self.column_header[i+1]:
                    self.tree_widget.currentItem().data(i+1, 0)
                for i in range(len(self.column_header[1:]))}
            cb.setText(str(data),
                       mode=cb.Mode.Clipboard)

    def get_station_path(self, titem: qt.QtWidgets.QTreeWidgetItem):
        key_list = []
        while titem.parent() is not None:
            key_list.append(str(titem.data(0, 0)))
            titem = titem.parent()
        key_list.reverse()

        if len(key_list) > 2:
            return key_list[0]+'.'+key_list[2]
        elif len(key_list) > 0:
            return key_list[0]
        else:
            return ''


class TreeItemViewer(qt.QtWidgets.QWidget):
    """
    Basic viewer for QTreeWidgetItem objects with features:
    - copying key and value (first and second column entry) to the clipboard
    - sorting alphabetically
    """

    def __init__(self, treeitem, screen, column_header):
        """
        Initialization of the TreeItemViewer
        Args:
            treeitem (QTreeWidgetItem): Tree item whose children are displayed
            in a QTreeWidget
        """
        super(TreeItemViewer, self).__init__()

        # Initialization of the tree widget
        self.tree_widget = qt.QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(column_header)
        self.tree_widget.header().setSectionResizeMode(
            qt.QtWidgets.QHeaderView.ResizeMode.Interactive)
        screen_geom = screen.size()
        # initializes the width of the column: 0.3*screen_size.width() is the
        # width of the entire window defined in AdditionalWindow
        self.tree_widget.header().resizeSection(
            0, int(.5 * .3 * screen_geom.width()))
        self.tree_widget.setExpandsOnDoubleClick(True)
        self.tree_widget.setSortingEnabled(True)

        # set up the tree
        for i in range(treeitem.childCount()):
            self.tree_widget.insertTopLevelItem(i, treeitem.child(i).clone())

        # creates the context menu
        self._create_actions()
        self._connect_actions()
        self.tree_widget.setContextMenuPolicy(
            qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(
            self._set_context_menu)

        # Add tree widget to the layout
        layout = qt.QtWidgets.QHBoxLayout()
        layout.addWidget(self.tree_widget)
        self.setLayout(layout)

    def _create_actions(self):
        """
        Function which creates the actions for the context menu and menu bars
        """
        self.copyKeyAction = qt.QAction("Copy key")
        self.copyValueAction = qt.QAction("Copy value")

    def _connect_actions(self):
        """
        Function which connects the actions defined in _createActions with
        events (functions).
        lambda function is needed for functions with arguments.
        """
        self.copyKeyAction.triggered.connect(lambda: self._copy_content(0))
        self.copyValueAction.triggered.connect(lambda: self._copy_content(1))

    def _set_context_menu(self, position):
        """
        Adds the actions to the context menu and displays it at a given position
        Args:
            position (QPoint): Position where the context menu pops up.
        """
        menu = qt.QtWidgets.QMenu()
        menu.addAction(self.copyKeyAction)
        menu.addAction(self.copyValueAction)
        menu.exec_(self.tree_widget.viewport().mapToGlobal(position))

    def _copy_content(self, column: int):
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
            TreeItemViewer(dict_view.tree_widget.currentItem(),
                           screen, dict_view.column_header))
        self.setWindowTitle(
            dict_view.get_dirtext(dict_view.tree_widget.currentItem()))
        screen_geom = screen.size()
        # layout options (x-coordinate, y-coordinate, width, height) in px
        self.setGeometry(int(0.2 * screen_geom.width()),
                         int(0.2 * screen_geom.height()),
                         int(0.3 * screen_geom.width()),
                         int(0.3 * screen_geom.height()))

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

    def __init__(self, dic: dict, title: str = '', screen=None,
                 timestamps=None):
        """
        Initialization of the main window
        Args:
            dic(dict): Dictionary which is displayed in the main window
            title(str): Title which pops up in the window
        """
        super(DictViewerWindow, self).__init__()
        self.dialogs = list()  # list of additional windows
        self.screen = screen
        if timestamps is None:
            widget = DictView(dic, title, screen, timestamps=timestamps)
            self.setWindowTitle("Snapshot Viewer")
        else:
            widget = ComparisonDictView(dic, title, screen, timestamps)
            self.setWindowTitle("Comparison Viewer")



        # open new window with double click
        # dict_view.tree_widget.itemDoubleClicked.connect(
        #   lambda: self.openNewWindow(dict_view))
        # open new window via content menu
        widget.openContentAction.triggered.connect(
            lambda: self.open_new_window(widget))

        self.setCentralWidget(widget)

        screen_geom = screen.size()
        # layout options (x-coordinate, y-coordinate, width, height) in px
        self.setGeometry(int(0.1 * screen_geom.width()),
                         int(0.1 * screen_geom.height()),
                         int(0.4 * screen_geom.width()),
                         int(0.7 * screen_geom.height()))

        self.show()

    def open_new_window(self, dict_view):
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
    def __init__(self, snapshot: dict, timestamp):
        self.snapshot = snapshot
        self.timestamp = timestamp

    def _prepare_new_process(self):
        """
        Helper function to start a new process. Sets the start method.
        """
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            if mp.get_start_method() != 'spawn':
                log.warning('Child process should be spawned')

    def spawn_viewer(self, new_process=False):
        """
        Spawns the dict viewer. Either spawns the snapshot viewer if
        self.timestamps (timestamps is set to None) or the comparison viewer if
        self.timestamps is a list if timestamps (timestamps set to this list)
        Args:
            new_process (bool): True if new process should be started, which
                does not block the IPython kernel. False by default because
                it takes some time to start the new process.
        """
        if new_process:
            self._prepare_new_process()
            from pycqed.gui.gui_process import dict_viewer_process
            qt_lib = qt.QtWidgets.__package__
            args = (self.snapshot, self.timestamp, qt_lib,)
            process = mp.Process(target=dict_viewer_process,
                                 args=args)
            process.daemon = False
            process.start()
        else:
            if isinstance(self.timestamp, list):
                title = 'Comparison of %s snapshots' % len(self.timestamp)
                timestamps = self.timestamp
            else:
                title = 'Snapshot timestamp: %s' % self.timestamp
                timestamps = None
            if not qt.QtWidgets.QApplication.instance():
                qt_app = qt.QtWidgets.QApplication(sys.argv)
            else:
                qt_app = qt.QtWidgets.QApplication.instance()
            screen = qt_app.primaryScreen()
            viewer = DictViewerWindow(
                dic=self.snapshot,
                title=title,
                screen=screen,
                timestamps=timestamps)
            qt_app.exec_()


def get_snapshot_from_filepath(filepath):
    """
    Returns the snapshot of the instrument settings of a given file.
    Args:
        filepath (str): OS path of the instrument settings file. File extension
            must be in pycqed.utilities.settings_manager.file_extensions.

    Returns: Snapshot as a dictionary
    """
    from pycqed.utilities.settings_manager import get_loader_from_file
    loader = get_loader_from_file(filepath=filepath)
    if hasattr(loader, 'get_snapshot'):
        return loader.get_snapshot()
    else:
        station = loader.get_station()
        return station.snapshot()


if __name__ == "__main__":
    """
    The next lines are needed to open a the dict viewer via the command line.
    E.g. "python dict_viewer --filepath %filepath%"
    This feature is used to open files via the context menu 
    (see pycqedscripts/scripts/open_instrument_settings)
    """
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath",
                        help="Filepath of the instrument settings file.",
                        required=True,
                        type=str)
    args = parser.parse_args()

    filepath = args.filepath
    snap = get_snapshot_from_filepath(filepath)

    snapshot_viewer = SnapshotViewer(
        snapshot=snap,
        timestamp=filepath)
    snapshot_viewer.spawn_viewer()
