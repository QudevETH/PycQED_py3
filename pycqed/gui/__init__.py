# This import ensures that qt_compat is run everytime anything from the gui
# package is imported. And qt_compat itself ensures that gui works regardless
# whether it uses PyQt or PySide (there are some method naming differences
# which are fixed with qt_compat)
from pycqed.gui import qt_compat
