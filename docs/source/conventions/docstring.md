# Docstring format

All docstrings for Python objects (modules, classes, functions, etc.) should
conform to
[Google's style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
of docstrings. Please refer to this document for more information and
justifications regarding this choice of format.

## Additional remarks

### Always use double-quotes for docstrings

```python
def correct_docstring():
"""Do."""

def incorrect_docstring():
'''Don't.'''
```

### Argument types

The arguments type(s) should be included as follows, if the code does not
contain a corresponding type annotation:
```python
def function(x, y, z):
    """Some function.

    Arguments:
        x (int): Some parameter.
        y (str): Other parameter.
        z (float|List[float]): Third parameter.
    """
```
Always indicate the acceptable types for `*args` or `**kwargs` arguments.

## Modifying existing docstrings

If you modify an existing docstring, use its current docstring format even if it
does not conform with our conventions. If you have to rewrite most of an
existing docstring, then use our docstring format. 

**If you are adding new docstrings in a file, you should use our docstring format
even if the other existing docstrings do not conform with our conventions.**

## Checking the proper formatting of the documentation

Please make sure to build the documentation locally (as described in this repo's
README) and correct the formatting of your docstrings if any error/warning is
thrown by Sphinx. As PycQED is a large codebase, the error/warning list can
rapidly become overwhelming if it is not kept under control.

## Example of docstrings

You can have a look at the
[docstring examples](https://documentation.qudev.phys.ethz.ch/template_python_project/main/included_source_files/example_class.html)
from the `Template Python Project` repository for examples of how to write
Google-styled docstrings. These examples also show how to cross-reference python
module/classes/functions.

TODO: Add real-case example of well-written docstring from PycQED conforming
with the Google style.

## Setting up your IDE to write docstrings more easily

Most modern IDEs (e.g. VS Code, PyCharm) include plugins or snippets to write
docstrings faster. Make sure to set your IDE in a way that encourages you to
write good docstrings.

Feel free to try the following plugins:
* VS Code extension: [Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
This extension can inspect your functions to produce a formatted docstring with
the arguments, return type and raised exceptions.
* TODO: Add Similar extension for PyCharm