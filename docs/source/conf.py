# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(os.path.abspath("../../../pycqed_scripts"))
sys.path.append(os.path.abspath("./ext"))

# Custom variable __sphinx_build__ which can be used to check inside the code
# if the documentation is being built.
import builtins
builtins.__sphinx_build__ = True


# -- Project information -----------------------------------------------------

project = "PycQED"
copyright = "2023, Quantum Device Laboratory"


# -- General configuration ---------------------------------------------------

import sphinx_rtd_theme

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
# FIXME: This can't handle our __init__ arguments which are sometimes added
# to the class docstrings.
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "myst_parser",
# FIXME: This one is broken
#    "autodoc_instrument",
]

# Enable writing pages both in ReStructuredText and Markdown
source_suffix = [".rst", ".md"]

# Include figure number
numfig = True

# Md -> ReST extension configuration
myst_enable_extensions = [
    "amsmath", # Latex math expressions
    "dollarmath", # Latex math expressions
    "tasklist", # Checklists
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Autodoc and autosummary -------------------------------------------------

# Do not prepend module names to avoid long, overflowing titles
# FIXME: This is actually a problem
add_module_names = False

# FIXME: Lots of bugs...
autosummary_generate = False

# FIXME: See above
autodoc_default_options = {
    "undoc-members": True, # Include members with no docstrings
    # Include following dunder functions in docs
    "special-members": "__init__, __iter__, __next__",
    # FIXME: autodoc just goes nuts here when using more than one core via Makefile
    #"exclude-members": "pycqed.instrument_drivers.physical_instruments.arduino_switch_control.SwitchError",
}


# -- intersphinx documentation -----------------------------------------------

intersphinx_mapping = {
    # Can be used to add links to Python Standard library
    "python": ("https://docs.python.org/3.9/", None),
    # Can be used to add links to qcodes objects
    "qcodes": ("https://qcodes.github.io/Qcodes/", None),
    "h5py": ('https://docs.h5py.org/en/latest', None),
    "numpy": ('https://numpy.org/doc/stable', None),
    "matplotlib": ('https://matplotlib.org/stable', None),
}


# -- Options for HTML output -------------------------------------------------

# Use Read-the-docs theme instead of default theme
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Hide "View page source" link
html_show_sourcelink = False


# -- autodoc_instrument extension --------------------------------------------

# FIXME: This one is broken
# autodoc_instrument_configs_file = os.path.abspath("./autodoc_instrument_configs.yaml")

# -- enable macOS doc building --------------------------------------------

import multiprocessing

try:
    multiprocessing.set_start_method('forkserver')
except RuntimeError:
    pass  # Method already set