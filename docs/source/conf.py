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

sys.path.insert(0, os.path.abspath('../..'))
sys.path.append(os.path.abspath("./ext"))

# Custom variable __sphinx_build__ which can be used to check inside the code
# if the documentation is being built.
import builtins
builtins.__sphinx_build__ = True

# -- Project information -----------------------------------------------------

project = 'PycQED'
copyright = '2021, QuDev'

# -- General configuration ---------------------------------------------------

import sphinx_rtd_theme

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'myst_parser',
    'autodoc_instrument',
]

source_suffix = ['.rst', '.md']

numfig = True

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "tasklist",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Autodoc and autosummary -------------------------------------------------

# Do not prepend module names to avoid long, overflowing titles
add_module_names = False

autosummary_generate = True

autodoc_default_options = {
    'undoc-members': True,
}

# -- Options for HTML output -------------------------------------------------

# Use Read-the-docs theme instead of default theme
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Hide "View page source" link
html_show_sourcelink = False


# -- autodoc_instrument extensin ---------------------------------------------

autodoc_instrument_configs_file = os.path.abspath("./autodoc_instrument_configs.yaml")
