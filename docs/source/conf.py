# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "Active Segmentation"
copyright = (
    '2022, Chair of Digital Health - Machine Learning @ Hasso-Plattner Institute'
)
author = 'Chair of Digital Health - Machine Learning @ Hasso-Plattner Institute'

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
]

# build the templated autosummary files
autosummary_generate = True
numpydoc_show_class_members = False
napoleon_use_ivar = True

# Disable docstring inheritance
autodoc_inherit_docstrings = False

# Disable displaying type annotations, these can be very verbose
autodoc_typehints = "none"

# Enable overriding of function signatures in the first line of the docstring.
autodoc_docstring_signature = True

html_css_files = ["https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.css"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "trojanzoo_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/HealthML/active-segmentation",
    "logo": "",
    "logo_dark": "",
    "logo_icon": "",
}
