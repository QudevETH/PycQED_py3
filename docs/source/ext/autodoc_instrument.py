"""Sphinx autodoc extension adding parameter docstrings.

This extension will try to instantiate the QCodes instrument subclasses to be
able to retrieve their instrument parametrs and associated documentation.

To enable this extension, add it to the list of extensions in your sphinx config
file (``conf.py``):

.. code:: python

    extensions = [
        # ...
        'autodoc_instrument',
    ]

You can also optionally specify in the Sphinx config a YAML file containing
custom parameters necessary when instantiating some instrument classes:

.. code:: python

    autodoc_instrument_configs_file = os.path.abspath("./autodoc_instrument_configs.yaml")

This file must have the following structure:

.. code:: yaml

    # These instruments will be ignored by the extension
    _skipped_instruments:
      - InstrumentX
      - InstrumentY
      # You can also specify the full import path to the instrument in case of
      # colliding names
      - pycqed.instrument_drivers.physical_instruments.instrument_z.InstrumentZ

    # Specify the name of an instrument class
    InstrumentA:
      # Specify its arguments values
      address: ""
      values: []

    InstrumentB:
      station:
        # If you need to instantiate a class for an input parameter, use its
        # full import path
        _class: "qcodes.station"
        # Optional constructor arguments
        _args:
          config_file: "config.yaml"

    InstrumentMonitor:
    station:
        _class: "qcodes.Station"
"""

from functools import lru_cache
from inspect import cleandoc
import os
import importlib
from typing import Any, Dict, Optional
from jinja2 import TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment
from sphinx.application import Sphinx
from sphinx.ext.autodoc import ClassDocumenter
from sphinx.util import logging
from sphinx.util.template import SphinxFileSystemLoader
from docutils.statemachine import StringList
from qcodes import Instrument
from qcodes.instrument.parameter import _BaseParameter
import yaml


logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def _get_template(template_path:str):

    # Create a template loader which looks in the correct folders, for
    # reference see 'sphinx.ext.autosummary.AutosummaryRenderer.__init__()'
    templates_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), '..', '_templates',
    )
    loader = SphinxFileSystemLoader(templates_path)
    env = SandboxedEnvironment(loader=loader)

    try:
        return env.get_template(template_path)
    except TemplateNotFound:
        logger.error("Could not find templates for instruments.")
        raise

def get_fully_qualified_name(object) -> str:
    return f"{object.__module__}.{object.__qualname__}"

def import_from_string(fully_qualified_name:str) -> object:
    """Imports an object from a fully qualified import string.

    Example::

        import_str = "pycqed.instrument_drivers.physical_instruments.ZurichInstruments.ZI_PQSC.ZI_PQSC"
        ZI_PQSC = import_class_from_string(import_str)
        instrument = ZI_PQSC(...)

    Args:
        fully_qualified_name: Full python import path for a class/function.

    Returns:
        [object]: Class, function, variable, ...
    """

    module_name, class_name = fully_qualified_name.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def clean_qcodes_doc(docstring:str) -> str:
    """Trim Qcodes appended data in docstring."""

    docstring = cleandoc(docstring)

    append_index = docstring.find("Parameter class:")
    if append_index != -1:
        docstring = docstring[:append_index]

    return docstring

class _NoInitialValue:
    pass

class InstrumentDocumenter(ClassDocumenter):
    """Documenter with custom behavior for QCodes' instrument subclasses."""

    objtype = 'class'
    directivetype = 'class'
    priority = 10 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)
    instrument_configs = {}

    @classmethod
    def can_document_member(cls, member:Any, membername:str, isattr:bool,
                            parent:Any) -> bool:
        return super().can_document_member(member, membername, isattr, parent)

    def add_content(self, more_content: Optional[StringList],
                    no_docstring: bool = False) -> None:
        """Add instrument parameters to the documentation."""

        super().add_content(more_content, no_docstring)

        if issubclass(self.object, Instrument):

            if more_content is not None:
                self.add_extra_doc()

    def add_extra_doc(self):

        if self.object.__name__ in \
            self.instrument_configs["_skipped_instruments"] or \
            get_fully_qualified_name(self.object) in \
            self.instrument_configs["_skipped_instruments"]:
            return

        # Try to retrieve additional params
        params = self.instrument_configs.get(self.object.__name__, {})

        # Instantiate classes if there are class parameters
        for p in params.keys():
            if isinstance(params[p], dict) and params[p].get("_class", None):
                param_cls = import_from_string(params[p]["_class"])
                params[p] = param_cls(**params[p].get("_args", {}))

        try:
            instrument:Instrument = self.object(
                name=self.object.__name__, **params
            )
        except:
            logger.warning("Could not instantiate instrument "\
                          f"'{self.object.__name__}' with only name and " \
                          f"additional params: {params}.")
            return

        self.add_parameter_table(instrument.parameters)

        for param in instrument.parameters.values():
            self.add_parameter_detail(param)

        instrument.close()

    def add_parameter_table(self, parameters:Dict[str, _BaseParameter]):
        """Add a section with a summary table of the parameters."""

        context = {
            "parameters": {
                name: {
                    "unit": param.unit,
                    "docstring": " ".join(clean_qcodes_doc(param.__doc__) \
                                    .splitlines()),
                }
                for (name, param) in parameters.items()
            },
        }

        table = _get_template("autosummary/parameter_table.rst").render(context)
        lines = table.splitlines()

        for l in lines:
            self.add_line(l, source="")

    def add_parameter_detail(self, param:_BaseParameter):
        """Add a detailed description of the parameter."""

        initial_value = _NoInitialValue

        # The following code does not work on our current QCodes fork
        # if param.gettable and hasattr(param, 'cache'):
        #     initial_value = param.cache.get(False)

        if initial_value is _NoInitialValue:
            initial_value = ""
        else:
            try:
                initial_value = repr(initial_value)
            except:
                initial_value = ""
                logger.warning("Could not convert initial value to string " \
                              f"for parameter '{param.name}' of instrument " \
                              f"'{param.instrument.name}'.")

        context = {
            "name": param.name,
            "unit": param.unit,
            "label": param.label,
            "label_contains_latex": any([c in param.label for c in "^}{"]),
            "vals": repr(param.vals),
            "initial_value": initial_value,
            "docstring": " ".join(clean_qcodes_doc(param.__doc__).splitlines()),
        }

        detail = _get_template("autosummary/parameter_detail.rst").render(context)

        lines = detail.splitlines()
        lines.append("")

        for l in lines:
            self.add_line(l, source="")


def load_config_file(app, config):
    filepath = config.autodoc_instrument_configs_file

    if filepath:
        with open(filepath) as file:
            InstrumentDocumenter.instrument_configs = yaml.safe_load(file)
            logger.info(f"autodoc_instrument: Loaded configs file:\n{filepath}")
    else:
        logger.info(f"autodoc_instrument: No configs file specified.")

def setup(app:Sphinx):
    """Setup the extension."""

    # Require autodoc extension
    app.setup_extension("sphinx.ext.autodoc")
    app.setup_extension("sphinx.ext.autosummary")
    app.add_autodocumenter(InstrumentDocumenter)

    # Add option for instrument configs YAML file
    app.add_config_value("autodoc_instrument_configs_file", "", "html", [str])
    app.connect('config-inited', load_config_file)

    return {
        "version": "0.2",
        "parallel_read_safe": True,
    }
