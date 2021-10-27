"""Sphinx autodoc extension adding parameter docstrings."""

from functools import lru_cache
from inspect import cleandoc
import os
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
    """Class documenter for QCodes' instrument subclasses."""

    objtype = 'class'
    directivetype = 'class'
    priority = 10 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)

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
                try:
                    instrument:Instrument = self.object(name=self.object.__name__)
                except:
                    logger.warning("Could not instantiate instrument "\
                                   f"'{self.object.__name__}' with only name.")
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


def setup(app:Sphinx):
    """Setup the extension."""

    # Require autodoc extension
    app.setup_extension('sphinx.ext.autodoc')
    app.setup_extension('sphinx.ext.autosummary')
    app.add_autodocumenter(InstrumentDocumenter)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
    }
