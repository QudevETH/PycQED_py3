"""Test checking that all modules in the package can be imported."""


import unittest
from importlib import import_module
from pkgutil import iter_modules
import os.path
from typing import List

import pycqed


def get_modules_recursive(path:str, prefix:str="") -> List[str]:
    """Find the list of all submodules inside a base module.

    Arguments:
        path: Path of the base module in which to look for submodules.
        prefix: Prefix for the name of the submodules. You might want to include
            a dot (``.``) at the end of it.

    Returns:
        List[str]: List of sumodules (build recursively).
    """

    modules = []

    for mod in iter_modules([path]):
        modules.append(f"{prefix}{mod.name}")

        if mod.ispkg:
            modules += get_modules_recursive(
                path=os.path.join(path, mod.name),
                prefix=f"{prefix}{mod.name}."
            )

    return modules


class TestModuleImports(unittest.TestCase):
    """Tries to import all pycqed modules recursively.

    This test is very convenient for finding trivial errors that prevent a
    module from being imported, including:
    * Syntax or indentation errors
    * Missing packages imported inside a module
    * Circular imports
    """

    @classmethod
    def setUpClass(cls):

        all_modules = get_modules_recursive(
            path=pycqed.__path__[0],
            prefix=f"{pycqed.__name__}."
        )

        # The modules listed here will be skipped during the test
        ignored_modules = [
            # For instance:
            # "pycqed.analysis_v2.cryo_scope_analysis",
        ]

        cls.modules = [m for m in all_modules if m not in ignored_modules]

    def test_module_imports(self):

        for module in self.modules:
            with self.subTest(module):
                import_module(module)
