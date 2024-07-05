import pycycle.utils as pycycle
import pathlib
import pytest

# Going down 2 levels because of our odd project layout!
PROJECT_ROOT = pathlib.Path(__file__).parent.parent 

def test_project_circular_imports(tmpdir):
    """Tests for circular imports in the entire project."""

    project_source = str(PROJECT_ROOT) 

    # Check for cycles
    root_node = pycycle.read_project(project_source, verbose=True, ignore=['tests', 'docs'])
    cycle_exists = pycycle.check_if_cycles_exist(root_node)

    # In case of cycle detection add some more information
    # e.g., the whole chain is shown in case of large cycles
    assert not cycle_exists, "Circular imports detected:\n{}".format(
        pycycle.get_cycle_path(root_node)
    )

# FIXME: More granular approach
# @pytest.mark.parametrize(
#     "source_path, expected_outcome, options", 
#     [
#         # baseline
#         (str(PROJECT_ROOT), False, {}),
# 
#         # In case only a part of the project shall be checked
#         (str(PROJECT_ROOT / 'known_cycles_dir'), True, {}), 
# 
#         #  --verbose (defaults to true already)
#         (str(PROJECT_ROOT), False, {'verbose': True}),  
# 
#         # Mark special encodings
#         (str(PROJECT_ROOT / 'special_encoding'), False, {'encoding': 'utf-16'}), 
# 
#         # --ignore
#         (str(PROJECT_ROOT), False, {'ignore': ['tests', 'docs']}) 
#     ])
# def test_circular_imports_parametrized(source_path, expected_outcome, options):
#     root_node = pycycle.read_project(source_path, **options)
#     cycle_exists = pycycle.check_if_cycles_exist(root_node)
# 
#     assert cycle_exists == expected_outcome, "Unexpected result for circular import detection"  
# 
# 