import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import glob

def run_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': '.'}})

notebooks = glob.glob('./notebooks/*.ipynb')
for notebook in notebooks:
    run_notebook(notebook)
