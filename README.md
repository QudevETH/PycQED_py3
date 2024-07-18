![Qudev logo](docs/source/images/qudev_logo.png)

# PycQED 

A python based measurement environment for circuit-QED experiments.

[[_TOC_]]

## Installation

Create a directory you want to work in and traverse into it.

For instance:

```bash
pwsh # open powershell
mkdir qudev
cd qudev
```

Clone the repository:

```bash
# cd a/convenient/directory
git clone https://gitlab.phys.ethz.ch/qudev/control_software/pycqed_py3.git ./pycqed
```

Install a Python environment which allows you to use Python 3.11. As of 2023 many use [Anaconda](https://www.anaconda.com/products/individual). If your default is already Python 3.11 you can skip installing Anaconda. You can also use a more modern setup. If you want this you probably know what you are doing and don't need further guidance.

Here a simple approach known to work on Windows 10 and 11:

```bash
conda create -n pycqed311 python=3.11
conda activate pycqed311

# Update pip within the virtual environment
python -m pip install --upgrade pip

# Install the required packages for the repository
pip install -e ./pycqed
```

Remember that everytime you open a new terminal, you need to activate the virtual environment:

```bash
conda activate pycqed311
```

Start a jupyter notebook without password

```bash
jupyter notebook --NotebookApp.token='' --NotebookApp.password=''
```

## Documentation

Further, general documentation and how to get started in depth can be found on
[documentation.qudev.phys.ethz.ch](https://documentation.qudev.phys.ethz.ch).

## Testing

For testing, make sure your correct environment is activated. See [Installation](#installation)

Then you can run the test suite in your _current_ environment via:

```
pip install '.[test]'
pytest -v --cov=pycqed --cov-report term -m "not hardware" pycqed/tests
```

## License

This software is released under the [MIT License](LICENSE.md)


## Disclaimer

This repository was originally forked from [PycQED](https://github.com/DiCarloLab-Delft/PycQED_py3) by the [DiCarlo group](http://dicarlolab.tudelft.nl/) at [QuTech](http://qutech.nl/), Delft University of Technology.
