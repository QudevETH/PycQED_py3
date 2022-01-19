![Qudev logo](docs/source/images/qudev_logo.png)

# PycQED 

A python based measurement environment for circuit-QED experiments.

This repository was originally forked from
[PycQED](https://github.com/DiCarloLab-Delft/PycQED_py3) by the
[DiCarlo group](http://dicarlolab.tudelft.nl/) at [QuTech](http://qutech.nl/),
Delft University of Technology.

## Installation

Clone the repository:
```bash
# cd a/convenient/directory
git clone https://gitlab.ethz.ch/qudev/control_software/pycqed_py3.git
```

Install [Anaconda](https://www.anaconda.com/products/individual) if you don't 
have it, and create a virtual environment:
```bash
conda create -n pycqed36 python=3.6
conda activate pycqed36

# Update pip within the virtual environment
python -m pip install --upgrade pip

# Install the required packages for the repository
pip install -r requirements.txt

# Only on measurement PCs where the NI DAQmx package is installed
# pip install nidaqmx  
```

Everytime you open a new terminal, you need to activate the virtual environment:
```bash
conda activate pycqed36 
```

Enable the [Jupyter notebook extensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html)
which adds a tab "Nbextensions" in the main Jupyter window where you can 
enable/disable any of the available extensions.
```bash
jupyter contrib nbextension install --user
```

PycQED depends on the following repositories, which should be installed as 
symbolic links in your virtual environment (`pip install -e`). This way, the 
changes inside these local repositories (branch switching or file modification)
are reflected when using the modules within PycQED.

```bash
# Make sure to have activated your virtual environment
# Go to the parent folder in which you cloned PycQED
pip install -e ./PycQED_py3
git clone https://gitlab.ethz.ch/qudev/control_software/qcodes.git
pip install -e ./Qcodes
git clone https://github.com/pyGSTio/pyGSTi.git
pip install -e ./pyGSTi
```

## Examples

TODO: Add here link to relevant notebook example.

## Documentation

The documentation for the `qudev_master` branch is available online
[here](https://documentation.qudev.phys.ethz.ch/control_software/pycqed_py3/qudev_master/).
The documentation for other branches can be found on
[documentation.qudev.phys.ethz.ch](https://documentation.qudev.phys.ethz.ch/control_software/pycqed_py3).

### Building the documentation locally

Assuming you have followed the installation steps, you can locally build the
documentation as follows:
```bash
# From the root of the repo
cd docs
make clean
make html
```

You can then access the documentation by opening `docs/build/html/index.html` in
your browser.

## Overview of the main modules

TODO: This section should be replaced by the short PycQED user manual/general
overview, to make this README lighter. 

Below follows an overview of the main structure of the code. It makes sense to take a look around here if your are new to get a feeling where to find things.
Also take a look at [this presentation](docs/160714_qcodes_meetup.pdf), where the relation to qcodes and the core concepts in the package are explained.
Mind however that the code is continuously under development so if you think something should be in a different location feel free to tap me (Adriaan) on the shoulder or create an issue to discuss it.

### Folder Structure
+ [docs](docs/)
+ [init](init/)
+ [analysis](analysis/)
+ [measurement](measurement/)
+ [utilities](utilities/)
+ [instrument_drivers](instrument_drivers/)
+ [scripts](scripts/)
    + [testing](scripts/testing/)
    + [personal_folders](scripts/personal_folders/)


### The init folder
Contains script that are to be used when setting up an experiment. Used to store configuration info and create instruments.

###The instruments folder

PycQED makes extensive use of instruments. Instruments are children of the qcodes instrument class and can be used as drivers for physical instruments,
but can also provide a layer of abstraction in the form of meta-instruments, which contain other instruments.

We use these qcodes instruments for several reasons; the class provides logging of variables, provides a standardized format for getting and setting parameters, protects the underlying instruments against setting 'bad' values. Additionally qcodes itself comes with drivers for most instruments we use in the lab.

We split the instrument folder up in several subfolders. The physical instruments are drivers that control physical instruments. Meta-instruments are higher level instruments that control other lower level instruments. The main point here is that commands only flow down and information flows up.

#### Measurement Control
The **Measurement Control** is a special object that is used to run experiments. It takes care of preparing an experiment, giving instructions to the instruments involved and saving the data.

Below is an example of running a homodyne experiment using the measurement control.

```python
MC.set_sweep_function(Source.frequency)
MC.set_sweep_points(np.arange(freq_start,freq_stop,freq_step))
MC.set_detector_function(det.HomodyneDetector())
MC.run()
```

A sweep_function determines what parameter is varied, a qcodes parameter that contains a .set method can also be inserted here.
A detector_function determines what parameter is measured, a qcodes parameter that has a .get method can also be inserted here.

#### The qubit object
The qubit object is a (meta) instrument but it defies the general categorization of the other instruments.

It is the object that one is actively manipulating during an experiment and as such contains functions such as qubit.measure_Rabi() and qubit.find_frequency_spec(). It is also used to store the known parameters of the physical qubit object.

Because the qubit object is also used to start experiments and update itself it makes active use of the measurement control.
It's functions are (generally) split into the following types, these correspond to prefixes in the function names.
* measure
* calibrate
* find
* tune

### The modules folder
The modules folder contains core modules of PyCQED. It is split into measurement and analysis. It also contains a utilities module which contains some general functions (such as send_email) which do not really fit anywhere else.

#### Measurement

The measurement module contains mostly modules with sweep and detector functions. These are used to define what instruments have to do when performing a measurement.
The sweep function defines what variable(s) are sweeped and the detector function defines how it is to be detected. Both sweep and detector functions are split into soft(ware) and hard(ware) controlled sweep and detector functions.

In a soft measurement the measurement loop is controlled completely by PyCQED which sets a sweep point on an instrument and then uses another instrument to 'probe' the measured quantity. An example of such a measurement is a Homodyne measurement.

In a hard measurement the measurement is prepared by the software and then triggered and controlled completely by the hardware which returns all the data after completing an iteration of the experiment. An example of such a measurement would be running an AWG-sequence.

#### Analysis
The measurement analysis currently (april 2015) contains three modules.

##### Measurement analysis
The measurement analysis module contains the main analysis objects. By instantiating one of these objects a dataset can be analyzed. It contains default methods for the most common experiments and makes extensive use of object oriented hereditary relations. This means that for example the Rabi_Analysis is a child of the TD_Analysis which is a child of the MeasurementAnalysis and has all the functions of it's parent classes.

When instantiating such an object you can pass it a timestamp and/or a label to determine what datafile to load. By giving it the argument auto=True the default analysis script is run.
##### Analysis Toolbox
This toolbox contains tools for analysis such as file-handling tools, plotting tools and some data analysis tools such as a peak finder.

##### Fitting models
This module contains the lmfit model definitions and fitting functions used for curve fitting.

## Dependencies

* Python 3.6
* [QCodes](https://gitlab.ethz.ch/qudev/control_software/qcodes) `0.1.11`:
Custom fork of [QCodes](https://github.com/QCoDeS/Qcodes), a data acquisition
framework.
* TODO: clean list of dependencies and document which version we use.

## License

This software is released under the [MIT License](LICENSE.md)

## Contributing

TODO: Clean contributing and put it somewhere else
Please see [Contributing.md](.github/CONTRIBUTING.md)
