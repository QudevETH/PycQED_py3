FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y libgl1 openssh-client git libhdf5-dev libnlopt-dev

# FIXME: Any more elegant way? The problem is the deps can change, however,  I think these are a good base.

RUN python -m venv .venv
RUN . .venv/bin/activate

RUN python -m pip install --upgrade pip

RUN pip install matplotlib numpy pandas scipy sympy qcodes qcodes-loop requests scikit-learn
RUN pip install blosc2 func-timeout lmfit more-itertools msgpack msgpack_numpy pyqtgraph pyside6
RUN pip install zhinst sphinx sphinx-rtd-theme myst-parser pylint jupyter jupyter-contrib-nbextensions
RUN pip install notebook psutil plotly pyyaml

# Problematic installs
RUN apt-get install -y build-essential cmake git pkg-config libblas-dev liblapack-dev
RUN pip install h5py

#RUN pip install nlopt
RUN apt-get install -y python3-nlopt

RUN pip install qutip

RUN pip install tensorflow

RUN pip install neupy

# FIXME: From source. Why?
RUN pip install git+https://github.com/pyGSTio/pyGSTi.git

# FIXME: Can I integrate this here?
# RUN pip install git+https://gitlab-ci-token:${CI_JOB_TOKEN}@${QCODES_CONTRIB_DRIVERS_REPOSITORY}
# RUN pip install git+https://gitlab-ci-token:${CI_JOB_TOKEN}@${VC707_PYTHON_INTERFACE_REPOSITORY}
# RUN pip install git+https://gitlab-ci-token:${CI_JOB_TOKEN}@${DEVICE_DB_CLIENT_REPOSITORY}
