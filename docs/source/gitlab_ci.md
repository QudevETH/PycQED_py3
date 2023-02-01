# CI/CD GitLab pipelines

GitLab can be configured to run continuous integration (CI) and continuous 
deployment (CD) pipelines. The pipelines are defined in the `.gitlab-ci.yml` 
file, and can be used for a variety of tasks, such as compiling documentation,
running unit tests, and even upload a python package to a registry (e.g. PyPI).

## Implemented jobs

### ci_image

This job builds a docker image with Python, and the associated system (i.e. 
`apt-get`) and Python (`pip`) dependencies. Pre-building substantially reduces
the time it takes to run subsequent jobs, as the custom docker image is already
ready to be used.

This pipeline runs only when the `Dockerfile` or `requirements.txt` files are
modified.

### pages

This jobs compiles the sphinx documentation, and uploads it to the QuDev 
documentation webshare: `documentation.qudev.phys.ethz.ch`.
