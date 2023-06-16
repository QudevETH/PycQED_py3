# CI/CD GitLab pipelines

GitLab can be configured to run continuous integration (CI) and continuous 
deployment (CD) pipelines. The pipelines are defined in the `.gitlab-ci.yml` 
file, and can be used for a variety of tasks, such as compiling documentation,
running unit tests, and even upload a python package to a registry (e.g. PyPI).

## Implemented jobs

### pages

This job builds a docker image with all the dependencies, required to 
compile full documentation and then compiles the sphinx documentation, and 
uploads it to the documentation webshare.
