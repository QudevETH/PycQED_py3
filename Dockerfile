FROM python:3.9-buster

# Copy requirements file from the repository inside the docker container
COPY requirements.txt requirements.txt

# ============================================================================
# Python virtual env
# ============================================================================
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install qcodes
RUN pip install git+https://github.com/pyGSTio/pyGSTi.git

RUN python --version
RUN pip list

# ============================================================================
# Configure SSH agent to be able to copy the doc via SSH
# ============================================================================
RUN which ssh-agent || ( apt-get update -y && apt-get install openssh-client git -y )
RUN eval $(ssh-agent -s)
RUN mkdir -p ~/.ssh
RUN chmod 700 ~/.ssh
RUN ssh-keyscan gitlab.ethz.ch >> ~/.ssh/known_hosts
RUN ssh-keyscan documentation.qudev.phys.ethz.ch >> ~/.ssh/known_hosts
RUN chmod 644 ~/.ssh/known_hosts
