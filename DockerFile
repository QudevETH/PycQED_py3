FROM python:3.9-buster

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
RUn mkdir -p ~/.ssh
RUn chmod 700 ~/.ssh
RUn ssh-keyscan gitlab.ethz.ch >> ~/.ssh/known_hosts
RUn ssh-keyscan documentation.qudev.phys.ethz.ch >> ~/.ssh/known_hosts
RUn chmod 644 ~/.ssh/known_hosts
