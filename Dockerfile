FROM python:3.10.6@sha256:745efdfb7e4aac9a8422bd8c62d8bc35a693e8979a240d29677cb03e6aa91052 
WORKDIR /app

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

#INIT
#---CHANGES BEFORE HERE WILL RERUN UPDATE---

#git & vim
RUN apt-get update && apt-get -y install \
    git \
    vim
RUN pip install --upgrade pip


# Install pip requirements
#COPY requirements.txt .

#PROJECT
COPY . /app
#pybind11 for submodule not being found - manually install reqs for submodule first
#   . removed from submodule reqs.txt
#   might also be able to pip install /app/xfmparser instead?
RUN python -m pip install -r /app/xfmparser/requirements.txt
RUN pip install -e /app/

#UIDs
ARG UNAME=user
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

# VSCODE DEFAULT:
# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
#RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
#USER appuser

USER $UNAME
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD ["xfmreadout"]
