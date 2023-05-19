FROM python:3.10.6@sha256:745efdfb7e4aac9a8422bd8c62d8bc35a693e8979a240d29677cb03e6aa91052 
WORKDIR /app

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

#-----------------------------
#INIT
#---CHANGES ABOVE HERE WILL TRIGGER FULL APT UPDATE---

#install git & vim
RUN apt-get update && apt-get -y install \
    git \
    vim
RUN pip install --upgrade pip

#-----------------------------
#PROJECT
COPY . /app

#manually install submodule reqs (needed to ensure pybind11 is recognised)
RUN python -m pip install -r /app/xfmparser/requirements.txt
#install main package
RUN pip install -e /app/

#-----------------------------
#UIDs
#Adjust UIDs so written files belong to user building container
ARG UNAME=user
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

USER $UNAME

#-----------------------------
#ENTRYPOINT
CMD ["xfmread-raw"]
