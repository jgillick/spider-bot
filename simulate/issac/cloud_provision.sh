#!/bin/bash
#######################################################
#
# Provision an Ubuntu AWS instance with Isaac Lab.
#
# Instructions:
#  1. Copy cloud.cfg.example to cloud.cfg
#  2. Open cloud.cfg and define the variables inside it.
#  3. Run ./cloud_provision.sh
#
#######################################################

set -e
set -x # Uncomment to debug

source cloud.cfg

CONNECT="${CLOUD_SSH_USER}@${CLOUD_IP}"
SSH_PREFIX="ssh -i ${CLOUD_SSH_KEY_PATH} -o StrictHostKeyChecking=no"
SSH="${SSH_PREFIX} ${CONNECT}"
PYENV_ACTIVATE="source ~/${CLOUD_PYENV}/bin/activate"


# Install dependencies
$SSH "sudo apt-get update -y"
$SSH "sudo apt-get upgrade -y"
$SSH "sudo add-apt-repository ppa:deadsnakes/ppa -y"
$SSH "sudo apt-get install -y \
  python3.10 python3.10-venv \
  cmake build-essential \
  nvidia-cuda-toolkit \
  libvulkan-dev\
  xserver-xorg \
  nvidia-driver-535-server"

# Create python env
$SSH "python3.10 -m venv ${CLOUD_PYENV}"

# NVIDIA
$SSH "curl -sLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && sudo dpkg -i cuda-keyring_1.1-1_all.deb"
$SSN "nvidia-smi -pm ENABLED" # Enable persistent mode
$SSN "nvidia-smi --ecc-config=0" # Disable ECC

# Isaac Sim installation
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-sim
$SSH "${PYENV_ACTIVATE} && pip install --upgrade pip"
$SSH "${PYENV_ACTIVATE} && pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121"
$SSH "${PYENV_ACTIVATE} && pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com"

# Isaac Lab installation
$SSH "if [[ ! -d "/home/ubuntu/IsaacLab" ]]; then \
  git clone https://github.com/isaac-sim/IsaacLab.git ${CLOUD_ISAACLAB_ROOT}; \
  else cd ${CLOUD_ISAACLAB_ROOT}; git pull; \
  fi"
$SSH "cd ${CLOUD_ISAACLAB_ROOT} && ${PYENV_ACTIVATE} && ./isaaclab.sh -i"

$SSH "${PYENV_ACTIVATE} && pip install tensorflow"
