#!/bin/bash
#######################################################
#
# Convert the mujoco model to USD on a cloud instance.
#
# Instructions:
#  1. Copy cloud.cfg.example to cloud.cfg
#  2. Open cloud.cfg and define the variables inside it.
#  3. Run ./cloud_mjcf_convert.sh
#
#######################################################

set -e
# set -x # Uncomment to debug

source cloud.cfg

TASK_DIR="${CLOUD_ISAACLAB_ROOT}/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/spider_locomotion"
ASSETS_PATH="${TASK_DIR}/assets"
USD_PATH="${ASSETS_PATH}/SpiderBot.usd"
CLOUD_HOME="/home/${CLOUD_SSH_USER}"

CONNECT="${CLOUD_SSH_USER}@${CLOUD_IP}"
SSH_PREFIX="ssh -i ${CLOUD_SSH_KEY_PATH} -o StrictHostKeyChecking=no"
SSH="${SSH_PREFIX} ${CONNECT}"
PYENV_ACTIVATE="source ~/${CLOUD_PYENV}/bin/activate"
ISAACLAB_SH="${PYENV_ACTIVATE} && ${CLOUD_ISAACLAB_ROOT}/isaaclab.sh"

ROBOT_DIR="../robot"
MUJOCO_FILE="SpiderBotNoEnv.xml"
USD_FILE="SpiderBot.usd"
LOCAL_ASSETS_DIR="./spider_locomotion/assets"

sync_with_cloud() {
    rsync -aiz -e "${SSH_PREFIX}" "$1" "$2"
}


# Sync conversion scripts
sync_with_cloud "./tools/" "${CONNECT}:${CLOUD_ISAACLAB_ROOT}/tools/"

# Send mujoco files to the cloud
mujoco_dir="${CLOUD_HOME}/mujoco"
$SSH "mkdir -p ${mujoco_dir}"
sync_with_cloud "${ROBOT_DIR}/${MUJOCO_FILE}" "${CONNECT}:${mujoco_dir}/${MUJOCO_FILE}"
sync_with_cloud "${ROBOT_DIR}/meshes/mujoco/" ${CONNECT}:${mujoco_dir}/meshes/mujoco/

# Run conversion script with IsaacLab
$SSH "${ISAACLAB_SH} -p ${CLOUD_ISAACLAB_ROOT}/tools/mjcf_2_usd.py \"${mujoco_dir}/${MUJOCO_FILE}\" \"${ASSETS_PATH}/${USD_FILE}\""

# Fetch USD file
sync_with_cloud "${CONNECT}:${ASSETS_PATH}/${USD_FILE}" "${LOCAL_ASSETS_DIR}/${USD_FILE}"

# For some reason the file syncs as a hidden file
chflags nohidden ${LOCAL_ASSETS_DIR}/${USD_FILE}
