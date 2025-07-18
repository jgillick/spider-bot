#!/bin/bash
#######################################################
#
# Push the isaac lab code to the cloud and run it.
#
# Instructions:
#  1. Copy cloud.cfg.example to cloud.cfg
#  2. Open cloud.cfg and define the variables inside it.
#  3. Run ./cloud_run.sh
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
URDF_FILE="SpiderBot.urdf"
USD_FILE="SpiderBot.usd"
LOCAL_ASSETS_DIR="./spider_locomotion/assets"
LOCAL_LOGS_DIR="./logs"

CLOUD_LOGS_DIR="${CLOUD_HOME}/logs"

#######################################################
# Helper functions
#######################################################

##
# Wrapper for rsync that only outputs the files that were synced, if any
#
sync_with_cloud() {
    local source="$1"
    local destination="$2"

    rsync -av --itemize-changes -e "${SSH_PREFIX}" "$source" "$destination" \
      | grep -E '^<f'\
      | cut -d' ' -f2

    changes=0
    if [ -n "$files_synced" ]; then
      echo "$files_synced"
    fi
}

##
# Start tensorboard
#
start_tensorboard() {
  # Create logs directory, if necessary
  $SSH "mkdir -p ${CLOUD_LOGS_DIR}"

  # Start tensorboard
  $SSH "${ISAACLAB_SH} -p -m tensorboard.main --logdir=logs --bind_all"
}

##
# Sync the latest training logs
#
sync_logs() {
  sync_with_cloud "${CONNECT}:${CLOUD_LOGS_DIR}/" "${LOCAL_LOGS_DIR}/"
  # rsync -av -e "${SSH_PREFIX}" "${CONNECT}:${CLOUD_LOGS_DIR}/" "${LOCAL_LOGS_DIR}/"
}

##
# Regularly download new training logs
#
watch_for_logs() {
  while true; do
    sync_logs
    sleep 30
  done
}

#######################################################
# Main program
#######################################################

# Does the USD file need to be (re)generated
generate_usd=0
local_usd_filepath="${LOCAL_ASSETS_DIR}/${USD_FILE}"
mkdir -p ${LOCAL_ASSETS_DIR}
if [ ! -f "${local_usd_filepath}" ]; then
  generate_usd=1
else
  mujoco_newer=$(find "${ROBOT_DIR}/${MUJOCO_FILE}" -type f -newer "${local_usd_filepath}" -print -quit)
  meshes_newer=$(find "${ROBOT_DIR}/meshes/" -type f -newer "${local_usd_filepath}" -print -quit)
  if [[ -n "$mujoco_newer" || -n "$meshes_newer" ]]; then
    generate_usd=1
  fi
fi

##
# Convert mujoco model to USD
#
if [[ $generate_usd -eq 1 ]]; then
  echo ""
  echo "########################################################"
  echo "# Converting mujoco model to USD"
  echo "########################################################"

  # Sync conversion scripts
  sync_with_cloud "./isaaclab.python.mjcf.kit" "${CONNECT}:${CLOUD_ISAACLAB_ROOT}/apps/isaaclab.python.mjcf.kit"
  sync_with_cloud "./mjcf_2_usd.py" "${CONNECT}:${CLOUD_ISAACLAB_ROOT}/scripts/tools/mjcf_2_usd.py"

  # Send mujoco files to the cloud
  mujoco_dir="${CLOUD_HOME}/mujoco"
  $SSH "mkdir -p ${mujoco_dir}"
  sync_with_cloud "${ROBOT_DIR}/${MUJOCO_FILE}" "${CONNECT}:${mujoco_dir}/${MUJOCO_FILE}"
  sync_with_cloud "${ROBOT_DIR}/meshes/" ${CONNECT}:${mujoco_dir}/meshes/

  # Run conversion script with IsaacLab
  $SSH "${ISAACLAB_SH} -p ${CLOUD_ISAACLAB_ROOT}/scripts/tools/mjcf_2_usd.py \"${mujoco_dir}/${MUJOCO_FILE}\" \"${ASSETS_PATH}/${USD_FILE}\""

  # Fetch USD file
  sync_with_cloud "${CONNECT}:${ASSETS_PATH}/${USD_FILE}" "${LOCAL_ASSETS_DIR}/${USD_FILE}"

  # For some reason the file syncs as a hidden file
  chflags nohidden ${LOCAL_ASSETS_DIR}/${USD_FILE}
fi

##
# Sync the training files
#
echo ""
echo "########################################################"
echo "# Sync training scripts with the cloud"
echo "########################################################"
sync_with_cloud "./spider_locomotion/" "${CONNECT}:${TASK_DIR}"

##
# Run the training script
#
echo ""
echo "########################################################"
echo "# Running the training script"
echo "#   - Videos will be synced to ${LOCAL_VIDEOS_DIR}"
echo "#   - Tensorboard: http://${CLOUD_IP}:6006"
echo "########################################################"

# Clean up on exit
trap "exit" INT TERM
trap "kill 0" EXIT

start_tensorboard &

# Start the logs downloader
watch_for_logs &

# Run the training script
$SSH "${PYENV_ACTIVATE} && \
      HYDRA_FULL_ERROR=1 \
      ${CLOUD_ISAACLAB_ROOT}/isaaclab.sh \
      -p ${CLOUD_ISAACLAB_ROOT}/scripts/reinforcement_learning/rsl_rl/train.py \
      --task ${SPIDER_TASK} \
      --num_envs 512 \
      --headless \
      --verbose \
      --enable_cameras \
      --video --video_length 1000 --video_interval 100"

# Download any final logs & videos
sync_logs
