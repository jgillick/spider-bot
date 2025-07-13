#!/bin/bash
set -e
# set -x # Uncomment to debug

source cloud.cfg

TASK_DIR="${CLOUD_ISAACLAB_ROOT}/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/spider_locomotion"
MUJOCO_XML_PATH="${TASK_DIR}/assets/SpiderBotNoEnv.xml"
USD_PATH="${TASK_DIR}/assets/SpiderBot.usd"
MUJOCO_CONVERSION_DIR="~/model_conversion"

CONNECT="${CLOUD_SSH_USER}@${CLOUD_IP}"
SSH_PREFIX="ssh -i ${CLOUD_SSH_KEY_PATH} -o StrictHostKeyChecking=no"
SSH="${SSH_PREFIX} ${CONNECT}"
ISAACLAB_SH="source ${CLOUD_PYENV_DIR}/bin/activate && ${CLOUD_ISAACLAB_ROOT}/isaaclab.sh"

ROBOT_DIR="../robot"
MUJOCO_FILE="SpiderBotNoEnv.xml"
USD_FILE="SpiderBot.usd"
LOCAL_USD_DIR="./spider_locomotion/assets"

# Wrapper for rsync that only outputs the files that were synced, if any
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

# Patch the headless kit
sync_with_cloud "./patch_headless_kit.sh" ${CONNECT}:~/patch_headless_kit.sh
$SSH "chmod 755 ~/patch_headless_kit.sh"
$SSH "~/patch_headless_kit.sh ${CLOUD_ISAACLAB_ROOT}/apps/isaaclab.python.headless.kit"

# Does the USD file need to be (re)generated from the mujoco files
generate_usd=0
local_usd_filepath="${LOCAL_USD_DIR}/${USD_FILE}"
mkdir -p ${LOCAL_USD_DIR}
if [ ! -f "${local_usd_filepath}" ]; then
  generate_usd=1
else
  set +e
  mujoco_newer=$(find "${ROBOT_DIR}/${MUJOCO_FILE}" -type f -newer "${local_usd_filepath}" -print -quit | grep -q .)
  meshes_newer=$(find "${ROBOT_DIR}/meshes/" -type f -newer "${local_usd_filepath}" -print -quit | grep -q .)
  set -e
  if [[ $mujoco_newer -eq 1 || $meshes_newer -eq 1 ]]; then
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

  # Send mujoco files to the cloud
  $SSH "mkdir -p ${MUJOCO_CONVERSION_DIR}/mujoco"
  $SSH "mkdir -p ${MUJOCO_CONVERSION_DIR}/usd"
  sync_with_cloud "${ROBOT_DIR}/${MUJOCO_FILE}" "${CONNECT}:${MUJOCO_CONVERSION_DIR}/mujoco/${MUJOCO_FILE}"
  sync_with_cloud "${ROBOT_DIR}/meshes/" ${CONNECT}:${MUJOCO_CONVERSION_DIR}/mujoco/meshes/

  # Run conversion script with IsaacLab
  $SSH ${ISAACLAB_SH} -p ${CLOUD_ISAACLAB_ROOT}/scripts/tools/convert_mjcf.py \
      --headless \
      --import-sites \
      --make-instanceable \
      "${MUJOCO_CONVERSION_DIR}/mujoco/${MUJOCO_FILE}" \
      "${MUJOCO_CONVERSION_DIR}/usd/${USD_FILE}" 2>&1

  # Place USD file in the task directory
  $SSH "mkdir -p ${TASK_DIR}/assets"
  $SSH "cp -r ${MUJOCO_CONVERSION_DIR}/usd/* ${TASK_DIR}/assets/"

  # Fetch USD file
  sync_with_cloud ${CONNECT}:${MUJOCO_CONVERSION_DIR}/usd/ "${LOCAL_USD_DIR}/"
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
echo "########################################################"

# Run the training script
$SSH "${ISAACLAB_SH} \
    -p ${CLOUD_ISAACLAB_ROOT}/scripts/reinforcement_learning/rsl_rl/train.py \
    --task ${SPIDER_TASK} \
    --num_envs 4096 \
    --headless \
    --video --video_length 500 --video_interval 1000"
