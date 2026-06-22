#!/usr/bin/bash

set -x

umask 007

REPO=${REPO:-"/2023133163/liuf/lingbot-va"}
CONDA_SH=${CONDA_SH:-"/root/miniconda3/etc/profile.d/conda.sh"}
CONDA_ENV=${CONDA_ENV:-"lingbot-va"}
PYTHON=${PYTHON:-"/root/miniconda3/envs/lingbot-va/bin/python"}

DATA_DIR=${DATA_DIR:-"/2023133163/datasets/lingbot/robotwin-clean-and-aug-lerobot"}
CKPT_DIR=${CKPT_DIR:-"/2023133163/checkpoints/lingbot/lingbot-va-base"}
SAVE_ROOT=${SAVE_ROOT:-"/2023133163/checkpoints/lingbot/robotwin-posttrain"}
EMPTY_EMB_PATH=${EMPTY_EMB_PATH:-"${DATA_DIR}/empty_emb.pt"}

NGPU=${NGPU:-"8"}
NNODES=${NNODES:-"1"}
NODE_RANK=${NODE_RANK:-"0"}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_lingbot_train"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

if [ -f ${CONDA_SH} ]; then
    source ${CONDA_SH}
    conda activate ${CONDA_ENV}
fi

if [ ! -x ${PYTHON} ]; then
    PYTHON=$(which python)
fi

cd ${REPO}
mkdir -p ${SAVE_ROOT}

export DATA_DIR=${DATA_DIR}
export CKPT_DIR=${CKPT_DIR}
export SAVE_ROOT=${SAVE_ROOT}
export EMPTY_EMB_PATH=${EMPTY_EMB_PATH}

# Set ENABLE_WANDB=1 and provide these variables if you want online wandb logging.
export ENABLE_WANDB=${ENABLE_WANDB:-"0"}
export WANDB_API_KEY=${WANDB_API_KEY:-""}
export WANDB_BASE_URL=${WANDB_BASE_URL:-""}
export WANDB_TEAM_NAME=${WANDB_TEAM_NAME:-""}
export WANDB_PROJECT=${WANDB_PROJECT:-"va_robotwin"}

## node setting
num_gpu=${NGPU}
num_nodes=${NNODES}
node_rank=${NODE_RANK}
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
save_root=${SAVE_ROOT}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
${PYTHON} -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --nnodes=${num_nodes} \
    --node_rank=${node_rank} \
    --master_addr=${master_addr} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} --save-root ${save_root} $overrides
