#!/bin/bash
#SBATCH --job-name=minigpt4
#SBATCH --account=project_2010633
#SBATCH --partition=gpusmall
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
## if local fast disk on a node is also needed, replace above line with:
#SBATCH --gres=gpu:a100:1,nvme:900
#

# export PATH="/projappl/project_2010633/Video-LLaVA/videollava_evn/bin:$PATH"
export PATH="/projappl/project_2010633/MiniGPT4-video/minigpt4_env/bin:$PATH"
export HF_DATASETS_CACHE=/scratch/project_2010633/minigpt4_video
export XDG_CACHE_HOME=/scratch/project_2010633/minigpt4_video
export PIP_CACHE_DIR=/scratch/project_2010633/videollkava_cache
export TRANSFORMERS_CACHE=/scratch/project_2010633/minigpt4_video
export HF_HOME=/scratch/project_2010633/minigpt4_video

bash run_demo.sh
