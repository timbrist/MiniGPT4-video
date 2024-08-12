#!/bin/bash
#SBATCH --job-name=minigpt4
#SBATCH --account=project_2010633
#SBATCH --partition=gpumedium
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4

# export PATH="/projappl/project_2010633/Video-LLaVA/videollava_evn/bin:$PATH"
export PATH="/projappl/project_2010633/MiniGPT4-video/minigpt4_env/bin:$PATH"
export HF_DATASETS_CACHE=/scratch/project_2010633/minigpt4_video
export XDG_CACHE_HOME=/scratch/project_2010633/minigpt4_video
export PIP_CACHE_DIR=/scratch/project_2010633/videollkava_cache
export TRANSFORMERS_CACHE=/scratch/project_2010633/minigpt4_video
export HF_HOME=/scratch/project_2010633/minigpt4_video


job_name="test" # Name of the experiment
cfg_path="train_configs/224_v2_llama2_video_stage_3.yaml" # path to the config file
number_of_gpus=1 # number of gpus
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done
echo "Port is $PORT"
torchrun --nproc-per-node $number_of_gpus train.py --job_name ${job_name} --cfg-path ${cfg_path}
