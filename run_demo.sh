export WORKSPACE=$(pwd)
export MODEL_PATH="/scratch/project_2010633/minigpt4_video/MiniGPT4-Video/checkpoints/video_llama_checkpoint_best.pth"
export VIDEO_PATH="/projappl/project_2010633/RAG-Driver/video_process/BDDX_Test/video/2895b5bc-16c48fdf_24672.mp4"
python minigpt4_video_inference.py \
 --ckpt ${MODEL_PATH} \
 --cfg-path test_configs/llama2_test_config.yaml \
 --video_path ${VIDEO_PATH} \
