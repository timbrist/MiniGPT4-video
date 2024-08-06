export WORKSPACE=$(pwd)
export MODEL_PATH="/scratch/project_2010633/minigpt4_video/MiniGPT4-Video/checkpoints/video_llama_checkpoint_best.pth"
export VIDEO_PATH="/scratch/project_2010633/BDD/samples-1k/videos/08022699-d03af7f6.mov"
python minigpt4_video_inference.py --ckpt ${MODEL_PATH} --cfg-path test_configs/llama2_test_config.yaml --video_path ${VIDEO_PATH}
