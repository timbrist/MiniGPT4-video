export WORKSPACE=$(pwd)
python minigpt4_video_demo.py --ckpt ${WORKSPACE}/models/MiniGPT4-Video/checkpoints/video_llama_checkpoint_best.pth --cfg-path test_configs/llama2_test_config.yaml