import torch
import webvtt
import os
import cv2 
from minigpt4.common.eval_utils import prepare_texts, init_model
from minigpt4.conversation.conversation import CONV_VISION,Conversation,SeparatorStyle
from torchvision import transforms
import json
from tqdm import tqdm
import soundfile as sf
import argparse
import moviepy.editor as mp
import gradio as gr
from pytubefix import YouTube
import shutil
from PIL import Image
from moviepy.editor import VideoFileClip
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import time
import transformers
import whisper
from datetime import timedelta

import textwrap


def get_arguments():
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--cfg-path", help="path to configuration file.",default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--ckpt", type=str,default='checkpoints/video_llama_checkpoint_last.pth', help="path to checkpoint")
    parser.add_argument("--add_subtitles",action= 'store_true',help="whether to add subtitles")
    parser.add_argument("--question", type=str, help="question to ask")
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
                "in xxx=yyy format will be merged into config file (deprecate), "
                "change to --cfg-options instead.",
    )
    return parser.parse_args()

ARGS=get_arguments()


SYSTEM_MESSAGE=textwrap.dedent(f"""
You are an AI assistant helping an autonomous driving vehicle analyze its behaviors. 
The vehicle's camera has captured a continuous driving scene, 
Your task is to describe the vehicle's current action and explain the reasoning behind it, 
based on any potential obstacles, traffic signs, pedestrians, other vehicles, and road conditions.
Please answer in a format by split the action and justification using ##, for examples: 
The car stops.##because there is a stop sign.
The car accelerates slowly to a maintained speed##because traffic is moving smoothly.
The car moves forward then turns right##because no traffic is blocking the way and the car can safely move forward.
The car merges into the lane to its right##due to traffic moving freely in that lane.
The car is slowly moving forward##since traffic is busy
                               """)

class Evaluator():
    def __init__(self) -> None:
        self.checkpoint_path =""
        self.video_folder = ""
        self.model_name = "llama2"

        # set up random configuration 
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

        # initialize 
        self.model, self.vis_processor,whisper_gpu_id,_,_ = init_model(ARGS)
        self.whisper_model=whisper.load_model("large").to(f"cuda:{whisper_gpu_id}")

        # set up conversation instance 
        self.conv = Conversation(
            # system="Give the following image: <Img>ImageContent</Img>. "
            system = SYSTEM_MESSAGE,
            roles = (r"[INST] ",r" [/INST]"),
            messages=[],
            offset=2,
            sep_style=SeparatorStyle.SINGLE,
            sep="<s>",
        )

    # get answer from minigpt4-video
    def inference(self,video_path,query):

        prepared_images,prepared_instruction= self.embed_query(video_path,query)
        if prepared_images is None:
            return "Video cann't be open ,check the video path again"
        length=len(prepared_images)
        print(f"the length of the prepared image: {length}")
        prepared_images=prepared_images.unsqueeze(0)

        self.conv.append_message(self.conv.roles[0], prepared_instruction)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = [self.conv.get_prompt()]
        answers = self.model.generate(prepared_images, prompt, 
                                      max_new_tokens=ARGS.max_new_tokens, do_sample=True,
                                      lengths=[length],num_beams=1)
        return answers[0]
    

    # embed the prompts/query before inference 
    def embed_query(self,video_path, query):
        cap = cv2.VideoCapture(video_path)
        total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

        # model configuration 
        if "mistral" in ARGS.ckpt :
            max_images_length=90
            max_sub_len = 800
        else:
            max_images_length = 45
            max_sub_len = 400

            images = []
        frame_count = 0
        sampling_interval = int(total_num_frames / max_images_length)
        if sampling_interval == 0:
            sampling_interval = 1
        img_placeholder = ""
        raw_frames=[]
        transform=transforms.Compose([
                    transforms.ToPILImage(),
                ])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Find the corresponding subtitle for the frame and combine the interval subtitles into one subtitle
            # we choose 1 frame for every 2 seconds,so we need to combine the subtitles in the interval of 2 seconds
            if frame_count % sampling_interval == 0:
                raw_frames.append(Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)))
                frame = transform(frame[:,:,::-1]) # convert to RGB
                frame = self.vis_processor(frame)
                images.append(frame)
                img_placeholder += '<Img><ImageHere>'
            frame_count += 1

            if len(images) >= max_images_length:
                break
        cap.release()
        cv2.destroyAllWindows()
        if len(images) == 0:
            # skip the video if no frame is extracted
            return None,None
        images = torch.stack(images)
        instruction = f"{img_placeholder} \n {query}"
        return images,instruction

def test():
    video_path = "path/to/video"
    prompt = "The current video records driving scenario:\n Control Signal until current Frame Sequence is: Speed: [3.47, 3.68, 3.91, 4.1, 4.27, 4.43, 3.97]\n Curvature: [-0.02, -0.0, 0.02, 0.02, -0.0, -0.02, -0.01]\n Acceleration: [0.14, 0.27, 0.42, 0.36, -0.08, -0.48, -0.52]\n Course: [0.0, 0.12, 0.18, 0.05, 0.07, 0.05, 0.04]\nWhat is the action of ego car?, and Why does the ego car doing this?"
    evaluator = Evaluator()
    t1=time.time()

    pred = evaluator.inference(video_path, prompt)

    print(pred)
    print("time taken : ",time.time()-t1)
    print("Number of output words : ",len(pred.split(' ')))

def eval_bddx():
    video_folder = "/scratch/project_2010633/minigpt4_video/bddx/eval/video"
    eval_json_path = "/scratch/project_2010633/minigpt4_video/bddx/minigpt4_instruct_eval.json"

    # Read the json to get video id and prompt
    with open(eval_json_path, 'r') as file:
        data =  json.load(file)

    #init our evaluator
    evaluator = Evaluator()

    # evaluate the data from the json file 
    for d in data:
        prompt = d['q']
        video_id = d['video_id']
        video_path = f"{video_folder}/{video_id}.mp4"

        t1=time.time()

        pred = evaluator.inference(video_path, prompt)

        # save the data TODO
        print(pred)
        print("time taken : ",time.time()-t1)
        print("Number of output words : ",len(pred.split(' ')))

        break


if __name__ == "__main__":
    test()