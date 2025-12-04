import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
import os 
import cv2
import re

# --- Setup ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

print('Hello.....')
video_dir = "WAN2.1-without-prompt"


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        set((i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n+1) for j in range(1, n+1) 
            if i*j <= max_num and i*j >= min_num), 
        key=lambda x: x[0]*x[1])
    
    best_ratio = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0]/r[1]))
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_video(video_path, input_size=448, max_num=1, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    transform = build_transform(input_size)
    frame_indices = np.linspace(0, max_frame, num_segments, dtype=int)
    
    pixel_values_list, num_patches_list = [], []
    for idx in frame_indices:
        img = Image.fromarray(vr[idx].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pv = torch.stack([transform(t) for t in tiles])
        pixel_values_list.append(pv)
        num_patches_list.append(pv.size(0))
    
    pixel_values = torch.cat(pixel_values_list, dim=0)
    return pixel_values, num_patches_list


# --- Load model ---
path = 'OpenGVLab/InternVL3_5-8B'
model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16,
                                  low_cpu_mem_usage=True, trust_remote_code=True,
                                  device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

modelnames = ['WAN2.1-without-prompt']
with open('SA.json','r') as f:
    data = json.load(f)

for modelname in modelnames:
    results = []
    for i in range(len(data)):
        try:
            data_item = {}
            concept_id = data[i]['concept_id']
            tpid = data[i]['teaching_point_id']
            teaching_point = data[i]['teaching_point']
            filename = f'Id{concept_id}_{tpid}.mp4'
            data_item['concept_id'] = concept_id
            data_item['teaching_point_id'] = tpid
            data_item['teaching_point'] = teaching_point

            # Get all object keys
            object_list = list(data[i]['objects'].keys())
            action = data[i]['action']
            data_item['objects'] = object_list

            # --- Load video ---
            video_path = f'Results/Combined_Videos/{modelname}/{filename}'  # Replace with your video
            pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            # --- Prepare question ---
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            object_string = ", ".join(object_list)

            question = video_prefix + f"""
            According to the key images from the video, evaluate if {object_string} is present. 
            Assign a score from 1 to 2:
            2: All objects present
            1: Some objects missing
            0: None present
            Provide JSON with keys 'score' and 'explanation', step by step.
            Keep all analysis within explanation inside json. Only and only return the json back.
            The score should not be below 0 or above 2.
            """

            # --- Ask the model ---
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                        num_patches_list=num_patches_list,
                                        history=None, return_history=True)
            try:
                print(response)
                clean_resp = re.sub(r"^```json|```$", "", response.strip(), flags=re.MULTILINE).strip()
                parsed_response = json.loads(clean_resp)
                print(parsed_response)
            except Exception as e:
                print(f'Json parsing error: {e} {clean_resp} {parsed_response}')

            data_item['object_score'] = parsed_response['score']
            data_item['object_score_explanation'] = parsed_response['explanation']
            question_video = video_prefix + f"According to the key images from the video, evaluate if {action} is presented in the video. Assign a score from 0 to 1 according to the criteria:\n \
            1: the action is presented in this video.\n \
            0: the action is not presented in this video.\n \
            Provide your analysis and explanation in JSON format with the following keys: score (e.g., 2) and then explain it, step by step. Keep all analysis within explanation inside json. Only and only return the json back"


            # --- Ask the model ---
            generation_config_ans = dict(max_new_tokens=1024, do_sample=True)
            response_act, history_act = model.chat(tokenizer, pixel_values, question_video, generation_config_ans,
                                        num_patches_list=num_patches_list,
                                        history=None, return_history=True)
            try:
                clean_resp_act = re.sub(r"^```json|```$", "", response_act.strip(), flags=re.MULTILINE).strip()
                parsed_response_act = json.loads(clean_resp_act)
                data_item['action_score'] = parsed_response_act['score']
                data_item['action_score_explanation'] = parsed_response_act.get('explanation') or parsed_response_act.get('explain')
            except Exception as e:
                print(f'Json parsing error: {e} {clean_resp_act} {parsed_response_act}')

            data_item['SA_internVL35'] = data_item['object_score'] + data_item['action_score']
            results.append(data_item)

            output_path = f'Results/SA-new/{modelname}.json'
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            print(f'Error occurred: {e}')