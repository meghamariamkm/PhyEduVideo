import os
import json
import base64
import requests
# import jsonlines
from multiprocessing import Pool
from openai import OpenAI
from functools import partial
import time
import os
import base64
import json
import random
import time

from multiprocessing import Pool

from functools import partial
import os
from PIL import Image
import io
import base64
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from PIL import Image



import t2v_metrics
import os
import json
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def encode_images(img_paths):
    encoded_images = []
    for img_path in img_paths:
        try:
            base64_image = encode_image(img_path)
            encoded_images.append(base64_image)
        except Exception as e:
            print(f'encode image error for {img_path}: {e}')
            return []
    return encoded_images

def get_video_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames

def sample_frames(video_path, num_frames):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((Image.fromarray(frame_rgb), idx))  
    
    video.release()
    return frames

def calculate_clip_scores(frames, prompt):
    images = [frame[0] for frame in frames]
    print(len(images),prompt)
    inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  
    return logits_per_image.squeeze().tolist()


def save_most_similar_frame(frames, scores, output_path):
    max_index = np.argmax(scores)
    most_similar_frame = frames[max_index][0]
    most_similar_frame.save(output_path)
    print(f"Most similar frame saved at: {output_path}")

def get_middle_frame(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  
    middle_frame_index = total_frames // 2 
    
    video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)  
    
    success, frame = video.read()
    video.release()
    
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    else:
        raise ValueError("Failed to retrieve the middle frame from the video.")

def save_middle_frame(video_path, output_path):
    middle_frame = get_middle_frame(video_path)
    middle_frame.save(output_path)
    print(f"Middle frame saved at: {output_path}")


def get_first_frame(video_path):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    success, frame = video.read()
    video.release()
    
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    else:
        raise ValueError("Failed to retrieve the first frame from the video.")

def save_first_frame(video_path, output_path):
    first_frame = get_first_frame(video_path)
    first_frame.save(output_path)
    print(f"First frame saved at: {output_path}")



def get_last_frame(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  
    video.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1) 
    
    success, frame = video.read()
    video.release()
    
    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    else:
        raise ValueError("Failed to retrieve the last frame from the video.")

def save_last_frame(video_path, output_path):
    last_frame = get_last_frame(video_path)
    last_frame.save(output_path)
    print(f"Last frame saved at: {output_path}")


def save_surrounding_frames(frames, scores, output_path):
    max_index = np.argmax(scores)
    start_index = max(max_index - 1, 0)
    end_index = min(max_index + 2, len(frames))
    
    for i in range(start_index, end_index):
        frame = frames[i][0]
        frame_output_path = f"{output_path.split('.jpg')[0]}_frame_{i}.jpg"
        frame.save(frame_output_path)
        print(f"Frame saved at: {frame_output_path}")
    return start_index, max_index, end_index

pretrix = "Answer me in Format:{'Choice':'Yes or No','Reason':'the reason'} "

with open('multi/LLaVA-NeXT-interleave_inference/llava/eval/LLaVA-NeXT/llava/eval/pc-2.json','r') as f:
    data = json.load(f)

directory = 'Results/Combined_Videos'


#modelname = 'phyt2v'

modelnames = ['phyt2v']

num_frames = 32  # 需要采样的帧数

# for modelname in modelnames:

for modelname in modelnames:
    result = []
    if not os.path.exists(os.path.join('Results/PC-2/NEW/multiimage_clips1',modelname)):
        os.makedirs(os.path.join('Results/PC-2/NEW/multiimage_clips1',modelname))
    video_directory = os.path.join(directory,modelname)
    for i in range(len(data)):
        try:
            T2V_prompt = data[i]["visual_explanation"]
            Physical_law = data[i]["concept"]
            concept_id = data[i]['concept_id']
            tpid = data[i]['teaching_point_id']
            filename = f'Id{concept_id}_{tpid}.mp4'
            #video_path = os.path.join(video_directory,f"{i+1:04d}_699_15.mp4")
            video_path = os.path.join(video_directory,filename)
            if not os.path.exists(video_path):
                print(f"❌ File not found: {video_path}")
                continue
            retrieval_question = data[i]['multiimage_question']
            retrieval_prompt = retrieval_question["Retrieval Prompt"]
            question_prompt1 = retrieval_question["Description1"]
            question_prompt2 = retrieval_question["Description2"]
            question_prompt3 = data[i]["visual_explanation"]
            
            
            output_first_image_path = os.path.join(os.path.join('Results/PC-2/NEW/multiimage_clips1',modelname),f'{filename}_first.jpg')
            output_middle_image_path = os.path.join(os.path.join('Results/PC-2/NEW/multiimage_clips1',modelname),f'{filename}_middle.jpg')
            output_last_image_path = os.path.join(os.path.join('Results/PC-2/NEW/multiimage_clips1',modelname),f'{filename}_last.jpg')


            total_frames = get_video_frame_count(video_path)
            print(f"Total frames in video: {total_frames}, {video_path}")

            # 平均采样帧
            sampled_frames = sample_frames(video_path, num_frames)

            save_first_frame(video_path,output_first_image_path)
            save_last_frame(video_path,output_last_image_path)

            if retrieval_prompt == 'Middle Frame':
                save_middle_frame(video_path,output_middle_image_path)


            else:
                scores = calculate_clip_scores(sampled_frames, retrieval_prompt)
                # 保存最相似的帧
                start_index, max_index, end_index = save_surrounding_frames(sampled_frames, scores, output_middle_image_path)
                scores_res = []
                cnt1 = 0
                scores1 = []
                for k in range(start_index, max_index+1):
                    
                    frame_output_path = f"{output_middle_image_path.split('.jpg')[0]}_frame_{k}.jpg"
                    image = [frame_output_path] # an image path in string format
                    text = [retrieval_prompt]
                    score = clip_flant5_score(images=image, texts=text)
                    score_re = score.item()
                    print('score_re: ',score_re, k)
                    data[i]["multiimage_question"][f"retrieval_1_{k}"] = score_re

                    print(f"retrieval_1_{k}")




                cnt2 = 0
                scores2 = []
                for m in range(max_index, end_index):
                    
                    frame_output_path = f"{output_middle_image_path.split('.jpg')[0]}_frame_{m}.jpg"
                    image = [frame_output_path] # an image path in string format
                    text = [retrieval_prompt]
                    score2 = clip_flant5_score(images=image, texts=text)
                    score_re2 = score2.item()
                    data[i]["multiimage_question"][f"retrieval_2_{m}"] = score_re2

                    print('score_re2: ',score_re2, m)

                    print(f"retrieval_2_{m}")


                data[i]["multiimage_question"]["CLIP_Index"] = f"{start_index},{max_index},{end_index}"
                # data[i]["multiimage_question"]["GPT4o_1_Score"] = score1_final
                # data[i]["multiimage_question"]["GPT4o_2_Score"] = score2_final
            result.append(data[i])
        except Exception as e:
            print('Error: ', e)
            result.append(data[i])


        

        

        
        # break

    print(len(result))
    output_path = f"Results/PC-2/NEW/{modelname}"
    os.makedirs(output_path, exist_ok=True)  # ✅ Create directory if not exists

    file_path = os.path.join(
        output_path,
        f"prompt_replace_augment_multi_question_{modelname}_res_imageclip.json"
    )
    with open(f'Results/PC-2/NEW/{modelname}/prompt_replace_augment_multi_question_{modelname}_res_imageclip.json','w') as f:
        json.dump(result,f, indent=4)
