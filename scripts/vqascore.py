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


def save_surrounding_frames(frames, scores, output_path):
    max_index = np.argmax(scores)
    start_index = max(max_index - 2, 0)
    end_index = min(max_index + 3, len(frames))

    for i in range(start_index, end_index):
        frame = frames[i][0]

        frame_output_path = f"{output_path.split('.jpg')[0]}_frame_{i}.jpg"
        # if not os.path.exists(frame_output_path):
            # print('need save')
        frame.save(frame_output_path)
        print(f"Frame saved at: {frame_output_path}")
    return end_index, start_index

def get_video_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames

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



with open('PhyGenBench/PhyGenEval/single/pc1.json','r') as f:
    data = json.load(f)


result = []
directory = 'Results/Combined_Videos'
modelname = 'Wan2.1'
modelnames = ['phyt2v']


for modelname in modelnames:
    num_frames = 32  

    video_directory = os.path.join(directory,modelname)
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)


    for i in range(len(data)):
        id = data[i]['concept_id']
        tpid = data[i]['teaching_point_id']
        filename = f'Id{id}_{tpid}.mp4'
        file = f'Id{id}_{tpid}'
        video_path = os.path.join(video_directory, filename)
        retrieval_questions = data[i]['questions']
        score_final_all = 0
        score_final_all_discrete = 0
        for j in range(len(retrieval_questions)):
            retrieval_question = retrieval_questions[j]
            retrieval_prompt = retrieval_question["retrieval_prompt"]
            question_prompt = retrieval_question["statement"]
            question_prompt_an = retrieval_question["antonym"]

        
            dir = os.path.join('Results/PC-1/singleimage_clips', modelname)

            # Create the directory if it doesn't exist
            os.makedirs(dir, exist_ok=True)
            output_image_path = os.path.join(os.path.join('Results/PC-1/singleimage_clips',modelname),f'{file}_{j}.jpg')
            
            total_frames = get_video_frame_count(video_path)
            print(f"Total frames in video: {total_frames}")

            sampled_frames = sample_frames(video_path, num_frames)


            if retrieval_prompt == 'Last Frame':
                print('retrieval_prompt == Last Frame')
                save_last_frame(video_path,output_image_path)


                image = [output_image_path] # an image path in string format
                text = [question_prompt]
                score = clip_flant5_score(images=image, texts=text)
                score1 = score.item()
                score_re = 0
                score_final = score1
                score_final_all += score1

                


            else:
                scores = calculate_clip_scores(sampled_frames, retrieval_prompt)
                end_index, start_index = save_surrounding_frames(sampled_frames, scores, output_image_path)
                scores_res = []

                for k in range(start_index, end_index):
                    frame_output_path = f"{output_image_path.split('.jpg')[0]}_frame_{k}.jpg"

                    
                    image = [frame_output_path] # an image path in string format
                    text = [retrieval_prompt]
                    score = clip_flant5_score(images=image, texts=text)
                    score_re = score.item()



                    image = [frame_output_path] # an image path in string format
                    text = [question_prompt]
                    score = clip_flant5_score(images=image, texts=text)
                    score1 = score.item()


                    image = [frame_output_path] # an image path in string format
                    text = [question_prompt_an]
                    score = clip_flant5_score(images=image, texts=text)
                    score1_no = score.item()

                    print(question_prompt, score1, score1_no)
                    score1 = score1 / (score1 + score1_no)



                    score_res = score_re + score1

                    scores_res.append(score_res)
                
                score_final = max(scores_res)

                score_final_all += score_final

                data[i]['questions'][j]['clip_retrieval'] = score_re
                

            print(score_re, score1, modelname, filename ,question_prompt)

            data[i]['questions'][j][modelname] = score_final

        # only use for two-question
        if score_final_all <= 1:
            score_final_all_discrete = 0
        elif score_final_all <= 1.5 and score_final_all > 1:
            score_final_all_discrete = 1
        elif score_final_all <= 2.25 and score_final_all > 1.5:
            score_final_all_discrete = 2
        elif score_final_all > 2.25:
            score_final_all_discrete = 3


        data[i][f'{modelname}_score'] = score_final_all
        data[i][f'{modelname}_descrete'] = score_final_all_discrete

        print(score_final_all_discrete,i)
        result.append(data[i])

    print(len(result))

    with open(f'Results/PC-1/prompt_replace_augment_single_question_{modelname}_res.json','w') as f:
        json.dump(result,f, indent=4)