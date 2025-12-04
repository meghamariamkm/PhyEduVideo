import argparse
import numpy as np
import os
import cv2
import json
import torch
from tqdm import tqdm
import time
from configs.config import Config, eval_dict_leaf
from configs.utils import retrieve_text, _frame_from_video, setup_internvideo2
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Video Score Calculation")
    parser.add_argument("--seed", type=int, default=1421538, help="Random seed")
    parser.add_argument('--model_names', nargs='+', default=["test"], help="Name of the models.")
    parser.add_argument("--input_folder", type=str, default="Results/Combined_Videos", help="Input folder containing videos")
    parser.add_argument("--output_folder", type=str, default="Results/PC-3-real/", help="Output folder for saving scores")
    parser.add_argument("--config_path", type=str, default='video/MTScore/configs/internvideo2_stage2_config.py', help="Path to config file")
    parser.add_argument("--model_pth", type=str, default='video/MTScore/InternVideo2-Stage2_6B-224p-f4/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt', help="Path to model checkpoint folder")
    parser.add_argument('--eval_type', type=int, choices=[150, 1649], default=150)
    return parser.parse_args()


def retry_setup(config, max_attempts=3, delay=2):
    attempts = 0
    while attempts < max_attempts:
        try:
            intern_model, tokenizer = setup_internvideo2(config)
            return intern_model, tokenizer
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
            if attempts == max_attempts:
                raise Exception("All attempts failed.")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

def calculate_video_score(video_path, text_to_index, text_candidates, intern_model, config):
    video = cv2.VideoCapture(video_path)
    frames = [x for x in _frame_from_video(video)]

    texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=4, config=config)


    return texts, probs

def calculate_average_score(scores):
    total_videos = len(scores)
    total_general_score = sum(score[0] for score in scores)
    total_metamorphic_score = sum(score[1] for score in scores)
    average_general_score = total_general_score / total_videos
    average_metamorphic_score = total_metamorphic_score / total_videos
    return average_general_score, average_metamorphic_score

def load_existing_scores(filepath):
    with open(filepath, 'r') as file:
        scores = json.load(file)
    return scores

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    try:
        config = Config.from_file(args.config_path)
    except:
        config = Config.from_file(args.config_path.replace("configs/", "MTScore/configs/"))

    config = eval_dict_leaf(config)

    #print(config)

    config['model']['vision_encoder']['pretrained'] = args.model_pth

    #print(f"This is model pretrained entry: {config['model']['vision_encoder']['pretrained']}")


    #print(f'config['model']['vision_encoder']['pretrained'])

    intern_model, tokenizer = retry_setup(config, 10, 2)


    modelnames = ['phyt2v', 'Video-MSG', 'VideoCrafter2', 'Wan2.1', 'CogVideoX']
    
    for modelname in modelnames:
        with open('PC-3.json','r') as f:
            data = json.load(f)


        result = []
        for i in range(len(data)):
            try:
                try:
                    concept_id = data[i]['concept_id']
                    tpid = data[i]['teaching_point_id']
                    filename = f'Id{concept_id}_{tpid}'
                    data_tmp = data[i]
                    text_candidates = [
                        f"Completely Fantastical:{data_tmp['video_question']['Description1']}",
                        f"Highly Unrealistic:{data_tmp['video_question']['Description2']}",
                        f"Slightly Unrealistic:{data_tmp['video_question']['Description3']}",
                        f"Almost Realistic:{data_tmp['video_question']['Description4']}"
                        
                    ]

                    text_to_index = {text: index for index, text in enumerate(text_candidates)}

                    video_path = f'Results/Combined_Videos/{modelname}/{filename}.mp4'
                    #print(video_path)
                    texts, probs = calculate_video_score(video_path, text_to_index, text_candidates, intern_model, config)

                    print(texts,probs)
                    # break
                    choice = texts[0].split(':')[0]

                    if 'Completely Fantastical' in choice:
                        score = 0
                    elif 'Highly Unrealistic' in choice:
                        score = 1
                    elif 'Slightly Unrealistic' in choice:
                        score = 2
                    elif 'Almost Realistic' in choice:
                        score = 3
                    
                    data_tmp['InternVideo2_score'] = score

                    result.append(data_tmp)
                    print(i)
                except Exception as e:
                    print(f"This is the error: {e}: \n {data[i]}  \n")
                    sys.exit()


                print(len(result))
                with open(f'Results/PC-3-real/prompt_replace_augment_video_question_{modelname}_res_intern.json','w') as f:
                    json.dump(result,f, indent=4)

            except Exception as e:
                print(f'Error:  {e}, \n Videopath =   {video_path}')
                sys.exit()
    

if __name__ == "__main__":
    main()