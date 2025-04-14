# filter images
from PIL import Image, ImageSequence
import requests
from tqdm import tqdm
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def load_frames(image: Image, mode='RGBA'):
    return np.array([
        np.array(frame.convert(mode))
        for frame in ImageSequence.Iterator(image)
    ])

img_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
img_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



def filter(gifs, input_image): 
    max_cosine = 0.9
    max_gif = []
    
    for gif in tqdm(gifs, total=len(gifs)):
        with Image.open(gif) as im:
            frames = load_frames(im)

        frames = np.array(frames)
        frames = frames[:, :, :, :3]
        frames = np.transpose(frames, (0, 3, 1, 2))[1:]



        image = Image.open(input_image)
        
        
        inputs = img_processor(images=frames, return_tensors="pt", padding=False)
        inputs_base = img_processor(images=image, return_tensors="pt", padding=False)
        
        with torch.no_grad():
            feat_img_base = img_model.get_image_features(pixel_values=inputs_base["pixel_values"])        
            feat_img_vid = img_model.get_image_features(pixel_values=inputs["pixel_values"])
        cos_avg = 0
        avg_score_for_vid = 0
        for i in range(len(feat_img_vid)):
                
            cosine_similarity = torch.nn.functional.cosine_similarity(
                feat_img_base, 
                feat_img_vid[0].unsqueeze(0), 
                dim=1)
            # print(cosine_similarity)
            cos_avg += cosine_similarity.item()
            
        cos_avg /= len(feat_img_vid)
        print("Current cosine similarity: ", cos_avg)
        print("Max cosine similarity: ", max_cosine)
        if cos_avg > max_cosine:
            # max_cosine = cos_avg
            max_gif.append(gif)
    return max_gif