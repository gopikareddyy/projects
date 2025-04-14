from flask import Flask, render_template, request, jsonify
import os
import cv2
import torch
import torchvision
import warnings
import numpy as np
from PIL import Image, ImageSequence
from moviepy.editor import VideoFileClip
import imageio
import uuid
import pickle
import argparse

from diffusers import (
    TextToVideoSDPipeline,
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet3DConditionModel,
)
import time
from transformers import CLIPTokenizer, CLIPTextModel

from diffusers.utils import export_to_video
from gifs_filter import filter
from invert_utils import ddim_inversion as dd_inversion
from text2vid_modded import TextToVideoSDPipelineModded

# Flask app setup

UPLOAD_FOLDER = 'static/uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Environment setup
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
LORA_CHECKPOINT = "checkpoint-2500"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

# Helper functions

def cleanup_old_files(directory, age_in_seconds = 600):
    """
    Deletes files older than a certain age in the specified directory.

    Args:
        directory (str): The directory to clean up.
        age_in_seconds (int): The age in seconds; files older than this will be deleted.
    """
    now = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Only delete files (not directories)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > age_in_seconds:
                try:
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
                    
def load_frames(image: Image, mode='RGBA'):
    return np.array([np.array(frame.convert(mode)) for frame in ImageSequence.Iterator(image)])

def save_gif(frames, path):
    imageio.mimsave(
        path,
        [frame.astype(np.uint8) for frame in frames],
        format="GIF",
        duration=1 / 10,
        loop=0  # 0 means infinite loop
    )
def load_image(imgname, target_size=None):
    pil_img = Image.open(imgname).convert('RGB')
    if target_size:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
    return torchvision.transforms.ToTensor()(pil_img).unsqueeze(0)  # Add batch dimension

def prepare_latents(pipe, x_aug):
    with torch.cuda.amp.autocast():
        batch_size, num_frames, channels, height, width = x_aug.shape
        x_aug = x_aug.reshape(batch_size * num_frames, channels, height, width)
        latents = pipe.vae.encode(x_aug).latent_dist.sample()
        latents = latents.view(batch_size, num_frames, -1, latents.shape[2], latents.shape[3])
        latents = latents.permute(0, 2, 1, 3, 4)
    return pipe.vae.config.scaling_factor * latents

@torch.no_grad()
def invert(pipe, inv, load_name, device="cuda", dtype=torch.bfloat16):
    input_img = [load_image(load_name, 256).to(device, dtype=dtype).unsqueeze(1)] * 5
    input_img = torch.cat(input_img, dim=1)
    latents = prepare_latents(pipe, input_img).to(torch.bfloat16)
    inv.set_timesteps(20)
    id_latents = dd_inversion(pipe, inv, video_latent=latents, num_inv_steps=20, prompt="")[-1].to(dtype)
    return torch.mean(id_latents, dim=2, keepdim=True)

def load_primary_models(pretrained_model_path):
    return (
        DDPMScheduler.from_config(pretrained_model_path, subfolder="scheduler"),
        CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer"),
        CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder"),
        AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae"),
        UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet"),
    )


def initialize_pipeline(model: str, device: str = "cuda"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(model)
    pipe = TextToVideoSDPipeline.from_pretrained(
        pretrained_model_name_or_path="damo-vilab/text-to-video-ms-1.7b",
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.bfloat16),
        vae=vae.to(device=device, dtype=torch.bfloat16),
        unet=unet.to(device=device, dtype=torch.bfloat16),
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe, pipe.scheduler

pipe_inversion, inv = initialize_pipeline(LORA_CHECKPOINT, device)
pipe = TextToVideoSDPipelineModded.from_pretrained(
    pretrained_model_name_or_path="damo-vilab/text-to-video-ms-1.7b",
    scheduler=pipe_inversion.scheduler,
    tokenizer=pipe_inversion.tokenizer,
    text_encoder=pipe_inversion.text_encoder,
    vae=pipe_inversion.vae,
    unet=pipe_inversion.unet,
).to(device)

@torch.no_grad()
def process(num_frames, num_seeds, generator, exp_dir, load_name, caption, lambda_, out_name):
    pipe_inversion.to(device)
    id_latents = invert(pipe_inversion, inv, load_name).to(device, dtype=dtype)
    latents = id_latents.repeat(num_seeds, 1, 1, 1, 1)
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(num_seeds)]
    video_frames = pipe(
        prompt=caption,
        negative_prompt="",
        num_frames=num_frames,
        num_inference_steps=20,
        inv_latents=latents,
        guidance_scale=9,
        generator=generator,
        lambda_=lambda_,
    ).frames
    
    gifs = []
    for seed in range(num_seeds):
        print(f"Processing seed {seed}")
        vid_name = f"{exp_dir}/mp4_logs/vid_{out_name}_{seed}.mp4"
        gif_name = f"{exp_dir}/gif_logs/vid_{out_name}_{seed}.gif"
        video_path = export_to_video(video_frames[seed], output_video_path=vid_name)
        VideoFileClip(vid_name).write_gif(gif_name)
        
        with Image.open(gif_name) as im:
            frames = load_frames(im)
        
        frames_collect = np.empty((0, 1024, 1024), int)
        
        for frame in frames:
            # Resize with aspect ratio kept intact
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            
            new_width = 1024  # Set the width
            new_height = int(new_width / aspect_ratio)  # Adjust height to keep aspect ratio
            
            resized_frame = cv2.resize(frame, (new_width, new_height))[:, :, :3]
            resized_frame = cv2.cvtColor(255 - resized_frame, cv2.COLOR_RGB2GRAY)
            _, resized_frame = cv2.threshold(255 - resized_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            frames_collect = np.append(frames_collect, [resized_frame], axis=0)
        
        save_gif(frames_collect, gif_name)
        gifs.append(gif_name)
    
    return gifs



# def process(num_frames, num_seeds, generator, exp_dir, load_name, caption, lambda_,out_name):
#     pipe_inversion.to(device)
#     id_latents = invert(pipe_inversion, inv, load_name).to(device, dtype=dtype)
#     latents = id_latents.repeat(num_seeds, 1, 1, 1, 1)
#     generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(num_seeds)]
#     video_frames = pipe(
#         prompt=caption, 
#         negative_prompt="",
#         num_frames=num_frames,
#         num_inference_steps=20, 
#         inv_latents=latents, 
#         guidance_scale=9,
#         generator=generator,
#         lambda_=lambda_,
#     ).frames
#     try: 
#         load_name = load_name.split("/")[-1]
#     except:
#         pass
#     gifs = []
#     for seed in range(num_seeds):
#         print(seed)
#         vid_name = f"{exp_dir}/mp4_logs/vid_{out_name}_{seed}.mp4"
#         gif_name = f"{exp_dir}/gif_logs/vid_{out_name}_{seed}.gif"
#         video_path = export_to_video(video_frames[seed], output_video_path=vid_name)
#         VideoFileClip(vid_name).write_gif(gif_name)
#         with Image.open(gif_name) as im:
#             frames = load_frames(im)

#         frames_collect = np.empty((0, 1024, 1024), int)
#         for frame in frames:
#             frame = cv2.resize(frame, (1024, 1024))[:, :, :3]
#             frame = cv2.cvtColor(255 - frame, cv2.COLOR_RGB2GRAY)

#             _, frame = cv2.threshold(255 - frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#             frames_collect = np.append(frames_collect, [frame], axis=0)

#         save_gif(frames_collect, gif_name)
#         gifs.append(gif_name)

#     return gifs
        

def generate_gifs(filepath, prompt,out_name, num_seeds=5, lambda_=0):
    exp_dir = "static/app_tmp"
    os.makedirs(exp_dir, exist_ok=True)
    gifs = process(
        num_frames=10,
        num_seeds=num_seeds,
        generator=None,
        exp_dir=exp_dir,
        load_name=filepath,
        caption=prompt,
        lambda_=lambda_,
        out_name=out_name
    )
    return gifs


def index():
    return render_template('index.html')


def generate(out_name,prompt="",num_gifs=3,lambda_value=0.5,selected_example=None,file=None):
    
    directories_to_clean = [
        UPLOAD_FOLDER,
        'static/app_tmp/mp4_logs',
        'static/app_tmp/gif_logs',
        'static/app_tmp/png_logs'
    ]

    # Perform cleanup
    os.makedirs('static/app_tmp', exist_ok=True)
    for directory in directories_to_clean:
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        cleanup_old_files(directory)

    if not file and not selected_example:
        return jsonify({'error': 'No image file provided or example selected'}), 400

    if selected_example:
        # Use the selected example image
        filepath = os.path.join('static', 'examples', selected_example)
        unique_id = None  # No need for unique ID
    else:
        # Save the uploaded image
        unique_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_FOLDER, f"{unique_id}_uploaded_image.png")
        file.save(filepath)

    generated_gifs = generate_gifs(filepath, prompt,out_name, num_seeds=num_gifs, lambda_=lambda_value)

    unique_id = str(uuid.uuid4())
    # Append unique id to each gif path
    for i in range(len(generated_gifs)):
        os.rename(generated_gifs[i], f"{generated_gifs[i].split('.')[0]}_{unique_id}.gif")
        generated_gifs[i] = f"{generated_gifs[i].split('.')[0]}_{unique_id}.gif"
        # Move the generated gifs to the static folder
        
    # with open(f"output_gifs_{unique_id}.pkl","wb") as f:
    #     pickle.dump(generated_gifs,f)
    filtered_gifs = filter(generated_gifs, filepath)
    return {'gifs': filtered_gifs, 'prompt': prompt}

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Arguments for flipsketch")

    # Add arguments
    parser.add_argument('out_name',type=str,help="Give a unique name for storing the gifs in static app_tmp")
    parser.add_argument('prompt', type=str, help='Give the prompt for motion generation')
    parser.add_argument('selected_example', type=str, help='Give the file name from the examples folder')
    parser.add_argument('num_gifs',type=int,default=3)
    parser.add_argument('lambda_value',type=float,default=0.5)
    
    # Parse the arguments
    args = parser.parse_args()

    out = generate(
      out_name=args.out_name,
      prompt=args.prompt,
      num_gifs=int(args.num_gifs),
      lambda_value=float(args.lambda_value),
      selected_example=args.selected_example,
      file=None)
