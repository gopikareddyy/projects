import os
import cv2
import torch
import gradio as gr
import torchvision
import warnings
import numpy as np
from PIL import Image, ImageSequence
from moviepy.editor import VideoFileClip
import imageio
from diffusers import (
    TextToVideoSDPipeline,
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet3DConditionModel,
)
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.utils import export_to_video
from typing import List
from text2vid_modded import TextToVideoSDPipelineModded
from invert_utils import ddim_inversion as dd_inversion
from gifs_filter import filter
import subprocess


def load_frames(image: Image, mode='RGBA'):
    return np.array([np.array(frame.convert(mode)) for frame in ImageSequence.Iterator(image)])


# def run_setup():
#     try:
#         # Step 1: Install Git LFS
#         subprocess.run(["git", "lfs", "install"], check=True)

#         # Step 2: Clone the repository
#         repo_url = "https://huggingface.co/Hmrishav/t2v_sketch-lora"
#         subprocess.run(["git", "clone", repo_url], check=True)

#         # Step 3: Move the checkpoint file
#         source = "t2v_sketch-lora/checkpoint-2500"
#         destination = "./checkpoint-2500/"
#         os.rename(source, destination)

#         print("Setup completed successfully!")
#     except subprocess.CalledProcessError as e:
#         print(f"Error during setup: {e}")
#     except FileNotFoundError as e:
#         print(f"File operation error: {e}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")

# # Automatically run setup during app initialization
# run_setup()


def save_gif(frames, path, target_size=(1024, 1024)):
    # Resize frames to the target size
    resized_frames = [cv2.resize(frame, target_size) for frame in frames]
    # Save the resized frames as a GIF
    imageio.mimsave(
        path,
        [frame.astype(np.uint8) for frame in resized_frames],
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
    return torchvision.transforms.ToTensor()(pil_img).unsqueeze(0)

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
    inv.set_timesteps(25)
    id_latents = dd_inversion(pipe, inv, video_latent=latents, num_inv_steps=25, prompt="")[-1].to(dtype)
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

# Initialize the models
LORA_CHECKPOINT = "checkpoint-2500"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16

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
@torch.no_grad()
def process_video(num_frames, num_seeds, generator, exp_dir, load_name, caption, lambda_):
    pipe_inversion.to(device)
    id_latents = invert(pipe_inversion, inv, load_name).to(device, dtype=dtype)
    latents = id_latents.repeat(num_seeds, 1, 1, 1, 1)
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(num_seeds)]
    video_frames = pipe(
        prompt=caption,
        negative_prompt="",
        num_frames=num_frames,
        num_inference_steps=25,
        inv_latents=latents,
        guidance_scale=9,
        generator=generator,
        lambda_=lambda_,
    ).frames

    gifs = []
    for seed in range(num_seeds):
        vid_name = f"{exp_dir}/mp4_logs/vid_{os.path.basename(load_name)[:-4]}-rand{seed}.mp4"
        gif_name = f"{exp_dir}/gif_logs/vid_{os.path.basename(load_name)[:-4]}-rand{seed}.gif"
        os.makedirs(os.path.dirname(vid_name), exist_ok=True)
        os.makedirs(os.path.dirname(gif_name), exist_ok=True)
        video_path = export_to_video(video_frames[seed], output_video_path=vid_name)
        VideoFileClip(vid_name).write_gif(gif_name)

        with Image.open(gif_name) as im:
            frames = load_frames(im)
            frames_collect = np.empty((0, 1024, 1024), int)
            for frame in frames:
                frame = cv2.resize(frame, (1024, 1024))[:, :, :3]
                frame = cv2.cvtColor(255 - frame, cv2.COLOR_RGB2GRAY)
                _, frame = cv2.threshold(255 - frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                frames_collect = np.append(frames_collect, [frame], axis=0)

            # Save the resized frames as a GIF
            save_gif(frames_collect, gif_name, target_size=(1024, 1024))

        gifs.append(gif_name)

    return gifs

def generate_output(image, prompt: str, num_seeds: int = 3, lambda_value: float = 0.5) -> List[str]:
    """Main function to generate output GIFs"""
    exp_dir = "static/app_tmp"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save the input image temporarily
    temp_image_path = os.path.join(exp_dir, "temp_input.png")
    image.save(temp_image_path)
    
    # Generate the GIFs
    generated_gifs = process_video(
        num_frames=10,
        num_seeds=num_seeds,
        generator=None,
        exp_dir=exp_dir,
        load_name=temp_image_path,
        caption=prompt,
        lambda_=1 - lambda_value
    )
    
    # Apply filtering (assuming filter function is imported)
    filtered_gifs = filter(generated_gifs, temp_image_path)
    
    return filtered_gifs


def create_gradio_interface():
    with gr.Blocks(css="""
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .example-gallery {
            margin: 20px 0;
            padding: 20px;
            background: #f7f7f7;
            border-radius: 8px;
        }
        .selected-example {
            margin: 20px 0;
            padding: 20px;
            background: #ffffff;
            border-radius: 8px;
            
        }
        .controls-section {
            background: #ffffff;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            
        }
        .output-gallery {
            min-height: 500px;
            margin: 20px 0;
            padding: 20px;
            background: #f7f7f7;
            border-radius: 8px;
        }
        .example-item {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
            cursor: pointer;
        }
        .example-item:hover {
            transform: scale(1.05);
        }
        /* Prevent gallery images from expanding */
        .gallery-image {
            height: 200px !important;
            width: 200px !important;
            object-fit: cover !important;
        }
        .generate-btn {
            width: 100%;
            margin-top: 1rem;
            
        }
        
        .generate-btn:disabled {
            opacity: 0.7;
            cursor: not-allowed;
        }
    """) as demo:
        gr.Markdown(
            """
                        
            <div align="center" id = "user-content-toc">
            <img align="left" width="70" height="70" src="https://github.com/user-attachments/assets/c61cec76-3c4b-42eb-8c65-f07e0166b7d8" alt="">
            
            # [FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations](https://hmrishavbandy.github.io/flipsketch-web/)
            ## [Hmrishav Bandyopadhyay](https://hmrishavbandy.github.io/) . [Yi-Zhe Song](https://personalpages.surrey.ac.uk/y.song/)
            </div>

            """
        )
        
        with gr.Tabs() as tabs:
            # First tab: Examples (Secure)
            with gr.Tab("Examples"):
                gr.Markdown("## Step 1 👉  &nbsp; &nbsp; &nbsp; Select a sketch from the gallery of sketches")
                examples_dir = "static/examples"
                if os.path.exists(examples_dir):
                    example_images = []
                    for example in os.listdir(examples_dir):
                        if example.endswith(('.png', '.jpg', '.jpeg')):
                            example_path = os.path.join(examples_dir, example)
                            example_images.append(Image.open(example_path))
                    
                    example_selection = gr.Gallery(
                        example_images,
                        label="Sketch Gallery",
                        elem_classes="example-gallery",
                        columns=4,
                        rows=2,
                        height="auto",
                        allow_preview=False,  # Disable preview expansion
                        show_share_button=False,
                        interactive=False,
                        selected_index=None  # Don't pre-select any image
                    )
                gr.Markdown("## Step 2 👉  &nbsp; &nbsp; &nbsp; Describe the motion you want to generate")
                with gr.Group(elem_classes="selected-example"):
                    with gr.Row():
                        selected_example = gr.Image(
                            type="pil",
                            label="Selected Sketch",
                            scale=1,
                            interactive=False,
                            show_download_button=False,
                            height=300  # Fixed height for consistency
                        )
                        with gr.Column(scale=2):
                            example_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe the motion...",
                                lines=3
                            )
                            with gr.Row():
                                example_num_seeds = gr.Slider(
                                    minimum=1, 
                                    maximum=10, 
                                    value=5, 
                                    step=1, 
                                    label="Seeds"
                                )
                                example_lambda = gr.Slider(
                                    minimum=0, 
                                    maximum=1, 
                                    value=0.5, 
                                    step=0.1, 
                                    label="Motion Strength"
                                )
                            example_generate_btn = gr.Button(
                                "Generate Animation",
                                variant="primary",
                                elem_classes="generate-btn",
                                interactive=True,
                            )

                            
                
                gr.Markdown("## Result 👉 &nbsp; &nbsp; &nbsp; Generated Animations ❤️")
                example_gallery = gr.Gallery(
                    label="Results",
                    elem_classes="output-gallery",
                    columns=3,
                    rows=2,
                    height="auto",
                    allow_preview=False,  # Disable preview expansion
                    show_share_button=False,
                    object_fit="cover",
                    preview=False
                )
            
            # Second tab: Upload
            with gr.Tab("Upload Your Sketch"):
                with gr.Group(elem_classes="selected-example"):
                    with gr.Row():
                        upload_image = gr.Image(
                            type="pil",
                            label="Upload Your Sketch",
                            scale=1,
                            height=300,  # Fixed height for consistency
                            show_download_button=False,
                            sources=["upload"],
                        )
                        with gr.Column(scale=2):
                            upload_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe what you want to generate...",
                                lines=3
                            )
                            with gr.Row():
                                upload_num_seeds = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=5,
                                    step=1,
                                    label="Number of Variations"
                                )
                                upload_lambda = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=0.5,
                                    step=0.1,
                                    label="Motion Strength"
                                )
                            upload_generate_btn = gr.Button(
                                "Generate Animation",
                                variant="primary",
                                elem_classes="generate-btn",
                                size="lg",
                                interactive=True,
                            )
                    
                gr.Markdown("## Result 👉 &nbsp; &nbsp; &nbsp; Generated Animations ❤️")
                upload_gallery = gr.Gallery(
                    label="Results",
                    elem_classes="output-gallery",
                    columns=3,
                    rows=2,
                    height="auto",
                    allow_preview=False,  # Disable preview expansion
                    show_share_button=False,
                    object_fit="cover",
                    preview=False
                )
        
        # Event handlers
        def select_example(evt: gr.SelectData):
            prompts = {'sketch1.png': 'The camel walks slowly',
                'sketch2.png': 'The wine in the wine glass sways from side to side',
                'sketch3.png': 'The squirrel is eating a nut',
                'sketch4.png': 'The surfer surfs on the waves',
                'sketch5.png': 'A galloping horse',
                'sketch6.png': 'The cat walks forward',
                'sketch7.png': 'The eagle flies in the sky',
                'sketch8.png': 'The flower is blooming slowly',
                'sketch9.png': 'The reindeer looks around',
                'sketch10.png': 'The cloud floats in the sky',
                'sketch11.png': 'The jazz saxophonist performs on stage with a rhythmic sway, his upper body sways subtly to the rhythm of the music.',
                'sketch12.png': 'The biker rides on the road',}
            if evt.index < len(example_images):
                example_img = example_images[evt.index]
                prompt_text = prompts.get(os.path.basename(example_img.filename), "")
                

                return [
                    example_img,
                    prompt_text
                ]
            return [None, ""]
        
        example_selection.select(
            select_example,
            None,
            [selected_example, example_prompt]
        )
        
        example_generate_btn.click(
            fn=generate_output,
            inputs=[
                selected_example,
                example_prompt,
                example_num_seeds,
                example_lambda
            ],
            outputs=example_gallery
        )
        
        upload_generate_btn.click(
            fn=generate_output,
            inputs=[
                upload_image,
                upload_prompt,
                upload_num_seeds,
                upload_lambda
            ],
            outputs=upload_gallery
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )
