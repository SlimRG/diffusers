# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple Video Outpaint using WAN 2.1

This script provides a simple interface for video outpaint using WAN 2.1.
It extends video boundaries and fills in missing content while maintaining
temporal consistency across frames.

Example usage:
    python wan_video_outpaint_simple.py
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video


def create_sample_video_frames(width=512, height=512, num_frames=8):
    """
    Create sample video frames for demonstration.
    In practice, you would load these from a video file.
    """
    frames = []
    for i in range(num_frames):
        # Create a simple gradient frame
        frame = Image.new('RGB', (width, height))
        pixels = frame.load()
        for x in range(width):
            for y in range(height):
                r = int(255 * x / width)
                g = int(255 * y / height)
                b = int(255 * i / num_frames)
                pixels[x, y] = (r, g, b)
        frames.append(frame)
    return frames


def main():
    """Main function demonstrating video outpaint."""
    
    # Configuration
    model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"  # or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"Using device: {device}")
    print(f"Loading WAN model: {model_id}")
    
    try:
        # Load models
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
        transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)
        
        # Create scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        print("Models loaded successfully!")
        
        # Create sample video frames (replace with your own video loading)
        print("Creating sample video frames...")
        original_frames = create_sample_video_frames(width=512, height=512, num_frames=8)
        
        # Outpaint configuration
        outpaint_right = 256  # Add 256 pixels to the right
        outpaint_bottom = 128  # Add 128 pixels to the bottom
        
        # Calculate new dimensions
        new_width = 512 + outpaint_right
        new_height = 512 + outpaint_bottom
        
        print(f"Original dimensions: 512x512")
        print(f"New dimensions: {new_width}x{new_height}")
        print(f"Outpainting: Right: {outpaint_right}px, Bottom: {outpaint_bottom}px")
        
        # Prepare frames for outpaint
        prepared_frames = []
        for frame in original_frames:
            # Create new frame with extended dimensions
            new_frame = Image.new('RGB', (new_width, new_height), (0, 0, 0))
            # Place original frame at top-left
            new_frame.paste(frame, (0, 0))
            prepared_frames.append(new_frame)
        
        # Move models to device
        vae = vae.to(device)
        text_encoder = text_encoder.to(device)
        transformer = transformer.to(device)
        
        # Encode text prompt
        prompt = "A beautiful landscape extending to the right and bottom with mountains, trees, and a flowing river"
        negative_prompt = "blurry, low quality, distorted, artifacts, text, watermark"
        
        print(f"Prompt: {prompt}")
        print(f"Negative prompt: {negative_prompt}")
        
        # Tokenize and encode
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=226,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = text_encoder(text_input_ids)[0]
        
        # Encode negative prompt
        neg_text_inputs = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=226,
            truncation=True,
            return_tensors="pt",
        )
        neg_text_input_ids = neg_text_inputs.input_ids.to(device)
        negative_prompt_embeds = text_encoder(neg_text_input_ids)[0]
        
        # Prepare latents
        batch_size = 1
        num_channels_latents = vae.config.latent_channels
        num_frames = len(prepared_frames)
        latent_height = new_height // 8  # VAE scale factor
        latent_width = new_width // 8
        
        # Create noise latents
        latents = torch.randn(
            (batch_size, num_channels_latents, num_frames, latent_height, latent_width),
            device=device,
            dtype=dtype
        )
        
        # Encode prepared frames
        frame_tensor = torch.stack([torch.from_numpy(np.array(frame)) for frame in prepared_frames])
        frame_tensor = frame_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # [1, 8, 3, H, W]
        frame_tensor = frame_tensor.to(device=device, dtype=torch.float32) / 255.0 * 2.0 - 1.0
        
        # Encode to latent space
        with torch.no_grad():
            image_latents = vae.encode(frame_tensor).latents
        
        # Set scheduler timesteps
        num_inference_steps = 20  # Reduced for faster demo
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        
        # Denoising loop
        print("Starting denoising loop...")
        guidance_scale = 7.5
        
        for i, t in enumerate(timesteps):
            print(f"Step {i+1}/{len(timesteps)}")
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = transformer(
                    latent_model_input,
                    t,
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                    return_dict=False,
                )[0]
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # Apply outpaint mask to preserve original content
            if i < len(timesteps) - 1:
                # Create mask for original content (top-left region)
                mask = torch.zeros_like(latents)
                mask[:, :, :, :latent_height, :latent_width] = 1
                
                # Blend original and generated content
                latents = latents * (1 - mask) + image_latents * mask
        
        print("Denoising completed!")
        
        # Decode latents
        print("Decoding latents...")
        latents = 1 / vae.config.scaling_factor * latents
        
        with torch.no_grad():
            video = vae.decode(latents).sample
        
        # Post-process
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().float().numpy()
        
        # Convert to PIL images
        video_frames = []
        for frame in video[0]:  # Remove batch dimension
            frame = (frame * 255).round().astype("uint8")
            frame = frame.transpose(1, 2, 0)  # CHW -> HWC
            video_frames.append(Image.fromarray(frame))
        
        # Save output video
        output_path = "wan_outpaint_output.mp4"
        print(f"Saving output video to: {output_path}")
        export_to_video(video_frames, output_path, fps=8)
        
        print("Video outpaint completed successfully!")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install diffusers transformers torch opencv-python pillow")


if __name__ == "__main__":
    main()