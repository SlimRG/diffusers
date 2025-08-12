#!/usr/bin/env python3
"""
Simple Video Outpaint Example using WAN 2.1

This script demonstrates basic video outpaint functionality using the WAN 2.1 model.
It shows how to extend video dimensions and fill in missing areas with coherent content.

Usage:
    python wan_video_outpaint_simple.py
"""

import torch
from diffusers import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# Import our custom outpaint pipeline
from wan_video_outpaint import WanVideoOutpaintPipeline


def main():
    """Simple example of video outpaint with WAN 2.1."""
    
    # Configuration
    model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"  # or use 1.3B for smaller model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"Using device: {device}")
    print(f"Model: {model_id}")
    
    # Load VAE (always use float32 for VAE)
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    
    # Load pipeline
    print("Loading pipeline...")
    pipe = WanVideoOutpaintPipeline.from_pretrained(
        model_id, 
        vae=vae, 
        torch_dtype=dtype
    )
    
    # Configure scheduler for 720p output
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, 
        flow_shift=flow_shift
    )
    
    # Move to device
    pipe.to(device)
    
    # Example 1: Basic text-to-video outpaint
    print("\n=== Example 1: Basic Text-to-Video Outpaint ===")
    
    prompt = "A majestic mountain landscape with snow-capped peaks, flowing rivers, and dramatic clouds, cinematic lighting, high quality, 8K"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    print(f"Generating video with prompt: {prompt}")
    
    # Generate video with outpaint to larger dimensions
    output = pipe.outpaint(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=720,      # Target height (original might be smaller)
        width=1280,      # Target width (original might be smaller)
        num_frames=81,   # Number of frames
        num_inference_steps=30,  # Fewer steps for faster generation
        guidance_scale=5.0,
        strength=0.8,    # Outpaint strength
        stg_applied_layers_idx=[8],  # STG layer for spatial-temporal guidance
        stg_scale=1.0,   # STG scale
    )
    
    # Save output
    frames = output.frames[0] if hasattr(output, 'frames') else output[0]
    output_path = "basic_outpaint_output.mp4"
    print(f"Saving to: {output_path}")
    export_to_video(frames, output_path, fps=16)
    
    # Example 2: High-quality outpaint with more steps
    print("\n=== Example 2: High-Quality Outpaint ===")
    
    prompt2 = "A serene forest scene with tall trees, dappled sunlight, gentle breeze moving leaves, peaceful atmosphere, high quality, cinematic"
    
    print(f"Generating high-quality video with prompt: {prompt2}")
    
    output2 = pipe.outpaint(
        prompt=prompt2,
        negative_prompt=negative_prompt,
        height=720,
        width=1280,
        num_frames=81,
        num_inference_steps=50,  # More steps for better quality
        guidance_scale=7.0,      # Higher guidance for better prompt adherence
        strength=0.9,            # Higher strength for more aggressive outpaint
        stg_applied_layers_idx=[3, 8, 16],  # Multiple STG layers
        stg_scale=1.0,
    )
    
    frames2 = output2.frames[0] if hasattr(output2, 'frames') else output2[0]
    output_path2 = "high_quality_outpaint_output.mp4"
    print(f"Saving to: {output_path2}")
    export_to_video(frames2, output_path2, fps=16)
    
    # Example 3: Different aspect ratio outpaint
    print("\n=== Example 3: Different Aspect Ratio Outpaint ===")
    
    prompt3 = "A futuristic cityscape with neon lights, flying cars, towering skyscrapers, cyberpunk aesthetic, high quality, cinematic"
    
    print(f"Generating widescreen video with prompt: {prompt3}")
    
    output3 = pipe.outpaint(
        prompt=prompt3,
        negative_prompt=negative_prompt,
        height=720,
        width=1920,      # Widescreen 16:9 aspect ratio
        num_frames=81,
        num_inference_steps=40,
        guidance_scale=6.0,
        strength=0.85,
        stg_applied_layers_idx=[8],
        stg_scale=1.0,
    )
    
    frames3 = output3.frames[0] if hasattr(output3, 'frames') else output3[0]
    output_path3 = "widescreen_outpaint_output.mp4"
    print(f"Saving to: {output_path3}")
    export_to_video(frames3, output_path3, fps=16)
    
    print("\n=== All examples completed! ===")
    print(f"Generated videos:")
    print(f"1. {output_path}")
    print(f"2. {output_path2}")
    print(f"3. {output_path3}")
    
    print("\nTips for better results:")
    print("- Use descriptive prompts with specific details")
    print("- Adjust guidance_scale (5.0-8.0) for prompt adherence")
    print("- Use strength 0.7-0.9 for balanced outpaint")
    print("- Experiment with different STG layer combinations")
    print("- Higher num_inference_steps = better quality but slower generation")


if __name__ == "__main__":
    main()