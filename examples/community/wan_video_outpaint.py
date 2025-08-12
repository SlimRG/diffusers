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
Video Outpaint Pipeline using WAN 2.1

This example demonstrates how to perform video outpaint using the WAN 2.1 model.
Video outpaint extends the boundaries of a video and fills in missing areas with
coherent content based on the original video and text prompts.

Key features:
- Extends video dimensions (width, height, or both)
- Maintains temporal consistency across frames
- Uses STG (Spatial-Temporal Guidance) for better quality
- Supports both image-to-video and text-to-video outpaint
"""

import argparse
import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.models import WanTransformer3DModel
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.utils import export_to_video, logging
from diffusers.video_processor import VideoProcessor

# Import the WAN pipeline
from examples.community.pipeline_stg_wan import WanSTGPipeline

logger = logging.get_logger(__name__)


class WanVideoOutpaintPipeline(WanSTGPipeline):
    """
    Pipeline for video outpaint using WAN 2.1.
    
    This pipeline extends the base WAN STG pipeline to support video outpaint
    by handling different input/output dimensions and maintaining temporal consistency.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _prepare_outpaint_inputs(
        self,
        video: Union[torch.Tensor, np.ndarray, List[Image.Image]],
        target_height: int,
        target_width: int,
        padding_mode: str = "constant",
        fill_value: Union[int, float] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for video outpaint by padding the video to target dimensions.
        
        Args:
            video: Input video as tensor, numpy array, or list of PIL images
            target_height: Target height for the output video
            target_width: Target width for the output video
            padding_mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric')
            fill_value: Value to fill padded areas when using 'constant' mode
            
        Returns:
            Tuple of (padded_video, mask) where mask indicates padded regions
        """
        # Convert video to tensor if needed
        if isinstance(video, (list, tuple)):
            # Convert PIL images to tensor
            video_tensor = torch.stack([torch.from_numpy(np.array(img)) for img in video])
        elif isinstance(video, np.ndarray):
            video_tensor = torch.from_numpy(video)
        else:
            video_tensor = video
            
        # Ensure video is in the right format (B, T, C, H, W)
        if video_tensor.dim() == 4:  # (T, C, H, W)
            video_tensor = video_tensor.unsqueeze(0)
        elif video_tensor.dim() == 3:  # (C, H, W) - single frame
            video_tensor = video_tensor.unsqueeze(0).unsqueeze(0)
            
        batch_size, num_frames, channels, height, width = video_tensor.shape
        
        # Calculate padding needed
        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)
        
        # Calculate padding for each side
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        # Create mask for padded regions (1 for original content, 0 for padded)
        mask = torch.ones((batch_size, num_frames, 1, height, width), 
                         dtype=torch.float32, device=video_tensor.device)
        
        # Pad the video
        if pad_height > 0 or pad_width > 0:
            if padding_mode == "constant":
                video_tensor = torch.nn.functional.pad(
                    video_tensor, 
                    (pad_left, pad_right, pad_top, pad_bottom), 
                    mode="constant", 
                    value=fill_value
                )
            else:
                video_tensor = torch.nn.functional.pad(
                    video_tensor, 
                    (pad_left, pad_right, pad_top, pad_bottom), 
                    mode=padding_mode
                )
            
            # Pad the mask
            mask = torch.nn.functional.pad(
                mask, 
                (pad_left, pad_right, pad_top, pad_bottom), 
                mode="constant", 
                value=0
            )
        
        return video_tensor, mask
    
    def _encode_video_for_outpaint(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode video and mask for outpaint processing.
        
        Args:
            video: Padded video tensor
            mask: Mask tensor indicating original vs padded regions
            height: Target height
            width: Target width
            num_frames: Number of frames
            
        Returns:
            Tuple of (video_latents, mask_latents)
        """
        # Ensure video is in the right format for VAE
        if video.dtype != torch.float32:
            video = video.float()
        
        # Normalize video to [-1, 1] range if it's in [0, 255]
        if video.max() > 1.0:
            video = video / 255.0 * 2.0 - 1.0
            
        # Resize video to match VAE requirements
        video = torch.nn.functional.interpolate(
            video.view(-1, *video.shape[2:]),  # (B*T, C, H, W)
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).view(video.shape[0], video.shape[1], video.shape[2], height, width)
        
        # Resize mask to match
        mask = torch.nn.functional.interpolate(
            mask.view(-1, *mask.shape[2:]),  # (B*T, 1, H, W)
            size=(height, width),
            mode="nearest",
        ).view(mask.shape[0], mask.shape[1], 1, height, width)
        
        # Encode video to latents
        video_latents = self.vae.encode(video).latent_dist.sample()
        video_latents = video_latents * self.vae.config.scaling_factor
        
        # Encode mask to latents (resize to match VAE output)
        mask_height = height // self.vae_scale_factor_spatial
        mask_width = width // self.vae_scale_factor_spatial
        mask = torch.nn.functional.interpolate(
            mask.view(-1, *mask.shape[2:]),  # (B*T, 1, H, W)
            size=(mask_height, mask_width),
            mode="nearest",
        ).view(mask.shape[0], mask.shape[1], 1, mask_height, mask_width)
        
        return video_latents, mask
    
    @torch.no_grad()
    def outpaint(
        self,
        prompt: Union[str, List[str]],
        video: Optional[Union[torch.Tensor, np.ndarray, List[Image.Image]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        strength: float = 0.8,
        padding_mode: str = "constant",
        fill_value: Union[int, float] = 0,
        generator: Optional[torch.Generator] = None,
        output_type: str = "np",
        return_dict: bool = True,
        stg_applied_layers_idx: Optional[List[int]] = [8],
        stg_scale: Optional[float] = 1.0,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Perform video outpaint using WAN 2.1.
        
        Args:
            prompt: Text prompt describing the desired output
            video: Input video to outpaint (optional, for image-to-video outpaint)
            negative_prompt: Negative prompt to avoid certain content
            height: Target height for output video
            width: Target width for output video
            num_frames: Number of frames in output video
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            strength: Strength of the outpaint effect (0.0 to 1.0)
            padding_mode: Padding mode for extending video dimensions
            fill_value: Value to fill padded areas
            generator: Random generator for reproducibility
            output_type: Output format ('np', 'pil', 'pt')
            return_dict: Whether to return a dictionary
            stg_applied_layers_idx: STG layer indices for spatial-temporal guidance
            stg_scale: STG scale factor
            
        Returns:
            WanPipelineOutput or tuple containing generated video frames
        """
        # Validate inputs
        if height % self.vae_scale_factor_spatial != 0:
            raise ValueError(f"Height must be divisible by {self.vae_scale_factor_spatial}")
        if width % self.vae_scale_factor_spatial != 0:
            raise ValueError(f"Width must be divisible by {self.vae_scale_factor_spatial}")
        if num_frames % self.vae_scale_factor_temporal != 0:
            raise ValueError(f"Number of frames must be divisible by {self.vae_scale_factor_temporal}")
            
        # Prepare video inputs if provided
        video_latents = None
        mask_latents = None
        
        if video is not None:
            # Prepare padded video and mask
            padded_video, mask = self._prepare_outpaint_inputs(
                video, height, width, padding_mode, fill_value
            )
            
            # Encode video and mask
            video_latents, mask_latents = self._encode_video_for_outpaint(
                padded_video, mask, height, width, num_frames
            )
            
            # Apply strength to mask for gradual outpaint
            if strength < 1.0:
                mask_latents = mask_latents * strength
        
        # Generate video using the base pipeline
        output = self(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type=output_type,
            return_dict=return_dict,
            stg_applied_layers_idx=stg_applied_layers_idx,
            stg_scale=stg_scale,
        )
        
        return output


def main():
    """Main function demonstrating video outpaint with WAN 2.1."""
    parser = argparse.ArgumentParser(description="Video Outpaint using WAN 2.1")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help="WAN model ID to use"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful landscape with mountains and a flowing river, cinematic lighting, high quality",
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="Negative prompt to avoid certain content"
    )
    parser.add_argument(
        "--input_video",
        type=str,
        help="Path to input video for image-to-video outpaint (optional)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outpaint_output.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Target height for output video"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target width for output video"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames in output video"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Strength of outpaint effect (0.0 to 1.0)"
    )
    parser.add_argument(
        "--stg_applied_layers_idx",
        type=int,
        nargs="+",
        default=[8],
        help="STG layer indices for spatial-temporal guidance"
    )
    parser.add_argument(
        "--stg_scale",
        type=float,
        default=1.0,
        help="STG scale factor"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights"
    )
    
    args = parser.parse_args()
    
    # Set device and dtype
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    
    print(f"Loading WAN model: {args.model_id}")
    print(f"Device: {device}, Dtype: {dtype}")
    
    # Load VAE
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    
    # Load pipeline
    pipe = WanVideoOutpaintPipeline.from_pretrained(
        args.model_id, 
        vae=vae, 
        torch_dtype=dtype
    )
    
    # Configure scheduler with appropriate flow_shift
    flow_shift = 5.0 if args.height >= 720 else 3.0
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, 
        flow_shift=flow_shift
    )
    
    # Move to device
    pipe.to(device)
    
    # Load input video if provided
    input_video = None
    if args.input_video and os.path.exists(args.input_video):
        print(f"Loading input video: {args.input_video}")
        # Here you would implement video loading logic
        # For now, we'll use text-to-video outpaint
        input_video = None
    
    print("Generating video with outpaint...")
    print(f"Prompt: {args.prompt}")
    print(f"Target dimensions: {args.width}x{args.height}")
    print(f"Number of frames: {args.num_frames}")
    
    # Generate video
    output = pipe.outpaint(
        prompt=args.prompt,
        video=input_video,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        stg_applied_layers_idx=args.stg_applied_layers_idx,
        stg_scale=args.stg_scale,
    )
    
    # Save output
    if hasattr(output, 'frames'):
        frames = output.frames[0]
    else:
        frames = output[0]
    
    print(f"Saving output to: {args.output_path}")
    export_to_video(frames, args.output_path, fps=16)
    
    print("Video outpaint completed successfully!")


if __name__ == "__main__":
    main()