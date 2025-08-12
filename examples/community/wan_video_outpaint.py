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
Video Outpaint using WAN 2.1

This script demonstrates how to perform video outpaint using the WAN 2.1 model.
Video outpaint extends the boundaries of a video and fills in the missing content
using AI generation, maintaining temporal consistency across frames.

Features:
- Extend video dimensions (width, height, or both)
- Maintain temporal consistency across frames
- Support for different outpaint directions (left, right, top, bottom)
- Configurable outpaint regions and overlap
- High-quality video generation with WAN 2.1

Usage:
    python wan_video_outpaint.py --input_video path/to/video.mp4 --output_path output.mp4 --outpaint_right 256
"""

import argparse
import os
import warnings
from typing import List, Optional, Tuple, Union, Callable, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, logging
from diffusers.video_processor import VideoProcessor

import torch.nn.functional as F
from torch import randn_tensor


logger = logging.get_logger(__name__)


class WanVideoOutpaintPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    """
    Pipeline for video outpaint using WAN 2.1.
    
    This pipeline extends video boundaries and fills in missing content
    while maintaining temporal consistency across frames.
    """
    
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        
        # Tokenize and encode the prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        if text_input_ids.shape[-1] > max_sequence_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, max_sequence_length:])
            logger.warning(
                f"The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, :max_sequence_length]
        
        text_embeddings = self.text_encoder(text_input_ids.to(device))[0]
        
        # Duplicate text embeddings for each generation per prompt
        seq_len = text_embeddings.shape[1]
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)
        
        return text_embeddings
    
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        
        # Encode the prompt
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
            dtype=dtype,
        )
        
        # Encode the negative prompt
        if negative_prompt is None:
            negative_prompt = ""
        negative_prompt_embeds = self._get_t5_prompt_embeds(
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
            dtype=dtype,
        )
        
        return prompt_embeds, negative_prompt_embeds
    
    def _prepare_video_frames(
        self,
        video_frames: List[Image.Image],
        target_width: int,
        target_height: int,
        outpaint_left: int = 0,
        outpaint_right: int = 0,
        outpaint_top: int = 0,
        outpaint_bottom: int = 0,
    ) -> List[Image.Image]:
        """
        Prepare video frames for outpaint by extending dimensions and creating masks.
        
        Args:
            video_frames: List of input video frames
            target_width: Target width after outpaint
            target_height: Target height after outpaint
            outpaint_left: Pixels to add on the left
            outpaint_right: Pixels to add on the right
            outpaint_top: Pixels to add on the top
            outpaint_bottom: Pixels to add on the bottom
            
        Returns:
            List of prepared frames with extended dimensions
        """
        prepared_frames = []
        
        for frame in video_frames:
            # Convert to numpy array
            frame_array = np.array(frame)
            h, w = frame_array.shape[:2]
            
            # Create new frame with target dimensions
            new_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate source and destination regions
            src_x = outpaint_left
            src_y = outpaint_top
            dst_x = 0
            dst_y = 0
            
            # Copy original frame to new frame
            new_frame[dst_y:dst_y+h, dst_x:dst_x+w] = frame_array
            
            # Convert back to PIL Image
            prepared_frame = Image.fromarray(new_frame)
            prepared_frames.append(prepared_frame)
        
        return prepared_frames
    
    def _create_outpaint_mask(
        self,
        width: int,
        height: int,
        outpaint_left: int = 0,
        outpaint_right: int = 0,
        outpaint_top: int = 0,
        outpaint_bottom: int = 0,
    ) -> np.ndarray:
        """
        Create a mask indicating which regions need to be outpainted.
        
        Args:
            width: Frame width
            height: Frame height
            outpaint_left: Pixels to add on the left
            outpaint_right: Pixels to add on the right
            outpaint_top: Pixels to add on the top
            outpaint_bottom: Pixels to add on the bottom
            
        Returns:
            Binary mask where 1 indicates regions to be outpainted
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Mark outpaint regions
        if outpaint_left > 0:
            mask[:, :outpaint_left] = 1
        if outpaint_right > 0:
            mask[:, -outpaint_right:] = 1
        if outpaint_top > 0:
            mask[:outpaint_top, :] = 1
        if outpaint_bottom > 0:
            mask[-outpaint_bottom:, :] = 1
        
        return mask
    
    def _prepare_latents(
        self,
        video_frames: List[Image.Image],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
    ):
        """
        Prepare latents for video generation.
        
        Args:
            video_frames: Input video frames
            batch_size: Batch size
            num_channels_latents: Number of latent channels
            height: Frame height
            width: Frame width
            num_frames: Number of frames
            dtype: Data type
            device: Device to use
            generator: Random generator
            latents: Pre-computed latents
            
        Returns:
            Prepared latents tensor
        """
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        video_frames: List[Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        outpaint_left: int = 0,
        outpaint_right: int = 0,
        outpaint_top: int = 0,
        outpaint_bottom: int = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Generate video with outpainted regions.
        
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt
            video_frames: Input video frames
            height: Output height
            width: Output width
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            outpaint_left: Pixels to add on the left
            outpaint_right: Pixels to add on the right
            outpaint_top: Pixels to add on the top
            outpaint_bottom: Pixels to add on the bottom
            generator: Random generator
            latents: Pre-computed latents
            output_type: Output type ("pil", "latent", "np")
            return_dict: Whether to return a dictionary
            callback: Callback function
            callback_steps: Callback frequency
            cross_attention_kwargs: Cross-attention kwargs
            
        Returns:
            Generated video frames
        """
        # 0. Default height and width to video dimensions
        if height is None:
            height = video_frames[0].height + outpaint_top + outpaint_bottom
        if width is None:
            width = video_frames[0].width + outpaint_left + outpaint_right
        
        # 1. Check inputs
        self.check_inputs(prompt, height, width, callback_steps)
        
        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        dtype = self.transformer.dtype
        
        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=1,
            device=device,
            dtype=dtype,
        )
        
        # 4. Prepare video frames
        prepared_frames = self._prepare_video_frames(
            video_frames=video_frames,
            target_width=width,
            target_height=height,
            outpaint_left=outpaint_left,
            outpaint_right=outpaint_right,
            outpaint_top=outpaint_top,
            outpaint_bottom=outpaint_bottom,
        )
        
        # 5. Create outpaint mask
        outpaint_mask = self._create_outpaint_mask(
            width=width,
            height=height,
            outpaint_left=outpaint_left,
            outpaint_right=outpaint_right,
            outpaint_top=outpaint_top,
            outpaint_bottom=outpaint_bottom,
        )
        
        # 6. Prepare latents
        num_channels_latents = self.vae.config.latent_channels
        latents = self._prepare_latents(
            video_frames=prepared_frames,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        
        # 7. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 8. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)
        
        # 9. Add image latents to the conditioning
        image_latents = self.vae.encode(prepared_frames).latents
        image_latents = image_latents.to(device=device, dtype=dtype)
        
        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    t,
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                
                # Perform guidance
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # Apply outpaint mask to maintain original content
                if i < len(timesteps) - 1:  # Don't apply on the last step
                    # Convert mask to latent space
                    mask_latent = torch.from_numpy(outpaint_mask).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    mask_latent = F.interpolate(
                        mask_latent.float(),
                        size=latents.shape[-2:],
                        mode="nearest"
                    ).to(device=device, dtype=dtype)
                    
                    # Blend original and generated content
                    latents = latents * (1 - mask_latent) + image_latents * mask_latent
                
                # Call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        # 11. Post-processing
        video = self.decode_latents(latents)
        
        # 12. Convert to PIL
        if output_type == "pil":
            video = self.numpy_to_pil(video)
        
        # 13. Offload all models
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return video
        
        return {"frames": video}
    
    def decode_latents(self, latents):
        """Decode latents to video frames."""
        latents = 1 / self.vae.config.scaling_factor * latents
        video = self.vae.decode(latents).sample
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().float().numpy()
        return video
    
    def numpy_to_pil(self, video):
        """Convert numpy video to PIL images."""
        video = (video * 255).round().astype("uint8")
        video = video.transpose(0, 2, 3, 1)
        pil_images = [Image.fromarray(frame) for frame in video]
        return pil_images


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[Image.Image]:
    """
    Load video frames from a video file.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to load
        
    Returns:
        List of PIL Image objects
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if max_frames and frame_count >= max_frames:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)
        frame_count += 1
    
    cap.release()
    return frames


def main():
    """Main function for video outpaint."""
    parser = argparse.ArgumentParser(description="Video Outpaint using WAN 2.1")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_path", type=str, required=True, help="Path for output video")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape with mountains and trees", 
                       help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, 
                       default="blurry, low quality, distorted, artifacts", 
                       help="Negative text prompt")
    parser.add_argument("--outpaint_left", type=int, default=0, help="Pixels to add on the left")
    parser.add_argument("--outpaint_right", type=int, default=0, help="Pixels to add on the right")
    parser.add_argument("--outpaint_top", type=int, default=0, help="Pixels to add on the top")
    parser.add_argument("--outpaint_bottom", type=int, default=0, help="Pixels to add on the bottom")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to generate")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--model_id", type=str, 
                       default="Wan-AI/Wan2.1-T2V-14B-Diffusers", 
                       help="WAN model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    
    args = parser.parse_args()
    
    # Check if any outpaint direction is specified
    if all(x == 0 for x in [args.outpaint_left, args.outpaint_right, args.outpaint_top, args.outpaint_bottom]):
        raise ValueError("At least one outpaint direction must be specified")
    
    # Set device and dtype
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    
    print(f"Loading WAN model: {args.model_id}")
    
    # Load models
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(args.model_id, subfolder="text_encoder", torch_dtype=dtype)
    transformer = WanTransformer3DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=dtype)
    
    # Create scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    
    # Create pipeline
    pipeline = WanVideoOutpaintPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )
    
    # Move to device
    pipeline.to(device)
    
    # Load video frames
    print(f"Loading video frames from: {args.input_video}")
    video_frames = load_video_frames(args.input_video, max_frames=args.num_frames)
    print(f"Loaded {len(video_frames)} frames")
    
    # Calculate output dimensions
    original_height = video_frames[0].height
    original_width = video_frames[0].width
    output_height = original_height + args.outpaint_top + args.outpaint_bottom
    output_width = original_width + args.outpaint_left + args.outpaint_right
    
    print(f"Original dimensions: {original_width}x{original_height}")
    print(f"Output dimensions: {output_width}x{output_height}")
    print(f"Outpaint: L:{args.outpaint_left} R:{args.outpaint_right} T:{args.outpaint_top} B:{args.outpaint_bottom}")
    
    # Generate outpainted video
    print("Generating outpainted video...")
    output = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        video_frames=video_frames,
        height=output_height,
        width=output_width,
        num_frames=len(video_frames),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        outpaint_left=args.outpaint_left,
        outpaint_right=args.outpaint_right,
        outpaint_top=args.outpaint_top,
        outpaint_bottom=args.outpaint_bottom,
    )
    
    # Save output video
    print(f"Saving output video to: {args.output_path}")
    export_to_video(output["frames"], args.output_path, fps=16)
    print("Video outpaint completed successfully!")


if __name__ == "__main__":
    main()