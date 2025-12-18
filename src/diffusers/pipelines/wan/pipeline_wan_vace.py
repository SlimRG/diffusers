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

import html
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .SlimLanPaint import SlimLanPaint
else:
    try:
        from .SlimLanPaint import SlimLanPaint
    except Exception:
        SlimLanPaint = None  # type: ignore
import inspect
import math
import re

import ftfy
import numpy as np
import PIL.Image
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from ...image_processor import VaeImageProcessor
from ...loaders import WanLoraLoaderMixin
from ...models import AutoencoderKLWan, WanVACETransformer3DModel
from ...models.attention_processor import Attention, AttentionProcessor
from ...models.embeddings import get_timestep_embedding
from ...models.modeling_utils import ModelMixin
from ...models.unets.unet_3d_blocks import UNetMidBlock3DCrossAttn
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    XLA_AVAILABLE,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from .pipeline_output import WanPipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline

        >>> # helper functions
        >>> import PIL.Image

        >>> def prepare_video_and_mask(first_img: PIL.Image.Image, last_img: PIL.Image.Image, height: int, width: int, num_frames: int):
        ...     first_img = first_img.resize((width, height))
        ...     last_img = last_img.resize((width, height))
        ...     frames = []
        ...     frames.append(first_img)
        ...     # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
        ...     # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
        ...     # match the original code.
        ...     frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
        ...     frames.append(last_img)
        ...     mask_black = PIL.Image.new("L", (width, height), 0)
        ...     mask_white = PIL.Image.new("L", (width, height), 255)
        ...     mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
        ...     return frames, mask

        >>> # Available checkpoints: Wan-AI/Wan2.1-VACE-1.3B-diffusers, Wan-AI/Wan2.1-VACE-14B-diffusers
        >>> model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        >>> pipe.to("cuda")

        >>> prompt = "CG animation style, a small blue bird take off from the beach at sunrise and soar into the vastness of the sky from a close-up, low-angle perspective."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works"
        >>> height, width = 480, 832
        >>> num_frames = 81

        >>> # prepare input video and mask (first and last frame are visible)
        >>> first_image = PIL.Image.open("frame_0001.png")
        >>> last_image = PIL.Image.open("frame_0081.png")
        >>> video, mask = prepare_video_and_mask(first_image, last_image, height, width, num_frames)

        >>> video = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     video=video,
        ...     mask=mask,
        ...     height=height,
        ...     width=width,
        ...     num_frames=num_frames,
        ...     guidance_scale=6.0,
        ...     num_inference_steps=50,
        ... ).frames[0]

        >>> export_to_video(video, "generated.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class WanVACEPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using WanVACE.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        tokenizer ([`AutoTokenizer`]):
            Tokenizer for text encoder.
        text_encoder ([`UMT5EncoderModel`]):
            Text-encoder.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        transformer ([`WanVACETransformer3DModel`]):
            The primary transformer model.
        transformer_2 ([`WanVACETransformer3DModel`], *optional*):
            A secondary transformer model used in Wan2.2 low-noise stage.
        boundary_ratio (`float`, *optional*):
            Ratio to split timesteps between high-noise (transformer) and low-noise (transformer_2) stages.
    """

    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    _optional_components = ["transformer_2"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: WanVACETransformer3DModel = None,
        transformer_2: WanVACETransformer3DModel = None,
        boundary_ratio: Optional[float] = None,
        lanpaint: Optional["SlimLanPaint"] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio)

        # Optional: tiled denoising helper (off by default)
        self.lanpaint = lanpaint

        self.vae_scale_factor_spatial = 8
        self.vae_scale_factor_temporal = 4
        self.video_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @property
    def transformer_dtype(self):
        return self.transformer.dtype

    @property
    def transformer_2_dtype(self):
        if self.transformer_2 is None:
            return None
        return self.transformer_2.dtype

    @property
    def _execution_device(self):
        if self.device.type != "meta":
            return self.device
        for name, module in self.components.items():
            if isinstance(module, ModelMixin) and module.device.type != "meta":
                return module.device

        # TODO: raise an error if execution device not found
        return self.device

    def _get_add_time_ids(self, fps, motion_bucket_id, noise_aug_strength, dtype):
        add_time_ids = torch.tensor([fps, motion_bucket_id, noise_aug_strength], dtype=dtype)
        add_time_ids = add_time_ids.flatten()
        return add_time_ids

    def encode_prompt(
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
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length}: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device), return_dict=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def prepare_video_and_mask(
        self,
        video: Union[List[PIL.Image.Image], List[np.ndarray], torch.Tensor],
        mask: Optional[Union[List[PIL.Image.Image], List[np.ndarray], torch.Tensor]] = None,
        reference_images: Optional[Union[List[PIL.Image.Image], List[np.ndarray], torch.Tensor]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.transformer.dtype

        if not isinstance(video, torch.Tensor):
            video = self.video_processor.preprocess(video, height=height, width=width).to(device=device, dtype=dtype)
            video = video.unsqueeze(0)

        if mask is None:
            mask = torch.ones_like(video[:, :1, :])
        elif not isinstance(mask, torch.Tensor):
            if isinstance(mask[0], PIL.Image.Image):
                mask = [x.convert("L") for x in mask]
            mask = self.video_processor.preprocess(mask, height=height, width=width).to(device=device, dtype=dtype)
            mask = mask.unsqueeze(0)

        if reference_images is not None and not isinstance(reference_images, torch.Tensor):
            if isinstance(reference_images[0], PIL.Image.Image):
                reference_images = [x.convert("RGB") for x in reference_images]
            reference_images = self.video_processor.preprocess(reference_images, height=height, width=width).to(
                device=device, dtype=dtype
            )

            if reference_images.ndim == 4:
                reference_images = reference_images.unsqueeze(0)
            reference_images_preprocessed = []
            image_size = reference_images.shape[-2:]
            for reference_image in reference_images:
                preprocessed_images = []
                for image in reference_image:
                    img_height, img_width = image.shape[-2:]
                    scale = min(image_size[0] / img_height, image_size[1] / img_width)
                    new_height, new_width = int(img_height * scale), int(img_width * scale)
                    resized_image = torch.nn.functional.interpolate(
                        image, size=(new_height, new_width), mode="bilinear", align_corners=False
                    ).squeeze(0)  # [C, H, W]
                    top = (image_size[0] - new_height) // 2
                    left = (image_size[1] - new_width) // 2
                    canvas = torch.ones(3, *image_size, device=device, dtype=dtype)
                    canvas[:, top : top + new_height, left : left + new_width] = resized_image
                    preprocessed_images.append(canvas)
                reference_images_preprocessed.append(preprocessed_images)

            reference_images_preprocessed = reference_images_preprocessed
        else:
            reference_images_preprocessed = None

        return video, mask, reference_images_preprocessed

    def prepare_video_latents(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        reference_images: Optional[List[List[torch.Tensor]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device

        if isinstance(generator, list):
            # TODO: support this
            raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")

        if reference_images is None:
            # For each batch of video, we set no reference image (as one or more can be passed by user)
            reference_images = [[None] for _ in range(video.shape[0])]
        else:
            if video.shape[0] != len(reference_images):
                raise ValueError(
                    f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
                )

        if video.shape[0] != 1:
            # TODO: support this
            raise ValueError(
                "Generating with more than one video is not yet supported. This may be supported in the future."
            )

        vae_dtype = self.vae.dtype

        video = video.to(dtype=vae_dtype)
        video = video.permute(0, 2, 1, 3, 4)
        mask = mask.to(dtype=vae_dtype)

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=vae_dtype).view(1, 4, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=vae_dtype).view(1, 4, 1, 1, 1)
        latents = retrieve_latents(self.vae.encode(video), generator, sample_mode="argmax")
        latents = ((latents.float() - latents_mean) * latents_std).to(vae_dtype)
        mask = self.prepare_masks(mask, num_frames=latents.shape[2]).to(device=device, dtype=vae_dtype)
        masked_latents = latents * (1 - mask)
        masked_latents = torch.cat([masked_latents, mask], dim=1).to(dtype=vae_dtype)
        latents = torch.cat([latents, torch.zeros_like(latents)], dim=1).to(dtype=vae_dtype)

        latent_list = []
        for latent, reference_images_batch in zip(latents, reference_images):
            for reference_image in reference_images_batch:
                assert reference_image.ndim == 3
                reference_image = reference_image.to(dtype=vae_dtype)
                reference_image = reference_image[None, :, None, :, :]  # [1, C, 1, H, W]
                reference_latent = retrieve_latents(self.vae.encode(reference_image), generator, sample_mode="argmax")
                reference_latent = ((reference_latent.float() - latents_mean) * latents_std).to(vae_dtype)
                reference_latent = reference_latent.squeeze(0)  # [C, 1, H, W]
                reference_latent = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=0)
                latent = torch.cat([reference_latent.squeeze(0), latent], dim=1)
            latent_list.append(latent)
        return torch.stack(latent_list)

    def prepare_masks(self, mask: torch.Tensor, num_frames: int):
        mask_list = []
        for mask_ in mask:
            if num_frames == mask_.shape[1]:
                mask_list.append(mask_)
                continue
            old_num_frames = mask_.shape[1]
            # We want to interpolate the mask in time to match the number of latent frames
            new_num_frames = num_frames
            assert old_num_frames > new_num_frames, "We only expect to downsample the mask in time."
            # Downsample the mask in time by taking a max over blocks
            # This mirrors the original Wan implementation which uses numpy reshape and max.
            # Reshape: [8x8, old_num_frames, new_height, new_width]
            new_height = mask_.shape[-2] // self.vae_scale_factor_spatial
            new_width = mask_.shape[-1] // self.vae_scale_factor_spatial
            mask_ = mask_.view(num_frames, new_height, self.vae_scale_factor_spatial, new_width, self.vae_scale_factor_spatial)
            mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [8x8, num_frames, new_height, new_width]
            mask_ = torch.nn.functional.interpolate(mask_.unsqueeze(0), size=(new_num_frames, new_height, new_width), mode="nearest-exact").squeeze(0)
            num_ref_images = len(reference_images_batch)
            if num_ref_images > 0:
                mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
                mask_ = torch.cat([mask_padding, mask_], dim=1)
            mask_list.append(mask_)
        return torch.stack(mask_list)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (batch_size, num_channels_latents, num_frames, height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        video: Union[List[PIL.Image.Image], List[np.ndarray], torch.Tensor] = None,
        mask: Optional[Union[List[PIL.Image.Image], List[np.ndarray], torch.Tensor]] = None,
        reference_images: Optional[Union[List[PIL.Image.Image], List[np.ndarray], torch.Tensor]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        guidance_scale_2: float = 6.0,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        # 0. Default height and width to transformer config
        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial

        # 1. Check inputs
        self.check_inputs(
            prompt,
            negative_prompt,
            video,
            mask,
            reference_images,
            height,
            width,
            num_frames,
            callback_on_step_end_tensor_inputs,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        if prompt_embeds is None:
            prompt_embeds = self.encode_prompt(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=self.transformer.config.max_sequence_length,
                device=device,
                dtype=self.text_encoder.dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt_embeds = self.encode_prompt(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=self.transformer.config.max_sequence_length,
                device=device,
                dtype=self.text_encoder.dtype,
            )

        # 4. Prepare video, mask and reference images
        video, mask, reference_images_preprocessed = self.prepare_video_and_mask(
            video=video,
            mask=mask,
            reference_images=reference_images,
            height=height,
            width=width,
            num_frames=num_frames,
            device=device,
            dtype=self.transformer.dtype,
        )

        # 5. Prepare latent variables
        conditioning_latents = self.prepare_video_latents(
            video=video,
            mask=mask,
            reference_images=reference_images_preprocessed,
            generator=generator,
            device=device,
        )

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames + (0 if reference_images_preprocessed is None else len(reference_images_preprocessed[0]) * self.vae_scale_factor_temporal),
            torch.float32,
            device,
            generator,
            latents,
        )

        if conditioning_latents.shape[2] != latents.shape[2]:
            logger.warning(
                "The number of frames in the conditioning latent does not match the number of frames to be generated. Generation quality may be affected."
            )

        transformer_dtype = self.transformer.dtype

        # 5.1 Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    if self.lanpaint is None:
                        noise_pred = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            control_hidden_states=conditioning_latents,
                            control_hidden_states_scale=conditioning_scale,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    else:
                        noise_pred = self.lanpaint.predict_noise(
                            model=current_model,
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            control_hidden_states=conditioning_latents,
                            control_hidden_states_scale=conditioning_scale,
                            attention_kwargs=attention_kwargs,
                        )

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        if self.lanpaint is None:
                            noise_uncond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                control_hidden_states=conditioning_latents,
                                control_hidden_states_scale=conditioning_scale,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        else:
                            noise_uncond = self.lanpaint.predict_noise(
                                model=current_model,
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                control_hidden_states=conditioning_latents,
                                control_hidden_states_scale=conditioning_scale,
                                attention_kwargs=attention_kwargs,
                            )
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            video = latents
        elif output_type == "pt":
            latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, 4, 1, 1, 1)
            latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, 4, 1, 1, 1)
            latents = latents[:, :4, :, :, :]
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, 4, 1, 1, 1)
            latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, 4, 1, 1, 1)
            latents = latents[:, :4, :, :, :]
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
