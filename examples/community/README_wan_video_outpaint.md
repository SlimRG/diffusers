# WAN 2.1 Video Outpaint

This directory contains examples for performing video outpaint using the WAN 2.1 model. Video outpaint extends the boundaries of a video and fills in missing areas with coherent content based on text prompts and the original video content.

## What is Video Outpaint?

Video outpaint is a technique that allows you to:
- **Extend video dimensions**: Increase width, height, or both while maintaining content coherence
- **Fill missing areas**: Generate content for regions that were previously outside the video frame
- **Maintain temporal consistency**: Ensure the generated content flows smoothly across frames
- **Preserve original content**: Keep the existing video content intact while expanding boundaries

## Features

- **Spatial-Temporal Guidance (STG)**: Uses WAN's advanced STG mechanism for better quality
- **Flexible dimensions**: Support for various output resolutions and aspect ratios
- **Multiple padding modes**: Constant, edge, reflect, and symmetric padding options
- **Strength control**: Adjustable outpaint strength for gradual content generation
- **High-quality generation**: Configurable inference steps and guidance scales

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Diffusers library
- Transformers library
- CUDA-compatible GPU (recommended)

## Installation

```bash
# Install required packages
pip install torch diffusers transformers accelerate

# Clone the diffusers repository if you haven't already
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
```

## Quick Start

### Basic Usage

```python
from diffusers import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from wan_video_outpaint import WanVideoOutpaintPipeline

# Load VAE and pipeline
vae = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", 
    subfolder="vae", 
    torch_dtype=torch.float32
)

pipe = WanVideoOutpaintPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", 
    vae=vae, 
    torch_dtype=torch.bfloat16
)

# Configure scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, 
    flow_shift=5.0  # 5.0 for 720P, 3.0 for 480P
)

pipe.to("cuda")

# Generate video with outpaint
output = pipe.outpaint(
    prompt="A beautiful landscape with mountains and flowing rivers",
    height=720,
    width=1280,
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=5.0,
    strength=0.8,
    stg_applied_layers_idx=[8],
    stg_scale=1.0,
)
```

### Command Line Usage

```bash
# Basic text-to-video outpaint
python wan_video_outpaint.py \
    --prompt "A majestic mountain landscape with snow-capped peaks" \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --output_path "mountain_landscape.mp4"

# High-quality outpaint with custom settings
python wan_video_outpaint.py \
    --prompt "A futuristic cityscape with neon lights and flying cars" \
    --height 720 \
    --width 1920 \
    --num_frames 81 \
    --num_inference_steps 50 \
    --guidance_scale 7.0 \
    --strength 0.9 \
    --stg_applied_layers_idx 3 8 16 \
    --output_path "cyberpunk_city.mp4"
```

## Examples

### 1. Basic Text-to-Video Outpaint

```python
# Simple landscape generation with outpaint
output = pipe.outpaint(
    prompt="A serene forest scene with tall trees and dappled sunlight",
    height=720,
    width=1280,
    num_frames=81,
    guidance_scale=5.0,
    strength=0.8,
)
```

### 2. High-Quality Outpaint

```python
# High-quality generation with more inference steps
output = pipe.outpaint(
    prompt="A cinematic ocean scene with crashing waves and dramatic clouds",
    height=720,
    width=1280,
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=7.0,
    strength=0.9,
    stg_applied_layers_idx=[3, 8, 16],
)
```

### 3. Different Aspect Ratios

```python
# Widescreen 16:9 aspect ratio
output = pipe.outpaint(
    prompt="A futuristic cityscape with towering skyscrapers",
    height=720,
    width=1920,  # 16:9 aspect ratio
    num_frames=81,
    guidance_scale=6.0,
    strength=0.85,
)
```

## Parameters

### Core Parameters

- **`prompt`**: Text description of desired video content
- **`height`**: Target height (must be divisible by 8)
- **`width`**: Target width (must be divisible by 8)
- **`num_frames`**: Number of frames (must be divisible by 4)

### Quality Parameters

- **`num_inference_steps`**: Number of denoising steps (20-100, higher = better quality)
- **`guidance_scale`**: Classifier-free guidance strength (5.0-8.0 recommended)
- **`strength`**: Outpaint effect strength (0.7-0.9 recommended)

### STG Parameters

- **`stg_applied_layers_idx`**: Layer indices for spatial-temporal guidance
  - For 14B model: [0-39]
  - For 1.3B model: [0-29]
  - Common values: [8], [3, 8, 16], [8, 16, 24]
- **`stg_scale`**: STG influence strength (0.0 for CFG, 1.0 for STG)

### Advanced Parameters

- **`padding_mode`**: Padding method for extending dimensions
  - `"constant"`: Fill with specified value
  - `"edge"`: Extend edge pixels
  - `"reflect"`: Mirror edge pixels
  - `"symmetric"`: Symmetric mirroring
- **`fill_value`**: Value for constant padding mode

## Model Variants

### WAN 2.1 Models

1. **Wan-AI/Wan2.1-T2V-14B-Diffusers**
   - Higher quality, more detailed output
   - Requires more VRAM (16GB+ recommended)
   - Better for professional applications

2. **Wan-AI/Wan2.1-T2V-1.3B-Diffusers**
   - Faster generation, lower VRAM usage
   - Good for prototyping and testing
   - Suitable for 8GB+ VRAM

### Scheduler Configuration

```python
# For 720P and above
flow_shift = 5.0

# For 480P and below
flow_shift = 3.0

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, 
    flow_shift=flow_shift
)
```

## Best Practices

### Prompt Engineering

- **Be specific**: Include details about lighting, style, and atmosphere
- **Use quality terms**: "high quality", "cinematic", "8K", "professional"
- **Avoid negatives**: Focus on what you want rather than what you don't want
- **Include context**: Describe the scene, mood, and visual elements

### Parameter Tuning

- **Start with defaults**: Use recommended values and adjust gradually
- **Balance quality vs speed**: More inference steps = better quality but slower
- **Experiment with STG**: Try different layer combinations for varied results
- **Adjust guidance**: Higher values = stronger prompt adherence

### Memory Management

- **Monitor VRAM**: Use `nvidia-smi` to track GPU memory usage
- **Batch size**: Reduce batch size if you encounter memory issues
- **Model offloading**: Use `pipe.enable_model_cpu_offload()` for large models
- **Gradient checkpointing**: Enable for memory efficiency during training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or number of frames
   - Use smaller model (1.3B instead of 14B)
   - Enable model offloading

2. **Poor Quality Output**
   - Increase `num_inference_steps`
   - Adjust `guidance_scale`
   - Improve prompt quality
   - Try different STG layer combinations

3. **Inconsistent Results**
   - Set random seed for reproducibility
   - Use consistent parameter values
   - Check input validation

### Performance Tips

- **Use appropriate dtype**: `bfloat16` for GPU, `float32` for CPU
- **Enable optimizations**: Use `pipe.enable_xformers_memory_efficient_attention()`
- **Batch processing**: Process multiple videos in sequence
- **Model caching**: Keep models in memory for repeated use

## Examples Directory

- **`wan_video_outpaint.py`**: Full-featured pipeline with command-line interface
- **`wan_video_outpaint_simple.py`**: Simple examples for learning and testing
- **`README_wan_video_outpaint.md`**: This documentation file

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{wan2024video,
  title={WAN 2.1: Advanced Video Generation with Spatial-Temporal Guidance},
  author={Wan Team},
  year={2024},
  url={https://github.com/Wan-AI/Wan2.1}
}
```

## License

This code is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the video outpaint functionality.

## Support

For questions and support:
- Check the [Diffusers documentation](https://huggingface.co/docs/diffusers)
- Visit the [WAN model page](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)
- Open an issue in the diffusers repository