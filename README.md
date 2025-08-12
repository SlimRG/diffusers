# WAN 2.1 Video Outpainting Toolkit

A comprehensive Python toolkit for video outpainting using the WAN 2.1 (Wide Area Network 2.1) VACE model. This toolkit provides both cloud-based (Alibaba Cloud API) and local (ComfyUI) implementations for extending video boundaries with AI-generated content.

## Features

- 🎥 **Video Outpainting**: Extend video boundaries in all directions
- ☁️ **Cloud & Local**: Support for both Alibaba Cloud API and local ComfyUI
- 🛠️ **Video Processing**: Comprehensive video validation and processing utilities
- 📊 **Progress Tracking**: Real-time progress monitoring and logging
- 🔧 **Flexible Configuration**: Customizable scaling factors and generation parameters
- 📈 **Batch Processing**: Support for processing multiple videos

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for video processing utilities)
- For local processing: ComfyUI with WAN 2.1 models

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Additional Setup

#### For Alibaba Cloud API:
1. Sign up for Alibaba Cloud DashScope
2. Obtain your API key
3. Set environment variable: `export DASHSCOPE_API_KEY=your_api_key`

#### For Local ComfyUI:
1. Install ComfyUI
2. Download required models:
   - `wan2.1_t2v_14B_fp8_e4m3fn.safetensors`
   - `Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors`
   - `Wan2_1_VAE_bf16.safetensors`
   - `umt5-xxl-enc-bf16.safetensors`

## Quick Start

### Using Alibaba Cloud API

```python
from wan_video_outpainting import WanVideoOutpainter

# Initialize with API key
outpainter = WanVideoOutpainter("your_api_key")

# Outpaint a video
result_path = outpainter.outpaint_video(
    video_url="https://example.com/your_video.mp4",
    prompt="A beautiful garden with blooming flowers extending beyond the frame",
    output_path="./outpainted_video.mp4",
    top_scale=1.5,
    bottom_scale=1.5,
    left_scale=1.5,
    right_scale=1.5
)

print(f"Outpainted video saved to: {result_path}")
```

### Command Line Usage - Cloud API

```bash
python wan_video_outpainting.py \
    --video-url "https://example.com/video.mp4" \
    --prompt "A serene landscape with mountains in the background" \
    --output "./result.mp4" \
    --top-scale 1.3 \
    --bottom-scale 1.3 \
    --left-scale 1.5 \
    --right-scale 1.5
```

### Using Local ComfyUI

```python
from comfyui_wan_outpainting import ComfyUIWanOutpainter

# Initialize ComfyUI outpainter
outpainter = ComfyUIWanOutpainter("http://127.0.0.1:8188")

# Outpaint a local video
result_path = outpainter.outpaint_video(
    video_path="./input_video.mp4",
    prompt="An elegant ballroom with crystal chandeliers",
    negative_prompt="blurry, low quality, distorted",
    top_scale=1.4,
    bottom_scale=1.4,
    left_scale=1.6,
    right_scale=1.6,
    steps=25,
    cfg=7.5
)
```

### Command Line Usage - ComfyUI

```bash
python comfyui_wan_outpainting.py \
    --video-path "./input.mp4" \
    --prompt "A magical forest with ethereal lighting" \
    --negative-prompt "dark, gloomy, low quality" \
    --top-scale 1.5 \
    --bottom-scale 1.2 \
    --left-scale 1.8 \
    --right-scale 1.8 \
    --steps 30 \
    --cfg 8.0
```

## Advanced Usage

### Video Processing Utilities

```python
from video_utils import VideoProcessor, calculate_outpaint_dimensions

# Get video information
info = VideoProcessor.get_video_info("video.mp4")
print(f"Resolution: {info['width']}x{info['height']}")
print(f"Duration: {info['duration']:.2f} seconds")
print(f"FPS: {info['fps']}")

# Validate video constraints
is_valid, issues = VideoProcessor.check_video_constraints(
    "video.mp4",
    max_duration=300,  # 5 minutes
    max_resolution=(1920, 1080),
    min_resolution=(480, 360)
)

if not is_valid:
    print("Video validation issues:", issues)

# Calculate final dimensions after outpainting
final_w, final_h = calculate_outpaint_dimensions(
    original_width=1920,
    original_height=1080,
    top_scale=1.5,
    bottom_scale=1.5,
    left_scale=1.3,
    right_scale=1.3
)
print(f"Final resolution will be: {final_w}x{final_h}")
```

### Video Upload for Cloud Processing

```python
from video_utils import VideoUploader

# Upload video to temporary hosting (for cloud API)
video_url = VideoUploader.upload_to_temp_host(
    "local_video.mp4",
    service="file.io",
    max_size_mb=100
)
print(f"Video uploaded: {video_url}")
```

### Create Comparison Videos

```python
from video_utils import create_comparison_video

# Create side-by-side comparison
comparison_path = create_comparison_video(
    original_path="original.mp4",
    outpainted_path="outpainted.mp4",
    output_path="comparison.mp4",
    side_by_side=True
)
```

## Configuration Parameters

### Scaling Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `top_scale` | Top expansion factor | 1.5 | 1.0-3.0 |
| `bottom_scale` | Bottom expansion factor | 1.5 | 1.0-3.0 |
| `left_scale` | Left expansion factor | 1.5 | 1.0-3.0 |
| `right_scale` | Right expansion factor | 1.5 | 1.0-3.0 |

### Generation Parameters (ComfyUI)

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `steps` | Denoising steps | 20 | 10-50 |
| `cfg` | CFG scale | 7.0 | 1.0-20.0 |
| `guidance_scale` | Guidance scale | 4.5 | 1.0-10.0 |
| `scheduler` | Scheduler type | "ddim" | ddim, euler, etc. |

## Examples

### Basic Outpainting

```bash
# Simple 1.5x expansion in all directions
python wan_video_outpainting.py \
    --video-url "https://example.com/dance.mp4" \
    --prompt "A grand theater stage with audience in the background" \
    --output "./dance_outpainted.mp4"
```

### Asymmetric Expansion

```bash
# Different scaling for each direction
python wan_video_outpainting.py \
    --video-url "https://example.com/portrait.mp4" \
    --prompt "A beautiful garden with flowers and trees" \
    --output "./portrait_expanded.mp4" \
    --top-scale 2.0 \
    --bottom-scale 1.2 \
    --left-scale 1.8 \
    --right-scale 1.8
```

### High Quality Processing (ComfyUI)

```bash
# High-quality local processing
python comfyui_wan_outpainting.py \
    --video-path "./input.mp4" \
    --prompt "An opulent palace ballroom with golden decorations" \
    --negative-prompt "blurry, low quality, artifacts, distorted" \
    --steps 40 \
    --cfg 8.5 \
    --guidance-scale 5.0 \
    --top-scale 1.6 \
    --bottom-scale 1.4 \
    --left-scale 1.7 \
    --right-scale 1.7
```

## Model Requirements

### For ComfyUI Setup

1. **WAN 2.1 Models** (place in `models/diffusion_models/`):
   - `wan2.1_t2v_14B_fp8_e4m3fn.safetensors` (~28GB)

2. **VACE Module** (place in `models/diffusion_models/`):
   - `Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors` (~28GB)

3. **VAE** (place in `models/vae/`):
   - `Wan2_1_VAE_bf16.safetensors` (~335MB)

4. **Text Encoder** (place in `models/text_encoders/`):
   - `umt5-xxl-enc-bf16.safetensors` (~21GB)

### Hardware Requirements

- **Minimum**: 24GB VRAM (RTX 4090, A6000)
- **Recommended**: 40GB+ VRAM (A100, H100)
- **CPU**: 16+ cores recommended
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ free space for models

## Troubleshooting

### Common Issues

1. **Out of Memory (VRAM)**
   - Reduce batch size
   - Use fp8 quantized models
   - Consider model offloading

2. **FFmpeg Not Found**
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/
   ```

3. **API Rate Limits**
   - Implement retry logic
   - Use exponential backoff
   - Consider upgrading API plan

4. **ComfyUI Connection Issues**
   - Verify ComfyUI is running on correct port
   - Check firewall settings
   - Ensure all models are properly loaded

### Performance Optimization

1. **For Cloud API**:
   - Upload videos to CDN for faster access
   - Use video compression before processing
   - Batch multiple requests

2. **For ComfyUI**:
   - Use FP8 quantized models
   - Enable model offloading
   - Optimize ComfyUI settings

## API Reference

### WanVideoOutpainter Class

```python
class WanVideoOutpainter:
    def __init__(self, api_key: str, base_url: str = "https://dashscope-intl.aliyuncs.com")
    
    def outpaint_video(
        self, 
        video_url: str, 
        prompt: str, 
        output_path: str,
        top_scale: float = 1.5,
        bottom_scale: float = 1.5,
        left_scale: float = 1.5,
        right_scale: float = 1.5,
        max_wait_time: int = 1800
    ) -> str
```

### ComfyUIWanOutpainter Class

```python
class ComfyUIWanOutpainter:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188")
    
    def outpaint_video(
        self,
        video_path: str,
        prompt: str,
        negative_prompt: str = "",
        top_scale: float = 1.5,
        bottom_scale: float = 1.5,
        left_scale: float = 1.5,
        right_scale: float = 1.5,
        steps: int = 20,
        cfg: float = 7.0,
        guidance_scale: float = 4.5,
        max_wait_time: int = 1800
    ) -> str
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WAN Video Team for the WAN 2.1 model
- Alibaba Cloud for the DashScope API
- ComfyUI community for the interface and workflows
- FFmpeg team for video processing capabilities

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review ComfyUI documentation for local setup
- Consult Alibaba Cloud docs for API issues
