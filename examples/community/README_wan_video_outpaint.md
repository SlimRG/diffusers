# WAN 2.1 Video Outpaint Examples

This directory contains examples for performing video outpaint using the WAN 2.1 model. Video outpaint extends the boundaries of a video and fills in missing content using AI generation while maintaining temporal consistency across frames.

## What is Video Outpaint?

Video outpaint is a technique that allows you to:
- **Extend video dimensions**: Add pixels to the left, right, top, or bottom of a video
- **Fill missing content**: Use AI to generate realistic content for the extended regions
- **Maintain consistency**: Ensure temporal and spatial coherence across all frames
- **Preserve original content**: Keep the original video content intact while adding new areas

## Available Examples

### 1. `wan_video_outpaint.py` - Full-Featured Pipeline
A complete pipeline implementation with all features:
- Command-line interface
- Support for all outpaint directions
- Configurable parameters
- Video file input/output
- Advanced masking and blending

### 2. `wan_video_outpaint_simple.py` - Simple Demo
A simplified example for learning and testing:
- Self-contained demonstration
- Sample video generation
- Basic outpaint functionality
- Easy to understand and modify

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 16GB VRAM for the 14B model, 8GB for the 1.3B model

### Install Dependencies
```bash
pip install -r requirements_wan_outpaint.txt
```

### Alternative Manual Installation
```bash
pip install torch torchvision diffusers transformers accelerate
pip install Pillow opencv-python numpy
pip install xformers  # Optional: for better performance
```

## Quick Start

### Simple Example
```bash
python wan_video_outpaint_simple.py
```

This will:
1. Load the WAN 2.1 model
2. Create sample video frames
3. Perform outpaint (right: 256px, bottom: 128px)
4. Save the result as `wan_outpaint_output.mp4`

### Full Pipeline Example
```bash
python wan_video_outpaint.py \
    --input_video input.mp4 \
    --output_path output.mp4 \
    --outpaint_right 256 \
    --outpaint_bottom 128 \
    --prompt "A beautiful landscape with mountains and trees" \
    --num_inference_steps 50
```

## Usage Examples

### Extend Video to the Right
```bash
python wan_video_outpaint.py \
    --input_video video.mp4 \
    --output_path extended_right.mp4 \
    --outpaint_right 512 \
    --prompt "Extending the scene to the right with more landscape"
```

### Extend Video to the Bottom
```bash
python wan_video_outpaint.py \
    --input_video video.mp4 \
    --output_path extended_bottom.mp4 \
    --outpaint_bottom 256 \
    --prompt "Adding more ground and foreground elements below"
```

### Extend in Multiple Directions
```bash
python wan_video_outpaint.py \
    --input_video video.mp4 \
    --output_path extended_all.mp4 \
    --outpaint_left 128 \
    --outpaint_right 256 \
    --outpaint_top 64 \
    --outpaint_bottom 128 \
    --prompt "Expanding the scene in all directions with natural elements"
```

## Parameters

### Required Parameters
- `--input_video`: Path to input video file
- `--output_path`: Path for output video

### Outpaint Parameters
- `--outpaint_left`: Pixels to add on the left
- `--outpaint_right`: Pixels to add on the right
- `--outpaint_top`: Pixels to add on the top
- `--outpaint_bottom`: Pixels to add on the bottom

### Generation Parameters
- `--prompt`: Text description of what to generate
- `--negative_prompt`: What to avoid generating
- `--num_frames`: Number of frames to process
- `--num_inference_steps`: Denoising steps (higher = better quality, slower)
- `--guidance_scale`: How closely to follow the prompt (higher = more adherence)

### Model Parameters
- `--model_id`: WAN model to use
- `--device`: Device to use (cuda/cpu)
- `--dtype`: Data type (bfloat16/float32)

## Model Options

### Available Models
1. **Wan2.1-T2V-14B-Diffusers** (Recommended)
   - Highest quality
   - Requires 16GB+ VRAM
   - Best for production use

2. **Wan2.1-T2V-1.3B-Diffusers**
   - Faster generation
   - Requires 8GB+ VRAM
   - Good for testing and development

### Model Selection
```bash
# For high quality (if you have enough VRAM)
--model_id "Wan-AI/Wan2.1-T2V-14B-Diffusers"

# For faster generation
--model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
```

## Tips for Best Results

### Prompt Engineering
- **Be specific**: "A mountain landscape with pine trees and a flowing river" vs "nature"
- **Include context**: Mention what should extend from the original scene
- **Use descriptive language**: "Rolling hills", "dense forest", "clear blue sky"

### Negative Prompts
Common negative prompts to avoid issues:
```
blurry, low quality, distorted, artifacts, text, watermark, 
duplicate, extra limbs, poorly drawn hands, poorly drawn face, 
mutation, deformed, ugly, blurry, bad anatomy, bad proportions
```

### Parameter Tuning
- **Guidance Scale**: 7.5-15.0 (higher = more prompt adherence)
- **Inference Steps**: 30-100 (higher = better quality, slower)
- **Frame Count**: Match your input video or use fewer for faster generation

### Outpaint Sizes
- **Small extensions** (64-128px): Good for subtle scene expansion
- **Medium extensions** (256-512px): Good for significant scene changes
- **Large extensions** (512px+): May require more steps and careful prompting

## Troubleshooting

### Common Issues

#### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Use the 1.3B model instead of 14B
- Reduce batch size or frame count
- Use CPU if GPU memory is insufficient
- Close other applications using GPU

#### Poor Quality Results
**Solutions:**
- Increase `num_inference_steps` (50-100)
- Improve your prompt description
- Use higher `guidance_scale` (10-15)
- Check that your negative prompt isn't too restrictive

#### Inconsistent Results
**Solutions:**
- Ensure your prompt describes the entire extended scene
- Use consistent terminology throughout the prompt
- Consider using the same seed for reproducible results

### Performance Optimization
- **Use xformers**: Install for better attention performance
- **Batch processing**: Process multiple videos in sequence
- **Model offloading**: Automatically handled by the pipeline
- **Mixed precision**: Use bfloat16 on supported GPUs

## Advanced Usage

### Custom Video Loading
Modify the `load_video_frames` function to support different video formats:
```python
def load_video_frames(video_path, max_frames=None, target_fps=None):
    # Your custom video loading logic here
    pass
```

### Custom Masking
Implement custom masking strategies for more control:
```python
def create_custom_mask(width, height, custom_regions):
    # Your custom masking logic here
    pass
```

### Batch Processing
Process multiple videos with different parameters:
```python
videos = [
    {"input": "video1.mp4", "output": "out1.mp4", "right": 256},
    {"input": "video2.mp4", "output": "out2.mp4", "bottom": 128},
]

for video_config in videos:
    # Process each video
    pass
```

## Examples Gallery

### Before and After
- **Original**: 512x512 video
- **After Right Outpaint**: 768x512 video (added 256px right)
- **After Bottom Outpaint**: 512x640 video (added 128px bottom)
- **After Multi-directional**: 640x640 video (added 128px all around)

### Use Cases
1. **Landscape Videos**: Extend scenic views
2. **Product Videos**: Add context around products
3. **Animation**: Expand animated scenes
4. **Documentary**: Add environmental context
5. **Creative Projects**: Expand artistic videos

## Contributing

Feel free to contribute improvements:
- Bug reports and fixes
- Performance optimizations
- New features and examples
- Documentation improvements

## License

This code follows the same license as the Diffusers library (Apache 2.0).

## Acknowledgments

- **WAN Team**: For the excellent WAN 2.1 model
- **Hugging Face**: For the Diffusers library
- **Community**: For feedback and contributions

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include your system specs and error messages

---

**Happy Video Outpainting! 🎬✨**