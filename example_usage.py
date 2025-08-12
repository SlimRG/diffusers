#!/usr/bin/env python3
"""
Example Usage Script for WAN 2.1 Video Outpainting
Demonstrates various use cases and configurations
"""

import os
import logging
from wan_video_outpainting import WanVideoOutpainter
from comfyui_wan_outpainting import ComfyUIWanOutpainter
from video_utils import VideoProcessor, VideoUploader, calculate_outpaint_dimensions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_cloud_api_basic():
    """
    Basic example using Alibaba Cloud API
    """
    print("\n=== Example 1: Basic Cloud API Usage ===")
    
    # Initialize with API key (from environment or directly)
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("Please set DASHSCOPE_API_KEY environment variable")
        return
    
    outpainter = WanVideoOutpainter(api_key)
    
    # Example video URL (replace with your own)
    video_url = "https://example.com/sample_video.mp4"
    
    try:
        result_path = outpainter.outpaint_video(
            video_url=video_url,
            prompt="A beautiful garden with blooming flowers and trees extending beyond the frame",
            output_path="./outputs/basic_outpainted.mp4",
            top_scale=1.5,
            bottom_scale=1.5,
            left_scale=1.5,
            right_scale=1.5
        )
        print(f"✅ Video outpainted successfully: {result_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_cloud_api_asymmetric():
    """
    Example with asymmetric scaling using Cloud API
    """
    print("\n=== Example 2: Asymmetric Scaling ===")
    
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("Please set DASHSCOPE_API_KEY environment variable")
        return
    
    outpainter = WanVideoOutpainter(api_key)
    
    video_url = "https://example.com/portrait_video.mp4"
    
    try:
        # Different scaling for each direction
        result_path = outpainter.outpaint_video(
            video_url=video_url,
            prompt="An elegant ballroom with ornate architecture and crystal chandeliers",
            output_path="./outputs/asymmetric_outpainted.mp4",
            top_scale=2.0,    # Expand more at the top
            bottom_scale=1.2, # Expand less at the bottom
            left_scale=1.6,   # Moderate expansion on sides
            right_scale=1.6
        )
        print(f"✅ Asymmetric outpainting completed: {result_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_local_comfyui():
    """
    Example using local ComfyUI setup
    """
    print("\n=== Example 3: Local ComfyUI Processing ===")
    
    # Initialize ComfyUI outpainter
    outpainter = ComfyUIWanOutpainter("http://127.0.0.1:8188")
    
    # Local video file
    video_path = "./sample_videos/input.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please place a video file at the specified path")
        return
    
    try:
        result_path = outpainter.outpaint_video(
            video_path=video_path,
            prompt="A magical forest with ethereal lighting and mystical creatures",
            negative_prompt="blurry, low quality, distorted, artifacts",
            top_scale=1.4,
            bottom_scale=1.4,
            left_scale=1.6,
            right_scale=1.6,
            steps=25,
            cfg=7.5,
            guidance_scale=4.8
        )
        print(f"✅ Local outpainting completed: {result_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_video_processing():
    """
    Example of video processing utilities
    """
    print("\n=== Example 4: Video Processing Utilities ===")
    
    video_path = "./sample_videos/test_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    try:
        # Get video information
        info = VideoProcessor.get_video_info(video_path)
        print(f"📹 Video Info:")
        print(f"   Resolution: {info['width']}x{info['height']}")
        print(f"   Duration: {info['duration']:.2f} seconds")
        print(f"   FPS: {info['fps']}")
        print(f"   Frame Count: {info['frame_count']}")
        
        # Validate video constraints
        is_valid, issues = VideoProcessor.check_video_constraints(
            video_path,
            max_duration=300,  # 5 minutes
            max_resolution=(1920, 1080),
            min_resolution=(480, 360),
            max_file_size=100 * 1024 * 1024  # 100MB
        )
        
        print(f"\n✅ Video Validation: {'PASSED' if is_valid else 'FAILED'}")
        if issues:
            for issue in issues:
                print(f"   ⚠️ {issue}")
        
        # Calculate outpaint dimensions
        final_w, final_h = calculate_outpaint_dimensions(
            original_width=info['width'],
            original_height=info['height'],
            top_scale=1.5,
            bottom_scale=1.5,
            left_scale=1.3,
            right_scale=1.3
        )
        print(f"\n📐 After outpainting:")
        print(f"   Original: {info['width']}x{info['height']}")
        print(f"   Final: {final_w}x{final_h}")
        print(f"   Size increase: {(final_w * final_h) / (info['width'] * info['height']):.1f}x")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")

def example_video_upload():
    """
    Example of uploading video for cloud processing
    """
    print("\n=== Example 5: Video Upload for Cloud Processing ===")
    
    local_video = "./sample_videos/small_video.mp4"
    
    if not os.path.exists(local_video):
        print(f"Video file not found: {local_video}")
        return
    
    try:
        # Upload video to temporary hosting
        video_url = VideoUploader.upload_to_temp_host(
            local_video,
            service="file.io",
            max_size_mb=50
        )
        print(f"📤 Video uploaded successfully: {video_url}")
        
        # Now use the uploaded URL with cloud API
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if api_key:
            outpainter = WanVideoOutpainter(api_key)
            result_path = outpainter.outpaint_video(
                video_url=video_url,
                prompt="A stunning sunset landscape with rolling hills",
                output_path="./outputs/uploaded_outpainted.mp4"
            )
            print(f"✅ Outpainting from uploaded video completed: {result_path}")
        else:
            print("Set DASHSCOPE_API_KEY to continue with outpainting")
            
    except Exception as e:
        print(f"❌ Error uploading video: {e}")

def example_batch_processing():
    """
    Example of batch processing multiple videos
    """
    print("\n=== Example 6: Batch Processing ===")
    
    # List of videos to process
    video_configs = [
        {
            "path": "./sample_videos/video1.mp4",
            "prompt": "A serene lake with mountains in the background",
            "scales": {"top": 1.3, "bottom": 1.3, "left": 1.5, "right": 1.5},
            "output": "./outputs/batch_video1_outpainted.mp4"
        },
        {
            "path": "./sample_videos/video2.mp4", 
            "prompt": "An ancient castle with dramatic clouds",
            "scales": {"top": 1.8, "bottom": 1.2, "left": 1.4, "right": 1.4},
            "output": "./outputs/batch_video2_outpainted.mp4"
        }
    ]
    
    # Initialize ComfyUI outpainter
    outpainter = ComfyUIWanOutpainter("http://127.0.0.1:8188")
    
    successful = 0
    failed = 0
    
    for i, config in enumerate(video_configs, 1):
        print(f"\n🎬 Processing video {i}/{len(video_configs)}: {config['path']}")
        
        if not os.path.exists(config['path']):
            print(f"   ❌ Video file not found: {config['path']}")
            failed += 1
            continue
        
        try:
            result_path = outpainter.outpaint_video(
                video_path=config['path'],
                prompt=config['prompt'],
                top_scale=config['scales']['top'],
                bottom_scale=config['scales']['bottom'],
                left_scale=config['scales']['left'],
                right_scale=config['scales']['right'],
                steps=20,
                cfg=7.0
            )
            print(f"   ✅ Completed: {result_path}")
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {successful/(successful+failed)*100:.1f}%")

def example_comparison_video():
    """
    Example of creating comparison videos
    """
    print("\n=== Example 7: Creating Comparison Videos ===")
    
    original_video = "./sample_videos/original.mp4"
    outpainted_video = "./outputs/outpainted.mp4"
    
    if not os.path.exists(original_video):
        print(f"Original video not found: {original_video}")
        return
    
    if not os.path.exists(outpainted_video):
        print(f"Outpainted video not found: {outpainted_video}")
        print("Run outpainting examples first to generate outpainted videos")
        return
    
    try:
        from video_utils import create_comparison_video
        
        # Create side-by-side comparison
        comparison_path = create_comparison_video(
            original_path=original_video,
            outpainted_path=outpainted_video,
            output_path="./outputs/side_by_side_comparison.mp4",
            side_by_side=True
        )
        print(f"✅ Side-by-side comparison created: {comparison_path}")
        
        # Create before/after comparison
        before_after_path = create_comparison_video(
            original_path=original_video,
            outpainted_path=outpainted_video,
            output_path="./outputs/before_after_comparison.mp4",
            side_by_side=False
        )
        print(f"✅ Before/after comparison created: {before_after_path}")
        
    except Exception as e:
        print(f"❌ Error creating comparison: {e}")

def main():
    """
    Main function to run all examples
    """
    print("🎥 WAN 2.1 Video Outpainting - Example Usage")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./sample_videos", exist_ok=True)
    
    # Run examples
    try:
        example_video_processing()
        example_cloud_api_basic()
        example_cloud_api_asymmetric()
        example_local_comfyui()
        example_video_upload()
        example_batch_processing()
        example_comparison_video()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    print("🎬 Examples completed!")
    print("\nNext steps:")
    print("- Place your video files in ./sample_videos/")
    print("- Set DASHSCOPE_API_KEY for cloud API examples")
    print("- Install and configure ComfyUI for local examples")
    print("- Check ./outputs/ for generated videos")

if __name__ == "__main__":
    main()