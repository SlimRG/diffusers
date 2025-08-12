#!/usr/bin/env python3
"""
Video Utilities for WAN 2.1 Video Outpainting
Utility functions for video processing, validation, and conversion
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import subprocess
import logging
from pathlib import Path
import tempfile
import json

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Utility class for video processing operations
    """
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, any]:
        """
        Get comprehensive video information using OpenCV and FFprobe
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Dict: Video information including duration, fps, resolution, etc.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        info = {}
        
        try:
            # OpenCV info
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info['fps'] = cap.get(cv2.CAP_PROP_FPS)
            info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
            
            cap.release()
            
            # FFprobe info for additional details
            try:
                ffprobe_cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', 
                    '-show_streams', video_path
                ]
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    ffprobe_info = json.loads(result.stdout)
                    
                    # Find video stream
                    video_stream = None
                    for stream in ffprobe_info.get('streams', []):
                        if stream.get('codec_type') == 'video':
                            video_stream = stream
                            break
                    
                    if video_stream:
                        info['codec'] = video_stream.get('codec_name', 'unknown')
                        info['bitrate'] = int(video_stream.get('bit_rate', 0))
                        info['pixel_format'] = video_stream.get('pix_fmt', 'unknown')
                        
                        # More accurate duration from format
                        format_info = ffprobe_info.get('format', {})
                        if 'duration' in format_info:
                            info['duration'] = float(format_info['duration'])
                            
            except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
                logger.warning("FFprobe not available or failed, using OpenCV data only")
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise
        
        return info
    
    @staticmethod
    def validate_video_format(video_path: str, supported_formats: List[str] = None) -> bool:
        """
        Validate if video format is supported
        
        Args:
            video_path (str): Path to the video file
            supported_formats (List[str]): List of supported formats (extensions)
            
        Returns:
            bool: True if format is supported
        """
        if supported_formats is None:
            supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        
        file_ext = Path(video_path).suffix.lower()
        return file_ext in supported_formats
    
    @staticmethod
    def check_video_constraints(
        video_path: str,
        max_duration: Optional[float] = None,
        max_resolution: Optional[Tuple[int, int]] = None,
        min_resolution: Optional[Tuple[int, int]] = None,
        max_file_size: Optional[int] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if video meets specified constraints
        
        Args:
            video_path (str): Path to the video file
            max_duration (float): Maximum duration in seconds
            max_resolution (Tuple[int, int]): Maximum width and height
            min_resolution (Tuple[int, int]): Minimum width and height
            max_file_size (int): Maximum file size in bytes
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check file size
            if max_file_size:
                file_size = os.path.getsize(video_path)
                if file_size > max_file_size:
                    issues.append(f"File size {file_size / (1024*1024):.1f}MB exceeds limit {max_file_size / (1024*1024):.1f}MB")
            
            # Get video info
            info = VideoProcessor.get_video_info(video_path)
            
            # Check duration
            if max_duration and info.get('duration', 0) > max_duration:
                issues.append(f"Duration {info['duration']:.1f}s exceeds limit {max_duration}s")
            
            # Check resolution
            width, height = info.get('width', 0), info.get('height', 0)
            
            if max_resolution:
                max_w, max_h = max_resolution
                if width > max_w or height > max_h:
                    issues.append(f"Resolution {width}x{height} exceeds limit {max_w}x{max_h}")
            
            if min_resolution:
                min_w, min_h = min_resolution
                if width < min_w or height < min_h:
                    issues.append(f"Resolution {width}x{height} below minimum {min_w}x{min_h}")
            
        except Exception as e:
            issues.append(f"Error checking video: {str(e)}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def extract_frames(
        video_path: str,
        output_dir: str,
        max_frames: Optional[int] = None,
        frame_interval: int = 1,
        resize: Optional[Tuple[int, int]] = None
    ) -> List[str]:
        """
        Extract frames from video
        
        Args:
            video_path (str): Path to input video
            output_dir (str): Directory to save frames
            max_frames (int): Maximum number of frames to extract
            frame_interval (int): Extract every Nth frame
            resize (Tuple[int, int]): Resize frames to (width, height)
            
        Returns:
            List[str]: Paths to extracted frame files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    if max_frames and extracted_count >= max_frames:
                        break
                    
                    if resize:
                        frame = cv2.resize(frame, resize)
                    
                    frame_filename = f"frame_{extracted_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    extracted_count += 1
                
                frame_count += 1
                
        finally:
            cap.release()
        
        logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
        return frame_paths
    
    @staticmethod
    def create_video_preview(
        video_path: str,
        output_path: str,
        duration: float = 10.0,
        start_time: float = 0.0,
        scale: Optional[str] = "640:360"
    ) -> str:
        """
        Create a preview/thumbnail video using FFmpeg
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path for output preview
            duration (float): Duration of preview in seconds
            start_time (float): Start time in original video
            scale (str): Scale filter for FFmpeg (e.g., "640:360")
            
        Returns:
            str: Path to created preview
        """
        try:
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'fast'
            ]
            
            if scale:
                cmd.extend(['-vf', f'scale={scale}'])
            
            cmd.append(output_path)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            logger.info(f"Created preview: {output_path}")
            return output_path
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Error creating preview: {e}")
            raise
    
    @staticmethod
    def convert_video_format(
        input_path: str,
        output_path: str,
        codec: str = 'libx264',
        crf: int = 23,
        preset: str = 'medium',
        audio_codec: str = 'aac'
    ) -> str:
        """
        Convert video to different format using FFmpeg
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path for output video
            codec (str): Video codec
            crf (int): Constant Rate Factor (quality)
            preset (str): Encoding preset
            audio_codec (str): Audio codec
            
        Returns:
            str: Path to converted video
        """
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', codec,
                '-crf', str(crf),
                '-preset', preset,
                '-c:a', audio_codec,
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
            
            logger.info(f"Converted video: {output_path}")
            return output_path
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Error converting video: {e}")
            raise

class VideoUploader:
    """
    Utility class for uploading videos to various services
    """
    
    @staticmethod
    def upload_to_temp_host(
        video_path: str,
        service: str = "file.io",
        max_size_mb: int = 100
    ) -> str:
        """
        Upload video to a temporary hosting service
        
        Args:
            video_path (str): Path to video file
            service (str): Hosting service to use
            max_size_mb (int): Maximum file size in MB
            
        Returns:
            str: Public URL of uploaded video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(f"File size {file_size_mb:.1f}MB exceeds limit {max_size_mb}MB")
        
        try:
            if service == "file.io":
                return VideoUploader._upload_file_io(video_path)
            else:
                raise ValueError(f"Unsupported service: {service}")
                
        except Exception as e:
            logger.error(f"Error uploading video: {e}")
            raise
    
    @staticmethod
    def _upload_file_io(video_path: str) -> str:
        """
        Upload to file.io service
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            str: Public URL
        """
        import requests
        
        url = "https://file.io"
        
        with open(video_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            if result.get('success'):
                public_url = result.get('link')
                logger.info(f"Video uploaded successfully: {public_url}")
                return public_url
            else:
                raise RuntimeError(f"Upload failed: {result.get('message', 'Unknown error')}")

def calculate_outpaint_dimensions(
    original_width: int,
    original_height: int,
    top_scale: float,
    bottom_scale: float,
    left_scale: float,
    right_scale: float
) -> Tuple[int, int]:
    """
    Calculate the final dimensions after outpainting
    
    Args:
        original_width (int): Original video width
        original_height (int): Original video height
        top_scale (float): Top expansion scale
        bottom_scale (float): Bottom expansion scale
        left_scale (float): Left expansion scale
        right_scale (float): Right expansion scale
        
    Returns:
        Tuple[int, int]: Final (width, height) after outpainting
    """
    # Calculate additional pixels for each direction
    top_pixels = int(original_height * (top_scale - 1))
    bottom_pixels = int(original_height * (bottom_scale - 1))
    left_pixels = int(original_width * (left_scale - 1))
    right_pixels = int(original_width * (right_scale - 1))
    
    # Calculate final dimensions
    final_width = original_width + left_pixels + right_pixels
    final_height = original_height + top_pixels + bottom_pixels
    
    return final_width, final_height

def estimate_processing_time(
    video_duration: float,
    video_resolution: Tuple[int, int],
    complexity_factor: float = 1.0
) -> float:
    """
    Estimate processing time for video outpainting
    
    Args:
        video_duration (float): Video duration in seconds
        video_resolution (Tuple[int, int]): Video resolution (width, height)
        complexity_factor (float): Complexity multiplier
        
    Returns:
        float: Estimated processing time in seconds
    """
    width, height = video_resolution
    pixel_count = width * height
    
    # Base processing time per second of video (rough estimates)
    base_time_per_second = 30  # seconds
    
    # Resolution factor (higher resolution takes longer)
    resolution_factor = pixel_count / (1920 * 1080)  # Normalized to 1080p
    
    # Calculate total estimated time
    estimated_time = (
        video_duration * 
        base_time_per_second * 
        resolution_factor * 
        complexity_factor
    )
    
    return max(60, estimated_time)  # Minimum 1 minute

def create_comparison_video(
    original_path: str,
    outpainted_path: str,
    output_path: str,
    side_by_side: bool = True
) -> str:
    """
    Create a comparison video showing original and outpainted versions
    
    Args:
        original_path (str): Path to original video
        outpainted_path (str): Path to outpainted video
        output_path (str): Path for comparison video
        side_by_side (bool): True for side-by-side, False for before/after
        
    Returns:
        str: Path to comparison video
    """
    try:
        if side_by_side:
            # Side-by-side comparison
            cmd = [
                'ffmpeg', '-y',
                '-i', original_path,
                '-i', outpainted_path,
                '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]',
                '-map', '[v]',
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                output_path
            ]
        else:
            # Before/after with text overlay
            cmd = [
                'ffmpeg', '-y',
                '-i', original_path,
                '-i', outpainted_path,
                '-filter_complex', (
                    '[0:v]drawtext=text=Original:fontcolor=white:fontsize=24:x=10:y=10[v0];'
                    '[1:v]drawtext=text=Outpainted:fontcolor=white:fontsize=24:x=10:y=10[v1];'
                    '[v0][v1]concat=n=2:v=1:a=0[v]'
                ),
                '-map', '[v]',
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                output_path
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg comparison failed: {result.stderr}")
        
        logger.info(f"Created comparison video: {output_path}")
        return output_path
        
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error creating comparison video: {e}")
        raise