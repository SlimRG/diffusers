#!/usr/bin/env python3
"""
WAN 2.1 Video Outpainting Script
Uses Alibaba Cloud's DashScope API for video outpainting with WAN 2.1 VACE model
"""

import os
import time
import json
import requests
import argparse
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanVideoOutpainter:
    """
    A class to handle video outpainting using WAN 2.1 VACE model via Alibaba Cloud API
    """
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope-intl.aliyuncs.com"):
        """
        Initialize the WAN Video Outpainter
        
        Args:
            api_key (str): DashScope API key
            base_url (str): Base URL for the API endpoint
        """
        self.api_key = api_key
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/v1/services/aigc/video-generation/video-synthesis"
        
        if not self.api_key:
            raise ValueError("API key is required. Set DASHSCOPE_API_KEY environment variable or pass it directly.")
    
    def validate_video_url(self, video_url: str) -> bool:
        """
        Validate if the video URL is accessible
        
        Args:
            video_url (str): URL of the video
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            parsed = urlparse(video_url)
            if not all([parsed.scheme, parsed.netloc]):
                return False
            
            response = requests.head(video_url, timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def submit_outpainting_job(
        self, 
        video_url: str, 
        prompt: str,
        top_scale: float = 1.5,
        bottom_scale: float = 1.5,
        left_scale: float = 1.5,
        right_scale: float = 1.5,
        model: str = "wan2.1-vace-plus"
    ) -> Dict:
        """
        Submit video outpainting job to the API
        
        Args:
            video_url (str): URL of the input video
            prompt (str): Description of the scene for outpainting
            top_scale (float): Scaling factor for top expansion
            bottom_scale (float): Scaling factor for bottom expansion
            left_scale (float): Scaling factor for left expansion
            right_scale (float): Scaling factor for right expansion
            model (str): Model to use for outpainting
            
        Returns:
            Dict: API response containing task ID
        """
        headers = {
            'X-DashScope-Async': 'enable',
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": model,
            "input": {
                "function": "video_outpainting",
                "prompt": prompt,
                "video_url": video_url
            },
            "parameters": {
                "top_scale": top_scale,
                "bottom_scale": bottom_scale,
                "left_scale": left_scale,
                "right_scale": right_scale
            }
        }
        
        try:
            logger.info(f"Submitting outpainting job for video: {video_url}")
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Job submitted successfully. Task ID: {result.get('output', {}).get('task_id', 'Unknown')}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Failed to submit job: {e}")
            raise
    
    def check_job_status(self, task_id: str) -> Dict:
        """
        Check the status of a submitted job
        
        Args:
            task_id (str): Task ID returned from submit_outpainting_job
            
        Returns:
            Dict: Job status and results
        """
        status_url = f"{self.base_url}/api/v1/tasks/{task_id}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.get(status_url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Failed to check job status: {e}")
            raise
    
    def wait_for_completion(self, task_id: str, max_wait_time: int = 1800, check_interval: int = 30) -> Dict:
        """
        Wait for job completion with periodic status checks
        
        Args:
            task_id (str): Task ID to monitor
            max_wait_time (int): Maximum time to wait in seconds
            check_interval (int): Interval between status checks in seconds
            
        Returns:
            Dict: Final job result
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_result = self.check_job_status(task_id)
            status = status_result.get('output', {}).get('task_status', 'UNKNOWN')
            
            logger.info(f"Job status: {status}")
            
            if status == 'SUCCEEDED':
                logger.info("Job completed successfully!")
                return status_result
            elif status == 'FAILED':
                error_msg = status_result.get('output', {}).get('message', 'Unknown error')
                logger.error(f"Job failed: {error_msg}")
                raise RuntimeError(f"Job failed: {error_msg}")
            elif status in ['PENDING', 'RUNNING']:
                logger.info(f"Job still processing... waiting {check_interval} seconds")
                time.sleep(check_interval)
            else:
                logger.warning(f"Unknown status: {status}. Continuing to wait...")
                time.sleep(check_interval)
        
        raise TimeoutError(f"Job did not complete within {max_wait_time} seconds")
    
    def download_result(self, result_url: str, output_path: str) -> str:
        """
        Download the resulting video from the API
        
        Args:
            result_url (str): URL of the result video
            output_path (str): Local path to save the video
            
        Returns:
            str: Path to the downloaded file
        """
        try:
            logger.info(f"Downloading result video to: {output_path}")
            response = requests.get(result_url, stream=True, timeout=60)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Video downloaded successfully: {output_path}")
            return output_path
            
        except requests.RequestException as e:
            logger.error(f"Failed to download result: {e}")
            raise
    
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
    ) -> str:
        """
        Complete video outpainting workflow
        
        Args:
            video_url (str): URL of the input video
            prompt (str): Description for outpainting
            output_path (str): Path to save the result
            top_scale (float): Top expansion scale
            bottom_scale (float): Bottom expansion scale
            left_scale (float): Left expansion scale
            right_scale (float): Right expansion scale
            max_wait_time (int): Maximum wait time in seconds
            
        Returns:
            str: Path to the outpainted video
        """
        # Validate input video URL
        if not self.validate_video_url(video_url):
            raise ValueError(f"Invalid or inaccessible video URL: {video_url}")
        
        # Submit job
        submit_result = self.submit_outpainting_job(
            video_url=video_url,
            prompt=prompt,
            top_scale=top_scale,
            bottom_scale=bottom_scale,
            left_scale=left_scale,
            right_scale=right_scale
        )
        
        task_id = submit_result.get('output', {}).get('task_id')
        if not task_id:
            raise RuntimeError("Failed to get task ID from API response")
        
        # Wait for completion
        final_result = self.wait_for_completion(task_id, max_wait_time)
        
        # Extract result URL
        result_url = final_result.get('output', {}).get('video_url')
        if not result_url:
            raise RuntimeError("No result video URL in API response")
        
        # Download result
        return self.download_result(result_url, output_path)


def main():
    """
    Main function to handle command line usage
    """
    parser = argparse.ArgumentParser(description='WAN 2.1 Video Outpainting Tool')
    parser.add_argument('--video-url', required=True, help='URL of the input video')
    parser.add_argument('--prompt', required=True, help='Description for outpainting')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--api-key', help='DashScope API key (or set DASHSCOPE_API_KEY env var)')
    parser.add_argument('--top-scale', type=float, default=1.5, help='Top expansion scale (default: 1.5)')
    parser.add_argument('--bottom-scale', type=float, default=1.5, help='Bottom expansion scale (default: 1.5)')
    parser.add_argument('--left-scale', type=float, default=1.5, help='Left expansion scale (default: 1.5)')
    parser.add_argument('--right-scale', type=float, default=1.5, help='Right expansion scale (default: 1.5)')
    parser.add_argument('--max-wait', type=int, default=1800, help='Maximum wait time in seconds (default: 1800)')
    
    args = parser.parse_args()
    
    # Get API key from argument or environment
    api_key = args.api_key or os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        logger.error("API key is required. Use --api-key or set DASHSCOPE_API_KEY environment variable.")
        return 1
    
    try:
        # Initialize outpainter
        outpainter = WanVideoOutpainter(api_key)
        
        # Perform outpainting
        result_path = outpainter.outpaint_video(
            video_url=args.video_url,
            prompt=args.prompt,
            output_path=args.output,
            top_scale=args.top_scale,
            bottom_scale=args.bottom_scale,
            left_scale=args.left_scale,
            right_scale=args.right_scale,
            max_wait_time=args.max_wait
        )
        
        logger.info(f"Video outpainting completed successfully!")
        logger.info(f"Result saved to: {result_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during video outpainting: {e}")
        return 1


if __name__ == "__main__":
    exit(main())