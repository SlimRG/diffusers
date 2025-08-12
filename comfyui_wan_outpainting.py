#!/usr/bin/env python3
"""
ComfyUI WAN 2.1 Video Outpainting Workflow
Local implementation using ComfyUI API for video outpainting with WAN 2.1 VACE
"""

import os
import json
import time
import requests
import argparse
import websocket
import uuid
from typing import Dict, Any, Optional
import logging
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComfyUIWanOutpainter:
    """
    A class to handle video outpainting using WAN 2.1 VACE model via ComfyUI
    """
    
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188"):
        """
        Initialize the ComfyUI WAN Outpainter
        
        Args:
            comfyui_url (str): ComfyUI server URL
        """
        self.comfyui_url = comfyui_url.rstrip('/')
        self.client_id = str(uuid.uuid4())
        
    def get_workflow_template(self) -> Dict[str, Any]:
        """
        Get the ComfyUI workflow template for WAN 2.1 video outpainting
        
        Returns:
            Dict: ComfyUI workflow configuration
        """
        workflow = {
            "1": {
                "inputs": {
                    "video": "",  # Will be set dynamically
                    "force_rate": 0,
                    "force_size": "Disabled",
                    "custom_width": 512,
                    "custom_height": 512,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1
                },
                "class_type": "VHS_LoadVideo",
                "_meta": {
                    "title": "Load Video (Path)"
                }
            },
            "2": {
                "inputs": {
                    "unet_name": "wan2.1_t2v_14B_fp8_e4m3fn.safetensors"
                },
                "class_type": "UNETLoader",
                "_meta": {
                    "title": "Load Diffusion Model"
                }
            },
            "3": {
                "inputs": {
                    "vae_name": "Wan2_1_VAE_bf16.safetensors"
                },
                "class_type": "VAELoader",
                "_meta": {
                    "title": "Load VAE"
                }
            },
            "4": {
                "inputs": {
                    "text_encoder_name": "umt5-xxl-enc-bf16.safetensors"
                },
                "class_type": "DualCLIPLoader",
                "_meta": {
                    "title": "DualCLIPLoader"
                }
            },
            "5": {
                "inputs": {
                    "module_name": "Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors"
                },
                "class_type": "WanVACELoader",
                "_meta": {
                    "title": "WanVACELoader"
                }
            },
            "6": {
                "inputs": {
                    "text": "",  # Will be set dynamically
                    "clip": ["4", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Prompt)"
                }
            },
            "7": {
                "inputs": {
                    "text": "",
                    "clip": ["4", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Negative)"
                }
            },
            "8": {
                "inputs": {
                    "function": "video_outpainting",
                    "prompt": ["6", 0],
                    "negative_prompt": ["7", 0],
                    "video": ["1", 0],
                    "vace_module": ["5", 0],
                    "top_scale": 1.5,
                    "bottom_scale": 1.5,
                    "left_scale": 1.5,
                    "right_scale": 1.5,
                    "steps": 20,
                    "cfg": 7.0,
                    "seed": -1,
                    "scheduler": "ddim",
                    "guidance_scale": 4.5
                },
                "class_type": "WanVACESampler",
                "_meta": {
                    "title": "WanVACESampler"
                }
            },
            "9": {
                "inputs": {
                    "samples": ["8", 0],
                    "vae": ["3", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAE Decode"
                }
            },
            "10": {
                "inputs": {
                    "filename_prefix": "wan_outpainted",
                    "fps": 8,
                    "loop_count": 0,
                    "filename_suffix": "_outpainted",
                    "format": "video/h264-mp4",
                    "pingpong": False,
                    "save_output": True,
                    "images": ["9", 0]
                },
                "class_type": "VHS_VideoCombine",
                "_meta": {
                    "title": "Video Combine"
                }
            }
        }
        return workflow
    
    def upload_video(self, video_path: str) -> str:
        """
        Upload video to ComfyUI server
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Uploaded filename on the server
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        upload_url = f"{self.comfyui_url}/upload/image"
        
        try:
            with open(video_path, 'rb') as f:
                files = {'image': (os.path.basename(video_path), f, 'video/mp4')}
                data = {'type': 'input', 'subfolder': ''}
                
                response = requests.post(upload_url, files=files, data=data, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Video uploaded successfully: {result['name']}")
                return result['name']
                
        except requests.RequestException as e:
            logger.error(f"Failed to upload video: {e}")
            raise
    
    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """
        Queue a prompt for execution on ComfyUI
        
        Args:
            workflow (Dict): ComfyUI workflow configuration
            
        Returns:
            str: Prompt ID
        """
        queue_url = f"{self.comfyui_url}/prompt"
        
        prompt_data = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        try:
            response = requests.post(queue_url, json=prompt_data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            prompt_id = result['prompt_id']
            logger.info(f"Prompt queued successfully: {prompt_id}")
            return prompt_id
            
        except requests.RequestException as e:
            logger.error(f"Failed to queue prompt: {e}")
            raise
    
    def wait_for_completion(self, prompt_id: str, max_wait_time: int = 1800) -> Dict[str, Any]:
        """
        Wait for prompt completion using WebSocket
        
        Args:
            prompt_id (str): Prompt ID to monitor
            max_wait_time (int): Maximum wait time in seconds
            
        Returns:
            Dict: Execution result
        """
        ws_url = f"ws://{self.comfyui_url.split('://')[1]}/ws?clientId={self.client_id}"
        
        start_time = time.time()
        result = None
        
        def on_message(ws, message):
            nonlocal result
            try:
                data = json.loads(message)
                
                if data['type'] == 'executing':
                    if data['data']['node'] is None and data['data']['prompt_id'] == prompt_id:
                        logger.info("Execution completed!")
                        result = data
                        ws.close()
                elif data['type'] == 'progress':
                    logger.info(f"Progress: {data['data']['value']}/{data['data']['max']}")
                elif data['type'] == 'execution_error':
                    logger.error(f"Execution error: {data}")
                    result = {'error': data}
                    ws.close()
                    
            except json.JSONDecodeError:
                pass
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
        
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Start WebSocket connection in a separate thread
            import threading
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for completion or timeout
            while result is None and time.time() - start_time < max_wait_time:
                time.sleep(1)
            
            if result is None:
                raise TimeoutError(f"Job did not complete within {max_wait_time} seconds")
            
            if 'error' in result:
                raise RuntimeError(f"Execution failed: {result['error']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error waiting for completion: {e}")
            raise
    
    def get_output_files(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get output files from completed execution
        
        Args:
            prompt_id (str): Prompt ID
            
        Returns:
            Dict: Output file information
        """
        history_url = f"{self.comfyui_url}/history/{prompt_id}"
        
        try:
            response = requests.get(history_url, timeout=30)
            response.raise_for_status()
            
            history = response.json()
            if prompt_id not in history:
                raise RuntimeError(f"No history found for prompt ID: {prompt_id}")
            
            outputs = history[prompt_id]['outputs']
            return outputs
            
        except requests.RequestException as e:
            logger.error(f"Failed to get output files: {e}")
            raise
    
    def download_output(self, filename: str, subfolder: str = "", output_type: str = "output") -> str:
        """
        Download output file from ComfyUI server
        
        Args:
            filename (str): Output filename
            subfolder (str): Subfolder path
            output_type (str): Type of output
            
        Returns:
            str: Local path to downloaded file
        """
        download_url = f"{self.comfyui_url}/view"
        params = {
            'filename': filename,
            'type': output_type,
            'subfolder': subfolder
        }
        
        try:
            response = requests.get(download_url, params=params, stream=True, timeout=60)
            response.raise_for_status()
            
            local_path = os.path.join("/workspace/outputs", filename)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Output downloaded: {local_path}")
            return local_path
            
        except requests.RequestException as e:
            logger.error(f"Failed to download output: {e}")
            raise
    
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
    ) -> str:
        """
        Complete video outpainting workflow using ComfyUI
        
        Args:
            video_path (str): Path to input video
            prompt (str): Description for outpainting
            negative_prompt (str): Negative prompt
            top_scale (float): Top expansion scale
            bottom_scale (float): Bottom expansion scale
            left_scale (float): Left expansion scale
            right_scale (float): Right expansion scale
            steps (int): Number of denoising steps
            cfg (float): CFG scale
            guidance_scale (float): Guidance scale
            max_wait_time (int): Maximum wait time in seconds
            
        Returns:
            str: Path to outpainted video
        """
        # Upload video
        uploaded_filename = self.upload_video(video_path)
        
        # Get workflow template
        workflow = self.get_workflow_template()
        
        # Configure workflow
        workflow["1"]["inputs"]["video"] = uploaded_filename
        workflow["6"]["inputs"]["text"] = prompt
        workflow["7"]["inputs"]["text"] = negative_prompt
        workflow["8"]["inputs"]["top_scale"] = top_scale
        workflow["8"]["inputs"]["bottom_scale"] = bottom_scale
        workflow["8"]["inputs"]["left_scale"] = left_scale
        workflow["8"]["inputs"]["right_scale"] = right_scale
        workflow["8"]["inputs"]["steps"] = steps
        workflow["8"]["inputs"]["cfg"] = cfg
        workflow["8"]["inputs"]["guidance_scale"] = guidance_scale
        
        # Queue prompt
        prompt_id = self.queue_prompt(workflow)
        
        # Wait for completion
        self.wait_for_completion(prompt_id, max_wait_time)
        
        # Get output files
        outputs = self.get_output_files(prompt_id)
        
        # Find video output
        video_output = None
        for node_id, node_outputs in outputs.items():
            if 'gifs' in node_outputs:
                video_output = node_outputs['gifs'][0]
                break
        
        if not video_output:
            raise RuntimeError("No video output found in execution results")
        
        # Download result
        return self.download_output(
            video_output['filename'],
            video_output.get('subfolder', ''),
            video_output.get('type', 'output')
        )


def main():
    """
    Main function for command line usage
    """
    parser = argparse.ArgumentParser(description='ComfyUI WAN 2.1 Video Outpainting Tool')
    parser.add_argument('--video-path', required=True, help='Path to input video')
    parser.add_argument('--prompt', required=True, help='Description for outpainting')
    parser.add_argument('--negative-prompt', default='', help='Negative prompt')
    parser.add_argument('--comfyui-url', default='http://127.0.0.1:8188', help='ComfyUI server URL')
    parser.add_argument('--top-scale', type=float, default=1.5, help='Top expansion scale')
    parser.add_argument('--bottom-scale', type=float, default=1.5, help='Bottom expansion scale')
    parser.add_argument('--left-scale', type=float, default=1.5, help='Left expansion scale')
    parser.add_argument('--right-scale', type=float, default=1.5, help='Right expansion scale')
    parser.add_argument('--steps', type=int, default=20, help='Number of denoising steps')
    parser.add_argument('--cfg', type=float, default=7.0, help='CFG scale')
    parser.add_argument('--guidance-scale', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--max-wait', type=int, default=1800, help='Maximum wait time in seconds')
    
    args = parser.parse_args()
    
    try:
        # Initialize outpainter
        outpainter = ComfyUIWanOutpainter(args.comfyui_url)
        
        # Perform outpainting
        result_path = outpainter.outpaint_video(
            video_path=args.video_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            top_scale=args.top_scale,
            bottom_scale=args.bottom_scale,
            left_scale=args.left_scale,
            right_scale=args.right_scale,
            steps=args.steps,
            cfg=args.cfg,
            guidance_scale=args.guidance_scale,
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