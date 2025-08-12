#!/usr/bin/env python3
"""
Video Outpainting with WAN 2.1 (DashScope VACE)
------------------------------------------------
This script shows how to use Alibaba Cloud's DashScope service to extend the borders of an existing
video (a.k.a *video outpainting*) with the WAN 2.1 VACE model.

Requirements:
    • Python 3.8+
    • requests 2.31+

Before running, set your DashScope API key in the environment variable `DASHSCOPE_API_KEY`.
Example:
    export DASHSCOPE_API_KEY="your_api_key_here"

Usage:
    python video_outpaint_wan21.py \
        --video-url "https://example.com/my_input.mp4" \
        --prompt "A serene sunset beach continuing beyond the frame" \
        --output outpainted.mp4 \
        --top 1.2 --bottom 1.2 --left 1.2 --right 1.2

The scale values (>1.0) control how far the frame is expanded in each direction.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any
import requests

# --- Constants -----------------------------------------------------------------
DASHSCOPE_API_BASE = "https://dashscope-intl.aliyuncs.com/api/v1"
SERVICE_ENDPOINT = f"{DASHSCOPE_API_BASE}/services/aigc/video-generation/video-synthesis"
TASK_ENDPOINT = f"{DASHSCOPE_API_BASE}/tasks"  # /{task_id}
MODEL_NAME = "wan2.1-vace-plus"  # WAN 2.1 VACE model ID (14B). Use wan2.1-vace if you are on 1.3B.
POLL_INTERVAL = 10  # seconds between status checks

# --- Helpers --------------------------------------------------------------------

def _headers(api_key: str, async_mode: bool = True) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if async_mode:
        headers["X-DashScope-Async"] = "enable"
    return headers


def submit_outpaint_request(api_key: str, video_url: str, prompt: str, scales: Dict[str, float]) -> str:
    """Submits the outpainting job and returns the task_id."""
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "input": {
            "function": "video_outpainting",
            "prompt": prompt,
            "video_url": video_url,
        },
        "parameters": {
            "top_scale": scales["top"],
            "bottom_scale": scales["bottom"],
            "left_scale": scales["left"],
            "right_scale": scales["right"],
        },
    }

    response = requests.post(SERVICE_ENDPOINT, headers=_headers(api_key, async_mode=True), json=payload)
    response.raise_for_status()
    data = response.json()
    task_id = data.get("request_id") or data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Unexpected response: {json.dumps(data, indent=2)}")
    print(f"[+] Submitted job. Task ID: {task_id}")
    return task_id


def poll_task(api_key: str, task_id: str) -> Dict[str, Any]:
    """Polls task status until completion. Returns the final task payload when succeeded."""
    url = f"{TASK_ENDPOINT}/{task_id}"
    while True:
        resp = requests.get(url, headers=_headers(api_key, async_mode=False))
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("status") or payload.get("output", {}).get("status")
        print(f"[*] Task status: {status}")
        if status in {"SUCCEEDED", "FAILED", "CANCELED"}:
            return payload
        time.sleep(POLL_INTERVAL)


def download_video(result_url: str, output_path: Path) -> None:
    print(f"[+] Downloading result video to {output_path} …")
    with requests.get(result_url, stream=True) as r:
        r.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("[✓] Download complete.")


# --- CLI ------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WAN 2.1 Video Outpainting")
    p.add_argument("--video-url", required=True, help="Publicly accessible URL to the input video")
    p.add_argument("--prompt", required=True, help="Text prompt describing the desired outpainted content")
    p.add_argument("--output", default="outpainted.mp4", help="Filename for the resulting video")
    p.add_argument("--top", type=float, default=1.2, help="Top scale factor (>=1.0)")
    p.add_argument("--bottom", type=float, default=1.2, help="Bottom scale factor (>=1.0)")
    p.add_argument("--left", type=float, default=1.2, help="Left scale factor (>=1.0)")
    p.add_argument("--right", type=float, default=1.2, help="Right scale factor (>=1.0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[!] Please set the DASHSCOPE_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    scales = {"top": args.top, "bottom": args.bottom, "left": args.left, "right": args.right}
    task_id = submit_outpaint_request(api_key, args.video_url, args.prompt, scales)

    result = poll_task(api_key, task_id)
    if result.get("status") == "SUCCEEDED":
        # DashScope returns the URL in different fields depending on version; try both.
        output_url = result.get("output", {}).get("video_url") or result.get("output", {}).get("video")
        if not output_url:
            print("[!] Job succeeded but could not find output URL:")
            print(json.dumps(result, indent=2))
            sys.exit(1)
        download_video(output_url, Path(args.output))
    else:
        print("[!] Task failed:")
        print(json.dumps(result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()