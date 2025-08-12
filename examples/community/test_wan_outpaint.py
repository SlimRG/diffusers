#!/usr/bin/env python3
"""
Test script for WAN Video Outpaint Examples

This script tests the basic functionality of the video outpaint examples
without requiring the full WAN model to be downloaded.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow {Image.__version__}")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, UMT5EncoderModel
        print("✓ Transformers")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        print("✓ Diffusers (WAN models)")
    except ImportError as e:
        print(f"✗ Diffusers WAN models import failed: {e}")
        return False
    
    try:
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        print("✓ Diffusers (schedulers)")
    except ImportError as e:
        print(f"✗ Diffusers schedulers import failed: {e}")
        return False
    
    try:
        from diffusers.utils import export_to_video
        print("✓ Diffusers (utils)")
    except ImportError as e:
        print(f"✗ Diffusers utils import failed: {e}")
        return False
    
    return True


def test_file_creation():
    """Test that sample video frames can be created."""
    print("\nTesting file creation...")
    
    try:
        from wan_video_outpaint_simple import create_sample_video_frames
        
        # Create sample frames
        frames = create_sample_video_frames(width=64, height=64, num_frames=4)
        
        if len(frames) == 4:
            print("✓ Sample video frames created successfully")
            
            # Check frame dimensions
            for i, frame in enumerate(frames):
                if frame.size == (64, 64):
                    print(f"  ✓ Frame {i+1}: {frame.size}")
                else:
                    print(f"  ✗ Frame {i+1}: Expected (64, 64), got {frame.size}")
                    return False
        else:
            print(f"✗ Expected 4 frames, got {len(frames)}")
            return False
            
    except Exception as e:
        print(f"✗ Sample frame creation failed: {e}")
        return False
    
    return True


def test_mask_creation():
    """Test that outpaint masks can be created."""
    print("\nTesting mask creation...")
    
    try:
        from wan_video_outpaint import WanVideoOutpaintPipeline
        
        # Create a dummy pipeline instance for testing
        pipeline = WanVideoOutpaintPipeline.__new__(WanVideoOutpaintPipeline)
        
        # Test mask creation
        mask = pipeline._create_outpaint_mask(
            width=100,
            height=100,
            outpaint_left=20,
            outpaint_right=30,
            outpaint_top=10,
            outpaint_bottom=15
        )
        
        if mask.shape == (100, 100):
            print("✓ Outpaint mask created successfully")
            
            # Check mask values
            left_region = mask[:, :20]
            right_region = mask[:, -30:]
            top_region = mask[:10, :]
            bottom_region = mask[-15:, :]
            
            if (left_region == 1).all() and (right_region == 1).all() and \
               (top_region == 1).all() and (bottom_region == 1).all():
                print("  ✓ Mask regions correctly marked")
            else:
                print("  ✗ Mask regions not correctly marked")
                return False
        else:
            print(f"✗ Expected mask shape (100, 100), got {mask.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Mask creation failed: {e}")
        return False
    
    return True


def test_video_loading():
    """Test that video loading functions exist and are callable."""
    print("\nTesting video loading functions...")
    
    try:
        from wan_video_outpaint import load_video_frames
        
        # Check if function exists and is callable
        if callable(load_video_frames):
            print("✓ Video loading function is callable")
        else:
            print("✗ Video loading function is not callable")
            return False
            
    except Exception as e:
        print(f"✗ Video loading function test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("WAN Video Outpaint - Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("File Creation Test", test_file_creation),
        ("Mask Creation Test", test_mask_creation),
        ("Video Loading Test", test_video_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The WAN video outpaint examples are ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())