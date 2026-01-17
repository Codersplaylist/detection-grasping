#!/usr/bin/env python3
"""
Complete Setup and Training Pipeline
Runs all steps: dataset conversion, model training, and deployment
"""
import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n✗ Failed: {description}")
        return False
    
    print(f"\n✓ Completed: {description}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'ultralytics',
        'torch',
        'cv2',
        'yaml',
        'PIL'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'yaml':
                import yaml
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies available")
    return True


def main():
    """Main setup pipeline"""
    
    print("="*60)
    print("YOLO Fruit Detection - Complete Setup")
    print("="*60)
    print("\nThis script will:")
    print("1. Install dependencies (if needed)")
    print("2. Convert dataset from XML to YOLO format")
    print("3. Train YOLO model")
    print("4. Export model to models/ directory")
    print("5. Test the model on validation images")
    print("\n" + "="*60)
    
    # Ask for confirmation
    response = input("\nDo you want to continue? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    # Check if dependencies are installed
    if not check_dependencies():
        print("\n" + "="*60)
        print("Installing dependencies...")
        print("="*60)
        
        if not run_command(
            "pip install -r requirements.txt",
            "Installing required packages"
        ):
            print("\n✗ Failed to install dependencies")
            print("Please install manually: pip install -r requirements.txt")
            return
    
    # Step 1: Convert dataset
    print("\n" + "="*60)
    print("Step 1: Converting Dataset")
    print("="*60)
    
    if not run_command(
        f"{sys.executable} convert_dataset.py",
        "Converting XML annotations to YOLO format"
    ):
        print("\n✗ Dataset conversion failed")
        return
    
    # Step 2: Train model
    print("\n" + "="*60)
    print("Step 2: Training Model")
    print("="*60)
    print("\n⚠️  This may take 30-60 minutes depending on your hardware")
    print("You can modify training parameters in train_model.py if needed")
    
    response = input("\nStart training? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("\nYou can train later by running:")
        print("  python train_model.py")
        return
    
    if not run_command(
        f"{sys.executable} train_model.py",
        "Training YOLO model"
    ):
        print("\n✗ Training failed")
        return
    
    # Success!
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\nYour trained model is ready at: models/my_model.pt")
    print("\nNext steps:")
    print("1. Run the camera application:")
    print("   python camera_yolo_integrated.py")
    print("\n2. Check training results:")
    print("   - Training plots: runs/train/fruit_detection/")
    print("   - Test predictions: runs/test/predictions/")
    print("\n3. Adjust settings in config.yaml if needed")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
