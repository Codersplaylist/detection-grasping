"""
Test script for YOLO model inference on sample images
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2


def test_model_loading(model_path):
    """Test if model can be loaded"""
    print(f"Testing model loading from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return False
    
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {model.task}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def test_inference_on_image(model_path, image_path):
    """Test inference on a single image"""
    print(f"\nTesting inference on: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return False
    
    try:
        model = YOLO(model_path)
        results = model(image_path, conf=0.25)
        
        # Process results
        for result in results:
            boxes = result.boxes
            print(f"✓ Inference successful")
            print(f"  Detections: {len(boxes)}")
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                print(f"  {i+1}. {class_name}: {confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False


def test_batch_inference(model_path, image_dir, output_dir="test_results"):
    """Test batch inference on multiple images"""
    print(f"\nTesting batch inference on directory: {image_dir}")
    
    if not os.path.exists(image_dir):
        print(f"✗ Directory not found: {image_dir}")
        return False
    
    try:
        model = YOLO(model_path)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
        
        if len(image_files) == 0:
            print(f"✗ No images found in {image_dir}")
            return False
        
        print(f"Found {len(image_files)} images")
        
        # Run batch inference
        results = model(image_dir, save=True, conf=0.25)
        
        print(f"✓ Batch inference complete")
        print(f"  Results saved to: runs/detect/predict/")
        
        # Count total detections
        total_detections = sum(len(r.boxes) for r in results)
        print(f"  Total detections: {total_detections}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch inference failed: {e}")
        return False


def main():
    """Main test function"""
    print("="*60)
    print("YOLO Model Testing")
    print("="*60)
    
    # Configuration
    model_path = "models/my_model.pt"
    test_image_dir = "test/test_zip/test"
    
    # Allow custom model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Test 1: Model loading
    print("\n1. Testing model loading...")
    if not test_model_loading(model_path):
        print("\n✗ Model loading failed. Please check:")
        print("  1. Model file exists at the specified path")
        print("  2. Model file is a valid YOLO model (.pt)")
        return
    
    # Test 2: Single image inference
    print("\n2. Testing single image inference...")
    # Find a test image
    test_images = list(Path(test_image_dir).glob("*.jpg"))
    if len(test_images) > 0:
        test_image = str(test_images[0])
        test_inference_on_image(model_path, test_image)
    else:
        print(f"✗ No test images found in {test_image_dir}")
    
    # Test 3: Batch inference
    print("\n3. Testing batch inference...")
    test_batch_inference(model_path, test_image_dir)
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
