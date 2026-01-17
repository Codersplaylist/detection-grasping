"""
YOLO Model Training Script
Trains a fresh YOLO model on the fruit detection dataset
"""
import os
from pathlib import Path
from ultralytics import YOLO
import torch


def check_dataset(data_yaml):
    """Verify dataset structure"""
    print("\nVerifying dataset...")
    
    data_path = Path(data_yaml).parent
    
    # Check required directories
    required_dirs = [
        data_path / 'train' / 'images',
        data_path / 'train' / 'labels',
        data_path / 'val' / 'images',
        data_path / 'val' / 'labels'
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"✗ Missing: {dir_path}")
            return False
        file_count = len(list(dir_path.glob('*')))
        print(f"✓ Found: {dir_path} ({file_count} files)")
    
    return True


def train_model(data_yaml, 
                model_name='yolo11n.pt',
                epochs=100,
                imgsz=640,
                batch=16,
                device=None):
    """
    Train YOLO model
    
    Args:
        data_yaml: Path to data.yaml configuration
        model_name: Pretrained model to start from (n/s/m/l/x)
        epochs: Number of training epochs
        imgsz: Training image size
        batch: Batch size (reduce if out of memory)
        device: Device to use (None=auto-detect, 'cpu', 'cuda', 'mps')
    """
    print("\n" + "="*60)
    print("YOLO Model Training")
    print("="*60)
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print(f"✓ Using Mac GPU (MPS)")
        else:
            device = 'cpu'
            print(f"✓ Using CPU (training will be slower)")
    
    # Load pretrained model
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print("\nThis may take a while depending on your hardware...")
    print("(On CPU: expect 30-60 minutes, on GPU: 10-20 minutes)")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project='runs/train',
        name='fruit_detection',
        patience=20,  # Early stopping
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return model, results


def validate_model(model, data_yaml):
    """Validate trained model"""
    print("\nValidating model on test set...")
    metrics = model.val(data=data_yaml)
    
    print("\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    
    return metrics


def export_model(model, output_path):
    """Export model to specified location"""
    print(f"\nExporting model to {output_path}...")
    
    # Get best weights
    best_weights = Path('runs/train/fruit_detection/weights/best.pt')
    
    if not best_weights.exists():
        print("✗ Best weights not found")
        return False
    
    # Copy to models directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.copy2(best_weights, output_path)
    
    print(f"✓ Model exported to: {output_path}")
    return True


def test_inference(model_path, test_image_dir):
    """Test model on sample images"""
    print(f"\nTesting model on sample images...")
    
    model = YOLO(model_path)
    results = model.predict(
        source=test_image_dir,
        save=True,
        conf=0.25,
        project='runs/test',
        name='predictions'
    )
    
    print(f"✓ Test predictions saved to: runs/test/predictions/")
    return results


def main():
    """Main training pipeline"""
    
    # Configuration
    data_yaml = 'dataset/data.yaml'
    output_model = 'models/my_model.pt'
    
    # Training parameters
    model_size = 'yolo11n.pt'  # n=nano (fastest), s=small, m=medium, l=large
    epochs = 100
    imgsz = 640
    batch_size = 16  # Reduce to 8 or 4 if out of memory
    
    print("="*60)
    print("YOLO Fruit Detection - Training Pipeline")
    print("="*60)
    
    # Step 1: Check dataset
    if not os.path.exists(data_yaml):
        print(f"\n✗ Dataset not found: {data_yaml}")
        print("\nPlease run 'python convert_dataset.py' first to prepare the dataset.")
        return
    
    if not check_dataset(data_yaml):
        print("\n✗ Dataset structure invalid")
        return
    
    # Step 2: Train model
    model, results = train_model(
        data_yaml=data_yaml,
        model_name=model_size,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size
    )
    
    # Step 3: Validate
    metrics = validate_model(model, data_yaml)
    
    # Step 4: Export to models directory
    export_model(model, output_model)
    
    # Step 5: Test on sample images
    test_image_dir = 'dataset/val/images'
    if os.path.exists(test_image_dir):
        test_inference(output_model, test_image_dir)
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print(f"\n✓ Trained model saved to: {output_model}")
    print(f"✓ Training results: runs/train/fruit_detection/")
    print(f"✓ Test predictions: runs/test/predictions/")
    print(f"\nYou can now run the camera application:")
    print(f"  python camera_yolo_integrated.py")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
