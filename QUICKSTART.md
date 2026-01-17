# Complete Setup Guide - Train Your Own Model

This guide will help you train a fresh YOLO model using your dataset and integrate it with the camera application.

## Quick Start (Automated)

The easiest way is to use the automated setup script:

```bash
python setup_and_train.py
```

This will:
1. ✅ Install all dependencies
2. ✅ Convert your XML dataset to YOLO format
3. ✅ Train a YOLO model
4. ✅ Export the model to `models/my_model.pt`
5. ✅ Test the model on validation images

Then run:
```bash
python camera_yolo_integrated.py
```

## Manual Setup (Step by Step)

If you prefer manual control, follow these steps:

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `ultralytics` - YOLO framework
- `torch` + `torchvision` - Deep learning
- `opencv-python` - Camera and image processing
- `numpy` - Numerical operations
- `Pillow` - Image processing for dataset conversion
- `PyYAML` - Configuration

### Step 2: Convert Dataset

Your dataset is in Pascal VOC XML format. Convert it to YOLO format:

```bash
python convert_dataset.py
```

This will:
- Read all XML annotations from `test/train_zip/train/` and `test/test_zip/test/`
- Convert bounding boxes to YOLO format (normalized x_center, y_center, width, height)
- Create `dataset/` directory with proper structure:
  ```
  dataset/
  ├── train/
  │   ├── images/  (240 images)
  │   └── labels/  (240 .txt files)
  ├── val/
  │   ├── images/  (60 images)
  │   └── labels/  (60 .txt files)
  ├── data.yaml
  └── classes.txt
  ```

### Step 3: Train Model

Train a YOLO model on your fruit dataset:

```bash
python train_model.py
```

**Training Configuration:**
- Model: YOLO11-nano (fastest, good accuracy)
- Epochs: 100 (with early stopping)
- Image size: 640x640
- Batch size: 16 (reduce to 8 or 4 if out of memory)

**Training Time:**
- CPU: 30-60 minutes
- Mac GPU (MPS): 15-25 minutes
- NVIDIA GPU (CUDA): 10-20 minutes

**What Happens:**
- Model trains on 240 training images
- Validates on 60 test images
- Saves best weights to `runs/train/fruit_detection/weights/best.pt`
- Copies best model to `models/my_model.pt`
- Creates training plots (loss, precision, recall, mAP)

### Step 4: Verify Model

After training, check the results:

```bash
# View training plots
open runs/train/fruit_detection/  # macOS
# Look for: results.png, confusion_matrix.png, F1_curve.png

# View test predictions
open runs/test/predictions/  # macOS
# Check if detections look good
```

### Step 5: Run Camera Application

```bash
python camera_yolo_integrated.py
```

Your freshly trained model will now detect apples, bananas, and oranges in real-time!

## Dataset Information

**Your Dataset:**
- **Training images**: 240 (80% split)
  - Apple: 76 images
  - Banana: 76 images
  - Orange: 68 images  
  - Mixed: 20 images (multiple fruits)

- **Validation images**: 60 (20% split)
  - Apple: 19 images
  - Banana: 18 images
  - Orange: 18 images
  - Mixed: 5 images

**Classes:**
1. apple (class 0)
2. banana (class 1)
3. orange (class 2)

## Customizing Training

You can modify `train_model.py` to adjust:

### Model Size
```python
model_size = 'yolo11n.pt'  # nano - fastest
# OR
model_size = 'yolo11s.pt'  # small - more accurate
# OR
model_size = 'yolo11m.pt'  # medium - even more accurate
```

### Training Duration
```python
epochs = 100  # Default
# OR
epochs = 50   # Faster training, may be less accurate
# OR
epochs = 150  # Longer training, potentially better results
```

### Batch Size
```python
batch_size = 16  # Default
# OR
batch_size = 8   # If out of memory
# OR
batch_size = 32  # If you have lots of GPU memory
```

## Troubleshooting

### "Out of Memory" Error

Reduce batch size in `train_model.py`:
```python
batch_size = 8  # or even 4
```

### Training is Very Slow

1. Check if GPU is being used:
   - The script will print "Using CUDA GPU" or "Using Mac GPU (MPS)"
   - If it says "Using CPU", training will be much slower

2. Use a smaller model:
   ```python
   model_size = 'yolo11n.pt'  # Nano is fastest
   ```

3. Reduce image size:
   ```python
   imgsz = 416  # Instead of 640
   ```

### Model Accuracy is Low

1. Train for more epochs:
   ```python
   epochs = 150
   ```

2. Use a larger model:
   ```python
   model_size = 'yolo11s.pt'  # or 'yolo11m.pt'
   ```

3. Check your dataset:
   - Make sure annotations are correct
   - Ensure good variety in training images

### Dataset Conversion Fails

Check that your image and XML files are in:
- `test/train_zip/train/` (training data)
- `test/test_zip/test/` (validation data)

## Expected Results

After training, you should see:

**Training Metrics (in console):**
```
Epoch 100/100: 
  - Loss: ~0.5-1.0
  - Precision: ~0.85-0.95
  - Recall: ~0.80-0.92
  - mAP50: ~0.90-0.98
```

**Files Created:**
- `models/my_model.pt` - Your trained model (ready to use!)
- `runs/train/fruit_detection/` - Training results and plots
- `runs/test/predictions/` - Test images with predictions
- `dataset/` - Converted YOLO format dataset

## Next Steps

Once training is complete:

1. **Test on Camera**:
   ```bash
   python camera_yolo_integrated.py
   ```

2. **Fine-tune Detection**:
   - Edit `config.yaml` to adjust confidence threshold
   - Lower `conf_threshold` if missing detections
   - Raise `conf_threshold` if getting false positives

3. **Improve Model**:
   - Collect more training images
   - Add more variety (different angles, lighting)
   - Retrain with larger model for better accuracy

## Summary

**Simple Method:**
```bash
python setup_and_train.py
python camera_yolo_integrated.py
```

**Manual Method:**
```bash
pip install -r requirements.txt
python convert_dataset.py
python train_model.py
python camera_yolo_integrated.py
```

Both methods will give you a fully trained, working object detection system! 🎉
