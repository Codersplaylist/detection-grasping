# Complete Step-by-Step Setup Guide

## Prerequisites

Before starting, make sure you have:
- Python 3.8 or higher installed
- Terminal access
- Internet connection (for downloading dependencies)

Check your Python version:
```bash
python3 --version
```

You should see something like `Python 3.9.x` or `Python 3.10.x` or higher.

---

## Step-by-Step Instructions

### Step 1: Navigate to Project Directory

Open Terminal and navigate to your project folder:

```bash
cd "/Users/mac/Documents/appa project all files"
```

### Step 2: Install System Dependencies (One Time)

Pygame requires SDL libraries. Install them via Homebrew:

```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

**If you don't have Homebrew installed**, install it first:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Time**: 1-2 minutes

---

### Step 3: Create Virtual Environment (Recommended)

macOS requires using a virtual environment for Python packages. Create one:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

You'll see `(venv)` appear at the start of your terminal prompt. This means the virtual environment is active.

**Important:** You need to activate the virtual environment every time you open a new terminal session.

### Step 4: Install Dependencies

Now install all required packages (with virtual environment activated):

```bash
pip3 install -r requirements.txt
```

**What this installs:**
- ultralytics (YOLO framework)
- torch & torchvision (deep learning)
- opencv-python (camera and image processing)
- numpy (numerical operations)
- Pillow (image processing)
- PyYAML (configuration files)
- pygame & PyOpenGL (display)

**Time**: 5-10 minutes (depending on internet speed)

**Expected output**: You'll see packages downloading and installing. Wait until you see "Successfully installed..."

---

### Step 5: Convert Dataset to YOLO Format

Convert your XML annotations to YOLO format:

```bash
python3 convert_dataset.py
```

**What happens:**
- Reads 240 training images with XML annotations
- Reads 60 validation images with XML annotations
- Converts bounding boxes to YOLO format
- Creates `dataset/` folder with proper structure

**Time**: 2-5 seconds

**Expected output:**
```
============================================================
XML to YOLO Converter
============================================================

1. Converting training data...
✓ Converted: 240
✗ Skipped: 0

2. Converting validation data...
✓ Converted: 60
✗ Skipped: 0

✓ Created data.yaml
✓ Created classes.txt

Dataset summary:
  Training images: 240
  Validation images: 60
  Classes: apple, banana, orange

Ready for training!
```

---

### Step 6: Train the YOLO Model

Train a fresh YOLO model on your fruit dataset:

```bash
python3 train_model.py
```

**What happens:**
- Loads YOLO11-nano pretrained model
- Trains on your 240 images for 100 epochs
- Validates on 60 test images
- Saves best model to `models/my_model.pt`
- Creates training plots and results

**Time**: 
- **Mac with M1/M2 (MPS)**: 15-25 minutes
- **Mac with Intel CPU**: 40-60 minutes
- **Linux/Windows with NVIDIA GPU**: 10-20 minutes

**Expected output:**
```
============================================================
YOLO Model Training
============================================================
✓ Using Mac GPU (MPS)  [or "Using CPU" or "Using CUDA GPU"]

Loading pretrained model: yolo11n.pt

Starting training...
  Epochs: 100
  Image size: 640
  Batch size: 16
  Device: mps

This may take a while...

Epoch 1/100: loss=1.2, precision=0.65, recall=0.58
Epoch 2/100: loss=1.1, precision=0.68, recall=0.62
...
Epoch 100/100: loss=0.4, precision=0.92, recall=0.89

============================================================
Training Complete!
============================================================

Validation Results:
  mAP50: 0.95
  mAP50-95: 0.82

✓ Model exported to: models/my_model.pt
✓ Training results: runs/train/fruit_detection/
✓ Test predictions: runs/test/predictions/

You can now run the camera application:
  python3 camera_yolo_integrated.py
```

**Important:** Don't close the terminal while training is running!

---

### Step 7: Verify Training Results

Check that the model was created successfully:

```bash
ls -lh models/my_model.pt
```

You should see a file around 5-6 MB in size.

View training plots (optional):

```bash
open runs/train/fruit_detection/results.png
```

View test predictions (optional):

```bash
open runs/test/predictions/
```

---

### Step 8: Run the Camera Application

Start the real-time object detection application:

```bash
python3 camera_yolo_integrated.py
```

**What happens:**
- Camera window opens
- Shows live video feed
- FPS counter in top-left corner
- Detects apples, bananas, oranges in real-time
- Shows bounding boxes, labels, and grasp points
- Prints grasp coordinates to terminal

**Expected output in terminal:**
```
============================================================
YOLO Object Detection with Grasp Prediction
============================================================

Controls:
  ESC - Exit application
  D - Toggle detection on/off
  G - Toggle grasp visualization
  C - Clear console output

============================================================

✓ Configuration loaded from config.yaml
✓ Camera initialized: 1920x1080
✓ YOLO model loaded from models/my_model.pt
✓ Using device: mps

[Camera window opens]
```

**When you show a fruit to the camera:**
```
============================================================
Detected 1 object(s) - Grasp coordinates:
1. Grasp: [apple] Position=(640, 360), Angle=45°, Width=120.5px, Quality=0.89
============================================================
```

---

### Step 9: Test with Objects

Hold different fruits in front of the camera:

1. **Apple** - Should show red bounding box with "apple" label
2. **Banana** - Should show yellow bounding box with "banana" label
3. **Orange** - Should show orange bounding box with "orange" label

Each detected object will have:
- ✅ Bounding box around it
- ✅ Class label (apple/banana/orange)
- ✅ Confidence score (e.g., 0.95)
- ✅ Magenta crosshair at grasp point
- ✅ Arrow showing approach angle

---

## Keyboard Controls

While the camera application is running:

| Key | Action |
|-----|--------|
| **ESC** | Exit the application |
| **D** | Toggle detection on/off |
| **G** | Toggle grasp point visualization |
| **C** | Clear console output |

---

## Quick Commands Summary

```bash
# 1. Navigate to project
cd "/Users/mac/Documents/appa project all files"

# 2. Install SDL libraries via Homebrew (one time)
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf

# 3. Create and activate virtual environment (one time setup)
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies (one time, with venv active)
pip3 install -r requirements.txt

# 4. Convert dataset (one time, with venv active)
python3 convert_dataset.py

# 5. Train model (one time, 20-60 minutes, with venv active)
python3 train_model.py

# 6. Run camera app (every time you want to use it)
# Remember to activate venv first if opening new terminal!
source venv/bin/activate
python3 camera_yolo_integrated.py
```

---

## Alternative: One-Command Setup

If you want to do steps 3-5 automatically (after creating venv):

```bash
# First create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Then run automated setup
python3 setup_and_train.py
```

This will:
1. Check if dependencies are installed
2. Convert the dataset
3. Train the model
4. Export and test

Then just run:
```bash
source venv/bin/activate  # If new terminal
python3 camera_yolo_integrated.py
```

---

## Troubleshooting

### Problem: "pip3: command not found"

**Solution:** Use `python3 -m pip` instead:
```bash
python3 -m pip install -r requirements.txt
```

### Problem: "Permission denied"

**Solution:** Add `sudo` or use `--user` flag:
```bash
pip3 install --user -r requirements.txt
```

### Problem: "ModuleNotFoundError: No module named 'ultralytics'"

**Solution:** Dependencies not installed. Run:
```bash
pip3 install -r requirements.txt
```

### Problem: "Model file not found"

**Solution:** The model hasn't been trained yet. Run:
```bash
python3 train_model.py
```

### Problem: "Camera failed to open"

**Solution 1:** Close other apps using the camera (Zoom, FaceTime, etc.)

**Solution 2:** Try different camera device ID. Edit `config.yaml`:
```yaml
camera:
  device_id: 1  # Try 1, then 2, then 3
```

### Problem: Training is very slow (>60 minutes)

**Solution 1:** Reduce batch size. Edit `train_model.py`:
```python
batch_size = 8  # Instead of 16
```

**Solution 2:** Use fewer epochs:
```python
epochs = 50  # Instead of 100
```

**Solution 3:** Use smaller image size:
```python
imgsz = 416  # Instead of 640
```

### Problem: "Out of memory" during training

**Solution:** Reduce batch size in `train_model.py`:
```python
batch_size = 4  # or even 2
```

### Problem: Low FPS in camera app (<5 FPS)

**Solution:** Reduce inference size in `config.yaml`:
```yaml
model:
  imgsz: 416  # Instead of 640
```

---

## Directory Structure After Setup

After completing all steps, your project should look like:

```
appa project all files/
├── camera_yolo_integrated.py    ← Camera app (run this!)
├── grasp_detector.py
├── config.yaml
├── convert_dataset.py
├── train_model.py
├── setup_and_train.py
├── requirements.txt
├── dataset/                      ← Created by convert_dataset.py
│   ├── train/
│   │   ├── images/ (240 files)
│   │   └── labels/ (240 files)
│   ├── val/
│   │   ├── images/ (60 files)
│   │   └── labels/ (60 files)
│   ├── data.yaml
│   └── classes.txt
├── models/
│   └── my_model.pt              ← Created by train_model.py (your trained model!)
└── runs/                         ← Created by train_model.py
    ├── train/fruit_detection/   (training results & plots)
    └── test/predictions/        (test images with detections)
```

---

## Expected Output Summary

### After Dataset Conversion:
✅ `dataset/` folder created  
✅ 240 training images + labels  
✅ 60 validation images + labels  
✅ `data.yaml` configuration file  

### After Training:
✅ `models/my_model.pt` (5-6 MB)  
✅ Training plots in `runs/train/fruit_detection/`  
✅ mAP50 around 0.90-0.98  
✅ Test predictions in `runs/test/predictions/`  

### Running Camera App:
✅ Camera window opens  
✅ FPS: 10-30 (depending on hardware)  
✅ Real-time detection of fruits  
✅ Bounding boxes and labels  
✅ Grasp coordinates in terminal  

---

## Daily Usage

Once setup is complete (one time), you only need:

```bash
cd "/Users/mac/Documents/appa project all files"
source venv/bin/activate  # Activate virtual environment
python3 camera_yolo_integrated.py
```

**Important:** Always activate the virtual environment (`source venv/bin/activate`) before running the app. You'll know it's active when you see `(venv)` in your terminal prompt.

---

## Need Help?

If you encounter any issues:

1. Check this guide's troubleshooting section
2. Make sure all commands use `python3` (not `python`)
3. Verify you're in the correct directory
4. Check that the model file exists: `ls models/my_model.pt`

**Common issue:** If training takes too long (>60 min on CPU), you can:
- Reduce epochs to 50
- Reduce batch size to 8
- Use smaller image size (416)

All these settings can be changed in `train_model.py` before running.
