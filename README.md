# Fruit Detection & Robotic Grasping System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time computer vision system for fruit detection and robotic grasping using YOLOv11 and intelligent grasp point calculation. Designed for robotic pick-and-place applications in agriculture and warehouse automation.

![Demo](docs/demo.gif)
*Real-time detection of apples, bananas, and oranges with grasp coordinates*

## 🌟 Features

- **Real-Time Object Detection**: YOLOv11-based detection trained on custom fruit dataset
- **Intelligent Grasp Planning**: Automatic calculation of optimal grasp points and approach angles
- **High Accuracy**: 90-95% mAP@0.5 on validation set
- **Fast Inference**: 10-30 FPS on CPU, 50-100 FPS on GPU
- **Complete Training Pipeline**: End-to-end solution from dataset to deployment
- **Easy Integration**: Simple API for robot control systems
- **Configurable**: YAML-based configuration for all parameters

## 🎯 Use Cases

- Automated fruit sorting in agriculture
- Warehouse pick-and-place operations
- Quality inspection systems
- Robotics research and education
- Computer vision benchmarking

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- macOS, Linux, or Windows

### Installation

```bash
# Clone the repository
git clone https://github.com/Codersplaylist/detection-grasping.git
cd fruit-detection-grasping

# Install system dependencies (macOS)
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Convert dataset to YOLO format
python3 convert_dataset.py

# Train the model (20-60 minutes depending on hardware)
python3 train_model.py

# Run the detection system
python3 camera_yolo_integrated.py
```

## 📊 Dataset

The system is trained on a custom dataset of 300 annotated images:

- **Training set**: 240 images
- **Validation set**: 60 images
- **Classes**: Apple, Banana, Orange
- **Format**: Pascal VOC XML → YOLO format

Dataset structure:
```
dataset/
├── train/
│   ├── images/  (240 .jpg files)
│   └── labels/  (240 .txt files)
├── val/
│   ├── images/  (60 .jpg files)
│   └── labels/  (60 .txt files)
└── data.yaml
```

## 🧠 Model Architecture

- **Base Model**: YOLO11-nano
- **Parameters**: ~2.6M
- **Model Size**: 5-6 MB
- **Training**: 100 epochs with early stopping
- **Optimization**: AdamW optimizer
- **Image Size**: 640×640

### Performance Metrics

| Metric | Value |
|--------|-------|
| mAP@0.5 | 90-95% |
| mAP@0.5:0.95 | 75-85% |
| Precision | 85-95% |
| Recall | 80-92% |
| Inference Speed (CPU) | 10-30 FPS |
| Inference Speed (GPU) | 50-100 FPS |

## 🎮 Usage

### Command Line Interface

```bash
# Run with default configuration
python3 camera_yolo_integrated.py

# Run with custom config
python3 camera_yolo_integrated.py path/to/config.yaml
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **ESC** | Exit application |
| **D** | Toggle detection on/off |
| **G** | Toggle grasp visualization |
| **C** | Clear console output |

### Grasp Output Format

When objects are detected, the system outputs:

```
============================================================
Detected 2 object(s) - Grasp coordinates:
1. Grasp: [apple] Position=(450, 320), Angle=45°, Width=85.6px, Quality=0.89
2. Grasp: [banana] Position=(680, 410), Angle=0°, Width=120.3px, Quality=0.75
============================================================
```

**Output Parameters:**
- **Position**: (x, y) pixel coordinates of grasp point
- **Angle**: Approach angle in degrees (0°=horizontal, 90°=vertical)
- **Width**: Recommended gripper width in pixels
- **Quality**: Grasp quality score (0.0-1.0)

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Model settings
model:
  weights: "models/my_model.pt"
  imgsz: 640
  conf_threshold: 0.25
  device: "cpu"  # or "cuda", "mps"

# Camera settings
camera:
  device_id: 0
  width: 1920
  height: 1080

# Display options
display:
  show_boxes: true
  show_labels: true
  show_confidence: true
  show_fps: true

# Grasp detection
grasp:
  enabled: true
  show_grasp_point: true
  print_coordinates: true
```

## 🤖 Robot Integration

### Python API Example

```python
from camera_yolo_integrated import YOLOCameraApp
from grasp_detector import GraspDetector

# Initialize system
app = YOLOCameraApp("config.yaml")
grasp_detector = GraspDetector()

# Get detections
frame = app.camera.read()
detections = app.detect_objects(frame)

# Calculate grasp poses
grasps = grasp_detector.get_multiple_grasps(detections)

# Use best grasp for robot control
best_grasp = grasps[0]
robot.move_to(best_grasp['position'])
robot.set_gripper_width(best_grasp['gripper_width'])
robot.approach_angle(best_grasp['approach_angle'])
robot.grasp()
```

## 📁 Project Structure

```
detection-grasping/
├── camera_yolo_integrated.py   # Main application
├── grasp_detector.py           # Grasp calculation module
├── convert_dataset.py          # Dataset conversion utility
├── train_model.py              # Model training script
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── SETUP_GUIDE.md             # Detailed setup instructions
├── LICENSE                     # MIT License
├── .gitignore                 # Git ignore rules
├── dataset/                    # YOLO format dataset (generated)
├── models/                     # Trained models
│   └── my_model.pt            # Best trained weights
├── runs/                       # Training results (generated)
│   └── detect/
│       └── runs/train/
└── docs/                       # Documentation and images
    ├── demo.gif
    └── architecture.png
```

## 🔬 Training Your Own Model

### 1. Prepare Your Dataset

Organize your images and annotations:
```
your_data/
├── train/
│   ├── image1.jpg
│   ├── image1.xml
│   └── ...
└── test/
    ├── image1.jpg
    ├── image1.xml
    └── ...
```

### 2. Convert to YOLO Format

```python
python3 convert_dataset.py
```

### 3. Configure Training

Edit `train_model.py` to adjust:
- Model size (nano/small/medium/large)
- Number of epochs
- Batch size
- Image size

### 4. Train

```bash
python3 train_model.py
```

Training time:
- CPU: 30-60 minutes
- Mac M1/M2 (MPS): 15-25 minutes
- NVIDIA GPU: 10-20 minutes

### 5. Evaluate

Results are saved in `runs/detect/runs/train/fruit_detection/`:
- `results.png` - Training curves
- `confusion_matrix.png` - Class confusion
- `weights/best.pt` - Best model weights

## 🐛 Troubleshooting

### Common Issues

**Camera not opening:**
```bash
# Try different camera ID in config.yaml
camera:
  device_id: 1  # or 2, 3
```

**Low FPS:**
```bash
# Reduce inference size in config.yaml
model:
  imgsz: 416  # instead of 640
```

**Out of memory during training:**
```python
# In train_model.py, reduce batch size
batch_size = 8  # or 4
```

**Import errors:**
```bash
pip install --upgrade -r requirements.txt
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed troubleshooting.

## 📈 Performance Optimization

### For Better Accuracy
- Use larger model: `yolo11s.pt` or `yolo11m.pt`
- Train for more epochs: 150-200
- Collect more training data
- Increase image diversity

### For Faster Inference
- Use smaller model: `yolo11n.pt`
- Reduce `imgsz` to 416 or 320
- Enable GPU acceleration
- Reduce camera resolution

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLOv11 implementation
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework

## 📧 Contact

**Project Link**: https://github.com/Codersplaylist/detection-grasping

**Sanghamitra** - sanghamitrarajagopal.21@gmail.com

---

⭐ If you found this project helpful, please consider giving it a star!
