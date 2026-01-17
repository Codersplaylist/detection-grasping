"""
YOLO Object Detection Camera Application with Grasp Detection
Real-time object detection using trained YOLO model with grasp coordinate calculation
"""
import cv2
import numpy as np
import time
import yaml
import sys
import os
from pathlib import Path
from ultralytics import YOLO
from grasp_detector import GraspDetector, format_grasp_coordinates


class YOLOCameraApp:
    """Main application class for YOLO-based object detection and grasp prediction"""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the application
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize camera
        self.camera = None
        self.initialize_camera()
        
        # Initialize YOLO model
        self.model = None
        self.load_model()
        
        # Initialize grasp detector
        marker_color = tuple(self.config['grasp']['marker_color'])
        marker_size = self.config['grasp']['marker_size']
        self.grasp_detector = GraspDetector(marker_color, marker_size)
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Class colors
        self.class_colors = self.config['display']['class_colors']
    
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"✗ Configuration file not found: {config_path}")
            print("Using default configuration...")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration if config file is missing"""
        return {
            'model': {
                'weights': 'models/my_model.pt',
                'imgsz': 640,
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'device': 'cpu'
            },
            'camera': {
                'device_id': 0,
                'width': 1920,
                'height': 1080,
                'frame_skip': 1
            },
            'display': {
                'show_boxes': True,
                'show_labels': True,
                'show_confidence': True,
                'show_fps': True,
                'box_thickness': 2,
                'font_scale': 0.6,
                'class_colors': {
                    'apple': [0, 0, 255],
                    'banana': [0, 255, 255],
                    'orange': [0, 165, 255]
                }
            },
            'grasp': {
                'enabled': True,
                'show_grasp_point': True,
                'marker_size': 10,
                'marker_color': [255, 0, 255],
                'print_coordinates': True
            },
            'classes': ['apple', 'banana', 'orange']
        }
    
    def initialize_camera(self):
        """Initialize camera with configured settings"""
        device_id = self.config['camera']['device_id']
        self.camera = cv2.VideoCapture(device_id)
        
        if not self.camera.isOpened():
            print(f"✗ Failed to open camera {device_id}")
            sys.exit(1)
        
        # Set resolution
        width = self.config['camera']['width']
        height = self.config['camera']['height']
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print(f"✓ Camera initialized: {width}x{height}")
    
    def load_model(self):
        """Load YOLO model"""
        model_path = self.config['model']['weights']
        
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            print("\nPlease ensure you have:")
            print("1. Trained your YOLO model in Colab")
            print("2. Downloaded the model weights (my_model.pt)")
            print("3. Placed the file in the correct location")
            print(f"   Expected location: {model_path}")
            print("\nYou can update the path in config.yaml")
            sys.exit(1)
        
        try:
            # Determine device
            device = self.config['model'].get('device', 'cpu')
            if device == 'cpu':
                # Auto-detect best available device
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available():
                    device = 'mps'
            
            self.model = YOLO(model_path)
            print(f"✓ YOLO model loaded from {model_path}")
            print(f"✓ Using device: {device}")
            
            # Move model to device
            self.model.to(device)
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            sys.exit(1)
    
    def detect_objects(self, frame):
        """
        Run YOLO detection on frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detections with bbox, class, and confidence
        """
        imgsz = self.config['model']['imgsz']
        conf = self.config['model']['conf_threshold']
        iou = self.config['model']['iou_threshold']
        
        # Run inference
        results = self.model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class': class_name,
                    'class_id': class_id
                })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with visualizations
        """
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            x1, y1, x2, y2 = bbox
            
            # Get color for this class
            color = self.class_colors.get(class_name, [255, 255, 255])
            color = tuple(color)
            
            # Draw bounding box
            if self.config['display']['show_boxes']:
                thickness = self.config['display']['box_thickness']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label and confidence
            if self.config['display']['show_labels']:
                label = class_name
                if self.config['display']['show_confidence']:
                    label += f" {confidence:.2f}"
                
                # Calculate label background size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = self.config['display']['font_scale']
                thickness = 1
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw label background
                cv2.rectangle(frame, 
                            (x1, y1 - label_height - 10), 
                            (x1 + label_width + 10, y1),
                            color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                          font, font_scale, (0, 0, 0), thickness)
        
        return frame
    
    def process_grasps(self, frame, detections):
        """
        Calculate and visualize grasp points
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with grasp visualizations
        """
        if not self.config['grasp']['enabled']:
            return frame
        
        # Calculate grasp poses for all detections
        grasps = self.grasp_detector.get_multiple_grasps(detections)
        
        # Visualize grasp points
        if self.config['grasp']['show_grasp_point']:
            for grasp in grasps:
                grasp_point = grasp['position']
                frame = self.grasp_detector.draw_grasp_point(frame, grasp_point, grasp)
        
        # Print coordinates to console
        if self.config['grasp']['print_coordinates'] and len(grasps) > 0:
            print("\n" + "="*60)
            print(f"Detected {len(grasps)} object(s) - Grasp coordinates:")
            for i, grasp in enumerate(grasps, 1):
                print(f"{i}. {format_grasp_coordinates(grasp)}")
            print("="*60)
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def draw_fps(self, frame):
        """Draw FPS counter on frame"""
        if self.config['display']['show_fps']:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("YOLO Object Detection with Grasp Prediction")
        print("="*60)
        print("\nControls:")
        print("  ESC - Exit application")
        print("  D - Toggle detection on/off")
        print("  G - Toggle grasp visualization")
        print("  C - Clear console output")
        print("\n" + "="*60 + "\n")
        
        detection_enabled = True
        window_name = "YOLO Object Detection"
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    print("✗ Failed to read frame from camera")
                    break
                
                # Run detection if enabled
                if detection_enabled:
                    detections = self.detect_objects(frame)
                    frame = self.draw_detections(frame, detections)
                    frame = self.process_grasps(frame, detections)
                
                # Update and draw FPS
                self.update_fps()
                frame = self.draw_fps(frame)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\nExiting application...")
                    break
                elif key == ord('d') or key == ord('D'):
                    detection_enabled = not detection_enabled
                    status = "enabled" if detection_enabled else "disabled"
                    print(f"Detection {status}")
                elif key == ord('g') or key == ord('G'):
                    self.config['grasp']['show_grasp_point'] = \
                        not self.config['grasp']['show_grasp_point']
                    status = "enabled" if self.config['grasp']['show_grasp_point'] else "disabled"
                    print(f"Grasp visualization {status}")
                elif key == ord('c') or key == ord('C'):
                    os.system('clear' if os.name != 'nt' else 'cls')
                    print("Console cleared")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        print("\n✓ Cleanup complete")


def main():
    """Main entry point"""
    # Check if config file exists
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create and run application
    app = YOLOCameraApp(config_path)
    app.run()


if __name__ == "__main__":
    main()
