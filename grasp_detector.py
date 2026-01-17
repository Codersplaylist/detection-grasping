"""
Grasp Detection Module
Calculates optimal grasp coordinates for detected objects
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional


class GraspDetector:
    """
    Simplified grasp detection based on YOLO bounding boxes.
    Calculates grasp points at the center of detected objects.
    """
    
    def __init__(self, marker_color=(255, 0, 255), marker_size=10):
        """
        Initialize grasp detector
        
        Args:
            marker_color: RGB color tuple for grasp point marker
            marker_size: Size of grasp point marker
        """
        self.marker_color = marker_color
        self.marker_size = marker_size
    
    def calculate_grasp_point(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Calculate grasp point from bounding box.
        Uses the center of the bounding box as the grasp point.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Tuple of (center_x, center_y) coordinates
        """
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return center_x, center_y
    
    def calculate_grasp_quality(self, bbox: Tuple[int, int, int, int], 
                               confidence: float) -> float:
        """
        Calculate grasp quality score based on bbox size and detection confidence.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            confidence: Detection confidence score
            
        Returns:
            Grasp quality score (0.0 - 1.0)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Normalize area (assuming max reasonable object size is 500x500)
        normalized_area = min(area / (500 * 500), 1.0)
        
        # Combine area and confidence (larger objects with high confidence = better grasp)
        # Weight: 70% confidence, 30% size
        quality = (0.7 * confidence) + (0.3 * normalized_area)
        
        return quality
    
    def get_grasp_pose(self, bbox: Tuple[int, int, int, int], 
                       class_name: str) -> dict:
        """
        Get complete grasp pose information including position and orientation.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            class_name: Name of detected class
            
        Returns:
            Dictionary containing grasp pose information
        """
        center_x, center_y = self.calculate_grasp_point(bbox)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Calculate approach angle based on object orientation
        # For elongated objects (banana), grasp along the major axis
        if height > width * 1.5:  # Vertical orientation
            approach_angle = 90  # degrees
        elif width > height * 1.5:  # Horizontal orientation
            approach_angle = 0  # degrees
        else:  # Square-ish object
            approach_angle = 45  # degrees (diagonal approach)
        
        return {
            'position': (center_x, center_y),
            'approach_angle': approach_angle,
            'gripper_width': min(width, height) * 0.8,  # 80% of smaller dimension
            'class': class_name,
            'bbox_size': (width, height)
        }
    
    def draw_grasp_point(self, frame: np.ndarray, 
                        grasp_point: Tuple[int, int],
                        grasp_pose: Optional[dict] = None) -> np.ndarray:
        """
        Draw grasp point and orientation on frame.
        
        Args:
            frame: Image frame
            grasp_point: (x, y) coordinates of grasp point
            grasp_pose: Optional grasp pose dictionary for additional visualization
            
        Returns:
            Frame with grasp visualization
        """
        x, y = grasp_point
        
        # Draw crosshair at grasp point
        cv2.drawMarker(frame, (x, y), self.marker_color, 
                      cv2.MARKER_CROSS, self.marker_size, 2)
        
        # Draw circle around grasp point
        cv2.circle(frame, (x, y), self.marker_size // 2, 
                  self.marker_color, 2)
        
        # If grasp pose available, draw approach direction
        if grasp_pose is not None:
            angle = np.radians(grasp_pose['approach_angle'])
            length = 30
            end_x = int(x + length * np.cos(angle))
            end_y = int(y + length * np.sin(angle))
            cv2.arrowedLine(frame, (x, y), (end_x, end_y), 
                          self.marker_color, 2, tipLength=0.3)
        
        return frame
    
    def get_multiple_grasps(self, detections: List[dict]) -> List[dict]:
        """
        Calculate grasp poses for multiple detected objects and rank them.
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'confidence', 'class'
            
        Returns:
            List of grasp poses sorted by quality (best first)
        """
        grasps = []
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Calculate grasp pose
            grasp_pose = self.get_grasp_pose(bbox, class_name)
            grasp_pose['confidence'] = confidence
            
            # Calculate quality score
            quality = self.calculate_grasp_quality(bbox, confidence)
            grasp_pose['quality'] = quality
            
            grasps.append(grasp_pose)
        
        # Sort by quality (highest first)
        grasps.sort(key=lambda g: g['quality'], reverse=True)
        
        return grasps


def format_grasp_coordinates(grasp_pose: dict) -> str:
    """
    Format grasp coordinates for display/logging.
    
    Args:
        grasp_pose: Grasp pose dictionary
        
    Returns:
        Formatted string
    """
    x, y = grasp_pose['position']
    angle = grasp_pose['approach_angle']
    width = grasp_pose['gripper_width']
    quality = grasp_pose.get('quality', 0.0)
    
    return (f"Grasp: [{grasp_pose['class']}] "
            f"Position=({x}, {y}), "
            f"Angle={angle}°, "
            f"Width={width:.1f}px, "
            f"Quality={quality:.2f}")
