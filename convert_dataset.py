"""
Convert Pascal VOC XML annotations to YOLO format
Processes all XML files and creates YOLO-format text labels
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil


class XMLtoYOLOConverter:
    """Convert Pascal VOC XML annotations to YOLO format"""
    
    def __init__(self, classes):
        """
        Initialize converter
        
        Args:
            classes: List of class names in order
        """
        self.classes = classes
        self.class_to_id = {name: i for i, name in enumerate(classes)}
    
    def convert_bbox(self, size, box):
        """
        Convert bbox from (xmin, ymin, xmax, ymax) to YOLO format
        YOLO format: (x_center, y_center, width, height) normalized to [0, 1]
        
        Args:
            size: Image size as (width, height)
            box: Bounding box as (xmin, ymin, xmax, ymax)
            
        Returns:
            Tuple of (x_center, y_center, width, height) normalized
        """
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        
        # Calculate center
        x_center = (box[0] + box[2]) / 2.0
        y_center = (box[1] + box[3]) / 2.0
        
        # Calculate width and height
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        # Normalize
        x_center = x_center * dw
        y_center = y_center * dh
        width = width * dw
        height = height * dh
        
        return (x_center, y_center, width, height)
    
    def convert_xml_to_yolo(self, xml_path):
        """
        Convert single XML file to YOLO format
        
        Args:
            xml_path: Path to XML annotation file
            
        Returns:
            List of YOLO format lines (class_id x_center y_center width height)
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find('size')
        if size is None:
            # If size not in XML, try to get from image
            print(f"Warning: No size info in {xml_path}, skipping...")
            return None
            
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        # Handle images with 0 width/height
        if w == 0 or h == 0:
            # Try to get actual image dimensions
            img_path = xml_path.replace('.xml', '.jpg')
            if os.path.exists(img_path):
                from PIL import Image
                with Image.open(img_path) as img:
                    w, h = img.size
            else:
                print(f"Warning: Invalid dimensions in {xml_path}, skipping...")
                return None
        
        yolo_lines = []
        
        # Process each object
        for obj in root.iter('object'):
            class_name = obj.find('name').text
            
            if class_name not in self.class_to_id:
                print(f"Warning: Unknown class '{class_name}' in {xml_path}")
                continue
            
            class_id = self.class_to_id[class_name]
            
            # Get bounding box
            xmlbox = obj.find('bndbox')
            bbox = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymax').text)
            )
            
            # Convert to YOLO format
            yolo_bbox = self.convert_bbox((w, h), bbox)
            
            # Format: class_id x_center y_center width height
            yolo_line = f"{class_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}"
            yolo_lines.append(yolo_line)
        
        return yolo_lines
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all XML files in a directory
        
        Args:
            input_dir: Directory containing XML and JPG files
            output_dir: Directory to save YOLO format labels
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        images_dir = output_path / 'images'
        labels_dir = output_path / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all XML files
        xml_files = list(input_path.glob('*.xml'))
        
        print(f"\nProcessing {len(xml_files)} annotations from {input_dir}")
        
        converted_count = 0
        skipped_count = 0
        
        for xml_file in xml_files:
            # Convert XML to YOLO
            yolo_lines = self.convert_xml_to_yolo(str(xml_file))
            
            if yolo_lines is None:
                skipped_count += 1
                continue
            
            # Get corresponding image
            img_file = xml_file.with_suffix('.jpg')
            if not img_file.exists():
                print(f"Warning: Image not found for {xml_file.name}")
                skipped_count += 1
                continue
            
            # Copy image to output
            output_img = images_dir / img_file.name
            shutil.copy2(img_file, output_img)
            
            # Save YOLO label
            output_label = labels_dir / img_file.with_suffix('.txt').name
            with open(output_label, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            converted_count += 1
        
        print(f"✓ Converted: {converted_count}")
        print(f"✗ Skipped: {skipped_count}")
        
        return converted_count


def main():
    """Main conversion function"""
    print("="*60)
    print("XML to YOLO Converter")
    print("="*60)
    
    # Define classes (must be in correct order)
    classes = ['apple', 'banana', 'orange']
    
    # Create converter
    converter = XMLtoYOLOConverter(classes)
    
    # Define paths
    base_dir = Path(__file__).parent
    train_input = base_dir / 'test' / 'train_zip' / 'train'
    test_input = base_dir / 'test' / 'test_zip' / 'test'
    
    output_dir = base_dir / 'dataset'
    train_output = output_dir / 'train'
    val_output = output_dir / 'val'
    
    # Convert training data
    print("\n1. Converting training data...")
    train_count = converter.process_directory(train_input, train_output)
    
    # Convert test/validation data
    print("\n2. Converting validation data...")
    val_count = converter.process_directory(test_input, val_output)
    
    # Create classes.txt
    classes_file = output_dir / 'classes.txt'
    with open(classes_file, 'w') as f:
        f.write('\n'.join(classes))
    print(f"\n✓ Created classes file: {classes_file}")
    
    # Create data.yaml for YOLO
    data_yaml = output_dir / 'data.yaml'
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_dir.absolute()}
train: train/images
val: val/images

# Number of classes
nc: {len(classes)}

# Class names
names: {classes}
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created data.yaml: {data_yaml}")
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"\nDataset summary:")
    print(f"  Training images: {train_count}")
    print(f"  Validation images: {val_count}")
    print(f"  Classes: {', '.join(classes)}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/  ({train_count} images)")
    print(f"    │   └── labels/  ({train_count} labels)")
    print(f"    ├── val/")
    print(f"    │   ├── images/  ({val_count} images)")
    print(f"    │   └── labels/  ({val_count} labels)")
    print(f"    ├── data.yaml")
    print(f"    └── classes.txt")
    print(f"\nReady for training!")


if __name__ == "__main__":
    main()
