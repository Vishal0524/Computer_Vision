import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Set


def scan_xml_files_for_classes(xml_dir: str) -> Set[str]:
    """Extract all unique class names from XML annotation files."""
    classes = set()
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                class_name = obj.find('name')
                if class_name is not None:
                    classes.add(class_name.text.strip())
        except ET.ParseError as e:
            print(f"Error parsing {xml_file}: {e}")
            continue
    
    return classes


def create_classes_file(classes: Set[str], output_dir: str) -> Dict[str, int]:
    """Create classes.txt file and return class name to ID mapping."""
    os.makedirs(output_dir, exist_ok=True)
    
    sorted_classes = sorted(list(classes))
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}
    
    classes_file = os.path.join(output_dir, 'classes.txt')
    with open(classes_file, 'w') as f:
        for cls_name in sorted_classes:
            f.write(f"{cls_name}\n")
    
    print(f"Created classes.txt with {len(sorted_classes)} classes")
    return class_to_id


def convert_bbox_to_yolo(bbox: Tuple[int, int, int, int], 
                        img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """Convert absolute bbox coordinates to YOLO format (normalized center, width, height)."""
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center coordinates and dimensions
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Normalize to 0-1 range
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    
    return center_x, center_y, width, height


def parse_xml_annotation(xml_path: str, class_to_id: Dict[str, int]) -> List[str]:
    """Parse XML file and return YOLO format annotation lines."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        if size is None:
            print(f"No size info in {xml_path}")
            return []
        
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        yolo_lines = []
        
        # Process each object
        for obj in root.findall('object'):
            class_name = obj.find('name').text.strip()
            
            if class_name not in class_to_id:
                print(f"Unknown class '{class_name}' in {xml_path}")
                continue
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Convert to YOLO format
            center_x, center_y, w, h = convert_bbox_to_yolo(
                (xmin, ymin, xmax, ymax), width, height
            )
            
            class_id = class_to_id[class_name]
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(yolo_line)
        
        return yolo_lines
        
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return []


def convert_xml_to_yolo(xml_dir: str, output_dir: str, class_to_id: Dict[str, int]) -> None:
    """Convert all XML files in directory to YOLO format."""
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    converted_count = 0
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        

        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_filename)
        
        yolo_lines = parse_xml_annotation(xml_path, class_to_id)
        
        if yolo_lines:
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            converted_count += 1
        else:
            # Create empty file if no valid annotations
            open(txt_path, 'w').close()
    
    print(f"Converted {converted_count}/{len(xml_files)} XML files to YOLO format")


def process_dataset(input_dir: str, output_dir: str) -> None:
    """Main processing function to convert XML dataset to YOLO format."""
    print(f"Processing dataset from: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Scan for unique classes
    print("Scanning XML files for unique classes...")
    classes = scan_xml_files_for_classes(input_dir)
    
    if not classes:
        print("No classes found in XML files!")
        return
    
    print(f"Found classes: {sorted(classes)}")
    
    # Step 2: Create classes.txt file
    class_to_id = create_classes_file(classes, output_dir)
    
    # Step 3: Convert XML files to YOLO format
    print("Converting XML annotations to YOLO format...")
    convert_xml_to_yolo(input_dir, output_dir, class_to_id)
    
    print("Dataset processing completed!")


def validate_yolo_dataset(labels_dir: str, classes_file: str) -> None:
    """Basic validation of converted YOLO dataset."""
    if not os.path.exists(classes_file):
        print("classes.txt not found!")
        return
    
    with open(classes_file, 'r') as f:
        num_classes = len(f.readlines())
    
    txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    total_annotations = 0
    
    for txt_file in txt_files:
        txt_path = os.path.join(labels_dir, txt_file)
        with open(txt_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            total_annotations += len(lines)
    
    print(f"Validation: {len(txt_files)} label files, {total_annotations} annotations, {num_classes} classes")