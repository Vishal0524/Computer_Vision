"""
YOLO Dataset Class Filter

This script filters a YOLO dataset to keep only specified classes,
updates class indices, and saves the filtered dataset to a new location.
"""

import os
import shutil
from pathlib import Path
from typing import List, Set, Dict, Tuple
import logging


class YOLODatasetFilter:
    """Filter YOLO dataset to keep only selected classes."""
    
    def __init__(self, dataset_path: str, selected_classes: List[str], 
                 output_path: str):
        """
        Initialize the dataset filter.
        
        Args:
            dataset_path: Path to the dataset folder containing images, txt files and classes.txt
            selected_classes: List of class names to keep
            output_path: Path where filtered dataset will be saved
        """
        self.dataset_path = Path(dataset_path)
        self.classes_file = self.dataset_path / 'classes.txt'
        self.selected_classes = selected_classes
        self.output_path = Path(output_path)
    
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.original_classes: List[str] = []
        self.class_mapping: Dict[int, int] = {}
        self.selected_class_indices: Set[int] = set()
        
    def load_classes(self) -> None:
        """Load class names from classes.txt file."""
        try:
            with open(self.classes_file, 'r', encoding='utf-8') as f:
                self.original_classes = [line.strip() for line in f.readlines()]
            
            self.logger.info(f"Loaded {len(self.original_classes)} classes from {self.classes_file}")
            new_index = 0
            for old_index, class_name in enumerate(self.original_classes):
                if class_name in self.selected_classes:
                    self.selected_class_indices.add(old_index)
                    self.class_mapping[old_index] = new_index
                    new_index += 1
            
            self.logger.info(f"Selected {len(self.selected_class_indices)} classes for filtering")
            
        except FileNotFoundError:
            self.logger.error(f"Classes file not found: {self.classes_file}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading classes: {e}")
            raise
    
    def create_output_structure(self) -> None:
        """Create output directory structure."""
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def filter_annotation_file(self, label_file: Path, output_file: Path) -> bool:
        """
        Filter a single annotation file to keep only selected classes.
        
        Args:
            label_file: Path to input label file
            output_file: Path to output label file
            
        Returns:
            bool: True if file has any valid annotations, False otherwise
        """
        try:
            filtered_lines = []
            
            with open(label_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) < 5:
                            self.logger.warning(f"Invalid annotation format in {label_file}:{line_num}")
                            continue
                        
                        old_class_id = int(parts[0])
            
                        if old_class_id in self.selected_class_indices:
                            new_class_id = self.class_mapping[old_class_id]
                            parts[0] = str(new_class_id)
                            filtered_lines.append(' '.join(parts))
                    
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Error parsing line {line_num} in {label_file}: {e}")
                        continue
   
            if filtered_lines:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(filtered_lines) + '\n')
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing {label_file}: {e}")
            return False
    
    def copy_image_file(self, image_file: Path, output_file: Path) -> None:
        """Copy image file to output location."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_file, output_file)
        except Exception as e:
            self.logger.error(f"Error copying image {image_file}: {e}")
    
    def get_image_extensions(self) -> Tuple[str, ...]:
        """Get common image file extensions."""
        return ('.jpg','.JPG', '.JPEG','.jpeg', '.png', '.PNG','.bmp', '.tiff', '.tif')
    
    def find_corresponding_image(self, label_file: Path) -> Path:
        """Find corresponding image file for a label file in the same directory."""
        label_stem = label_file.stem
        
        for ext in self.get_image_extensions():
            image_file = self.dataset_path / f"{label_stem}{ext}"
            if image_file.exists():
                return image_file
        
        return None
    
    def process_dataset(self) -> None:
        """Process the entire dataset."""
        self.logger.info("Starting dataset filtering process...")
        
        processed_files = 0
        kept_files = 0
        

        for label_file in self.dataset_path.glob('*.txt'):
            if label_file.name == 'classes.txt':
                continue
            
            output_label_file = self.output_path / label_file.name
            has_valid_annotations = self.filter_annotation_file(label_file, output_label_file)
            processed_files += 1
            
            if has_valid_annotations:
                kept_files += 1
                corresponding_image = self.find_corresponding_image(label_file)
                
                if corresponding_image:
                    output_image_file = self.output_path / corresponding_image.name
                    self.copy_image_file(corresponding_image, output_image_file)
                else:
                    self.logger.warning(f"No corresponding image found for {label_file}")

            if processed_files % 100 == 0:
                self.logger.info(f"Processed {processed_files} files, kept {kept_files}")
        
        self.logger.info(f"Dataset filtering complete. Processed {processed_files} files, "
                        f"kept {kept_files} files with valid annotations.")
    
    def create_filtered_classes_file(self) -> None:
        """Create new classes.txt file with only selected classes."""
        filtered_classes = [cls for cls in self.original_classes if cls in self.selected_classes]
        
        output_classes_file = self.output_path / 'classes.txt'
        with open(output_classes_file, 'w', encoding='utf-8') as f:
            for class_name in filtered_classes:
                f.write(f"{class_name}\n")
        
        self.logger.info(f"Created filtered classes file: {output_classes_file}")
        self.logger.info(f"New class mapping: {dict(zip(filtered_classes, range(len(filtered_classes))))}")
    
    def run(self) -> None:
        """Run the complete filtering process."""
        try:
            self.logger.info("Starting YOLO dataset filtering...")
            self.load_classes()
            
            invalid_classes = set(self.selected_classes) - set(self.original_classes)
            if invalid_classes:
                self.logger.error(f"Invalid class names: {invalid_classes}")
                return

            self.create_output_structure()
            self.process_dataset()
            self.create_filtered_classes_file()
            
            self.logger.info("Dataset filtering completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error during dataset filtering: {e}")
            raise


def main():
    """Main function to run the dataset filter."""

    DATASET_PATH = "/home/katomaran/Downloads/vishal/archive/FINAL"  
    OUTPUT_PATH = "/home/katomaran/Downloads/vishal/archive/Filter_Final"  
    

    SELECTED_CLASSES = [
        "car",
        "rickshaw", 
        "bus",
        "three wheelers (CNG)"
    ]
    

    filter_instance = YOLODatasetFilter(
        dataset_path=DATASET_PATH,
        selected_classes=SELECTED_CLASSES,
        output_path=OUTPUT_PATH
    )
    
    filter_instance.run()


if __name__ == "__main__":
    main()
