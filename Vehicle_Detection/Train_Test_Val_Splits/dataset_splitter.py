"""
YOLO Dataset Splitter - Train/Val/Test split with image-label correspondence
"""
import os
import shutil
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict


class YOLODatasetSplitter:
    def __init__(self, dataset_path: str, output_path: str, 
                 train_ratio: float = 0.7, val_ratio: float = 0.2, 
                 test_ratio: float = 0.1, random_state: int = 42):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        random.seed(random_state)
        
        # Supported image formats
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def find_image_label_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching image-label pairs."""
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.dataset_path.glob(f"*{ext}"))
            image_files.extend(self.dataset_path.glob(f"*{ext.upper()}"))
        
        pairs = []
        for img_path in image_files:
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                pairs.append((img_path, label_path))
            else:
                print(f"Warning: No label file for {img_path.name}")
        
        return pairs
    
    def create_directory_structure(self) -> None:
        """Create train/val/test directory structure."""
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                (self.output_path / split / subdir).mkdir(parents=True, exist_ok=True)
    
    def split_dataset(self, pairs: List[Tuple[Path, Path]]) -> Dict[str, List[Tuple[Path, Path]]]:
        """Split pairs into train/val/test sets."""
        random.shuffle(pairs)
        
        total = len(pairs)
        train_count = int(total * self.train_ratio)
        val_count = int(total * self.val_ratio)
        
        splits = {
            'train': pairs[:train_count],
            'val': pairs[train_count:train_count + val_count],
            'test': pairs[train_count + val_count:]
        }
        
        return splits
    
    def copy_files(self, splits: Dict[str, List[Tuple[Path, Path]]]) -> None:
        """Copy files to respective directories."""
        for split_name, pairs in splits.items():
            print(f"Copying {len(pairs)} pairs to {split_name}...")
            
            for img_path, label_path in pairs:
                # Copy image
                dst_img = self.output_path / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Copy label
                dst_label = self.output_path / split_name / 'labels' / label_path.name
                shutil.copy2(label_path, dst_label)
    
    def copy_classes_file(self) -> None:
        """Copy classes.txt to output directory if exists."""
        classes_file = self.dataset_path / 'classes.txt'
        if classes_file.exists():
            shutil.copy2(classes_file, self.output_path / 'classes.txt')
            print("Copied classes.txt")

    def _get_class_names(self) -> List[str]:
        """Helper to read class names from the generated classes.txt file."""
        classes_file = self.output_path / 'classes.txt'
        if not classes_file.exists():
            print("Error: classes.txt not found. Cannot generate data.yaml.")
            return []
        
        with open(classes_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        return names

    def generate_data_yaml(self) -> None:
        """
        Creates the data.yaml file required for YOLO training configuration.
        """
        class_names = self._get_class_names()
        if not class_names:
            return

        nc = len(class_names)
        
        # Format the class names into a YAML list string (e.g., 'class1', 'class2')
        names_yaml_list = ', '.join(f"'{name}'" for name in class_names)

        # Construct the YAML content
        # Using the base path is critical for relative dataset paths to work
        yaml_content = f"""# YOLOv8 dataset configuration
# The 'path' should point to the directory containing 'train', 'val', 'test' folders.
path: {self.output_path.resolve()} 

# Training/Validation/Test sets relative to 'path'
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {nc}

# Class names
names: [{names_yaml_list}]"""
        
        yaml_path = self.output_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Generated data.yaml file: {yaml_path}")
    
    def generate_summary(self, splits: Dict[str, List[Tuple[Path, Path]]]) -> None:
        """Generate and save split summary."""
        total = sum(len(pairs) for pairs in splits.values())
        
        summary = []
        summary.append("YOLO Dataset Split Summary")
        summary.append("=" * 30)
        summary.append(f"Total pairs: {total}")
        summary.append(f"Train: {len(splits['train'])} ({len(splits['train'])/total*100:.1f}%)")
        summary.append(f"Val: {len(splits['val'])} ({len(splits['val'])/total*100:.1f}%)")
        summary.append(f"Test: {len(splits['test'])} ({len(splits['test'])/total*100:.1f}%)")
        
        with open(self.output_path / 'split_summary.txt', 'w') as f:
            f.write('\n'.join(summary))
        
        print('\n'.join(summary))
    
    def split(self) -> None:
        """Execute the complete splitting process."""
        print(f"Splitting dataset from: {self.dataset_path}")
        print(f"Output directory: {self.output_path}")
        print(f"Ratios - Train: {self.train_ratio}, Val: {self.val_ratio}, Test: {self.test_ratio}")
        
        # Find image-label pairs
        pairs = self.find_image_label_pairs()
        if not pairs:
            raise ValueError("No image-label pairs found!")
        
        print(f"Found {len(pairs)} image-label pairs")
        self.create_directory_structure()
        splits = self.split_dataset(pairs)
        self.copy_files(splits)
        self.copy_classes_file()
        self.generate_data_yaml() 
        self.generate_summary(splits)
        
        print(f"\nDataset split completed! Output saved to: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description='Split YOLO dataset into train/val/test')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input dataset directory path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for split dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input directory does not exist: {args.input}")
        return
    
    splitter = YOLODatasetSplitter(
        dataset_path=args.input,
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )
    
    try:
        splitter.split()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        input_dir = input("Enter input dataset path: ").strip()
        output_dir = input("Enter output directory path: ").strip()
        
        if not os.path.exists(input_dir):
            print("Input directory does not exist!")
            exit(1)

        custom = input("Use custom ratios? (y/n, default: n): ").strip().lower()
        if custom == 'y':
            train = float(input("Train ratio (default 0.7): ") or 0.7)
            val = float(input("Val ratio (default 0.2): ") or 0.2)
            test = float(input("Test ratio (default 0.1): ") or 0.1)
        else:
            train, val, test = 0.7, 0.2, 0.1
        
        splitter = YOLODatasetSplitter(input_dir, output_dir, train, val, test)
        splitter.split()
    else:
        main()