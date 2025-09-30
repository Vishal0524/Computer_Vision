"""
Concise YOLO Dataset EDA - Object Count, Distribution & Heatmap
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


class YOLODatasetAnalyzer:
    def __init__(self, dataset_path: str, output_dir: str = "eda_output"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.classes = self._load_classes()
        self.annotations_data = []
        
    def _load_classes(self) -> List[str]:
        classes_file = self.dataset_path / "classes.txt"
        if not classes_file.exists():
            raise FileNotFoundError(f"classes.txt not found in {self.dataset_path}")
        
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def _parse_yolo_annotation(self, txt_file: Path) -> List[Dict]:
        annotations = []
        if not txt_file.exists() or txt_file.stat().st_size == 0:
            return annotations
        
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:])
                    
                    annotations.append({
                        'class_name': self.classes[class_id] if class_id < len(self.classes) else f'unknown_{class_id}',
                        'center_x': center_x,
                        'center_y': center_y
                    })
                except (ValueError, IndexError):
                    continue
        
        return annotations
    
    def load_dataset(self) -> None:
        txt_files = [f for f in self.dataset_path.glob("*.txt") if f.name != "classes.txt"]
        print(f"Found {len(txt_files)} annotation files")
        
        for txt_file in txt_files:
            annotations = self._parse_yolo_annotation(txt_file)
            self.annotations_data.extend(annotations)
        
        print(f"Loaded {len(self.annotations_data)} annotations")
    
    def plot_visualizations(self, df: pd.DataFrame) -> None:
        """Generate three separate visualization files."""
        class_counts = df['class_name'].value_counts()
        
        # 1. Object Count by Class
        plt.figure(figsize=(14, 8))
        sns.barplot(data=pd.DataFrame({'class': class_counts.index, 'count': class_counts.values}),
                x='class', y='count')
        plt.title('Object Count by Class', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Objects')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'object_count.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Class Distribution Pie Chart
        plt.figure(figsize=(10, 10))
        threshold = len(df) * 0.02
        major_classes = class_counts[class_counts >= threshold]
        minor_classes_sum = class_counts[class_counts < threshold].sum()
        
        if minor_classes_sum > 0:
            plot_data = major_classes.copy()
            plot_data['Others'] = minor_classes_sum
        else:
            plot_data = major_classes
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
        plt.pie(plot_data.values, labels=plot_data.index, autopct='%1.1f%%',
            colors=colors, startangle=90, labeldistance=1.1)
        plt.title('Class Distribution (Percentage)', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Object Density Heatmap
        plt.figure(figsize=(10, 8))
        plt.hist2d(df['center_x'], df['center_y'], bins=25, cmap='Blues')
        plt.title('Object Density Heatmap', fontsize=16)
        plt.xlabel('Center X (Normalized)')
        plt.ylabel('Center Y (Normalized)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'density_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_analysis(self) -> None:
        print("Starting YOLO Dataset Analysis...")
        self.load_dataset()
        
        if not self.annotations_data:
            print("No annotations found!")
            return
        
        df = pd.DataFrame(self.annotations_data)
        self.plot_visualizations(df)
        
        # Summary stats
        class_counts = df['class_name'].value_counts()
        print(f"\nSummary:")
        print(f"Total Objects: {len(df)}")
        print(f"Classes: {len(self.classes)}")
        print(f"Top 3 classes: {', '.join(class_counts.head(3).index.tolist())}")
        print(f"Results saved in: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='YOLO Dataset EDA Analysis')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to YOLO dataset folder')
    parser.add_argument('--output_dir', type=str, default='eda_output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Dataset path does not exist: {args.dataset_path}")
        return
    
    analyzer = YOLODatasetAnalyzer(args.dataset_path, args.output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        dataset_path = input("Enter dataset folder path: ").strip()
        if os.path.exists(dataset_path):
            analyzer = YOLODatasetAnalyzer(dataset_path)
            analyzer.run_analysis()
        else:
            print("Invalid path!")
    else:
        main()