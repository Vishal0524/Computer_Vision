"""
Main script to convert XML annotations to YOLO format.
"""

import os
from dataset_processing import process_dataset, validate_yolo_dataset


def main():
    """Main function to run dataset conversion."""
    

    input_directory = "/home/katomaran/Downloads/vishal/archive/train/Final_Train_Dataset"
    output_directory = "./yolo_dataset"
    input_directory = os.path.expanduser(input_directory)
    

    if not os.path.exists(input_directory):
        print(f"Input directory not found: {input_directory}")
        print("Please update the input_directory path in main.py")
        return
    
    xml_files = [f for f in os.listdir(input_directory) if f.endswith('.xml')]
    if not xml_files:
        print(f"No XML files found in: {input_directory}")
        return
    
    print(f"Found {len(xml_files)} XML files to process")
    

    try:
        process_dataset(input_directory, output_directory)
        labels_dir = os.path.join(output_directory, 'labels')
        classes_file = os.path.join(output_directory, 'classes.txt')
        validate_yolo_dataset(labels_dir, classes_file)
        
        print(f"\nOutput files:")
        print(f"- Classes: {classes_file}")
        print(f"- Labels: {labels_dir}")
        
    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main()