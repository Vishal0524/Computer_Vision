import os
from typing import List, Set

def validate_image_labels(folder_path: str):
    """
    Checks if all image files (.jpg) in a folder have a matching YOLO label file (.txt).
    """
    print(f"--- Checking Label Correspondence in: {folder_path} ---")

    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return


    all_files: List[str] = os.listdir(folder_path)
    image_names: Set[str] = set()
    label_names: Set[str] = set()

    for file_name in all_files:
        base_name, ext = os.path.splitext(file_name)
        
        if ext.lower() in ('.jpg', '.jpeg', '.png'):
            image_names.add(base_name)
        elif ext.lower() == '.txt':
            label_names.add(base_name)
    
    # Missing labels: Image exists, but no corresponding .txt file
    missing_labels: Set[str] = image_names.difference(label_names)
    # Orphan labels: .txt file exists, but no corresponding image file
    orphan_labels: Set[str] = label_names.difference(image_names)


    if not image_names:
        print("No image files (.jpg, .jpeg, .png) found.")
    elif not missing_labels and not orphan_labels:
        print(f"Success! All {len(image_names)} images have corresponding label files.")
    else:
        print("\n DATASET INCONSISTENCIES FOUND:")
        
        if missing_labels:
            print(f"\n- MISSING LABEL FILES ({len(missing_labels)}):")
            for name in sorted(list(missing_labels)):
                print(f"  > {name}.jpg is missing a {name}.txt file.")
                
        if orphan_labels:
            print(f"\n- ORPHAN LABEL FILES ({len(orphan_labels)}):")
            for name in sorted(list(orphan_labels)):
                print(f"  > {name}.txt is missing a corresponding image file.")


folder_to_check = "/home/katomaran/Downloads/vishal/archive/FINAL"
validate_image_labels(folder_to_check)