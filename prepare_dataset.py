import json
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def coco_to_yolo_seg(coco_ann, img_width, img_height):
    """Convert COCO polygon annotation to YOLO segmentation format"""
    segmentation = coco_ann['segmentation'][0]
    points = np.array(segmentation).reshape(-1, 2)

    # Normalize coordinates
    points[:, 0] /= img_width
    points[:, 1] /= img_height

    # Flatten to YOLO format: [class_id, x1, y1, x2, y2, ..., xn, yn]
    yolo_seg = points.flatten().tolist()

    return yolo_seg

def prepare_yolo_dataset(coco_json_path, img_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    """
    Convert COCO format to YOLO segmentation format and split into train/val/test
    """
    print("Loading COCO annotations...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Create image_id to annotations mapping
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    # Create image info mapping
    img_id_to_info = {img['id']: img for img in coco_data['images']}

    # Get all image IDs
    all_img_ids = list(img_id_to_info.keys())

    # Split dataset
    train_ids, temp_ids = train_test_split(all_img_ids, train_size=train_ratio, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, train_size=val_ratio/(1-train_ratio), random_state=42)

    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    print(f"\nDataset split:")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val:   {len(val_ids)} images")
    print(f"  Test:  {len(test_ids)} images")

    # Process each split
    for split_name, img_ids in splits.items():
        print(f"\nProcessing {split_name} split...")

        for img_id in img_ids:
            img_info = img_id_to_info[img_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            # Source and destination paths
            src_img_path = os.path.join(img_dir, img_filename)
            dst_img_path = os.path.join(output_dir, 'images', split_name, img_filename)

            # Copy image
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"  Warning: Image not found: {src_img_path}")
                continue

            # Create label file
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            label_path = os.path.join(output_dir, 'labels', split_name, label_filename)

            # Convert annotations to YOLO format
            yolo_lines = []
            if img_id in img_id_to_anns:
                for ann in img_id_to_anns[img_id]:
                    yolo_seg = coco_to_yolo_seg(ann, img_width, img_height)
                    # Class 0 for lesion_area
                    line = f"0 " + " ".join([f"{coord:.6f}" for coord in yolo_seg])
                    yolo_lines.append(line)

            # Write label file (even if empty - for images without lesions)
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))

        print(f"  Processed {len(img_ids)} images for {split_name}")

    # Create data.yaml for YOLOv8
    yaml_content = f"""# Anthracnose Lesion Detection Dataset
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: lesion_area

# Number of classes
nc: 1
"""

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ Dataset prepared successfully!")
    print(f"✓ data.yaml created at: {yaml_path}")
    print(f"✓ Ready for YOLOv8 training")

    return yaml_path

if __name__ == "__main__":
    coco_json = "/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_1/annotations/instances_default.json"
    images_dir = "/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_1/images"
    output_dir = "/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_1/dataset_yolo"

    yaml_path = prepare_yolo_dataset(coco_json, images_dir, output_dir)
    print(f"\nTo train: python train_model.py")
