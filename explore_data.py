import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_sample(img_path, annotations, save_path=None):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create overlay for annotations
    overlay = img_rgb.copy()

    for ann in annotations:
        # Get polygon points
        segmentation = ann['segmentation'][0]
        points = np.array(segmentation).reshape(-1, 2).astype(np.int32)

        # Draw filled polygon
        cv2.fillPoly(overlay, [points], color=(255, 0, 0))

        # Draw polygon outline
        cv2.polylines(img_rgb, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Blend overlay with original image
    result = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

    # Add text with area info
    for i, ann in enumerate(annotations):
        bbox = ann['bbox']
        x, y = int(bbox[0]), int(bbox[1])
        area = ann['area']
        cv2.putText(result, f"Lesion {i+1}: {area:.0f}px",
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (255, 255, 0), 1)

    if save_path:
        plt.figure(figsize=(8, 12))
        plt.imshow(result)
        plt.title(f"Image: {os.path.basename(img_path)}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return result

def analyze_dataset(json_path, img_dir):
    data = load_annotations(json_path)

    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    # Basic stats
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])

    print(f"\nTotal Images: {num_images}")
    print(f"Total Lesion Annotations: {num_annotations}")
    print(f"Average Lesions per Image: {num_annotations / num_images:.2f}")

    # Images without annotations
    image_ids_with_ann = set(ann['image_id'] for ann in data['annotations'])
    images_without_ann = num_images - len(image_ids_with_ann)
    print(f"Images WITHOUT lesions: {images_without_ann}")
    print(f"Images WITH lesions: {len(image_ids_with_ann)}")

    # Lesion area statistics
    areas = [ann['area'] for ann in data['annotations']]
    print(f"\nLesion Area Statistics (pixels):")
    print(f"  Min: {min(areas):.0f}")
    print(f"  Max: {max(areas):.0f}")
    print(f"  Mean: {np.mean(areas):.0f}")
    print(f"  Median: {np.median(areas):.0f}")
    print(f"  Std: {np.std(areas):.0f}")

    # Image size statistics
    widths = [img['width'] for img in data['images']]
    heights = [img['height'] for img in data['images']]
    print(f"\nImage Size Statistics:")
    print(f"  Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.0f}")
    print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.0f}")

    # Create mapping for visualization
    img_id_to_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_annotations:
            img_id_to_annotations[img_id] = []
        img_id_to_annotations[img_id].append(ann)

    # Visualize samples
    print(f"\n{'=' * 60}")
    print("Generating visualizations...")

    os.makedirs('visualizations', exist_ok=True)

    # Sample images with different lesion counts
    sample_counts = [1, 2]  # Images with 1 and 2+ lesions
    samples_visualized = 0

    for img_info in data['images'][:20]:  # Check first 20 images
        img_id = img_info['id']
        if img_id in img_id_to_annotations:
            annotations = img_id_to_annotations[img_id]
            lesion_count = len(annotations)

            if lesion_count in sample_counts or samples_visualized < 5:
                img_path = os.path.join(img_dir, img_info['file_name'])
                save_path = f"visualizations/sample_{img_info['file_name']}"

                if os.path.exists(img_path):
                    visualize_sample(img_path, annotations, save_path)
                    print(f"  Saved: {save_path} (Lesions: {lesion_count})")
                    samples_visualized += 1

                    if lesion_count in sample_counts:
                        sample_counts.remove(lesion_count)

            if samples_visualized >= 10:
                break

    print(f"\nVisualizations saved to 'visualizations/' folder")
    print("=" * 60)

    return data, img_id_to_annotations

if __name__ == "__main__":
    json_path = "ann_images/annotations/instances_default.json"
    img_dir = "ann_images"

    data, img_to_ann = analyze_dataset(json_path, img_dir)
