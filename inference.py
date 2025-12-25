from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np


MODEL_PATH = '/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_1/model_trained/anthracnose_model/weights/best.pt'  # Path to trained model
INPUT_FOLDER = 'test-folder-2'  # Folder containing images to test
OUTPUT_FOLDER = 'test-folder-2/predictions_0.5'  # Where to save prediction results
CONFIDENCE = 0.5  # Confidence threshold (0-1)



def run_inference(input_folder, output_folder, model_path, conf_threshold=0.25):
    """
    Run inference on all images in input folder and save to output folder
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to save prediction results
        model_path: Path to trained YOLO model
        conf_threshold: Confidence threshold for detections
    """
    
    print("=" * 70)
    print("YOLO INFERENCE - BATCH PREDICTION")
    print("=" * 70)
    print(f"Model:       {model_path}")
    print(f"Input:       {input_folder}")
    print(f"Output:      {output_folder}")
    print(f"Confidence:  {conf_threshold}")
    print("=" * 70)
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f" Error: Input folder not found: {input_folder}")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f" Error: Model not found: {model_path}")
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model = YOLO(model_path)
    print("✓ Model loaded successfully")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
    input_path = Path(input_folder)
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    if len(image_files) == 0:
        print(f" No images found in {input_folder}")
        return
    
    print(f"\nFound {len(image_files)} images")
    print("\nProcessing images...")
    print("-" * 70)
    
    # Run predictions
    results_list = model.predict(
        source=input_folder,
        conf=conf_threshold,
        save=True,
        project=output_folder,
        name='results',
        show_labels=True,
        show_conf=True,
        line_width=2,
        stream=True
    )
    
    # Process and display results
    total_detections = 0
    detection_summary = []
    
    for i, result in enumerate(results_list, 1):
        num_detections = len(result.boxes)
        total_detections += num_detections
        img_name = Path(result.path).name
        
        detection_summary.append({
            'image': img_name,
            'detections': num_detections
        })
        
        print(f"[{i}/{len(image_files)}] {img_name:40s} → {num_detections} lesions detected")
    
    # Print summary
    print("-" * 70)
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {len(image_files)}")
    print(f"Total lesions detected: {total_detections}")
    print(f"Average per image:      {total_detections/len(image_files):.2f}")
    print(f"\nResults saved to: {output_folder}/results/")
    print("=" * 70)
    
    return detection_summary


def run_inference_with_custom_viz(input_folder, output_folder, model_path, conf_threshold=0.25):
    """
    Run inference with custom colored visualization - side by side comparison
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to save prediction results
        model_path: Path to trained YOLO model
        conf_threshold: Confidence threshold for detections
    """
    
    print("=" * 70)
    print("YOLO INFERENCE - SIDE-BY-SIDE VISUALIZATION")
    print("=" * 70)
    print(f"Model:       {model_path}")
    print(f"Input:       {input_folder}")
    print(f"Output:      {output_folder}")
    print(f"Confidence:  {conf_threshold}")
    print("=" * 70)
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f" Error: Input folder not found: {input_folder}")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f" Error: Model not found: {model_path}")
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model = YOLO(model_path)
    print("✓ Model loaded successfully")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
    input_path = Path(input_folder)
    image_files = sorted([f for f in input_path.iterdir() 
                         if f.is_file() and f.suffix in image_extensions])
    
    if len(image_files) == 0:
        print(f" No images found in {input_folder}")
        return
    
    print(f"\nFound {len(image_files)} images")
    print("\nProcessing images...")
    print("-" * 70)
    
    # Define colors for lesions
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]
    
    total_detections = 0
    
    for i, img_path in enumerate(image_files, 1):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠ Warning: Could not read {img_path.name}, skipping...")
            continue
            
        original = image.copy()
        h, w = original.shape[:2]
        
        # Run inference
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False
        )[0]
        
        num_detections = len(results.boxes)
        total_detections += num_detections
        
        # Create annotated version
        annotated = original.copy()
        
        # Draw results
        if results.masks is not None and len(results.masks) > 0:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.data.cpu().numpy()
            
            overlay = annotated.copy()
            
            for j, (mask, box) in enumerate(zip(masks, boxes)):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (w, h))
                mask_bool = mask_resized > 0.5
                
                # Get color
                color = colors[j % len(colors)]
                
                # Apply colored mask
                overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color) * 0.5
                
                # Draw bounding box
                x1, y1, x2, y2, conf, cls = box
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                
                # Add label with background
                label = f"Lesion #{j+1}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (int(x1), int(y1) - label_size[1] - 12),
                            (int(x1) + label_size[0] + 10, int(y1)), color, -1)
                cv2.putText(overlay, label, (int(x1) + 5, int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Blend
            annotated = cv2.addWeighted(original, 0.4, overlay, 0.6, 0)
            
            # Add summary text on annotated image
            summary_text = f"Detected: {len(masks)} lesions"
            text_size, _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(annotated, (5, 5), (text_size[0] + 20, 50), (0, 0, 0), -1)
            cv2.putText(annotated, summary_text, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            # No detections
            cv2.rectangle(annotated, (5, 5), (350, 50), (0, 0, 0), -1)
            cv2.putText(annotated, "No lesions detected", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Resize images to same height if needed (for better side-by-side)
        target_height = 800
        if h > target_height:
            scale = target_height / h
            new_w = int(w * scale)
            original_resized = cv2.resize(original, (new_w, target_height))
            annotated_resized = cv2.resize(annotated, (new_w, target_height))
        else:
            original_resized = original
            annotated_resized = annotated
        
        # Add labels
        label_height = 50
        h_resized, w_resized = original_resized.shape[:2]
        
        # Create label for original
        original_label = np.zeros((label_height, w_resized, 3), dtype=np.uint8)
        cv2.putText(original_label, "ORIGINAL", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Create label for prediction
        prediction_label = np.zeros((label_height, w_resized, 3), dtype=np.uint8)
        cv2.putText(prediction_label, "PREDICTION", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Stack labels with images
        original_with_label = np.vstack([original_label, original_resized])
        annotated_with_label = np.vstack([prediction_label, annotated_resized])
        
        # Create side-by-side comparison with gap
        gap = 20
        gap_img = np.ones((h_resized + label_height, gap, 3), dtype=np.uint8) * 50
        side_by_side = np.hstack([original_with_label, gap_img, annotated_with_label])
        
        # Add filename at bottom
        footer_height = 40
        footer = np.zeros((footer_height, side_by_side.shape[1], 3), dtype=np.uint8)
        filename_text = f"File: {img_path.name}"
        cv2.putText(footer, filename_text, (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        side_by_side = np.vstack([side_by_side, footer])
        
        # Save result
        output_path = os.path.join(output_folder, f"comparison_{img_path.name}")
        cv2.imwrite(output_path, side_by_side)
        
        print(f"[{i}/{len(image_files)}] {img_path.name:40s} → {num_detections} lesions")
    
    # Print summary
    print("-" * 70)
    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {len(image_files)}")
    print(f"Total lesions detected: {total_detections}")
    print(f"Average per image:      {total_detections/len(image_files):.2f}")
    print(f"\nSide-by-side results saved to: {output_folder}/")
    print("=" * 70)


if __name__ == "__main__":
    # Choose inference mode:
    
    # Mode 1: Standard YOLO visualization (saves to output_folder/results/)
    #run_inference(
    #    input_folder=INPUT_FOLDER,
    #    output_folder=OUTPUT_FOLDER,
    #    model_path=MODEL_PATH,
    #    conf_threshold=CONFIDENCE
    #)
    
    # Mode 2: Custom colored visualization (saves directly to output_folder/)
     run_inference_with_custom_viz(
         input_folder=INPUT_FOLDER,
         output_folder=OUTPUT_FOLDER,
         model_path=MODEL_PATH,
         conf_threshold=CONFIDENCE
         
    )