import cv2
import numpy as np
from ultralytics import YOLO
from measure_lesions import LesionMeasurement, convert_pixels_to_mm
import os
from pathlib import Path
import pandas as pd

class AnthracnoseDetector:
    """
    Complete pipeline for anthracnose detection and measurement
    """

    def __init__(self, model_path, pixels_per_mm=None):
        """
        Initialize detector

        Parameters:
        -----------
        model_path : str
            Path to trained YOLOv8-seg model
        pixels_per_mm : float (optional)
            Calibration factor for converting pixels to mm
        """
        self.model = YOLO(model_path)
        self.pixels_per_mm = pixels_per_mm

    def detect_and_measure(self, image_path, conf_threshold=0.25):
        """
        Detect lesions and measure their length and area

        Parameters:
        -----------
        image_path : str
            Path to input image
        conf_threshold : float
            Confidence threshold for detection

        Returns:
        --------
        dict with detection results and measurements
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Run inference
        results = self.model.predict(image_path, conf=conf_threshold, verbose=False)[0]

        # Process results
        detections = []

        if results.masks is not None:
            for i, (mask, box, conf) in enumerate(zip(results.masks.data, results.boxes.data, results.boxes.conf)):
                # Convert mask to numpy
                mask_np = mask.cpu().numpy()

                # Resize mask to image size
                mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # Measure lesion
                measurements = LesionMeasurement.measure_lesion(mask_binary, method='all')

                # Convert to mm if calibration is available
                measurements_mm = {}
                if self.pixels_per_mm is not None:
                    measurements_mm['area_mm2'] = measurements['area_pixels'] / (self.pixels_per_mm ** 2)
                    measurements_mm['length_mm'] = measurements['length_feret'] / self.pixels_per_mm

                detection = {
                    'lesion_id': i + 1,
                    'confidence': float(conf.cpu().numpy()),
                    'mask': mask_binary,
                    'measurements_px': measurements,
                    'measurements_mm': measurements_mm if measurements_mm else None
                }

                detections.append(detection)

        result = {
            'image_path': image_path,
            'image_shape': img.shape,
            'num_lesions': len(detections),
            'detections': detections,
            'image': img
        }

        return result

    def visualize_results(self, result, save_path=None):
        """
        Visualize detection and measurement results
        """
        img = result['image'].copy()

        # Draw each detection
        for det in result['detections']:
            mask = det['mask']
            measurements = det['measurements_px']
            conf = det['confidence']

            # Create colored overlay
            colored_mask = np.zeros_like(img)
            colored_mask[mask > 0] = [0, 255, 0]
            img = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)

            # Draw contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

            # Draw oriented bounding box
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                if len(largest_contour) >= 5:
                    rect = cv2.minAreaRect(largest_contour)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(img, [box], 0, (0, 255, 255), 2)

                    # Get centroid for text placement
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = largest_contour[0][0]

                    # Add measurement text
                    if det['measurements_mm'] is not None:
                        text = f"L:{det['measurements_mm']['length_mm']:.1f}mm"
                    else:
                        text = f"L:{measurements['length_feret']:.0f}px"

                    cv2.putText(img, text, (cx - 40, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Add summary info
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Lesions detected: {result['num_lesions']}",
                   (10, y_offset), font, 0.7, (255, 255, 255), 2)

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"Visualization saved to: {save_path}")

        return img

    def process_batch(self, image_dir, output_dir, conf_threshold=0.25):
        """
        Process multiple images and save results
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

        results_data = []

        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(image_dir).glob(ext))

        print(f"Processing {len(image_files)} images...")

        for img_path in image_files:
            print(f"\nProcessing: {img_path.name}")

            try:
                # Detect and measure
                result = self.detect_and_measure(str(img_path), conf_threshold)

                # Visualize
                vis_path = os.path.join(output_dir, 'visualizations', f"result_{img_path.name}")
                self.visualize_results(result, vis_path)

                # Collect data for CSV
                for det in result['detections']:
                    row = {
                        'image_name': img_path.name,
                        'lesion_id': det['lesion_id'],
                        'confidence': det['confidence'],
                        'area_pixels': det['measurements_px']['area_pixels'],
                        'length_feret_pixels': det['measurements_px']['length_feret'],
                        'length_minAreaRect_pixels': det['measurements_px']['length_minAreaRect'],
                        'length_skeleton_pixels': det['measurements_px']['length_skeleton'],
                    }

                    if det['measurements_mm'] is not None:
                        row['area_mm2'] = det['measurements_mm']['area_mm2']
                        row['length_mm'] = det['measurements_mm']['length_mm']

                    results_data.append(row)

                print(f"  Found {result['num_lesions']} lesions")

            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")

        # Save results to CSV
        if results_data:
            df = pd.DataFrame(results_data)
            csv_path = os.path.join(output_dir, 'measurements.csv')
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Results saved to: {csv_path}")

            # Print summary statistics
            print("\n" + "=" * 60)
            print("SUMMARY STATISTICS")
            print("=" * 60)
            print(f"Total images processed: {len(image_files)}")
            print(f"Total lesions detected: {len(results_data)}")
            print(f"\nLength (Feret diameter) in pixels:")
            print(f"  Mean: {df['length_feret_pixels'].mean():.2f}")
            print(f"  Std:  {df['length_feret_pixels'].std():.2f}")
            print(f"  Min:  {df['length_feret_pixels'].min():.2f}")
            print(f"  Max:  {df['length_feret_pixels'].max():.2f}")
            print(f"\nArea in pixels:")
            print(f"  Mean: {df['area_pixels'].mean():.2f}")
            print(f"  Std:  {df['area_pixels'].std():.2f}")

        return results_data

def main():
    """
    Example usage
    """
    import argparse

    parser = argparse.ArgumentParser(description='Anthracnose Lesion Detection and Measurement')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image_dir', type=str, help='Directory of images')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--pixels_per_mm', type=float, help='Calibration: pixels per mm')

    args = parser.parse_args()

    # Initialize detector
    detector = AnthracnoseDetector(args.model, pixels_per_mm=args.pixels_per_mm)

    if args.image:
        # Process single image
        result = detector.detect_and_measure(args.image, args.conf)
        os.makedirs(args.output, exist_ok=True)
        detector.visualize_results(result, os.path.join(args.output, 'result.jpg'))

        print("\nDetection Results:")
        print(f"Lesions found: {result['num_lesions']}")
        for det in result['detections']:
            print(f"\nLesion {det['lesion_id']}:")
            print(f"  Confidence: {det['confidence']:.3f}")
            print(f"  Area: {det['measurements_px']['area_pixels']:.0f} pixels")
            print(f"  Length (Feret): {det['measurements_px']['length_feret']:.1f} pixels")

    elif args.image_dir:
        # Process batch
        detector.process_batch(args.image_dir, args.output, args.conf)

    else:
        print("Error: Please provide either --image or --image_dir")

if __name__ == "__main__":
    main()
