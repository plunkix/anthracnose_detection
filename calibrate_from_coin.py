"""
Detect coins in original images and calculate pixel-to-mm calibration
Coin diameter = 22mm
"""

import cv2
import numpy as np
from pathlib import Path
import json

def detect_coin_in_image(image_path, coin_diameter_mm=22):
    """
    Detect circular coin in image and calculate pixels per mm

    Parameters:
    -----------
    image_path : str
        Path to image with coin
    coin_diameter_mm : float
        Real-world coin diameter in mm (default 22mm)

    Returns:
    --------
    dict : {
        'pixels_per_mm': float,
        'coin_center': (x, y),
        'coin_radius_pixels': float,
        'confidence': str
    }
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=150
    )

    if circles is None:
        return None

    circles = np.uint16(np.around(circles))

    # Find the most circular/prominent circle (likely the coin)
    # For simplicity, take the first detected circle
    # In production, you might want to filter by size/color
    x, y, radius = circles[0][0]

    # Calculate calibration
    coin_diameter_pixels = radius * 2
    pixels_per_mm = coin_diameter_pixels / coin_diameter_mm

    result = {
        'image': Path(image_path).name,
        'pixels_per_mm': float(pixels_per_mm),
        'coin_center': (int(x), int(y)),
        'coin_radius_pixels': float(radius),
        'coin_diameter_pixels': float(coin_diameter_pixels),
        'coin_diameter_mm': coin_diameter_mm,
        'confidence': 'high' if len(circles[0]) == 1 else 'medium'
    }

    return result

def process_all_images(image_dir='anth-images', coin_diameter_mm=22):
    """
    Process all images in directory to detect coins and calculate calibration
    """
    image_dir = Path(image_dir)
    results = {}

    print("=" * 60)
    print("COIN DETECTION & CALIBRATION")
    print("=" * 60)
    print(f"\nSearching for coins in: {image_dir}")
    print(f"Reference: Coin diameter = {coin_diameter_mm}mm\n")

    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(image_dir.glob(ext))

    detected_count = 0

    for img_path in sorted(image_files):
        print(f"Processing: {img_path.name}...", end=' ')

        result = detect_coin_in_image(str(img_path), coin_diameter_mm)

        if result:
            results[img_path.name] = result
            detected_count += 1
            print(f"[OK] Found coin! {result['pixels_per_mm']:.2f} pixels/mm")
        else:
            print("[X] No coin detected")

    # Calculate average calibration
    if results:
        avg_pixels_per_mm = np.mean([r['pixels_per_mm'] for r in results.values()])
        std_pixels_per_mm = np.std([r['pixels_per_mm'] for r in results.values()])

        print("\n" + "=" * 60)
        print("CALIBRATION RESULTS")
        print("=" * 60)
        print(f"Images processed: {len(image_files)}")
        print(f"Coins detected: {detected_count}")
        print(f"\nAverage calibration: {avg_pixels_per_mm:.2f} ± {std_pixels_per_mm:.2f} pixels/mm")
        print(f"This means: 1mm = {avg_pixels_per_mm:.2f} pixels")
        print(f"Or: 1 pixel = {1/avg_pixels_per_mm:.4f} mm")

        # Add summary to results
        results['_summary'] = {
            'average_pixels_per_mm': float(avg_pixels_per_mm),
            'std_pixels_per_mm': float(std_pixels_per_mm),
            'total_images': len(image_files),
            'coins_detected': detected_count,
            'coin_diameter_mm': coin_diameter_mm
        }

        # Save to JSON
        output_file = 'calibration_data.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[OK] Calibration data saved to: {output_file}")
        print("=" * 60)

        # Create visualization for first few detections
        visualize_coin_detection(image_dir, results, max_images=3)

    else:
        print("\n[WARNING] No coins detected in any images!")
        print("Try adjusting the detection parameters or check if coins are visible.")

    return results

def visualize_coin_detection(image_dir, results, max_images=3):
    """
    Create visualizations showing detected coins
    """
    import os
    os.makedirs('calibration_visualizations', exist_ok=True)

    print(f"\nCreating visualizations...")

    count = 0
    for img_name, data in results.items():
        if img_name == '_summary' or count >= max_images:
            continue

        img_path = Path(image_dir) / img_name
        img = cv2.imread(str(img_path))

        if img is None:
            continue

        # Draw circle
        center = data['coin_center']
        radius = int(data['coin_radius_pixels'])

        cv2.circle(img, center, radius, (0, 255, 0), 3)
        cv2.circle(img, center, 2, (0, 0, 255), 3)

        # Add text
        text = f"Coin: {data['coin_diameter_pixels']:.1f}px = {data['coin_diameter_mm']}mm"
        cv2.putText(img, text, (center[0] - 100, center[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        text2 = f"Calibration: {data['pixels_per_mm']:.2f} pixels/mm"
        cv2.putText(img, text2, (center[0] - 100, center[1] - radius - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save
        output_path = f'calibration_visualizations/coin_detected_{img_name}'
        cv2.imwrite(output_path, img)
        count += 1

    print(f"[OK] Saved {count} visualization(s) to: calibration_visualizations/")

if __name__ == "__main__":
    # Detect coins in all original images
    results = process_all_images(image_dir='anth-images', coin_diameter_mm=22)
