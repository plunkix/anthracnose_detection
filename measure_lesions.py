import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import ndimage
    from skimage.morphology import skeletonize
    SKELETON_AVAILABLE = True
except ImportError:
    SKELETON_AVAILABLE = False
    print("Warning: scipy/scikit-image not available. Skeleton-based measurement disabled.")

class LesionMeasurement:
    """
    Class for measuring anthracnose lesion length and area from segmentation masks
    """

    @staticmethod
    def calculate_area(mask):
        """Calculate area in pixels"""
        return np.sum(mask > 0)

    @staticmethod
    def calculate_length_minAreaRect(mask):
        """
        Calculate length using minimum area rectangle (oriented bounding box)
        Returns the longer side of the fitted rectangle
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return 0

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) < 5:
            return 0

        # Fit minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        width, height = rect[1]

        # Return the longer dimension as length
        length = max(width, height)

        return length

    @staticmethod
    def calculate_length_skeleton(mask):
        """
        Calculate length using morphological skeleton
        Finds longest path through the skeleton
        """
        if not SKELETON_AVAILABLE:
            return 0

        if np.sum(mask) == 0:
            return 0

        # Skeletonize the mask
        skeleton = skeletonize(mask > 0)

        if np.sum(skeleton) == 0:
            return 0

        # Distance transform from skeleton
        skeleton_coords = np.column_stack(np.where(skeleton))

        if len(skeleton_coords) < 2:
            return 0

        # Find two endpoints or points that are farthest apart
        # This approximates the length along the skeleton
        max_dist = 0
        for i in range(len(skeleton_coords)):
            for j in range(i + 1, len(skeleton_coords)):
                dist = np.linalg.norm(skeleton_coords[i] - skeleton_coords[j])
                if dist > max_dist:
                    max_dist = dist

        return max_dist

    @staticmethod
    def calculate_length_feret(mask):
        """
        Calculate Feret diameter (maximum distance between any two points on boundary)
        Most accurate for irregular shapes like lesions
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return 0

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        points = largest_contour.reshape(-1, 2)

        if len(points) < 2:
            return 0

        # Calculate maximum distance between any two boundary points
        # For efficiency, we can subsample points if there are too many
        if len(points) > 100:
            indices = np.linspace(0, len(points) - 1, 100, dtype=int)
            points = points[indices]

        max_dist = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_dist:
                    max_dist = dist

        return max_dist

    @staticmethod
    def measure_lesion(mask, method='all'):
        """
        Comprehensive lesion measurement

        Parameters:
        -----------
        mask : numpy array
            Binary mask of the lesion
        method : str
            'minAreaRect', 'skeleton', 'feret', or 'all'

        Returns:
        --------
        dict with measurement results
        """
        area = LesionMeasurement.calculate_area(mask)

        measurements = {
            'area_pixels': area,
        }

        if method in ['minAreaRect', 'all']:
            measurements['length_minAreaRect'] = LesionMeasurement.calculate_length_minAreaRect(mask)

        if method in ['skeleton', 'all']:
            measurements['length_skeleton'] = LesionMeasurement.calculate_length_skeleton(mask)

        if method in ['feret', 'all']:
            measurements['length_feret'] = LesionMeasurement.calculate_length_feret(mask)

        # Recommended length (Feret diameter is most accurate for irregular lesions)
        if 'length_feret' in measurements:
            measurements['length_recommended'] = measurements['length_feret']

        return measurements

    @staticmethod
    def visualize_measurements(image, mask, measurements, save_path=None):
        """
        Visualize lesion with measurements overlaid
        """
        # Create visualization
        vis_img = image.copy()

        # Overlay mask
        colored_mask = np.zeros_like(vis_img)
        colored_mask[mask > 0] = [0, 255, 0]
        vis_img = cv2.addWeighted(vis_img, 0.7, colored_mask, 0.3, 0)

        # Draw contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 2)

        # Draw minimum area rectangle
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) >= 5:
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(vis_img, [box], 0, (0, 255, 255), 2)

        # Add text with measurements
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        cv2.putText(vis_img, f"Area: {measurements['area_pixels']:.0f} px",
                   (10, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += 30

        if 'length_feret' in measurements:
            cv2.putText(vis_img, f"Length (Feret): {measurements['length_feret']:.1f} px",
                       (10, y_offset), font, font_scale, (255, 255, 0), thickness)
            y_offset += 30

        if 'length_minAreaRect' in measurements:
            cv2.putText(vis_img, f"Length (MinRect): {measurements['length_minAreaRect']:.1f} px",
                       (10, y_offset), font, font_scale, (200, 200, 200), thickness)
            y_offset += 30

        if 'length_skeleton' in measurements:
            cv2.putText(vis_img, f"Length (Skeleton): {measurements['length_skeleton']:.1f} px",
                       (10, y_offset), font, font_scale, (200, 200, 200), thickness)

        if save_path:
            cv2.imwrite(save_path, vis_img)

        return vis_img

def convert_pixels_to_mm(pixels, pixels_per_mm=None, reference_length_mm=None, reference_length_px=None):
    """
    Convert pixel measurements to millimeters

    Parameters:
    -----------
    pixels : float
        Measurement in pixels
    pixels_per_mm : float (optional)
        Known conversion factor
    reference_length_mm : float (optional)
        Known real-world length in mm
    reference_length_px : float (optional)
        Corresponding pixel length

    Returns:
    --------
    float : measurement in mm
    """
    if pixels_per_mm is not None:
        return pixels / pixels_per_mm

    if reference_length_mm is not None and reference_length_px is not None:
        pixels_per_mm = reference_length_px / reference_length_mm
        return pixels / pixels_per_mm

    # If no calibration provided, return pixels
    print("Warning: No calibration provided. Returning measurement in pixels.")
    return pixels

if __name__ == "__main__":
    # Example usage
    import json

    print("Testing measurement algorithms...")

    # Load a sample annotation
    with open('ann_images/annotations/instances_default.json', 'r') as f:
        data = json.load(f)

    # Get first annotation
    ann = data['annotations'][0]
    img_info = data['images'][0]

    # Load image
    img = cv2.imread(f"ann_images/{img_info['file_name']}")

    # Create mask from polygon
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
    segmentation = ann['segmentation'][0]
    points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)

    # Measure
    measurements = LesionMeasurement.measure_lesion(mask, method='all')

    print("\nMeasurement Results:")
    print("-" * 40)
    for key, value in measurements.items():
        print(f"{key}: {value:.2f}")

    # Visualize
    vis = LesionMeasurement.visualize_measurements(img, mask, measurements, 'test_measurement.jpg')
    print("\nVisualization saved to: test_measurement.jpg")
