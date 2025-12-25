"""
Anthracnose Lesion Detection - User-Friendly Web Interface
Upload chilli images and get automated lesion detection with measurements
"""

import gradio as gr
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import zipfile
from datetime import datetime
from ultralytics import YOLO
from measure_lesions import LesionMeasurement
import os

class AnthracnoseApp:
    def __init__(self, model_path):
        """Initialize the application with trained model"""
        self.model = YOLO(model_path)

    def categorize_severity(self, length_px, area_px, grade_thresholds):
        """
        Categorize lesion severity based on length

        Parameters:
        -----------
        length_px : float
            Lesion length in pixels
        area_px : float
            Lesion area in pixels
        grade_thresholds : dict
            Dictionary with grade names and their length thresholds
            Example: {'Mild': 200, 'Moderate': 350, 'Severe': 500}

        Returns:
        --------
        str : Grade name
        """
        # Sort thresholds by value
        sorted_grades = sorted(grade_thresholds.items(), key=lambda x: x[1])

        for grade_name, threshold in sorted_grades:
            if length_px <= threshold:
                return grade_name

        # If exceeds all thresholds, return the highest grade
        return sorted_grades[-1][0] if sorted_grades else "Ungraded"

    def process_images(self,
                      images,
                      conf_threshold,
                      grade_mild_threshold,
                      grade_moderate_threshold,
                      grade_severe_threshold,
                      include_visualizations):
        """
        Process uploaded images and return results

        Parameters:
        -----------
        images : list
            List of uploaded image paths
        conf_threshold : float
            Confidence threshold for detection
        grade_mild_threshold : int
            Max length (px) for Mild grade
        grade_moderate_threshold : int
            Max length (px) for Moderate grade
        grade_severe_threshold : int
            Max length (px) for Severe grade
        include_visualizations : bool
            Whether to generate visualization images

        Returns:
        --------
        tuple : (excel_path, zip_path, summary_text, gallery_images)
        """
        if not images:
            return None, None, "⚠️ Please upload at least one image", []

        # Create grade thresholds dictionary
        grade_thresholds = {
            'Mild': grade_mild_threshold,
            'Moderate': grade_moderate_threshold,
            'Severe': grade_severe_threshold,
            'Critical': float('inf')  # Anything above Severe
        }

        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        vis_dir = os.path.join(temp_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        results_data = []
        gallery_images = []

        total_lesions = 0
        processed_images = 0

        # Process each image
        for img_path in images:
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_name = Path(img_path).name

                # Run inference with test-time augmentation (try multiple rotations)
                results = self.model.predict(
                    img_path,
                    conf=conf_threshold,
                    verbose=False,
                    augment=True  # Enable test-time augmentation for better rotation handling
                )[0]

                # Process detections
                lesion_data_for_image = []
                if results.masks is not None:
                    for i, (mask, conf) in enumerate(zip(results.masks.data, results.boxes.conf)):
                        # Convert mask to numpy
                        mask_np = mask.cpu().numpy()
                        mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)

                        # Measure lesion
                        measurements = LesionMeasurement.measure_lesion(mask_binary, method='all')

                        # Categorize severity
                        severity = self.categorize_severity(
                            measurements['length_feret'],
                            measurements['area_pixels'],
                            grade_thresholds
                        )

                        # Store results
                        row = {
                            'Image': img_name,
                            'Lesion_ID': i + 1,
                            'Confidence': float(conf.cpu().numpy()),
                            'Length_pixels': round(measurements['length_feret'], 2),
                            'Area_pixels': int(measurements['area_pixels']),
                            'Severity_Grade': severity
                        }

                        # Add millimeter measurements if calibration available
                        if self.pixels_per_mm:
                            row['Length_mm'] = round(measurements['length_feret'] / self.pixels_per_mm, 2)
                            row['Area_mm2'] = round(measurements['area_pixels'] / (self.pixels_per_mm ** 2), 2)
                        results_data.append(row)
                        lesion_data_for_image.append(row)
                        total_lesions += 1

                # Create visualization if requested (even if no lesions detected)
                if include_visualizations:
                    vis_img = self.create_visualization(
                        img, results, lesion_data_for_image, img_name
                    )
                    vis_path = os.path.join(vis_dir, f"result_{img_name}")
                    cv2.imwrite(vis_path, vis_img)
                    gallery_images.append(vis_path)

                processed_images += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # Create Excel file and summary
        excel_path = None
        zip_path = None

        if results_data:
            df = pd.DataFrame(results_data)

            # Calculate summary statistics
            grade_counts = df['Severity_Grade'].value_counts().to_dict()

            excel_path = os.path.join(temp_dir, f'lesion_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Write main results
                df.to_excel(writer, sheet_name='Lesion_Measurements', index=False)

                # Write summary statistics
                summary_df = pd.DataFrame([{
                    'Total_Images': processed_images,
                    'Total_Lesions': total_lesions,
                    'Avg_Length_px': df['Length_pixels'].mean(),
                    'Avg_Area_px': df['Area_pixels'].mean(),
                    **{f'{grade}_Count': grade_counts.get(grade, 0) for grade in ['Mild', 'Moderate', 'Severe', 'Critical']}
                }])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Create summary text
            summary_text = f"""
## ✅ Analysis Complete!

**Images Processed:** {processed_images}
**Total Lesions Detected:** {total_lesions}

**Severity Distribution:**
- 🟢 Mild: {grade_counts.get('Mild', 0)} lesions
- 🟡 Moderate: {grade_counts.get('Moderate', 0)} lesions
- 🟠 Severe: {grade_counts.get('Severe', 0)} lesions
- 🔴 Critical: {grade_counts.get('Critical', 0)} lesions

**Average Measurements:**"""

            # Add pixel measurements
            if 'Length_mm' in df.columns:
                summary_text += f"""
- Length: {df['Length_mm'].mean():.2f} mm ({df['Length_pixels'].mean():.1f} pixels)
- Area: {df['Area_mm2'].mean():.2f} mm² ({df['Area_pixels'].mean():.0f} pixels)

📏 **Calibrated using 22mm coin reference**"""
            else:
                summary_text += f"""
- Length: {df['Length_pixels'].mean():.1f} pixels
- Area: {df['Area_pixels'].mean():.0f} pixels

⚠️ No calibration data - measurements in pixels only"""

            summary_text += "\n\n📊 Download the Excel file below for detailed measurements."
        else:
            # No lesions detected
            summary_text = f"""
## ✅ Analysis Complete!

**Images Processed:** {processed_images}
**Total Lesions Detected:** 0

✨ **No lesions detected in any images!**

This could mean:
- The chillies are healthy
- The confidence threshold is too high (try lowering it)
- The lesions are too small or faint to detect

💡 Try adjusting the confidence threshold to 15-20% if you expect lesions.
"""

        # Create zip of visualizations if requested
        if include_visualizations and gallery_images:
            zip_path = os.path.join(temp_dir, f'visualizations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for img_path in gallery_images:
                    zipf.write(img_path, os.path.basename(img_path))

        return excel_path, zip_path, summary_text, gallery_images

    def create_visualization(self, img, results, lesion_data, img_name):
        """Create visualization with annotations"""
        vis_img = img.copy()
        num_lesions = 0

        if results.masks is not None and len(lesion_data) > 0:
            num_lesions = len(results.masks)
            for i, (mask, box, conf) in enumerate(zip(results.masks.data, results.boxes.data, results.boxes.conf)):
                # Convert mask
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # Color based on severity grade
                severity = lesion_data[i]['Severity_Grade']
                color_map = {
                    'Mild': [0, 255, 0],       # Green
                    'Moderate': [0, 255, 255],  # Yellow
                    'Severe': [0, 165, 255],    # Orange
                    'Critical': [0, 0, 255]     # Red
                }
                color = color_map.get(severity, [0, 255, 0])

                # Overlay colored mask
                colored_mask = np.zeros_like(img)
                colored_mask[mask_binary > 0] = color
                vis_img = cv2.addWeighted(vis_img, 0.7, colored_mask, 0.3, 0)

                # Draw contour
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, (255, 255, 255), 2)

                # Draw bounding box and text
                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if len(largest_contour) >= 5:
                        rect = cv2.minAreaRect(largest_contour)
                        box_pts = cv2.boxPoints(rect)
                        box_pts = np.intp(box_pts)
                        cv2.drawContours(vis_img, [box_pts], 0, (255, 0, 255), 2)

                        # Get centroid
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = largest_contour[0][0]

                        # Add text
                        length_px = lesion_data[i]['Length_pixels']
                        if 'Length_mm' in lesion_data[i]:
                            length_mm = lesion_data[i]['Length_mm']
                            text = f"{severity}: {length_mm:.1f}mm"
                        else:
                            text = f"{severity}: {length_px:.0f}px"
                        cv2.putText(vis_img, text, (cx - 60, cy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No lesions detected - add message on image
            h, w = vis_img.shape[:2]
            cv2.putText(vis_img, "NO LESIONS DETECTED",
                       (w // 2 - 180, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.putText(vis_img, "(Healthy or try lowering confidence)",
                       (w // 2 - 220, h // 2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add header with image info
        header_height = 40
        header = np.zeros((header_height, vis_img.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, f"Image: {img_name} | Lesions: {num_lesions}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vis_img = np.vstack([header, vis_img])

        return vis_img

# Initialize the app
MODEL_PATH = "runs/segment/anthracnose_rotation_robust/weights/best.pt"

# Load calibration data
try:
    import json
    with open('calibration_data.json', 'r') as f:
        calibration_data = json.load(f)
        PIXELS_PER_MM = calibration_data['_summary']['average_pixels_per_mm']
        print(f"[OK] Loaded calibration: {PIXELS_PER_MM:.2f} pixels/mm")
except:
    PIXELS_PER_MM = None
    print("[WARNING] No calibration data found - measurements will be in pixels only")

app = AnthracnoseApp(MODEL_PATH)

# Create Gradio interface
def process_wrapper(images, conf, mild, moderate, severe, include_vis):
    """Wrapper function for Gradio"""
    # Set calibration if available
    app.pixels_per_mm = PIXELS_PER_MM if PIXELS_PER_MM else None

    excel, zip_file, summary, gallery = app.process_images(
        images, conf/100, mild, moderate, severe, include_vis
    )
    return excel, zip_file, summary, gallery

# Custom CSS
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

# Build interface
with gr.Blocks(css=css, title="Anthracnose Detection System") as demo:
    calibration_status = f"📏 Calibrated: {PIXELS_PER_MM:.2f} px/mm" if PIXELS_PER_MM else "⚠️ Not calibrated"

    gr.HTML(f"""
    <div class="header">
        <h1>🌶️ Anthracnose Lesion Detection System</h1>
        <p>Upload cropped chilli images to detect and measure anthracnose lesions</p>
        <p style="font-size: 14px; margin-top: 5px;">{calibration_status}</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📤 Upload Images")
            image_input = gr.File(
                file_count="multiple",
                label="Upload Cropped Chilli Images",
                file_types=[".jpg", ".jpeg", ".png"]
            )

            gr.Markdown("## ⚙️ Detection Settings")
            confidence_slider = gr.Slider(
                minimum=10,
                maximum=90,
                value=15,
                step=5,
                label="Confidence Threshold (%)",
                info="Start with 15-20% for best results. Lower = more detections"
            )

            gr.Markdown("## 🎯 Severity Grade Thresholds (pixels)")
            gr.Markdown("*Define length thresholds for each severity grade*")

            mild_threshold = gr.Number(
                value=250,
                label="Mild (max length in pixels)",
                info="Lesions ≤ this length = Mild"
            )
            moderate_threshold = gr.Number(
                value=350,
                label="Moderate (max length in pixels)",
                info="Lesions ≤ this length = Moderate"
            )
            severe_threshold = gr.Number(
                value=450,
                label="Severe (max length in pixels)",
                info="Lesions ≤ this length = Severe, above = Critical"
            )

            include_viz = gr.Checkbox(
                value=True,
                label="Generate Visualizations",
                info="Create annotated images (can be downloaded as ZIP)"
            )

            process_btn = gr.Button("🚀 Analyze Images", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("## 📊 Results")
            summary_output = gr.Markdown(value="*Upload images and click 'Analyze Images' to start*")

            gr.Markdown("## 💾 Downloads")
            excel_output = gr.File(label="📑 Excel Report (Measurements + Summary)")
            zip_output = gr.File(label="🖼️ Visualizations ZIP")

            gr.Markdown("## 🖼️ Preview Visualizations")
            gallery_output = gr.Gallery(
                label="Detected Lesions",
                columns=2,
                height="auto",
                object_fit="contain"
            )

    # Examples section
    gr.Markdown("---")
    gr.Markdown("""
    ## 💡 How to Use:
    1. **Upload Images**: Select one or more cropped chilli images (.jpg, .png)
    2. **Adjust Settings**:
       - Set confidence threshold (25% recommended)
       - Define severity grade thresholds based on your criteria
    3. **Click Analyze**: Process images and view results
    4. **Download**: Get Excel report and visualizations

    ## 📏 Grade Thresholds Guide:
    - **Mild**: Early-stage lesions (small, localized)
    - **Moderate**: Developing lesions (medium spread)
    - **Severe**: Advanced lesions (large spread)
    - **Critical**: Extensive damage (exceeds severe threshold)

    *Default thresholds: Mild ≤250px, Moderate ≤350px, Severe ≤450px, Critical >450px*
    """)

    # Connect components
    process_btn.click(
        fn=process_wrapper,
        inputs=[
            image_input,
            confidence_slider,
            mild_threshold,
            moderate_threshold,
            severe_threshold,
            include_viz
        ],
        outputs=[excel_output, zip_output, summary_output, gallery_output]
    )

if __name__ == "__main__":
    print("=" * 60)
    print("🌶️  ANTHRACNOSE DETECTION SYSTEM")
    print("=" * 60)
    print("\nStarting web interface...")
    print("Open your browser and go to the URL shown below\n")

    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
