import cv2
import os
import numpy as np

def extract_chillies_from_image(img_path, out_dir, min_area=2000):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red threshold (two ranges due to HSV wrap)
    lower_red1 = np.array([0, 100, 60])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 60])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # clean it
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_name = os.path.splitext(os.path.basename(img_path))[0]

    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        crop = img[y:y+h, x:x+w]

        save_path = os.path.join(out_dir, f"{img_name}_chilli_{count}.jpg")
        cv2.imwrite(save_path, crop)
        count += 1

    print(f"Extracted {count} chillies from {img_path}")

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_folder, file)
            extract_chillies_from_image(img_path, output_folder)


if __name__ == "__main__":
    input_folder = "original_images/data_2"    # folder containing images
    output_folder = "data-2-cropped"

    process_folder(input_folder, output_folder)
