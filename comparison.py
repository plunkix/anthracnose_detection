import cv2
import numpy as np
import os
from ultralytics import YOLO
from tqdm import tqdm

# ===================== PATHS =====================
MODEL_PATH = "/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_1/model_trained/anthracnose_model/weights/best.pt"
VAL_IMG_DIR = "/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_1/dataset_yolo/images/val"
VAL_LABEL_DIR = "/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_1/dataset_yolo/labels/val"
OUT_DIR = "./gt_vs_pred"
IMG_SIZE = 640
CONF_THRES = 0.25
# =================================================

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

def load_gt_masks(label_path, img_shape):
    h, w = img_shape[:2]
    masks = []

    if not os.path.exists(label_path):
        return masks

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = list(map(float, line.strip().split()))
            coords = parts[1:]

            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            masks.append(pts.astype(np.int32))

    return masks

def draw_masks(img, masks, color):
    overlay = img.copy()
    for pts in masks:
        cv2.fillPoly(overlay, [pts], color)
    return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

for img_name in tqdm(os.listdir(VAL_IMG_DIR)):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(VAL_IMG_DIR, img_name)
    label_path = os.path.join(
        VAL_LABEL_DIR, os.path.splitext(img_name)[0] + ".txt"
    )

    img = cv2.imread(img_path)
    if img is None:
        continue

    # ---------- GT ----------
    gt_masks = load_gt_masks(label_path, img.shape)
    gt_img = draw_masks(img.copy(), gt_masks, (0, 255, 0))  # GREEN

    # ---------- Prediction ----------
    preds = model.predict(
        source=img,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        verbose=False
    )

    pred_img = img.copy()
    if preds[0].masks is not None:
        for mask in preds[0].masks.xy:
            pts = np.array(mask, dtype=np.int32)
            cv2.fillPoly(pred_img, [pts], (0, 0, 255))  # RED

    pred_img = cv2.addWeighted(pred_img, 0.4, img, 0.6, 0)

    # ---------- Side-by-side ----------
    combined = np.hstack([gt_img, pred_img])
    out_path = os.path.join(OUT_DIR, img_name)
    cv2.imwrite(out_path, combined)

print("✅ GT vs Prediction comparison saved to:", OUT_DIR)
