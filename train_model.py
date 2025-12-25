from ultralytics import YOLO
import torch
import os

# ==================== HARDCODE YOUR PATHS HERE ====================
DATA_YAML = '/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_1/dataset_yolo/data.yaml'  # Will be created by prepare_dataset.py
MODEL_SIZE = 'n'  # Model size: 'n', 's', 'm', 'l', 'x'
EPOCHS = 100
BATCH_SIZE = 32
IMG_SIZE = 640
PROJECT_DIR = '/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_1/model_trained_2'  # Where to save training results (just folder name, no full path)
MODEL_NAME = 'anthracnose_model_2'  # Name for this training run (just name, no path)
# ==================================================================

def train_model():
    """Train YOLOv8 segmentation model"""
    
    print("=" * 60)
    print("TRAINING YOLOV8 SEGMENTATION MODEL")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Model: yolov8{MODEL_SIZE}-seg")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60)
    
    # Load pretrained model
    model = YOLO(f'yolov8{MODEL_SIZE}-seg.pt')
    
    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=PROJECT_DIR,
        name=MODEL_NAME,
        patience=10,
        save=True,
        plots=True,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        
        
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Get actual save path
    save_dir = os.path.join(PROJECT_DIR, MODEL_NAME)
    print(f"Best model: {save_dir}/weights/best.pt")
    print(f"Last model: {save_dir}/weights/last.pt")
    
    return model, save_dir


def evaluate_model(model_dir):
    """Evaluate trained model"""
    
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    # Load best model using the actual path
    best_model_path = os.path.join(model_dir, 'weights', 'best.pt')
    print(f"Loading model from: {best_model_path}")
    
    if not os.path.exists(best_model_path):
        print(f"Error: Model not found at {best_model_path}")
        return None
    
    model = YOLO(best_model_path)
    
    # Evaluate on validation set
    results = model.val(
        data=DATA_YAML,
        split='val',
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=f'{MODEL_NAME}_val',
        plots=True,
        save_json=True
    )
    
    # Print metrics
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"Box mAP50-95: {results.box.map:.4f}")
    print(f"Box mAP50:    {results.box.map50:.4f}")
    print(f"Mask mAP50-95: {results.seg.map:.4f}")
    print(f"Mask mAP50:    {results.seg.map50:.4f}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Train
    model, save_dir = train_model()
    
    # Evaluate
    evaluate_model(save_dir)