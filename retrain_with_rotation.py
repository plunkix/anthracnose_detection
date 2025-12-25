"""
Retrain YOLOv8-seg with STRONG rotation augmentation
This will make the model robust to ALL orientations (vertical, horizontal, diagonal)
"""

from ultralytics import YOLO

def train_rotation_robust():
    """
    Train model with aggressive rotation augmentation
    """
    print("=" * 60)
    print("RETRAINING WITH ROTATION-ROBUST AUGMENTATION")
    print("=" * 60)
    print("\nThis training will handle chillies in ANY orientation:")
    print("  - Horizontal")
    print("  - Vertical")
    print("  - Diagonal")
    print("  - Any angle in between")
    print("\nStarting training...\n")

    # Load pretrained model
    model = YOLO('yolov8n-seg.pt')

    # Train with STRONG rotation augmentation
    model.train(
        data='dataset_yolo/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device='cpu',
        name='anthracnose_rotation_robust',
        patience=20,
        save=True,
        plots=True,

        # AGGRESSIVE rotation augmentation
        degrees=180,  # ±180 degrees = all possible rotations!
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,  # Increased vertical flip

        # Keep other augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,

        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
    )

    print("\n" + "=" * 60)
    print("ROTATION-ROBUST TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best model: runs/segment/anthracnose_rotation_robust/weights/best.pt")
    print("\nThis model can now handle chillies at ANY angle!")
    print("\nTo use in the app, update MODEL_PATH in app.py to:")
    print('  MODEL_PATH = "runs/segment/anthracnose_rotation_robust/weights/best.pt"')

    return model

if __name__ == "__main__":
    train_rotation_robust()
