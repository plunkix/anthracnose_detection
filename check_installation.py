"""
Quick check to verify installation and setup
"""

def check_installation():
    print("=" * 60)
    print("ANTHRACNOSE DETECTION SYSTEM - INSTALLATION CHECK")
    print("=" * 60)

    all_good = True

    # Check Python packages
    print("\n1. Checking Python packages...")
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'ultralytics': 'ultralytics',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
    }

    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            print(f"   [OK] {package_name}")
        except ImportError:
            print(f"   [MISSING] {package_name} - (run: pip install {package_name})")
            all_good = False

    # Check optional packages
    print("\n2. Checking optional packages (for skeleton measurement)...")
    optional_packages = {
        'scipy': 'scipy',
        'skimage': 'scikit-image',
    }

    for module_name, package_name in optional_packages.items():
        try:
            __import__(module_name)
            print(f"   [OK] {package_name}")
        except ImportError:
            print(f"   - {package_name} - Not installed (skeleton measurement disabled)")

    # Check dataset
    print("\n3. Checking dataset...")
    import os

    if os.path.exists('dataset_yolo/data.yaml'):
        print("   ✓ Dataset prepared (dataset_yolo/)")
        print(f"     - Train: {len(os.listdir('dataset_yolo/images/train/'))} images")
        print(f"     - Val:   {len(os.listdir('dataset_yolo/images/val/'))} images")
        print(f"     - Test:  {len(os.listdir('dataset_yolo/images/test/'))} images")
    else:
        print("   [MISSING] Dataset not prepared (run: python prepare_dataset.py)")
        all_good = False

    # Check for trained model
    print("\n4. Checking for trained model...")
    model_path = 'runs/segment/anthracnose_detection/weights/best.pt'
    if os.path.exists(model_path):
        print(f"   ✓ Trained model found: {model_path}")
    else:
        print(f"   - No trained model yet (run: python train_model.py)")

    # Check GPU
    print("\n5. Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"     - CUDA version: {torch.version.cuda}")
        else:
            print("   - No GPU detected (will use CPU - training will be slower)")
    except ImportError:
        print("   - PyTorch not installed (required for training)")
        all_good = False

    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("SYSTEM READY!")
        print("\nNext steps:")
        print("  1. Train model: python train_model.py")
        print("  2. Run inference: python detect_and_measure.py --help")
    else:
        print("WARNING: SETUP INCOMPLETE")
        print("\nPlease install missing packages:")
        print("  pip install ultralytics opencv-python numpy pandas matplotlib scikit-learn")
    print("=" * 60)

if __name__ == "__main__":
    check_installation()
