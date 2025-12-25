import os
import json
import shutil
from collections import defaultdict

# ==========================
# HARD CODE YOUR PATHS HERE
# ==========================

IMG_DIR_1 = r"/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_2"
ANN_1     = r"/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_2/annotations/instances_default.json"

IMG_DIR_2 = r"/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_3"
ANN_2     = r"/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_3/annotations/instances_default.json"

MERGED_IMG_DIR = r"/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_full"
MERGED_ANN     = r"/mnt/c/Users/NERI_Works/Downloads/lesion_detection/dataset_full/annotations/instances_default.json"


# ==========================
# UTILS
# ==========================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ==========================
# 1. DATASET CONSISTENCY CHECK
# ==========================

def check_coco_consistency(img_dir, ann_path):
    coco = load_json(ann_path)

    images_json = {img["file_name"]: img["id"] for img in coco["images"]}
    images_disk = set(os.listdir(img_dir))

    print("\n=== IMAGE CHECK ===")

    missing_on_disk = set(images_json.keys()) - images_disk
    missing_in_json = images_disk - set(images_json.keys())

    print(f"Images in JSON but missing on disk: {len(missing_on_disk)}")
    for i in list(missing_on_disk)[:10]:
        print("  ", i)

    print(f"\nImages on disk but missing in JSON: {len(missing_in_json)}")
    for i in list(missing_in_json)[:10]:
        print("  ", i)

    print("\n=== ANNOTATION CHECK ===")

    valid_image_ids = set(images_json.values())
    orphan_anns = [
        ann for ann in coco["annotations"]
        if ann["image_id"] not in valid_image_ids
    ]

    print(f"Orphan annotations: {len(orphan_anns)}")

    print("\n=== SUMMARY ===")
    print("JSON images:", len(images_json))
    print("Disk images:", len(images_disk))
    print("Annotations:", len(coco["annotations"]))


# ==========================
# 2. MERGE TWO COCO DATASETS
# ==========================

def merge_coco_datasets():
    coco1 = load_json(ANN_1)
    coco2 = load_json(ANN_2)

    os.makedirs(MERGED_IMG_DIR, exist_ok=True)

    # ---- CATEGORY HANDLING ----
    # Assume same categories by name
    cat_name_to_id = {}
    merged_categories = []

    for cat in coco1["categories"]:
        merged_categories.append(cat)
        cat_name_to_id[cat["name"]] = cat["id"]

    next_cat_id = max(cat_name_to_id.values()) + 1

    for cat in coco2["categories"]:
        if cat["name"] not in cat_name_to_id:
            new_cat = cat.copy()
            new_cat["id"] = next_cat_id
            cat_name_to_id[cat["name"]] = next_cat_id
            merged_categories.append(new_cat)
            next_cat_id += 1

    # ---- IMAGE + ANNOTATION OFFSET ----
    img_id_offset = max(img["id"] for img in coco1["images"]) + 1
    ann_id_offset = max(ann["id"] for ann in coco1["annotations"]) + 1

    merged_images = coco1["images"].copy()
    merged_annotations = coco1["annotations"].copy()

    # ---- COPY + REGISTER IMAGES (DATASET 2) ----
    for img in coco2["images"]:
        new_img = img.copy()
        new_img["id"] += img_id_offset
        merged_images.append(new_img)

        src = os.path.join(IMG_DIR_2, img["file_name"])
        dst = os.path.join(MERGED_IMG_DIR, img["file_name"])

        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # ---- ANNOTATIONS ----
    for ann in coco2["annotations"]:
        new_ann = ann.copy()
        new_ann["id"] += ann_id_offset
        new_ann["image_id"] += img_id_offset

        # remap category_id by name
        cat_name = next(
            c["name"] for c in coco2["categories"]
            if c["id"] == ann["category_id"]
        )
        new_ann["category_id"] = cat_name_to_id[cat_name]

        merged_annotations.append(new_ann)

    # ---- COPY IMAGES FROM DATASET 1 ----
    for img in coco1["images"]:
        src = os.path.join(IMG_DIR_1, img["file_name"])
        dst = os.path.join(MERGED_IMG_DIR, img["file_name"])
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    merged_coco = {
        "info": coco1.get("info", {}),
        "licenses": coco1.get("licenses", []),
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": merged_categories
    }

    save_json(merged_coco, MERGED_ANN)

    print("\n=== MERGE COMPLETE ===")
    print("Images:", len(merged_images))
    print("Annotations:", len(merged_annotations))
    print("Categories:", len(merged_categories))


# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    print("\nChecking Dataset 1")
    check_coco_consistency(IMG_DIR_1, ANN_1)

    print("\nChecking Dataset 2")
    check_coco_consistency(IMG_DIR_2, ANN_2)

    merge_coco_datasets()
