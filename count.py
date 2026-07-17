import os
import random
import shutil

# =====================================
# SETTINGS
# =====================================

SOURCE_DATASET = "dataset"
TARGET_DATASET = "dataset_small"

TRAIN_LIMIT = 120
VALID_LIMIT = 20

random.seed(42)

# =====================================
# FUNCTION
# =====================================

def reduce_split(source_dir, target_dir, limit):

    os.makedirs(target_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):

        source_class = os.path.join(source_dir, class_name)

        if not os.path.isdir(source_class):
            continue

        target_class = os.path.join(target_dir, class_name)
        os.makedirs(target_class, exist_ok=True)

        images = [
            img for img in os.listdir(source_class)
            if os.path.isfile(os.path.join(source_class, img))
        ]

        selected = random.sample(
            images,
            min(limit, len(images))
        )

        for img in selected:

            shutil.copy(
                os.path.join(source_class, img),
                os.path.join(target_class, img)
            )

        print(
            f"{class_name}: {len(selected)} copied"
        )

# =====================================
# TRAIN
# =====================================

print("\nReducing TRAIN dataset...\n")

reduce_split(
    os.path.join(SOURCE_DATASET, "train"),
    os.path.join(TARGET_DATASET, "train"),
    TRAIN_LIMIT
)

# =====================================
# VALID
# =====================================

print("\nReducing VALID dataset...\n")

reduce_split(
    os.path.join(SOURCE_DATASET, "valid"),
    os.path.join(TARGET_DATASET, "valid"),
    VALID_LIMIT
)


total_images = 0

for root, dirs, files in os.walk(TARGET_DATASET):
    total_images += len(files)

print(f"\nTotal images in '{TARGET_DATASET}': {total_images}")
print("\nDataset reduction complete!")