import json
import csv
import os
import random
from collections import defaultdict

# === CONFIG ===
json_path = "data/inat21/train_mini.json"   # Path to your JSON file
image_root = "train_mini"                   # Folder containing images
shots = 8                                   # Number of images per class
train_csv_out = "datasets/inat21/train_k8.csv"
test_csv_out = "datasets/inat21/test.csv"   # Optional

# === LOAD DATA ===
with open(json_path, 'r') as f:
    data = json.load(f)

# Group images by category
class_to_images = defaultdict(list)
for item in data['images']:
    file_name = item['file_name']
    category = file_name.split('/')[1]  # Extract class folder
    class_to_images[category].append(file_name)

print(f"Found {len(class_to_images)} unique classes.")

# === SPLIT INTO TRAIN AND TEST CSVs ===
train_rows = []
test_rows = []

for category, images in class_to_images.items():
    if len(images) < shots:
        print(f"Skipping class {category} (only {len(images)} images)")
        continue

    random.shuffle(images)
    selected = images[:shots]
    remaining = images[shots:]

    # Add 8-shot samples
    for img in selected:
        train_rows.append([os.path.join(image_root, img), category])

    # Add remaining to test set
    for img in remaining:
        test_rows.append([os.path.join(image_root, img), category])

print(f"Train samples: {len(train_rows)}")
print(f"Test samples:  {len(test_rows)}")

# === SAVE TO CSV ===
os.makedirs(os.path.dirname(train_csv_out), exist_ok=True)

with open(train_csv_out, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label'])
    writer.writerows(train_rows)

with open(test_csv_out, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label'])
    writer.writerows(test_rows)

print("CSV files created successfully.")
