import os
from PIL import Image

# Input (raw images) and output (resized images) paths
RAW_DATA_DIR = "raw_data"
OUTPUT_DIR = "data"

# Categories you want
categories = ["cars", "trucks", "motorcycles", "pedestrians", "cyclists"]

# Make sure output folders exist
for category in categories:
    os.makedirs(os.path.join(OUTPUT_DIR, category), exist_ok=True)

# Resize function
def resize_images(category):
    input_path = os.path.join(RAW_DATA_DIR, category)
    output_path = os.path.join(OUTPUT_DIR, category)
    
    for filename in os.listdir(input_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_path, filename)
            img = Image.open(img_path).convert("RGB")  # ensure 3 channels
            # img = img.resize((32, 32), Image.ANTIALIAS)
            img = img.resize((32, 32), Image.Resampling.LANCZOS)
            img.save(os.path.join(output_path, filename), optimize=True, quality=85)

# Run for all categories
for category in categories:
    resize_images(category)
    print(f"Resized images saved to {OUTPUT_DIR}/{category}/")
