import os
import cv2
import numpy as np
from PIL import Image

# Dataset directory
dataset_dir = "mundari_dataset"
processed_dir = "processed_dataset"
os.makedirs(processed_dir, exist_ok=True)

# Preprocessing function
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize to 128x128 pixels
    img = img / 255.0  # Normalize pixel values
    return img

# Process images
for img_name in os.listdir(dataset_dir):
    img_path = os.path.join(dataset_dir, img_name)
    processed_img = preprocess_image(img_path)

    # Save processed image
    processed_img_path = os.path.join(processed_dir, img_name)
    cv2.imwrite(processed_img_path, (processed_img * 255).astype(np.uint8))
    print(f"Processed: {processed_img_path}")

print("Dataset preprocessing completed!")
