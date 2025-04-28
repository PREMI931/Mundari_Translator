{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab684c20-8146-49eb-a1b9-0b9fdc4daeb5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Paths\n",
    "dataset_dir = \"processed_dataset\"\n",
    "csv_path = \"labels.csv\"\n",
    "output_data = \"dataset.npy\"\n",
    "output_labels = \"labels.npy\"\n",
    "\n",
    "# Load labels\n",
    "df = pd.read_csv(csv_path)\n",
    "image_files = df[\"image_filename\"].tolist()\n",
    "labels = df[\"label\"].tolist()\n",
    "\n",
    "X, y = [], []\n",
    "\n",
    "# Process images\n",
    "for img_name, label in zip(image_files, labels):\n",
    "    img_path = os.path.join(dataset_dir, img_name)\n",
    "\n",
    "    if os.path.exists(img_path):  # Check if file exists\n",
    "        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image\n",
    "        img = cv2.resize(img, (128, 128))  # Resize\n",
    "        img = img / 255.0  # Normalize\n",
    "        X.append(img)\n",
    "        y.append(label)\n",
    "    else:\n",
    "        print(f\"Warning: {img_name} not found!\")\n",
    "\n",
    "# Convert to NumPy arrays\n",
    "X = np.array(X).reshape(-1, 128, 128, 1)  # Shape (samples, height, width, channels)\n",
    "y = np.array(y)\n",
    "\n",
    "# Save dataset\n",
    "np.save(output_data, X)\n",
    "np.save(output_labels, y)\n",
    "\n",
    "print(\"Dataset saved successfully!\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
