# knn_classifier.py — robust eval + Step 0 bundle save (copy-paste ready)

import os
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

# -------------------------------
# Config
# -------------------------------
data_dir = "data"
image_size = (64, 64)    # keep consistent for training & prediction
n_components = 150
k_neighbors = 5
test_size = 0.2
random_seed = 42

# -------------------------------
# Helpers
# -------------------------------
def list_categories(root):
    """Stable, sorted class names (folder names)."""
    cats = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    cats.sort()
    return cats

def load_dataset(root, categories, image_size):
    """Load images as flattened float32 vectors in [0,1]."""
    X, y = [], []

    # Pillow compatibility for resample enum
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    for label, category in enumerate(categories):
        folder_path = os.path.join(root, category)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(folder_path, filename)
                img = Image.open(path).convert("RGB").resize(image_size, resample)
                arr = (np.asarray(img, dtype=np.float32) / 255.0).reshape(-1)
                X.append(arr)
                y.append(label)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

# -------------------------------
# Load data
# -------------------------------
if not os.path.isdir(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

categories = list_categories(data_dir)
if not categories:
    raise RuntimeError(f"No class folders found under {data_dir}")

print(f"Classes ({len(categories)}): {categories}")

X, y = load_dataset(data_dir, categories, image_size)
print(f"Loaded {len(X)} images → feature dim = {X.shape[1]}")

# -------------------------------
# Train/test split (prefer stratified; fallback if a class has <2 samples)
# -------------------------------
counts = Counter(y.tolist())
print(f"Class counts: {dict(counts)}")
min_count = min(counts.values())

try:
    if min_count < 2:
        raise ValueError("At least one class has <2 samples; cannot stratify.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    print("Using STRATIFIED split.")
except ValueError as e:
    print(f"WARNING: {e}")
    print("Falling back to UNSTRATIFIED split. "
          "Results may be biased; consider adding more images to tiny classes.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, shuffle=True, stratify=None
    )

# -------------------------------
# PCA (fit on TRAIN only) → transform
# -------------------------------
pca = PCA(n_components=n_components, random_state=random_seed).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca  = pca.transform(X_test)

# -------------------------------
# KNN on PCA features
# -------------------------------
knn = KNeighborsClassifier(n_neighbors=k_neighbors)
knn.fit(X_train_pca, y_train)

# -------------------------------
# Evaluate (handle missing classes in test)
# -------------------------------
y_pred = knn.predict(X_test_pca)

all_labels = np.arange(len(categories))  # ensure report covers all classes
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")

print("Classification Report:\n")
print(
    classification_report(
        y_test, y_pred,
        labels=all_labels,
        target_names=categories,
        digits=4,
        zero_division=0
    )
)

print("Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred, labels=all_labels))

# -------------------------------
# STEP 0: Save portable bundle
# -------------------------------
os.makedirs("models", exist_ok=True)
bundle = {
    "pca": pca,
    "knn": knn,
    "image_size": image_size,
    "categories": categories,
}
out_path = "models/knn_pca.joblib"
dump(bundle, out_path)
print(f"\nSaved model bundle → {out_path}")
print("Bundle includes keys: ['pca','knn','image_size','categories']")
