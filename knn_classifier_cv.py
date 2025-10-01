# knn_classifier_cv.py — cached dataset + CV tuning + rich eval + pipeline bundle
import os, json, time, hashlib
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
from joblib import dump
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, f1_score)

from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# -------------------------------
# Config
# -------------------------------
DATA_DIR = "data"
IMAGE_SIZE = (64, 64)
TEST_SIZE = 0.2
SEED = 42
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# search space (small but meaningful)
PARAM_GRID = {
    "pca__n_components": [80, 120, 150, 200],
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan", "cosine"],
}

# -------------------------------
# Helpers
# -------------------------------
def list_categories(root):
    cats = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    cats.sort()
    return cats

def resample_mode():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS

def dataset_manifest(root, image_exts=(".png",".jpg",".jpeg")):
    """Return a stable manifest of (relpath, size, mtime) to detect changes."""
    entries = []
    for cls in list_categories(root):
        folder = os.path.join(root, cls)
        for fn in os.listdir(folder):
            if fn.lower().endswith(image_exts):
                p = os.path.join(folder, fn)
                st = os.stat(p)
                entries.append({
                    "rel": os.path.relpath(p, root).replace("\\","/"),
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                })
    entries.sort(key=lambda e: e["rel"])
    blob = json.dumps(entries, separators=(",",":")).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()
    return entries, h

def maybe_load_cached_dataset(root, image_size):
    cats = list_categories(root)
    if not cats: raise RuntimeError(f"No class folders found under {root}")
    entries, h = dataset_manifest(root)
    cache_npz = os.path.join(CACHE_DIR, f"dataset_{image_size[0]}x{image_size[1]}_{h[:16]}.npz")
    cache_meta = os.path.join(CACHE_DIR, f"dataset_{image_size[0]}x{image_size[1]}_{h[:16]}.json")
    if os.path.exists(cache_npz) and os.path.exists(cache_meta):
        data = np.load(cache_npz)
        with open(cache_meta, "r") as f:
            meta = json.load(f)
        # sanity checks
        if (tuple(meta["image_size"]) == tuple(image_size)
            and meta["categories"] == cats):
            return cats, entries, h, data["X"], data["y"]
    # build fresh
    print("⏳ Building dataset cache (images → arrays)…")
    X, y = [], []
    rmode = resample_mode()
    for label, cls in enumerate(cats):
        folder = os.path.join(root, cls)
        for fn in os.listdir(folder):
            if fn.lower().endswith((".png",".jpg",".jpeg")):
                p = os.path.join(folder, fn)
                img = Image.open(p).convert("RGB").resize(image_size, rmode)
                arr = (np.asarray(img, np.float32) / 255.0).reshape(-1)
                X.append(arr); y.append(label)
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int32)
    np.savez_compressed(cache_npz, X=X, y=y)
    with open(cache_meta, "w") as f:
        json.dump({"hash": h, "image_size": image_size, "categories": cats}, f)
    print(f"✅ Cached → {cache_npz}")
    return cats, entries, h, X, y

def show_class_counts(y, cats):
    counts = Counter(y.tolist())
    print("Class counts:", {cats[k]: counts.get(k,0) for k in range(len(cats))})

# ---- HOG transformer (works inside Pipeline) ----
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
import numpy as np

class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, image_size=(64, 64),
                 pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2),
                 orientations=9,
                 luminance_weights=True):
        self.image_size = image_size
        self.pixels_per_cell = pixels_per_cell     # <-- match param name
        self.cells_per_block = cells_per_block     # <-- match param name
        self.orientations = orientations
        self.luminance_weights = luminance_weights

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        H, W = self.image_size
        x = X.reshape(-1, H, W, 3)
        if self.luminance_weights:
            # standard luma instead of simple mean
            Xg = 0.299 * x[:, :, :, 0] + 0.587 * x[:, :, :, 1] + 0.114 * x[:, :, :, 2]
        else:
            Xg = x.mean(axis=3)

        feats = []
        for img in Xg:
            f = hog(
                img,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=False,
                feature_vector=True,
            )
            feats.append(f)
        return np.asarray(feats, dtype=np.float32)

# -------------------------------
# Load (with cache) and split
# -------------------------------
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

categories, manifest, ds_hash, X, y = maybe_load_cached_dataset(DATA_DIR, IMAGE_SIZE)
print(f"Classes ({len(categories)}): {categories}")
print(f"Loaded {len(X)} images → dim = {X.shape[1]}")
show_class_counts(y, categories)

# prefer stratified; fallback if tiny class
min_count = min(Counter(y.tolist()).values())
try:
    if min_count < 2: raise ValueError("A class has <2 samples; cannot stratify.")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    print("Using STRATIFIED split.")
except ValueError as e:
    print("WARNING:", e, "Falling back to unstratified split.")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=None)

# -------------------------------
# Two representations to compare:
# A) RAW → PCA → KNN     (your current)
# B) HOG → (scale) → PCA → KNN
# -------------------------------
raw_pca_knn = Pipeline([
    ("pca", PCA(random_state=SEED)),
    ("knn", KNeighborsClassifier(n_jobs=-1)),
])

hog_pca_knn = Pipeline([
    ("hog", HOGTransformer(image_size=IMAGE_SIZE, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9)),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(random_state=SEED)),
    ("knn", KNeighborsClassifier(n_jobs=-1)),
])

# Param grids for each branch
RAW_PARAM_GRID = {
    "pca__n_components": [80, 120, 150, 200],
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan", "cosine"],
}

HOG_PARAM_GRID = {
    "hog__pixels_per_cell": [(8,8), (12,12)],
    "hog__cells_per_block": [(2,2), (3,3)],
    "hog__orientations": [9, 12],
    "pca__n_components": [50, 80, 120],  # HOG is already compact
    "knn__n_neighbors": [5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "cosine"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def run_grid(model, grid, tag):
    gs = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring=["balanced_accuracy","f1_macro","accuracy"],
        refit="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.time()
    gs.fit(X_tr, y_tr)
    t1 = time.time()
    print(f"\n[{tag}] Best params: {gs.best_params_}")
    print(f"[{tag}] CV best (balanced_acc): {gs.best_score_:.4f}  | fit_time={t1-t0:.1f}s")
    return gs

print("\n=== RAW → PCA → KNN ===")
gs_raw = run_grid(raw_pca_knn, RAW_PARAM_GRID, "RAW")

print("\n=== HOG → (scale) → PCA → KNN ===")
gs_hog = run_grid(hog_pca_knn, HOG_PARAM_GRID, "HOG")

# Pick the better one by balanced accuracy on CV first, then confirm on test
if gs_hog.best_score_ >= gs_raw.best_score_:
    best_model = gs_hog.best_estimator_
    best_params = gs_hog.best_params_
    best_tag = "HOG"
else:
    best_model = gs_raw.best_estimator_
    best_params = gs_raw.best_params_
    best_tag = "RAW"

print(f"\n>>> Selected representation by CV: {best_tag}")


# -------------------------------
# Evaluate on held-out test set
# -------------------------------
y_pred = best_model.predict(X_te)
acc = accuracy_score(y_te, y_pred)
bacc = balanced_accuracy_score(y_te, y_pred)
f1m = f1_score(y_te, y_pred, average="macro")

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Balanced Acc: {bacc:.4f}")
print(f"Test Macro-F1: {f1m:.4f}\n")

print("Classification Report:")
print(classification_report(y_te, y_pred, target_names=categories, digits=4, zero_division=0))

cm = confusion_matrix(y_te, y_pred, labels=np.arange(len(categories)))
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

# also show row-normalized for readability
with np.errstate(divide="ignore", invalid="ignore"):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
print("\nConfusion Matrix (row-normalized):")
np.set_printoptions(precision=3, suppress=True)
print(cm_norm)

# -------------------------------
# Simple error analysis: nearest neighbors for a few mistakes
# -------------------------------
def nearest_indices(model, Xq, k=5):
    """
    Compute neighbors for queries Xq using the trained pipeline 'model'.
    This applies all steps before the final KNN (e.g., HOG → Scaler → PCA),
    then runs kneighbors in the trained KNN space.
    """
    # Final step must be KNN
    assert list(model.steps)[-1][0] == "knn", "Pipeline last step must be 'knn'"

    # Transform queries through all pre-knn steps
    Xt = Xq
    for name, step in model.steps[:-1]:
        Xt = step.transform(Xt)

    knn = model.named_steps["knn"]
    dists, idxs = knn.kneighbors(Xt, n_neighbors=k, return_distance=True)
    return dists, idxs

mist = np.where(y_te != y_pred)[0][:10]  # first 10 mistakes
if len(mist):
    print("\nSample mistakes and their top-5 neighbor labels:")
    d, idxs = nearest_indices(best_model, X_te[mist], k=5)
    for j, i in enumerate(mist):
        neigh_labels = y_tr[idxs[j]].tolist()
        neigh_names = [categories[t] for t in neigh_labels]
        print(f"- true={categories[y_te[i]]}, pred={categories[y_pred[i]]}  | nn={neigh_names}")

# -------------------------------
# Save a single portable bundle (pipeline style)
# -------------------------------
bundle = {
    "model": best_model,
    "image_size": IMAGE_SIZE,
    "categories": categories,
    "dataset_hash": ds_hash,       # helps trace which dataset build this model used
    "params": best_params,
    "selected_representation": best_tag,
}
out_path = os.path.join(MODEL_DIR, "knn_pca_pipeline.joblib")
dump(bundle, out_path)
print(f"\nSaved pipeline bundle → {out_path}")
print("Bundle keys: ['model','image_size','categories','dataset_hash','params','selected_representation']")
