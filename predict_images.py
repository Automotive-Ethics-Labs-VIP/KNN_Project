# predict_images.py — seamless image prediction CLI
# Supports two bundle styles saved at training time:
#  1) {"model", "image_size", "categories"}                  # sklearn Pipeline (Scaler→PCA→KNN)
#  2) {"pca", "knn", "image_size", "categories"}             # separate objects

import os
import argparse
import numpy as np
from PIL import Image
from joblib import load

import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# Try to import the real implementation if available
try:
    from hog_features import HOGTransformer as _RealHOG
except Exception:
    _RealHOG = None

# Backward-compatibility shim for bundles pickled with __main__.HOGTransformer
if _RealHOG is not None:
    # expose it as __main__.HOGTransformer for unpickling
    setattr(sys.modules[__name__], "HOGTransformer", _RealHOG)
else:
    # Fallback minimal implementation (needs scikit-image)
    try:
        from skimage.feature import hog
        from skimage.color import rgb2gray
    except Exception:
        hog = None
        rgb2gray = None

    class HOGTransformer(BaseEstimator, TransformerMixin):  # name must match!
        def __init__(self,
                 image_size=(64, 64),
                 pixels_per_cell=(12, 12),
                 cells_per_block=(2, 2),
                 orientations=9,
                 block_norm="L2-Hys",
                 transform_sqrt=True):
            self.image_size = tuple(image_size)
            self.pixels_per_cell = tuple(pixels_per_cell)
            self.cells_per_block = tuple(cells_per_block)
            self.orientations = orientations
            self.block_norm = block_norm
            self.transform_sqrt = transform_sqrt

        # Backward-compat for old pickles that lack newer attributes
        def __setstate__(self, state):
            self.__dict__.update(state)
            # fill any missing attrs with sensible defaults
            self.image_size = tuple(getattr(self, "image_size", (64, 64)))
            self.pixels_per_cell = tuple(getattr(self, "pixels_per_cell", (12, 12)))
            self.cells_per_block = tuple(getattr(self, "cells_per_block", (2, 2)))
            self.orientations = getattr(self, "orientations", 9)
            self.block_norm = getattr(self, "block_norm", "L2-Hys")
            self.transform_sqrt = getattr(self, "transform_sqrt", True)

        def fit(self, X, y=None):  # stateless
            return self

        def transform(self, X):
            if hog is None or rgb2gray is None:
                raise ImportError("scikit-image is required for HOGTransformer fallback")
            H, W = self.image_size
            X = np.asarray(X, dtype=np.float32).reshape(-1, H, W, 3)
            G = rgb2gray(X)  # expects float in [0,1]
            feats = [hog(img,
                        orientations=self.orientations,
                        pixels_per_cell=self.pixels_per_cell,
                        cells_per_block=self.cells_per_block,
                        block_norm=getattr(self, "block_norm", "L2-Hys"),
                        transform_sqrt=getattr(self, "transform_sqrt", True),
                        feature_vector=True)
                    for img in G]
            return np.vstack(feats).astype(np.float32)


def load_bundle(model_path: str):
    b = load(model_path)
    if "model" in b:                      # pipeline bundle
        return {"kind": "pipeline", **b}
    if "pca" in b and "knn" in b:         # split bundle
        return {"kind": "split", **b}
    raise ValueError(
        "Unrecognized bundle format. Expected keys: "
        "['model','image_size','categories'] or "
        "['pca','knn','image_size','categories']."
    )

def preprocess_image(path: str, image_size):
    # Pillow compatibility for resample enum
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    img = Image.open(path).convert("RGB").resize(tuple(image_size), resample)
    arr = (np.asarray(img, dtype=np.float32) / 255.0).reshape(1, -1)
    return arr

def predict_one(bundle, path: str):
    X = preprocess_image(path, bundle["image_size"])
    categories = bundle["categories"]

    if bundle["kind"] == "pipeline":
        model = bundle["model"]
        y = model.predict(X)[0]
        conf = None
        if hasattr(model, "predict_proba"):
            try:
                conf = float(np.max(model.predict_proba(X)))
            except Exception:
                conf = None
        return categories[y], conf

    # split bundle path (pca + knn)
    pca = bundle["pca"]
    knn = bundle["knn"]
    Xp = pca.transform(X)
    y = knn.predict(Xp)[0]
    conf = None
    if hasattr(knn, "predict_proba"):
        try:
            conf = float(np.max(knn.predict_proba(Xp)))
        except Exception:
            conf = None
    return categories[y], conf

def iter_image_paths(input_path: str):
    if os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            if fname.lower().endswith(IMAGE_EXTS):
                yield os.path.join(input_path, fname)
    else:
        yield input_path

def main():
    ap = argparse.ArgumentParser(description="Predict class for an image file or an entire folder.")
    ap.add_argument("--model", default="models/knn_pca.joblib", help="Path to saved model bundle")
    ap.add_argument("--input", required=True, help="Path to image file OR folder of images")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model bundle not found: {args.model}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path not found: {args.input}")

    bundle = load_bundle(args.model)

    is_dir = os.path.isdir(args.input)
    if is_dir:
        print(f"Predicting for folder: {args.input}\n")

    found = False
    for path in iter_image_paths(args.input):
        found = True
        try:
            label, conf = predict_one(bundle, path)
            base = os.path.basename(path)
            if conf is None:
                print(f"{base} → {label}")
            else:
                print(f"{base} → {label}  (conf≈{conf:.2f})")
        except Exception as e:
            print(f"ERROR on {path}: {e}")

    if not found:
        print("No images found. Accepted extensions: .png .jpg .jpeg")

if __name__ == "__main__":
    main()
