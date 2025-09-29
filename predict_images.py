# predict_images.py — seamless image prediction CLI
# Supports two bundle styles saved at training time:
#  1) {"model", "image_size", "categories"}                  # sklearn Pipeline (Scaler→PCA→KNN)
#  2) {"pca", "knn", "image_size", "categories"}             # separate objects

import os
import argparse
import numpy as np
from PIL import Image
from joblib import load

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

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
