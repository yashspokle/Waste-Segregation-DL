
import json
import time
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
MISCLASSIFIED = RESULTS / "misclassified"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_test_data():
    paths, labels = [], []

    for label, class_name in enumerate(CLASSES):
        for path in (DATA_DIR / class_name).rglob("*"):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                paths.append(str(path))
                labels.append(label)

    # Must match the stratified split and seed used in train.py.
    from sklearn.model_selection import train_test_split

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    _, test_paths, _, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50,
        random_state=SEED, stratify=temp_labels
    )

    def preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, test_paths, np.array(test_labels)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    MISCLASSIFIED.mkdir(parents=True, exist_ok=True)

    dataset, test_paths, y_true = load_test_data()
    results = {}

    for model_name in ["mobilenetv3", "efficientnet_lite"]:
        model_path = ROOT / "models" / model_name / "best_model.keras"

        if not model_path.exists():
            print(f"Skipping {model_name}: model file not found: {model_path}")
            continue

        print(f"\nEvaluating {model_name}...")
        model = tf.keras.models.load_model(model_path)

        # Warm-up before timing inference.
        for images, _ in dataset.take(1):
            model(images, training=False)

        start = time.perf_counter()
        predictions = model.predict(dataset, verbose=0)
        elapsed = time.perf_counter() - start

        y_pred = np.argmax(predictions, axis=1)

        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(PLOTS / f"{model_name}_confusion_matrix.png", dpi=200)
        plt.close()

        report = classification_report(
            y_true, y_pred, labels=range(len(CLASSES)),
            target_names=CLASSES, output_dict=True, zero_division=0
        )

        wrong_dir = MISCLASSIFIED / model_name
        wrong_dir.mkdir(parents=True, exist_ok=True)

        # Clear old error images so they don't mix with this run.
        for old_file in wrong_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()

        for i, (actual, predicted) in enumerate(zip(y_true, y_pred)):
            if actual != predicted:
                source = Path(test_paths[i])
                destination = wrong_dir / (
                    f"actual_{CLASSES[actual]}_pred_{CLASSES[predicted]}_{i}"
                    f"{source.suffix.lower()}"
                )
                shutil.copy2(source, destination)

        num_images = len(y_true)
        results[model_name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )),
            "macro_recall": float(recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )),
            "macro_f1": float(f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )),
            "test_images": num_images,
            "inference_seconds": round(elapsed, 4),
            "average_latency_ms_per_image": round(elapsed / num_images * 1000, 3),
            "fps": round(num_images / elapsed, 2),
            "classification_report": report,
            "misclassified_images": int(np.sum(y_true != y_pred)),
        }

        print(f"Accuracy: {results[model_name]['accuracy']:.4f}")
        print(f"Macro F1: {results[model_name]['macro_f1']:.4f}")
        print(f"Latency: {results[model_name]['average_latency_ms_per_image']} ms/image")
        print(f"FPS: {results[model_name]['fps']}")
        print(f"Misclassified: {results[model_name]['misclassified_images']}")

    output_file = RESULTS / "metrics" / "evaluation_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nResults saved to: {output_file}")
    print(f"Confusion matrices saved in: {PLOTS}")
    print(f"Misclassified images saved in: {MISCLASSIFIED}")


if __name__ == "__main__":
    main()
