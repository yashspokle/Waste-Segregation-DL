
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_HEAD = 8
EPOCHS_FINE = 5

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def collect_images():
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        folder = DATA_DIR / name
        if not folder.exists():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp"
            }:
                paths.append(str(path))
                labels.append(label)
    if not paths:
        raise RuntimeError(f"No images found in {DATA_DIR}")
    return np.array(paths), np.array(labels)


def decode_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32), tf.one_hot(y, len(CLASSES))),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(architecture):
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ])
    x = augmentation(inputs)

    if architecture == "mobilenetv3":
        base = tf.keras.applications.MobileNetV3Small(
            include_top=False, weights="imagenet",
            input_shape=(*IMG_SIZE, 3)
        )
        model_dir = ROOT / "models" / "mobilenetv3"
    else:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(*IMG_SIZE, 3)
        )
        model_dir = ROOT / "models" / "efficientnet_lite"

    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    return model, base, model_dir


def main():
    paths, labels = collect_images()
    train_paths, temp_paths, train_y, temp_y = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_paths, test_paths, val_y, test_y = train_test_split(
        temp_paths, temp_y, test_size=0.50, random_state=SEED, stratify=temp_y
    )

    print(f"Images: {len(paths)} | Train: {len(train_paths)} | "
          f"Validation: {len(val_paths)} | Test: {len(test_paths)}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for architecture in ["mobilenetv3", "efficientnet_lite"]:
        print(f"\nTraining {architecture}...")
        model, base, model_dir = build_model(architecture)
        train_ds = make_dataset(train_paths, train_y, training=True)
        val_ds = make_dataset(val_paths, val_y)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(model_dir / "best_model.keras"),
                monitor="val_loss", save_best_only=True
            ),
        ]

        model.fit(
            train_ds, validation_data=val_ds,
            epochs=EPOCHS_HEAD, callbacks=callbacks
        )

        # Fine-tune the last portion of the backbone.
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(
            train_ds, validation_data=val_ds,
            epochs=EPOCHS_FINE, callbacks=callbacks
        )

        model = tf.keras.models.load_model(model_dir / "best_model.keras")
        test_ds = make_dataset(test_paths, test_y)
        probabilities = model.predict(test_ds, verbose=0)
        predictions = probabilities.argmax(axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y, predictions, average="macro", zero_division=0
        )
        accuracy = accuracy_score(test_y, predictions)
        summary[architecture] = {
            "accuracy": float(accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "test_images": int(len(test_y)),
        }
        print(summary[architecture])

        del model, base
        tf.keras.backend.clear_session()

    with open(RESULTS_DIR / "metrics" / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nTraining complete. Results saved in results/metrics/comparison.json")


if __name__ == "__main__":
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    main()
