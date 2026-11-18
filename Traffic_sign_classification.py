# Traffic Sign Classification - DenseNet121 Transfer Learning
# Complete, speed-optimized version (fast post-processing)

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -> no blocking windows
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

# Limit CPU thread oversubscription on Windows (optional but helps)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

import keras
from keras import layers
from keras.applications.densenet import DenseNet121, preprocess_input

# Paths 
BASE     = r"C:\Users\HP\Downloads\kaggle"
CSV_PATH = os.path.join(BASE, "labels.csv")
DATA_DIR = os.path.join(BASE, "traffic_Data", "DATA")
TEST_DIR = os.path.join(BASE, "traffic_Data", "TEST")

# Speed knobs 
IMG_SIZE   = (128, 128)
BATCH_SIZE = 64
EPOCHS     = 8
SEED       = 123

# 1. Dataset info

df = pd.read_csv(CSV_PATH)
num_classes = df["ClassId"].nunique()
id_to_name  = dict(zip(df["ClassId"], df["Name"]))
print(f"Classes: {num_classes}")

# Loader's lexicographic class order
probe = keras.utils.image_dataset_from_directory(
    DATA_DIR, image_size=IMG_SIZE, batch_size=1
)
loader_class_names = probe.class_names
loader_idx_to_name = {i: id_to_name[int(n)] for i, n in enumerate(loader_class_names)}
del probe

# 2. Display 9 sample images

classes = sorted(os.listdir(DATA_DIR))
rng = np.random.RandomState(SEED)
sample_cls = rng.choice(classes, size=9, replace=False)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for ax, cls in zip(axes.flat, sample_cls):
    folder = os.path.join(DATA_DIR, cls)
    fname  = rng.choice(os.listdir(folder))
    img    = Image.open(os.path.join(folder, fname))
    ax.imshow(img)
    ax.set_title(f"{cls}: {id_to_name[int(cls)]}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("samples.png", dpi=100)
plt.close()
print("Saved samples.png")

# 3. Train / validation datasets

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int",
)
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int",
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# 4. Build DenseNet121 transfer-learning model

base_model = DenseNet121(
    weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3),
)
base_model.trainable = False

inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# 5. Train

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Assignment 1: Visualize training results

a, va = history.history["accuracy"], history.history["val_accuracy"]
l, vl = history.history["loss"],     history.history["val_loss"]
r = range(len(a))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(r, a, label="train"); plt.plot(r, va, label="val")
plt.title("Accuracy"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(r, l, label="train"); plt.plot(r, vl, label="val")
plt.title("Loss"); plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=100)
plt.close()
print("Saved training_curves.png")

# Assignment 2: Classification accuracy 

loss_val, acc_val = model.evaluate(val_ds, verbose=0)
print(f"Validation loss: {loss_val:.4f} | Validation accuracy: {acc_val:.4f}")

# Full pass over all validation batches — every class will appear
y_true, y_pred = [], []
for imgs, labels_batch in val_ds:
    probs = model.predict(imgs, verbose=0)
    y_true.extend(labels_batch.numpy())
    y_pred.extend(np.argmax(probs, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

from sklearn.metrics import classification_report, confusion_matrix
names  = [loader_idx_to_name[i] for i in range(num_classes)]
labels = list(range(num_classes))

report = classification_report(
    y_true, y_pred,
    labels=labels,
    target_names=names,
    zero_division=0,
)
print(report)
print("Confusion matrix shape:", confusion_matrix(y_true, y_pred, labels=labels).shape)

with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("Saved classification_report.txt")

# Also save the confusion matrix as an image for the report
cm = confusion_matrix(y_true, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(12, 12))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix (validation set)")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
ax.set_xticklabels(names, rotation=90, fontsize=6)
ax.set_yticklabels(names, fontsize=6)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()
print("Saved confusion_matrix.png")

# Assignment 3: Test on new images (single batched predict)

test_files = sorted(os.listdir(TEST_DIR))[:9]
imgs = []
for fname in test_files:
    img = keras.utils.load_img(os.path.join(TEST_DIR, fname), target_size=IMG_SIZE)
    imgs.append(keras.utils.img_to_array(img))
batch = np.stack(imgs)                     # (9, H, W, 3)
probs = model.predict(batch, verbose=0)    # one call for all 9

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for ax, fname, prob in zip(axes.flat, test_files, probs):
    idx = int(np.argmax(prob))
    ax.imshow(keras.utils.load_img(os.path.join(TEST_DIR, fname),
                                   target_size=IMG_SIZE))
    ax.set_title(f"{loader_idx_to_name[idx]}\n{prob[idx]:.1%}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("test_predictions.png", dpi=100)
plt.close()
print("Saved test_predictions.png")

# Save model

model.save("traffic_sign_densenet121.keras")
print("\nAll done.")