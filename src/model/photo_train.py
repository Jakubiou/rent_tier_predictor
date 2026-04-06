import os, json
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "clean_listings1.csv")
PHOTO_DIR    = os.path.join(PROJECT_ROOT, "data", "photos")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "data", "photo_model.keras")
OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "data", "photo_quality1.json")

if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "clean_listings.csv")

IMG_SIZE = 64
N_CLASSES = 3

# ── 1. Načtení dat a příprava labelů ──────────────────
data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"Načteno {len(data)} řádků")

data["cena_za_m2"] = data["cena_kc"] / data["plocha_m2"]

med = data.groupby(["mesto", "dispozice_skore"])["cena_za_m2"].transform("median")
global_med = data["cena_za_m2"].median()
med = med.fillna(global_med).replace(0, global_med)
data["rel_price"] = data["cena_za_m2"] / med

data["quality"] = pd.qcut(
    data["rel_price"], q=3, labels=[0, 1, 2], duplicates="drop"
).astype(int)

QUALITY_NAMES = {0: "starší", 1: "standardní", 2: "moderní"}

print(f"\nNačítám fotky z {PHOTO_DIR}…")
images, labels, ids = [], [], []

for _, row in data.iterrows():
    path = os.path.join(PHOTO_DIR, f"{row['id']}.jpg")
    if not os.path.exists(path):
        continue
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        matrix = np.asarray(img)

        if matrix.shape != (IMG_SIZE, IMG_SIZE, 3):
            continue

        matrix = matrix.astype("float32") / 255

        images.append(matrix)
        labels.append(row["quality"])
        ids.append(str(row["id"]))
    except Exception:
        continue

print(f"Načteno {len(images)} fotek")
print(f"Distribuce: { {QUALITY_NAMES[k]: v for k, v in zip(*np.unique(labels, return_counts=True))} }")

if len(images) < 200:
    print(f"\nMálo fotek ({len(images)}). Spusť nejdřív collector.py.")
    print("Tento krok můžeš přeskočit – train.py funguje i bez photo_quality.json.")
    exit()

X = np.array(images)
y = np.array(labels)

print(f"\nRozměry vstupních dat X: {X.shape}")
print(f"Rozměry výstupních dat y: {y.shape}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train, N_CLASSES)
y_test_cat  = to_categorical(y_test, N_CLASSES)

print(f"Trénovací sada: {len(X_train)}")
print(f"Testovací sada:  {len(X_test)}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dense(N_CLASSES))
model.add(Activation("softmax"))

print(model.summary())

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.fit(
    X_train, y_train_cat,
    batch_size=32,
    epochs=30,
    verbose=1,
    validation_data=(X_test, y_test_cat),
)

y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test_cat, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix

acc = accuracy_score(y_test_class, y_pred_class)
print(f"\nAccuracy: {acc*100:.1f}%")
print(metrics.classification_report(
    y_test_class, y_pred_class,
    target_names=[QUALITY_NAMES[i] for i in range(N_CLASSES)],
    digits=4,
))

print("Confusion matrix:")
print(confusion_matrix(y_test_class, y_pred_class))

model.save(MODEL_PATH)
print(f"\nCNN model uložen → {MODEL_PATH}")

all_pred = model.predict(X)
quality_scores = all_pred[:, 1] * 0.5 + all_pred[:, 2] * 1.0

quality_map = {id_str: round(float(score), 4) for id_str, score in zip(ids, quality_scores)}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(quality_map, f, indent=2)

print(f"Photo quality skóre pro {len(quality_map)} bytů → {OUTPUT_PATH}")