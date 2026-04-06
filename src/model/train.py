import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     Embedding, Flatten, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "clean_listings1.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "model1.pkl")
CITY_IDX_PATH = os.path.join(PROJECT_ROOT, "data", "city_price_index1.json")

data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print(f"Načteno {len(data)} řádků, {len(data.columns)} sloupců")

input_features = [
    "plocha_m2",
    "dispozice_skore",
    "lat", "lon",
    "mhd_dostupnost",
    "skola_m",
    "park_m",
    "mhd_m",
    "lekarna_m",
    "supermarket_m",
    "photo_score",
]
target_feature = "tier"
city_feature = "mesto_enc"

X = data[input_features].values.astype("float32")
X_cat = data[city_feature].values.astype(int)
y = data[target_feature].values.astype(int)
y_cat = to_categorical(y, num_classes=3)

n_mest = data[city_feature].nunique()
embed_dim = 10

X_train, X_test, Xc_train, Xc_test, y_train, y_test, yr_train, yr_test = train_test_split(
    X, X_cat, y_cat, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

inp_num = Input(shape=(len(input_features),), name="numeric")
inp_cat = Input(shape=(1,), name="mesto")

emb = Embedding(
    input_dim=n_mest + 1,
    output_dim=embed_dim,
    embeddings_regularizer=l2(1e-3),
    name="mesto_embedding",
)(inp_cat)
emb = Flatten()(emb)
emb = Dropout(0.3)(emb)

x = Dense(128, activation="relu", kernel_regularizer=l2(5e-4))(inp_num)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)
x = Dense(64, activation="relu", kernel_regularizer=l2(5e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

merged = Concatenate()([x, emb])
merged = Dense(48, activation="relu", kernel_regularizer=l2(5e-4))(merged)
merged = Dropout(0.2)(merged)
out = Dense(3, activation="softmax")(merged)

model = Model(inputs=[inp_num, inp_cat], outputs=out)

model.compile(
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    optimizer="adam",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(
    [X_train_std, Xc_train], y_train,
    batch_size=64,
    epochs=400,
    verbose=1,
    validation_data=([X_test_std, Xc_test], y_test),
    callbacks=[
        EarlyStopping(
            monitor="val_accuracy",
            patience=60,
            restore_best_weights=True,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=20,
            min_lr=1e-6,
        ),
    ],
)

y_pred_proba = model.predict([X_test_std, Xc_test])
y_pred = np.argmax(y_pred_proba, axis=1)

acc = accuracy_score(yr_test, y_pred)
print(f"\nAccuracy: {acc * 100:.1f} %")
print(classification_report(yr_test, y_pred, target_names=["Levné", "Střední", "Drahé"]))

cm = confusion_matrix(yr_test, y_pred)
print("Confusion matrix:")
for i, l in enumerate(["Real.Levné", "Real.Střední", "Real.Drahé"]):
    print(f"  {l:13s}  {cm[i]}")

best_train = max(history.history["accuracy"])
best_val = max(history.history["val_accuracy"])
print(f"\nTrain accuracy: {best_train * 100:.1f}% | Val accuracy: {best_val * 100:.1f}%")

print("\nFeature importance:")
importances = {}
for i, feat in enumerate(input_features):
    X_p = X_test_std.copy()
    np.random.seed(42)
    X_p[:, i] = np.random.permutation(X_p[:, i])
    imp = acc - accuracy_score(yr_test, model.predict([X_p, Xc_test], verbose=0).argmax(axis=1))
    importances[feat] = round(imp, 4)
    print(f"  {feat:25s}  {imp * 100:+5.1f}pp")

imp_emb = acc - accuracy_score(
    yr_test,
    model.predict([X_test_std, np.zeros_like(Xc_test)], verbose=0).argmax(axis=1)
)
importances["mesto (embedding)"] = round(imp_emb, 4)
print(f"  {'mesto (embedding)':25s}  {imp_emb * 100:+5.1f}pp")

le_mesto = LabelEncoder()
le_mesto.fit(data["mesto"].fillna("neznámo"))

poi_medians = {col: float(data[col].median()) for col in
               ["skola_m", "park_m", "mhd_m", "lekarna_m", "supermarket_m"]}

city_index = {}
if os.path.exists(CITY_IDX_PATH):
    with open(CITY_IDX_PATH, encoding="utf-8") as f:
        city_index = json.load(f)

tier_meta_path = os.path.join(PROJECT_ROOT, "data", "tier_meta1.json")
with open(tier_meta_path, encoding="utf-8") as f:
    tier_meta = json.load(f)

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "input_features": input_features,
        "numeric_features": input_features,
        "poi_medians": poi_medians,
        "has_photo": "photo_score" in data.columns,
        "le_mesto": le_mesto,
        "n_mest": n_mest,
        "city_index": city_index,
        "importances": importances,
        "city_disp_medians": tier_meta.get("city_disp_medians", {}),
        "disp_medians": tier_meta.get("disp_medians", {}),
        "t1_rel": tier_meta["t1_rel"],
        "t2_rel": tier_meta["t2_rel"],
        "model_type": "embedding",
    }, f)

print(f"\nModel uložen: {MODEL_PATH}")
print(f"Accuracy: {acc * 100:.1f} %")