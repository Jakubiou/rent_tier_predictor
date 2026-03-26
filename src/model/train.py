import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "clean_listings_normalized.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "data", "model.pkl")

data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

target = "cenove_pasmo"

X = data.drop(columns=[target])
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Trénovací sada: {len(X_train)} řádků")
print(f"Testovací sada:  {len(X_test)} řádků")
print(f"Features:        {len(X.columns)} sloupců")
print()

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Přesnost (accuracy): {acc:.1%}")
print()
print("Detail podle tříd:")
print(classification_report(
    y_test, y_pred,
    target_names=["levný", "střední", "drahý"]
))

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Model uložen: {MODEL_PATH}")