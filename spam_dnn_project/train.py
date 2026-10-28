"""Spam Email Detection using a Deep Neural Network (TF-IDF + Dense layers)."""
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

SEED = 42
DATA_PATH = "data/spam.csv"      # columns: label (spam/ham), text
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
np.random.seed(SEED); tf.random.set_seed(SEED)


def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " urltoken ", t)
    t = re.sub(r"\$\d+|\d+", " numtoken ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# 1. Load data
df = pd.read_csv(DATA_PATH).dropna(subset=["label", "text"]).drop_duplicates()
df["y"] = (df["label"].str.lower().str.strip() == "spam").astype(int)
df["clean"] = df["text"].apply(clean_text)
print("Class counts:\n", df["label"].value_counts())

# 2. Split (stratified): 70 / 15 / 15
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    df["clean"], df["y"], test_size=0.30, stratify=df["y"], random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

# 3. TF-IDF features (fit on train only to avoid leakage)
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
Xtr = vec.fit_transform(X_tr).toarray()
Xval = vec.transform(X_val).toarray()
Xte = vec.transform(X_te).toarray()

# 4. DNN model
model = models.Sequential([
    layers.Input(shape=(Xtr.shape[1],)),
    layers.Dense(512, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])
model.summary()

cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
class_weight = {0: cw[0], 1: cw[1]}

cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]
hist = model.fit(Xtr, y_tr, validation_data=(Xval, y_val), epochs=30,
                 batch_size=64, class_weight=class_weight, callbacks=cbs, verbose=2)

# 5. Evaluate on the untouched test set
prob = model.predict(Xte, verbose=0).ravel()
pred = (prob >= 0.5).astype(int)
report = classification_report(y_te, pred, target_names=["ham", "spam"], digits=4)
auc = roc_auc_score(y_te, prob)
print("\n=== TEST RESULTS ===\n", report, f"ROC-AUC: {auc:.4f}")
(OUT / "report.txt").write_text(report + f"\nROC-AUC: {auc:.4f}\n")

# 6. Plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val")
ax[1].set_title("Accuracy"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "training_curves.png", dpi=150); plt.close()

plt.figure(figsize=(4.5, 4))
sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close()

fpr, tpr, _ = roc_curve(y_te, prob)
plt.figure(figsize=(4.5, 4)); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0, 1], [0, 1], "--", c="gray"); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", dpi=150); plt.close()

# 7. Save artifacts
model.save(OUT / "spam_dnn.keras")
joblib.dump(vec, OUT / "tfidf.joblib")
print("Saved model, vectorizer, plots to", OUT)
