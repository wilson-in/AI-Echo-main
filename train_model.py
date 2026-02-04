# ======================================================
# AI Echo - Sentiment Analysis
# ======================================================

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# PATH SETUP
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_df.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv(DATA_PATH)

# Rename label column to match documentation
df = df.rename(columns={"sentiment_score": "Sentiment"})

# Keep only required columns
df = df[["Final_cleaned_review", "Sentiment"]]

# ======================================================
# CLASS BALANCING (KEY FIX)
# ======================================================

# Separate classes
df_negative = df[df["Sentiment"] == "Negative"]
df_positive = df[df["Sentiment"] == "Positive"]
df_neutral  = df[df["Sentiment"] == "Neutral"]

# Find majority class size
max_size = max(len(df_negative), len(df_positive), len(df_neutral))

# Upsample minority classes
df_negative_up = resample(
    df_negative,
    replace=True,
    n_samples=max_size,
    random_state=42
)

df_positive_up = resample(
    df_positive,
    replace=True,
    n_samples=max_size,
    random_state=42
)

df_neutral_up = resample(
    df_neutral,
    replace=True,
    n_samples=max_size,
    random_state=42
)

# Combine into balanced dataset
df_balanced = pd.concat([df_negative_up, df_positive_up, df_neutral_up])

# Shuffle dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ======================================================
# FEATURES & TARGET
# ======================================================

X = df_balanced["Final_cleaned_review"]
y = df_balanced["Sentiment"]

# ======================================================
# TRAIN / TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# MODEL PIPELINE
# ======================================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            stop_words="english"
        )
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# ======================================================
# TRAIN MODEL
# ======================================================

pipeline.fit(X_train, y_train)

# ======================================================
# EVALUATION
# ======================================================

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}\n")

print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=pipeline.classes_,
    yticklabels=pipeline.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Sentiment Classification (Balanced)")
plt.tight_layout()
plt.show()

# ======================================================
# SAVE MODEL
# ======================================================

joblib.dump(pipeline, MODEL_PATH)
print(f"\n💾 Balanced model saved successfully at:\n{MODEL_PATH}")
