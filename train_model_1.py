# train_model.py
# -----------------------------------------------
# Document Classifier - Model Training Script
# -----------------------------------------------
# This script loads the 20 Newsgroups dataset,
# trains a TF-IDF + Logistic Regression model,
# evaluates it, and saves the model files.
# -----------------------------------------------

import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Define the 5 categories we want to classify ──────────────────────────
CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "talk.politics.misc",
    "soc.religion.christian",
]

print("=" * 60)
print("  Document Classifier – Model Training")
print("=" * 60)

# ── 2. Load training data ────────────────────────────────────────────────────
# remove='headers,footers,quotes' strips metadata so the model learns
# from actual content, not email/newsgroup artifacts.
print("\n[1/5] Loading 20 Newsgroups training data...")
train_data = fetch_20newsgroups(
    subset="train",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    random_state=42,
)

# ── 3. Load test data ────────────────────────────────────────────────────────
print("[2/5] Loading 20 Newsgroups test data...")
test_data = fetch_20newsgroups(
    subset="test",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    random_state=42,
)

print(f"      Training samples : {len(train_data.data)}")
print(f"      Test samples     : {len(test_data.data)}")

# ── 4. Convert text to TF-IDF feature vectors ────────────────────────────────
# TF-IDF (Term Frequency–Inverse Document Frequency) turns raw text into
# numeric features the model can learn from.
# max_features=5000 keeps only the 5000 most important words.
# stop_words="english" removes common words like "the", "is", "at".
print("\n[3/5] Vectorising text with TF-IDF...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train = vectorizer.fit_transform(train_data.data)   # learn vocab + transform
X_test  = vectorizer.transform(test_data.data)         # transform only (no leakage)

y_train = train_data.target
y_test  = test_data.target

print(f"      Feature matrix shape (train): {X_train.shape}")

# ── 5. Train the Logistic Regression classifier ──────────────────────────────
# Logistic Regression works well for text classification and is easy to interpret.
# max_iter=1000 ensures the solver converges on this dataset.
print("\n[4/5] Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ── 6. Evaluate on test set ──────────────────────────────────────────────────
print("\n[5/5] Evaluating model on test set...")
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  ✅ Accuracy: {accuracy * 100:.2f}%\n")
print("  Classification Report:")
print("  " + "-" * 56)
report = classification_report(
    y_test, y_pred, target_names=train_data.target_names
)
# Indent each line for readability
for line in report.splitlines():
    print("  " + line)

# ── 7. Save model artifacts ──────────────────────────────────────────────────
# We save three files so app.py can load them without re-training.
print("\n  Saving model files...")

with open("model.pkl",      "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("labels.pkl",     "wb") as f:
    pickle.dump(train_data.target_names, f)

print("  ✅ model.pkl       saved")
print("  ✅ vectorizer.pkl  saved")
print("  ✅ labels.pkl      saved")
print("\n" + "=" * 60)
print("  Training complete!  Run:  streamlit run app.py")
print("=" * 60 + "\n")
