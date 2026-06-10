# 📄 Document Classifier

> An AI/ML web application that predicts the category of any pasted document text — built with scikit-learn and Streamlit.

---


## 📸 Screenshots

> <img width="959" height="505" alt="image" src="https://github.com/user-attachments/assets/7cf25051-9172-42d8-887e-f48ccbd90cc1" />


---

## ✨ Features

- Paste any document text and classify it instantly
- Powered by **TF-IDF + Logistic Regression** — no paid APIs needed
- Confidence score and full probability breakdown per category
- Clean, professional UI with a sidebar summary and sample texts
- Fully reproducible — train from scratch with a single command

---

## 🛠️ Technologies Used

| Layer | Tool |
|---|---|
| Language | Python 3.9+ |
| ML Framework | scikit-learn |
| Vectoriser | TfidfVectorizer |
| Classifier | LogisticRegression |
| Web UI | Streamlit |
| Model Persistence | pickle |

---

## 📦 Dataset

**20 Newsgroups** (built into scikit-learn — no manual download needed)

Five categories used:

| Raw Label | Friendly Name |
|---|---|
| `comp.graphics` | 💻 Computer Graphics |
| `rec.sport.baseball` | ⚾ Sports – Baseball |
| `sci.med` | 🩺 Science & Medicine |
| `talk.politics.misc` | 🗳️ Politics |
| `soc.religion.christian` | ✝️ Religion – Christianity |

---

## 🧠 How the Model Works

```
Raw Text
   │
   ▼
TfidfVectorizer (max_features=5000, stop_words="english")
   │  Converts text to a sparse numeric matrix
   ▼
LogisticRegression (max_iter=1000)
   │  Learns decision boundaries per category
   ▼
Predicted Category + Confidence Score
```

1. **TF-IDF** scores each word by how often it appears in a document relative to all documents, filtering noise words.
2. **Logistic Regression** learns a set of weights for each category and picks the one with the highest probability.
3. The model is saved as `model.pkl` (along with `vectorizer.pkl` and `labels.pkl`) so the app loads instantly.

---

## 🖥️ Run Locally

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/document-classifier.git
cd document-classifier

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (creates model.pkl, vectorizer.pkl, labels.pkl)
python train_model.py

# 5. Launch the web app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Cloud

1. Push the repository to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**.
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
5. Click **Deploy**.

> **Note:** Streamlit Cloud does NOT have the `.pkl` files yet (they're in `.gitignore`).  
> Remove `model.pkl`, `vectorizer.pkl`, and `labels.pkl` from `.gitignore`, train locally, commit those files, and push. Alternatively, add a startup script — see the next section.

### Option A – Commit model files (simplest for a demo)

```bash
# Temporarily allow pkl files
echo "*.pkl" >> .gitignore  # remove this line instead — edit .gitignore manually
git add model.pkl vectorizer.pkl labels.pkl
git commit -m "Add trained model files for Streamlit Cloud deployment"
git push
```

### Option B – Train on first run (advanced)

Add this to the top of `app.py`:

```python
import os, subprocess
if not os.path.exists("model.pkl"):
    subprocess.run(["python", "train_model.py"], check=True)
```

---

## 📁 Project Structure

```
document-classifier/
├── train_model.py      # Data loading, training, evaluation, saving
├── app.py              # Streamlit web application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Files excluded from Git
```

---

## 📊 Model Performance

After training you should see output similar to:

```
✅ Accuracy: ~95%

                        precision  recall  f1-score  support
comp.graphics               0.95    0.93      0.94      389
rec.sport.baseball          0.98    0.97      0.97      397
sci.med                     0.95    0.96      0.95      396
soc.religion.christian      0.97    0.98      0.97      398
talk.politics.misc          0.93    0.95      0.94      310
```

*(Exact values may vary slightly.)*

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📜 License

MIT

---

*Built as an AI/ML internship project using open-source tools only.*
