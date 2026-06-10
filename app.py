# app.py
# -----------------------------------------------
# Document Classifier – Streamlit Web Application
# -----------------------------------------------
# Run with:  streamlit run app.py
# -----------------------------------------------

import pickle
import streamlit as st

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a clean, professional look ────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background and font */
    .main { background-color: #f8f9fb; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.8rem;
    }
    .hero h1 { font-size: 2.2rem; margin: 0 0 0.4rem 0; }
    .hero p  { font-size: 1rem; margin: 0; opacity: 0.9; }

    /* Card-style result box */
    .result-card {
        background: #ffffff;
        border-left: 5px solid #1a73e8;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
    .result-card h3 { margin: 0 0 0.3rem 0; color: #1a73e8; }
    .result-card p  { margin: 0; font-size: 1.05rem; color: #333; }

    /* Category pills in sidebar */
    .cat-pill {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        margin: 3px 2px;
    }

    /* Subtle footer */
    .footer {
        text-align: center;
        color: #9aa0a6;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load saved model artifacts ───────────────────────────────────────────────
@st.cache_resource   # cache so files are loaded only once
def load_artifacts():
    """Load the trained model, vectorizer, and label names from disk."""
    with open("model.pkl",      "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("labels.pkl",     "rb") as f:
        labels = pickle.load(f)
    return model, vectorizer, labels

# ── Map raw newsgroup names to friendly display names ───────────────────────
FRIENDLY_NAMES = {
    "comp.graphics":        "💻 Computer Graphics",
    "rec.sport.baseball":   "⚾ Sports – Baseball",
    "sci.med":              "🩺 Science & Medicine",
    "talk.politics.misc":   "🗳️  Politics",
    "soc.religion.christian": "✝️  Religion – Christianity",
}

CATEGORY_DESCRIPTIONS = {
    "comp.graphics":        "Topics related to computer graphics, rendering, image processing, and visual computing.",
    "rec.sport.baseball":   "Discussions about baseball games, players, teams, statistics, and events.",
    "sci.med":              "Medical questions, health advice, research findings, and clinical discussions.",
    "talk.politics.misc":   "Political opinions, news commentary, policy debates, and social issues.",
    "soc.religion.christian": "Christian theology, biblical discussion, faith experiences, and church life.",
}

# ── Try loading model; show helpful error if files are missing ───────────────
try:
    model, vectorizer, labels = load_artifacts()
    model_ready = True
except FileNotFoundError:
    model_ready = False

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📘 Project Info")
    st.markdown("---")
    st.markdown("**Project**")
    st.markdown("Document Classifier")
    st.markdown("**Model**")
    st.markdown("Logistic Regression")
    st.markdown("**Vectoriser**")
    st.markdown("TF-IDF (5,000 features)")
    st.markdown("**Dataset**")
    st.markdown("20 Newsgroups (5 categories)")
    st.markdown("---")
    st.markdown("**Supported Categories**")
    for raw, friendly in FRIENDLY_NAMES.items():
        st.markdown(f"<span class='cat-pill'>{friendly}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Your text is cleaned and tokenised.\n"
        "2. TF-IDF converts it to a numeric vector.\n"
        "3. Logistic Regression predicts the category."
    )

# ── Hero banner ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>📄 Document Classifier</h1>
        <p>Paste any document text and the AI model will predict its category instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Main content ─────────────────────────────────────────────────────────────
if not model_ready:
    st.error(
        "⚠️ **Model files not found.**\n\n"
        "Please train the model first by running:\n"
        "```\npython train_model.py\n```"
    )
    st.stop()

st.markdown("### ✏️ Enter Your Document Text")
st.markdown(
    "Paste a paragraph or more of text below. "
    "The model works best with at least **2–3 sentences**."
)

# ── Text input area ──────────────────────────────────────────────────────────
user_input = st.text_area(
    label="Document Text",
    placeholder=(
        "e.g.  The pitcher threw a curveball in the final inning, "
        "securing the home team's victory in an intense nine-inning game…"
    ),
    height=220,
    label_visibility="collapsed",
)

# ── Classify button ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    classify_btn = st.button("🔍  Classify Document", use_container_width=True, type="primary")

# ── Prediction logic ─────────────────────────────────────────────────────────
if classify_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before clicking **Classify Document**.")
    else:
        with st.spinner("Analysing your document…"):
            # Transform text using the same vectoriser used during training
            X = vectorizer.transform([user_input])
            # Predict the class index
            prediction_idx = model.predict(X)[0]
            # Get the raw label name (e.g. "sci.med")
            raw_label      = labels[prediction_idx]
            # Map to friendly display name
            friendly_label = FRIENDLY_NAMES.get(raw_label, raw_label)
            description    = CATEGORY_DESCRIPTIONS.get(raw_label, "")

            # Get prediction probabilities for a confidence bar
            proba      = model.predict_proba(X)[0]
            confidence = proba[prediction_idx] * 100

        # ── Display result ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">
                <h3>{friendly_label}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**Confidence:** {confidence:.1f}%")
        st.progress(int(confidence))

        # ── Probability breakdown ─────────────────────────────────────────
        with st.expander("📊 View full probability breakdown"):
            st.markdown("Probability for each category:")
            for i, label in enumerate(labels):
                fname = FRIENDLY_NAMES.get(label, label)
                pct   = proba[i] * 100
                st.markdown(f"**{fname}**")
                st.progress(int(pct))
                st.caption(f"{pct:.1f}%")

# ── Categories reference section ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📂 Supported Categories")
cols = st.columns(2)
for idx, (raw, friendly) in enumerate(FRIENDLY_NAMES.items()):
    with cols[idx % 2]:
        st.markdown(f"**{friendly}**")
        st.caption(CATEGORY_DESCRIPTIONS[raw])

# ── Sample texts ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("💡 Try a sample text"):
    samples = {
        "⚾ Baseball":    "The shortstop made a diving catch in the seventh inning, preserving the lead and thrilling the crowd. The bullpen held firm as the closer struck out the side to seal the win.",
        "💻 Graphics":   "Ray tracing calculates pixel colour by simulating the path of light from the camera through the scene. GPU acceleration has made real-time ray tracing feasible in modern games.",
        "🩺 Medicine":   "The patient presented with acute chest pain radiating to the left arm. An ECG revealed ST-segment elevation, confirming myocardial infarction. Thrombolytic therapy was administered immediately.",
        "🗳️ Politics":   "The senator argued that the proposed tax reform would disproportionately benefit high-income earners. Critics called for a more progressive fiscal policy ahead of the upcoming election.",
        "✝️ Religion":   "The sermon focused on the Sermon on the Mount and the importance of humility and forgiveness in the Christian life. The congregation gathered for communion after the closing prayer.",
    }
    for label, text in samples.items():
        st.markdown(f"**{label}**")
        st.code(text, language=None)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>Built with Streamlit · scikit-learn · 20 Newsgroups Dataset</div>",
    unsafe_allow_html=True,
)
