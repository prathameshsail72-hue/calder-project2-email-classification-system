"""
app.py — Streamlit Email Classifier
Calder AI/ML Internship — Project 2 — Prathamesh Sail
"""

import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

st.set_page_config(
    page_title="Email Classifier | Calder",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    model      = joblib.load("model/classifier.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"\{[^}]+\}", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

CAT = {
    "Inquiry":   {"emoji":"🔍","color":"#1a3a8a","bg":"#eef2fb","desc":"Customer requesting information or clarification"},
    "Complaint": {"emoji":"⚠️","color":"#8a1a1a","bg":"#fbeeed","desc":"Customer expressing dissatisfaction or reporting an issue"},
    "Feedback":  {"emoji":"💬","color":"#1a5a3a","bg":"#eef8f2","desc":"Customer sharing opinions, suggestions, or appreciation"},
}

with st.sidebar:
    st.markdown("## 📧 Email Classifier")
    st.markdown("*Calder AI/ML Internship*")
    st.markdown("---")
    st.markdown("""
**Categories:**
- 🔍 **Inquiry** — Info requests
- ⚠️ **Complaint** — Issues & problems
- 💬 **Feedback** — Opinions & suggestions

**Tech Stack:**
- Scikit-learn
- NLTK + TF-IDF
- Logistic Regression
- Streamlit

**Dataset:**
- 1,500 balanced emails
- Kaggle Customer Support Tickets
    """)
    st.markdown("---")
    st.caption("Built by Prathamesh Sail")

st.title("📧 Email Classification System")
st.markdown("*Automatically categorizes incoming emails into Inquiry, Complaint, or Feedback*")
st.markdown("---")

try:
    model, vectorizer = load_model()
    st.success("✅ Model loaded and ready")
except FileNotFoundError:
    st.error("⚠️ Model not found. Run `python src/classifier.py` first.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", "Logistic Regression")
c2.metric("Categories", "3")
c3.metric("Dataset", "1,500 emails")
c4.metric("Vectorizer", "TF-IDF Bigrams")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔎 Single Email", "📋 Batch Classification", "📊 How It Works"])

with tab1:
    st.subheader("Classify a Single Email")
    samples = {
        "Select a sample...": "",
        "📦 Inquiry — Pricing question":
            "I would like to know the pricing details for your premium subscription plan. Could you also explain the difference between the basic and standard tiers?",
        "😤 Complaint — Order not arrived":
            "My order has not arrived after 10 days and your support team has not responded to any of my emails. This is completely unacceptable.",
        "😊 Feedback — Positive review":
            "The new dashboard update is fantastic. It has made my workflow much smoother and the interface is very intuitive. Great work.",
        "❓ Inquiry — Password reset":
            "How do I reset my account password? I cannot log in and the reset email is not arriving in my inbox.",
        "💳 Complaint — Double charge":
            "I was charged twice this month for the same subscription and I need an immediate refund. Please look into this urgently.",
        "💡 Feedback — Feature suggestion":
            "I would suggest adding a dark mode option and keyboard shortcuts. These would make the platform much more efficient.",
    }

    selected    = st.selectbox("Try a sample:", list(samples.keys()))
    email_input = st.text_area("Enter email content:", value=samples[selected],
                               height=150, placeholder="Paste or type email content here...")

    if st.button("🔍 Classify Email", type="primary", use_container_width=True):
        if email_input.strip():
            cleaned = preprocess(email_input)
            tfidf   = vectorizer.transform([cleaned])
            pred    = model.predict(tfidf)[0]
            probs   = model.predict_proba(tfidf)[0]
            scores  = {c: round(float(p)*100, 2) for c, p in zip(model.classes_, probs)}
            cfg     = CAT[pred]

            st.markdown("---")
            st.markdown("### 📊 Result")
            left, right = st.columns([1, 1.5])

            with left:
                st.markdown(f"""
                <div style="background:{cfg['bg']};border:2px solid {cfg['color']};
                            border-radius:12px;padding:28px 20px;text-align:center;">
                    <div style="font-size:48px">{cfg['emoji']}</div>
                    <div style="font-size:26px;font-weight:700;color:{cfg['color']};margin:8px 0">
                        {pred}</div>
                    <div style="font-size:12px;color:#666;margin-bottom:12px">{cfg['desc']}</div>
                    <div style="font-size:20px;font-weight:600;color:{cfg['color']}">
                        {scores[pred]:.1f}% confident</div>
                </div>""", unsafe_allow_html=True)

            with right:
                st.markdown("**Confidence breakdown:**")
                for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    c = CAT[cat]
                    st.markdown(f"{c['emoji']} **{cat}** — {score:.1f}%")
                    st.progress(score / 100)

            with st.expander("🔬 Preprocessing details"):
                st.markdown(f"**Original:** {email_input[:200]}...")
                st.markdown(f"**Preprocessed:** {cleaned[:200]}...")
                st.caption(f"Token count: {len(cleaned.split())}")
        else:
            st.warning("Please enter an email.")

with tab2:
    st.subheader("Classify Multiple Emails")
    st.caption("Enter one email per line")

    default_batch = """I need to know your subscription pricing and available plans.
My account has been charged incorrectly twice this month. I need a refund.
The new interface update is excellent. Great improvement to the platform.
How do I add additional users to my existing account?
The product stopped working after the latest update. Very frustrated.
I would suggest adding an export to PDF feature in a future release."""

    batch_input = st.text_area("Emails (one per line):", value=default_batch, height=200)

    if st.button("🔍 Classify All", type="primary", use_container_width=True):
        if batch_input.strip():
            emails  = [e.strip() for e in batch_input.split("\n") if e.strip()]
            results = []
            for email in emails:
                cleaned = preprocess(email)
                tfidf   = vectorizer.transform([cleaned])
                pred    = model.predict(tfidf)[0]
                probs   = model.predict_proba(tfidf)[0]
                conf    = max(float(p)*100 for p in probs)
                cfg     = CAT[pred]
                results.append({
                    "Email Preview"   : email[:75]+"..." if len(email)>75 else email,
                    "Category"        : f"{cfg['emoji']}  {pred}",
                    "Confidence (%)"  : f"{conf:.1f}%",
                })

            st.markdown("---")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.markdown("**Summary:**")
            m1, m2, m3 = st.columns(3)
            cats = [r["Category"] for r in results]
            m1.metric("🔍 Inquiries",  sum(1 for c in cats if "Inquiry"   in c))
            m2.metric("⚠️ Complaints", sum(1 for c in cats if "Complaint" in c))
            m3.metric("💬 Feedback",   sum(1 for c in cats if "Feedback"  in c))
        else:
            st.warning("Please enter at least one email.")

with tab3:
    st.subheader("Pipeline Overview")
    st.markdown("""
```
Raw Email Text
      ↓
Text Preprocessing
  • Lowercase
  • Remove {placeholder} tokens
  • Remove special characters
  • Stopword removal (NLTK)
  • Lemmatization
      ↓
TF-IDF Vectorization
  • Unigrams + Bigrams
  • Max 5,000 features
  • Sublinear TF scaling
      ↓
Logistic Regression Classifier
  • 80/20 train/test split
  • 5-fold cross validation
      ↓
Predicted Category + Confidence Score
```

### Label Mapping from Kaggle Dataset

| Original Ticket Type | Mapped Category |
|---|---|
| Product inquiry | 🔍 Inquiry |
| Billing inquiry | 🔍 Inquiry |
| Technical issue | ⚠️ Complaint |
| Refund request | ⚠️ Complaint |
| Cancellation request | ⚠️ Complaint |
| Synthetic emails | 💬 Feedback |
    """)