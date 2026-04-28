import pandas as pd
import numpy as np
import re
import os
import joblib
import nltk
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

os.makedirs("model", exist_ok=True)

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\{[^}]+\}", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

def load_data(path="data/emails_clean.csv"):
    df = pd.read_csv(path)
    print(f"\n{'='*50}")
    print("DATA LOADING")
    print(f"{'='*50}")
    print(f"Total emails: {len(df)}")
    print(f"\nCategory distribution:")
    print(df["category"].value_counts())
    print("\nPreprocessing text...")
    df["cleaned_text"] = df["email_text"].apply(preprocess)
    print("Preprocessing complete")
    return df

def train_and_evaluate(df):
    X = df["cleaned_text"]
    y = df["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n{'='*50}")
    print("TRAIN / TEST SPLIT")
    print(f"{'='*50}")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"\n{'='*50}")
    print("MODEL TRAINING")
    print(f"{'='*50}")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_tfidf, y_train)
    lr_preds = lr.predict(X_test_tfidf)
    lr_acc   = accuracy_score(y_test, lr_preds)
    print(f"Logistic Regression Accuracy : {lr_acc*100:.2f}%")
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    nb_preds = nb.predict(X_test_tfidf)
    nb_acc   = accuracy_score(y_test, nb_preds)
    print(f"Naive Bayes Accuracy         : {nb_acc*100:.2f}%")
    if lr_acc >= nb_acc:
        best_model, best_preds, model_name = lr, lr_preds, "Logistic Regression"
    else:
        best_model, best_preds, model_name = nb, nb_preds, "Naive Bayes"
    print(f"\nBest Model: {model_name} ({max(lr_acc,nb_acc)*100:.2f}%)")
    print(f"\n{'='*50}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*50}")
    print(classification_report(y_test, best_preds))
    print(f"{'='*50}")
    print("CONFUSION MATRIX")
    print(f"{'='*50}")
    labels = ["Complaint", "Feedback", "Inquiry"]
    cm = confusion_matrix(y_test, best_preds, labels=labels)
    cm_df = pd.DataFrame(cm,
        index=[f"Actual: {l}" for l in labels],
        columns=[f"Pred: {l}" for l in labels])
    print(cm_df)
    print(f"\n{'='*50}")
    print("5-FOLD CROSS VALIDATION")
    print(f"{'='*50}")
    X_full = vectorizer.transform(X)
    cv = cross_val_score(best_model, X_full, y, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")
    joblib.dump(best_model, "model/classifier.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print(f"\nModel saved to model/classifier.pkl")
    print(f"Vectorizer saved to model/vectorizer.pkl")
    return best_model, vectorizer

def predict_single(text, model, vectorizer):
    cleaned  = preprocess(text)
    tfidf    = vectorizer.transform([cleaned])
    category = model.predict(tfidf)[0]
    probs    = model.predict_proba(tfidf)[0]
    scores   = {cls: round(float(p)*100, 2)
                for cls, p in zip(model.classes_, probs)}
    return {"email": text, "predicted_category": category,
            "confidence_scores": scores, "top_confidence": scores[category]}

def predict_batch(emails, model, vectorizer):
    rows = []
    for email in emails:
        r = predict_single(email, model, vectorizer)
        rows.append({
            "Email Preview": email[:80]+"..." if len(email)>80 else email,
            "Predicted Category": r["predicted_category"],
            "Confidence (%)": f"{r['top_confidence']:.1f}%"})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = load_data("data/emails_clean.csv")
    model, vectorizer = train_and_evaluate(df)
    print(f"\n{'='*50}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*50}")
    test_emails = [
        "I would like to know the pricing details for your premium subscription plan.",
        "My order has not arrived after 10 days and your support team has not responded.",
        "The new dashboard update is fantastic and has improved my workflow significantly.",
        "Can you explain the difference between the basic and standard plans?",
        "I was charged twice this month and I need an immediate refund urgently.",
        "I would suggest adding a dark mode feature to improve the user experience.",
        "How do I reset my account password? I cannot log in to my account.",
        "The app keeps crashing every time I try to open it on my phone.",
        "Thank you for the quick resolution. Your support team is truly excellent.",
    ]
    results = predict_batch(test_emails, model, vectorizer)
    print(results.to_string(index=False))