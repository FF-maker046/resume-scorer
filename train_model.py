"""
Model Training & Evaluation Script:
Trains the Resume Quality Classifier and extracts the Top-tier Benchmark Centroid.
Saves model bundle to app/models/quality_model.joblib.
"""
import os
import json
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import jieba

from app.scorer.rules import (
    evaluate_contact,
    evaluate_sections,
    evaluate_quantification,
    extract_skills,
    evaluate_readability_and_length
)
from app.scorer.ml_model import jieba_tokenize


def extract_structural_features(text: str) -> np.ndarray:
    """Extract handcrafted numerical features for quality classification."""
    contact = evaluate_contact(text)
    sections = evaluate_sections(text)
    quant = evaluate_quantification(text)
    skills = extract_skills(text)
    readability = evaluate_readability_and_length(text)

    features = [
        contact["score"] / 100.0,
        sections["score"] / 100.0,
        min(1.0, quant["quant_count"] / 5.0),
        min(1.0, quant["action_count"] / 6.0),
        min(1.0, quant["star_count"] / 3.0),
        min(1.0, skills["total_count"] / 10.0),
        min(1.0, readability["char_count"] / 1500.0),
        readability["score"] / 100.0
    ]
    return np.array(features, dtype=float)


def train():
    corpus_path = os.path.join("data", "resumes_corpus.json")
    if not os.path.exists(corpus_path):
        from data.build_dataset import build_and_save
        build_and_save()

    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_samples = data["positive"] + data["negative"]
    texts = [s["text"] for s in all_samples]
    labels = np.array([s["label"] for s in all_samples])

    print(f"Loaded {len(texts)} samples (Positive: {len(data['positive'])}, Negative: {len(data['negative'])})...")

    # 1. TF-IDF Text Vectorizer
    vectorizer = TfidfVectorizer(
        tokenizer=jieba_tokenize,
        token_pattern=None,
        min_df=1,
        max_features=800,
        sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(texts)

    # 2. Text Classifier (Logistic Regression)
    text_clf = LogisticRegression(C=2.0, max_iter=300)
    text_clf.fit(X_tfidf, labels)

    # 3. Structural Feature Matrix & Classifier (Random Forest)
    X_struct = np.array([extract_structural_features(t) for t in texts])
    struct_clf = RandomForestClassifier(n_estimators=30, random_state=42, max_depth=4)
    struct_clf.fit(X_struct, labels)

    # 4. Compute Positive Benchmark Centroid (大厂高分基准向量中心)
    pos_indices = [i for i, s in enumerate(all_samples) if s["label"] == 1]
    pos_tfidf_vectors = X_tfidf[pos_indices].toarray()
    pos_centroid = np.mean(pos_tfidf_vectors, axis=0, keepdims=True)
    # Normalize centroid
    pos_centroid = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-9)

    # 5. Evaluate Metrics
    pred_text_probs = text_clf.predict_proba(X_tfidf)[:, 1]
    pred_struct_probs = struct_clf.predict_proba(X_struct)[:, 1]
    ensemble_probs = 0.5 * pred_text_probs + 0.5 * pred_struct_probs
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    acc = accuracy_score(labels, ensemble_preds)
    print("\n" + "="*50)
    print(f"📊 模型训练完成！集成评估准确率 (Accuracy): {acc * 100:.2f}%")
    print("="*50)
    print(classification_report(labels, ensemble_preds, target_names=["缺陷/待优化", "优质/大厂级"]))

    # 6. Feature Importance / Vocabulary weights
    vocab = np.array(vectorizer.get_feature_names_out())
    coefs = text_clf.coef_[0]
    top_pos_idx = np.argsort(coefs)[::-1][:10]
    top_neg_idx = np.argsort(coefs)[:10]

    print("🌟 显著正向贡献关键词（高分特征）：")
    print(", ".join([f"{vocab[i]}(+{coefs[i]:.2f})" for i in top_pos_idx]))

    print("\n⚠️ 显著负向贡献关键词（缺陷特征）：")
    print(", ".join([f"{vocab[i]}({coefs[i]:.2f})" for i in top_neg_idx]))

    # 7. Save Model Bundle
    model_bundle = {
        "vectorizer": vectorizer,
        "text_clf": text_clf,
        "struct_clf": struct_clf,
        "pos_centroid": pos_centroid,
        "vocab": vocab.tolist(),
        "coefs": coefs.tolist()
    }
    os.makedirs(os.path.join("app", "models"), exist_ok=True)
    out_model_file = os.path.join("app", "models", "quality_model.joblib")
    joblib.dump(model_bundle, out_model_file)
    print(f"\n✅ 模型已持久化存储至: {out_model_file}\n")


if __name__ == "__main__":
    train()
