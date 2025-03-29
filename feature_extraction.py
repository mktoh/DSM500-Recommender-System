import open_clip
import torch
import re
import numpy as np
import spacy
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer

# Init models globally
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = open_clip.create_model("ViT-B-32", pretrained="laion2b_s34b_b79k").to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
try:
    nlp = spacy.load('en_core_web_lg', disable=["parser"])
except OSError:
    from spacy.cli import download
    download('en_core_web_lg')
    nlp = spacy.load('en_core_web_lg', disable=["parser"])
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def batch_process_text_embeddings(texts, batch_size=32):
    texts = [clean_text(text) for text in texts]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(batch_texts).to(device)
        with torch.no_grad():
            features = clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.extend(features.cpu().numpy())
    return np.array(embeddings)

def extract_keywords(texts, batch_size=64):
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        entities = [ent.text for ent in doc.ents]
        adjectives = [t.text for t in doc if t.pos_ == "ADJ"]
        nouns = [t.text for t in doc if t.pos_ == "NOUN"]
        proper_nouns = [t.text for t in doc if t.pos_ == "PROPN"]
        results.append(list(set(entities + adjectives + nouns + proper_nouns)))
    return results

def analyze_sentiment_and_emotions(review, weighted_rating):
    sentiment_score = sia.polarity_scores(review)['compound']
    adjusted_score = sentiment_score * weighted_rating
    sentiment = 'positive' if adjusted_score >= 0.05 else 'negative' if adjusted_score <= -0.05 else 'neutral'
    emotion = 'happiness' if adjusted_score > 0.5 else 'excitement' if adjusted_score > 0.05 else \
              'disappointment' if adjusted_score < -0.5 else 'frustration' if adjusted_score < -0.05 else 'neutral'

    sentiment_score = 1 if sentiment == 'positive' else 0 if sentiment == 'neutral' else -1
    emotion_score = {'excitement': 2, 'happiness': 1, 'neutral': 0, 'disappointment': -1, 'frustration': -2}.get(emotion, 0)

    return sentiment_score, emotion_score

def enrich_review_features(review_text, rating):
    sentiment_score, emotion_score = analyze_sentiment_and_emotions(review_text, rating)
    keyword_list = extract_keywords([review_text])[0]
    keywords_text = " ".join(keyword_list)
    review_feature = batch_process_text_embeddings([review_text])[0]
    keyword_feature = batch_process_text_embeddings([keywords_text])[0]
    return {
        "review_features": review_feature,
        "review_keyword_features": keyword_feature,
        "sentiment_score": sentiment_score,
        "emotion_score": emotion_score
    }