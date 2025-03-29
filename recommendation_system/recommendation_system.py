import numpy as np
import pandas as pd
import tensorflow as tf
from recommendation_system.cb.cb_recommender import build_cb_model_pipeline
from recommendation_system.cf.cf_recommender import build_cf_model_pipeline
from recommendation_system.mm.mm_recommender import build_mm_model_pipeline

def build_models(df):
    best_cb_model = build_cb_model_pipeline(df)
    best_cf_model = build_cf_model_pipeline(df)
    best_mm_model = build_mm_model_pipeline(df)
    return best_cb_model, best_cf_model, best_mm_model

def get_hybrid_recommendations(input_features, dataset,  best_cf_model, best_cb_model, best_mm_model,top_n=10, weight_cf=0.3, weight_cb=0.3, weight_mm=0.4):
    """
    Generates top-N hybrid recommendations by integrating Collaborative Filtering (CF),
    Content-Based Filtering (CB), and Multimodal (MM) recommendation models.

    Parameters:
    - input_features: DataFrame containing user and item features.
    - dataset: DataFrame containing item features.
    - unique_item_ids: List of all unique item IDs.
    - top_n: Number of recommendations to return.
    - weight_cf: Weight for Collaborative Filtering recommendations.
    - weight_cb: Weight for Content-Based Filtering recommendations.
    - weight_mm: Weight for Multimodal Filtering recommendations.
    - text_weight: Weight for text-based similarity in MM recommendations.
    - image_weight: Weight for image-based similarity in MM recommendations.
    - sentiment_weight: Weight for sentiment-based similarity in MM recommendations.

    Returns:
    - Sorted list of top-N recommended item IDs ranked based on weighted predicted ratings.
    """

    def prepare_cb_features(df):
        """Prepares features specifically for the CB model."""
        return {
            "item_desc_features": tf.convert_to_tensor(np.stack(df["item_desc_features"].to_list()), dtype=tf.float32),
            "item_desc_keyword_features": tf.convert_to_tensor(np.stack(df["item_desc_keyword_features"].to_list()), dtype=tf.float32),
            "image_features": tf.convert_to_tensor(np.stack(df["image_features"].to_list()), dtype=tf.float32),
            "average_rating": tf.convert_to_tensor(df["average_rating"].values.astype(np.float32), dtype=tf.float32),
            "rating_number": tf.convert_to_tensor(df["rating_number"].values.astype(np.float32), dtype=tf.float32),
        }

    def prepare_mm_features(df):
        """Prepares features specifically for the MM model."""
        return {
            "item_desc_features": tf.convert_to_tensor(np.stack(df["item_desc_features"].to_list()), dtype=tf.float32),
            "item_desc_keyword_features": tf.convert_to_tensor(np.stack(df["item_desc_keyword_features"].to_list()), dtype=tf.float32),
            "image_features": tf.convert_to_tensor(np.stack(df["image_features"].to_list()), dtype=tf.float32),
            "review_features": tf.convert_to_tensor(np.stack(df["review_features"].to_list()), dtype=tf.float32),
            "review_keyword_features": tf.convert_to_tensor(np.stack(df["review_keyword_features"].to_list()), dtype=tf.float32),
            "sentiment_score": tf.convert_to_tensor(df["sentiment_score"].values.astype(np.float32), dtype=tf.float32),
            "emotion_score": tf.convert_to_tensor(df["emotion_score"].values.astype(np.float32), dtype=tf.float32),
        }

    def get_cf_recommendations(user_id, model, df, top_n=100):
        if weight_cf == 0 or user_id not in dataset["user_id"].unique():
            return {}

        items = np.array(np.unique(df['parent_asin']), dtype=str)
        input_dict = {
            "user_id": tf.convert_to_tensor([user_id] * len(items)),
            "item_id": tf.convert_to_tensor(items)
        }

        _, _, ratings = model(input_dict)
        ratings = ratings.numpy().squeeze()  # shape: (num_items,)

        top_indices = np.argsort(ratings)[::-1][:top_n]
        return {items[idx]: ratings[idx] for idx in top_indices}

    def get_cb_recommendations(input_features, model, df, top_n=100):
        """Generates top CB recommendations and predicted ratings."""
        if weight_cb == 0:
            return {}  # Skip CB if weight is 0

        # Extract CB features
        cb_features = df[['parent_asin', 'item_desc_features', 'item_desc_keyword_features', 'image_features',
                        'average_rating', 'rating_number']].drop_duplicates(subset=["parent_asin"]).reset_index(drop=True)
        # Prepare features for embedding
        input_features_tensor = prepare_cb_features(input_features)
        input_embedding = model(input_features_tensor)["retrieval"].numpy()
        
        # Prepare CB features as tensor
        cb_features_tensor = prepare_cb_features(cb_features)
        cb_embeddings = model(cb_features_tensor)["retrieval"].numpy()

        # Compute similarity in **one operation**
        similarity_scores = np.dot(input_embedding, cb_embeddings.T).flatten()
        top_indices = np.argsort(-similarity_scores)[:top_n]

        # **Vectorized retrieval of parent_asins**
        top_asins = cb_features.iloc[top_indices]["parent_asin"].values

        # Batch prediction for all items at once
        predicted_ratings = model(cb_features_tensor)["rating"].numpy().flatten()

        # Map parent_asin to its predicted rating
        return dict(zip(top_asins, predicted_ratings[top_indices]))

    def get_mm_recommendations(input_features, model, df, top_n=100):
        """Generates top MM recommendations and predicted ratings."""
        if weight_mm == 0:
            return {}  # Skip MM if weight is 0
        
        # Extract MM features
        mm_features = df[['parent_asin', 'item_desc_features', 'item_desc_keyword_features', 'image_features',
                        'review_features', 'review_keyword_features', 'sentiment_score', 'emotion_score']
                        ].drop_duplicates(subset=["parent_asin"]).reset_index(drop=True)

        # Convert input features to tensor and get embedding
        input_features_tensor = prepare_mm_features(input_features)
        input_embedding = model(input_features_tensor)["retrieval"].numpy()

        # Convert MM dataset to tensor and get embeddings
        mm_features_tensor = prepare_mm_features(mm_features)
        mm_embeddings = model(mm_features_tensor)["retrieval"].numpy()

        # Compute similarity in **one operation**
        similarity_scores = np.dot(input_embedding, mm_embeddings.T).flatten()
        top_indices = np.argsort(-similarity_scores)[:top_n]
        
        # **Vectorized retrieval of parent_asins**
        top_asins = mm_features.iloc[top_indices]["parent_asin"].values

        # Batch prediction for all items at once (instead of calling model per iteration)
        predicted_ratings = model(mm_features_tensor)["rating"].numpy().flatten()

        # Map parent_asin to its predicted rating using vectorized operations
        return dict(zip(top_asins, predicted_ratings[top_indices]))


    user_id = input_features.iloc[0]["user_id"]

    # Get recommendations for each model, ensuring zero CF scores for new users
    cf_recommendations = get_cf_recommendations(user_id, best_cf_model, dataset, top_n=100)
    cb_recommendations = get_cb_recommendations(input_features, best_cb_model, dataset, top_n=100)
    mm_recommendations = get_mm_recommendations(input_features, best_mm_model, dataset, top_n=100)

    # Merge and weight recommendations
    merged_scores = {}
    all_items = set(cf_recommendations.keys()).union(cb_recommendations.keys(), mm_recommendations.keys())

    for item_id in all_items:
        cf_score = cf_recommendations.get(item_id, 0)
        cb_score = cb_recommendations.get(item_id, 0)
        mm_score = mm_recommendations.get(item_id, 0)
        weighted_score = (weight_cf * cf_score) + (weight_cb * cb_score) + (weight_mm * mm_score)
        merged_scores[item_id] = weighted_score

    # Rank items based on weighted predicted rating
    top_items = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return top_items
