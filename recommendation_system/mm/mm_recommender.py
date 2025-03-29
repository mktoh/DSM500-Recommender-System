import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

def build_mm_model_pipeline(df):
    #Dependencies
    mm_features_df = df[ ['parent_asin','item_desc_features','item_desc_keyword_features','image_features','review_features','review_keyword_features','sentiment_score','emotion_score']]
    unique_mm_features_df = mm_features_df.groupby('parent_asin').agg({
        'item_desc_features': 'first',  
        'item_desc_keyword_features': 'first',  
        'image_features': 'first',  
        'review_features': lambda x: np.mean(np.vstack(x.tolist()), axis=0) if len(x) > 0 else np.array([]),
        'review_keyword_features': lambda x: np.mean(np.vstack(x.tolist()), axis=0) if len(x) > 0 else np.array([]),
        'sentiment_score': 'mean',
        'emotion_score': 'mean'
    }).reset_index()

    item_ids = tf.convert_to_tensor(unique_mm_features_df["parent_asin"].astype(str).values, dtype=tf.string)

    # Convert numerical features to tf.float32 tensors
    emotion_score = tf.convert_to_tensor(unique_mm_features_df["emotion_score"].values, dtype=tf.float32)
    emotion_score = tf.expand_dims(emotion_score, axis=-1)

    sentiment_score = tf.convert_to_tensor(unique_mm_features_df["sentiment_score"].values, dtype=tf.float32)
    sentiment_score = tf.expand_dims(sentiment_score, axis=-1)

    # Convert all list features to tf.float32 tensors and ensure all are the same type
    item_desc_features = tf.convert_to_tensor(unique_mm_features_df['item_desc_features'].tolist(), dtype=tf.float32)
    item_desc_keyword_features = tf.convert_to_tensor(unique_mm_features_df['item_desc_keyword_features'].tolist(), dtype=tf.float32)
    image_features = tf.convert_to_tensor(unique_mm_features_df['image_features'].tolist(), dtype=tf.float32)
    review_features = tf.convert_to_tensor(unique_mm_features_df['review_keyword_features'].tolist(), dtype=tf.float32)
    review_keyword_features = tf.convert_to_tensor(unique_mm_features_df['review_keyword_features'].tolist(), dtype=tf.float32)


    # Stack all features together
    mm_features = tf.concat([
        item_desc_features,
        item_desc_keyword_features,
        image_features,
        review_features,
        review_keyword_features,
        emotion_score,
        sentiment_score
    ], axis=1)       
    
    class MMModel(tfrs.Model):
        def __init__(self, num_layers, units_per_layer, attention_type, dropout_rate, l2_reg_dense, l2_reg_embedding,
                    num_heads=None, attention_units=64, rating_weight=1.0, retrieval_weight=1.0):
            super().__init__()
            self.attention_type = attention_type

            # Feature Processing Networks
            self.item_desc_dense = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_dense)
            self.item_desc_dense.add(tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dense)))

            self.review_dense = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_dense)
            self.review_dense.add(tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dense)))

            self.image_dense = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_dense)
            self.image_dense.add(tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dense)))

            self.sentiment_dense = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_dense)
            self.sentiment_dense.add(tf.keras.layers.Dense(2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dense)))

            # Conditionally define the attention layer
            if attention_type == 'standard':
                self.attention = tf.keras.layers.Attention(use_scale=True)
            elif attention_type == 'multihead':
                self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=attention_units)

            # Item embeddings and Rating model
            self.item_model = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_embedding)

            self.rating_model = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_dense)
            self.rating_model.add(tf.keras.layers.Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dense)))

            # Compute embeddings dynamically and store properly structured tensors
            self.candidate_ids = tf.reshape(item_ids, (-1,))  # Ensure correct shape
            self.candidate_embeddings = self.item_model(mm_features)

            # Use BruteForce Index for Retrieval
            self.index = tfrs.layers.factorized_top_k.BruteForce()
            self.index.index(self.candidate_embeddings, self.candidate_ids)

            # Set up FactorizedTopK retrieval with Properly Structured Data
            self.retrieval_task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(candidates=self.index)
            )
            self.rating_task = tfrs.tasks.Ranking(
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

            self.rating_weight = rating_weight
            self.retrieval_weight = retrieval_weight

        def _create_dense_block(self, num_layers, units_per_layer, dropout_rate, l2_reg):
            """Creates a sequential dense block with dropout & L2 regularization."""
            model = tf.keras.Sequential()
            for _ in range(num_layers):
                model.add(tf.keras.layers.Dense(
                    units_per_layer, activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
                ))
                model.add(tf.keras.layers.Dropout(dropout_rate))  # Apply dropout
            return model

        def call(self, features, training=False):
            # Combine feature processing
            combined_features = tf.concat([
                self.item_desc_dense(tf.concat([features["item_desc_features"], features["item_desc_keyword_features"]], axis=1)),
                self.image_dense(features["image_features"]),
                self.review_dense(tf.concat([features["review_features"], features["review_keyword_features"]], axis=1)),
                self.sentiment_dense(tf.concat([tf.expand_dims(features["sentiment_score"], axis=-1), tf.expand_dims(features["emotion_score"], axis=-1)], axis=1))
            ], axis=1)

            # Ensure proper dimensions for attention layers
            combined_features = tf.expand_dims(combined_features, axis=1)  # Shape: [batch, 1, feature_dim]

            attention_scores = None  # Default

            # Apply attention based on selected type
            if self.attention_type == "standard":
                attended_output, attention_scores = self.attention([combined_features, combined_features], return_attention_scores=True)

            elif self.attention_type == "multihead":
                # MultiHeadAttention
                attended_output = self.attention(query=combined_features, value=combined_features, key=combined_features)

                # Retrieve attention scores from MultiHeadAttention
                attention_scores = self.attention.compute_mask(  # This is a workaround
                    tf.ones_like(combined_features), combined_features
                )

            attended_output = tf.squeeze(attended_output, axis=1)  # Remove extra dim

            item_embeddings = self.item_model(attended_output)
            rating_predictions = self.rating_model(item_embeddings)

            return {
                "rating": rating_predictions,
                "retrieval": item_embeddings,
                "attention_scores": attention_scores  # Now correctly returned
            }

        def compute_loss(self, features, training=False):
            outputs = self(features, training)
            rating_labels = features["weighted_rating"]

            # Ensure embeddings are float32 for TFRS compatibility
            retrieval_embeddings = tf.cast(outputs["retrieval"], tf.float32)


            retrieval_loss = self.retrieval_task(
                query_embeddings=retrieval_embeddings,  # Item embedding as query
                candidate_embeddings=self.item_model(mm_features)
            )
            rating_loss = self.rating_task(labels=rating_labels, predictions=outputs["rating"])

            return self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss

    best_mm_model = MMModel(
        num_layers=1,
        units_per_layer=256,
        attention_type='multihead',
        dropout_rate=0.4,
        l2_reg_dense= 0.006439485147843038,
        l2_reg_embedding=0.0003167431843224537,
        num_heads=2,
        attention_units=32,
        rating_weight=0.9,
        retrieval_weight=0.1
    )

    best_mm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    best_mm_model.load_weights('recommendation_system/mm/mm_weights/weights')
    return best_mm_model