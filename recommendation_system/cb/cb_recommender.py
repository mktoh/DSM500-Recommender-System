import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

def build_cb_model_pipeline(df):    
    # Dependencies
    item_features = df[['parent_asin', 'item_desc_features', 'item_desc_keyword_features', 'image_features', 'average_rating', 'rating_number']]
    item_features_df = pd.DataFrame(item_features).drop_duplicates(subset=["parent_asin"]).reset_index(drop=True)

    # Convert all features to tensors
    item_ids = tf.convert_to_tensor(item_features_df["parent_asin"].astype(str).values, dtype=tf.string)

    average_rating = tf.convert_to_tensor(item_features_df["average_rating"].values, dtype=tf.float32)
    rating_number = tf.convert_to_tensor(item_features_df["rating_number"].values, dtype=tf.float32)

    item_desc_features = tf.convert_to_tensor(item_features_df['item_desc_features'].tolist(), dtype=tf.float32)
    item_desc_keyword_features = tf.convert_to_tensor(item_features_df['item_desc_keyword_features'].tolist(), dtype=tf.float32)
    image_features = tf.convert_to_tensor(item_features_df['image_features'].tolist(), dtype=tf.float32)

    # Stack all features together (but do NOT compute embeddings yet)
    all_features = tf.concat([
        item_desc_features, item_desc_keyword_features, image_features,
        tf.expand_dims(average_rating, axis=-1),
        tf.expand_dims(rating_number, axis=-1)
    ], axis=1)
    class CBModel(tfrs.Model):
        def __init__(self, num_layers, units_per_layer, dropout_rate, l2_reg_dense, l2_reg_embedding, rating_weight=1.0, retrieval_weight=1.0):
            super().__init__()

            # Feature Processing Networks (Each Feature Group Gets Its Own Layers)
            self.text_dense = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_dense)
            self.text_dense.add(tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dense)))

            self.image_dense = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_dense)
            self.image_dense.add(tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dense)))

            self.numeric_dense = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_dense)
            self.numeric_dense.add(tf.keras.layers.Dense(2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dense)))

            # Item embeddings (NOW computed inside the model)
            self.item_model = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_embedding)

            # Rating model (Predicting Ratings)
            self.rating_model = self._create_dense_block(num_layers, units_per_layer, dropout_rate, l2_reg_dense)
            self.rating_model.add(tf.keras.layers.Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dense)))

            # Compute embeddings dynamically and store properly structured tensors
            self.candidate_ids = tf.reshape(item_ids, (-1,))  # Ensure correct shape
            self.candidate_embeddings = self.item_model(all_features)

            # Use BruteForce Index for Retrieval
            self.index = tfrs.layers.factorized_top_k.BruteForce()
            self.index.index(self.candidate_embeddings, self.candidate_ids)

            # Set up FactorizedTopK retrieval with Properly Structured Data
            self.retrieval_task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(candidates=self.index)
            )

            # Rating Task (Ranking Task)
            self.rating_task = tfrs.tasks.Ranking(
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

            # Loss Weights
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

        def call(self, features):
            """
            Calls the model with a dictionary of features.
            """
            # Process Features
            text_output = self.text_dense(tf.concat([features["item_desc_features"], features["item_desc_keyword_features"]], axis=1))
            image_output = self.image_dense(features["image_features"])

            numeric_features = tf.concat([
                tf.expand_dims(features["average_rating"], axis=-1),
                tf.expand_dims(features["rating_number"], axis=-1)
            ], axis=1)

            numeric_output = self.numeric_dense(numeric_features)
            # Combine All Features
            retrieval_features = tf.concat([text_output, image_output, numeric_output], axis=1)
            item_embeddings = self.item_model(retrieval_features)
            rating_predictions = self.rating_model(item_embeddings)

            return {
                "rating": rating_predictions,
                "retrieval": item_embeddings
            }

        def compute_loss(self, features, training=False):
            """Compute combined retrieval and ranking loss."""
            rating_labels = features["weighted_rating"]
            outputs = self(features)

            retrieval_loss = self.retrieval_task(
                query_embeddings=outputs["retrieval"],
                candidate_embeddings=self.item_model(all_features)
            )

            rating_loss = self.rating_task(labels=rating_labels, predictions=outputs["rating"])

            return self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss
    
    best_cb_model = CBModel(
        num_layers=1,
        units_per_layer=64,
        dropout_rate=0.30000000000000004,
        l2_reg_dense=0.0036593395774260525,
        l2_reg_embedding=0.00020043172008712458,
        rating_weight=0.9,
        retrieval_weight=0.1
    )
    best_cb_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    best_cb_model.load_weights('recommendation_system/cb/cb_weights/weights')
    return best_cb_model