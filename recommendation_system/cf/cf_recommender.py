import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

def build_cf_model_pipeline(df):
    unique_item_ids = np.array(df['parent_asin'].unique()) 
    unique_user_ids = np.array(df['user_id'].unique())
    item_ids = tf.data.Dataset.from_tensor_slices(unique_item_ids)
    #Dependancies
    class CFModel(tfrs.Model):
        def __init__(self, embedding_dim, num_layers, units_per_layer, dropout_rate, l2_reg_dense, l2_reg_embedding,
                    rating_weight, retrieval_weight):
            super().__init__()

            # User & Item Embeddings with L2 Regularization
            self.user_model = tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim,
                                        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_embedding))
            ])

            self.item_model = tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=unique_item_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_item_ids) + 1, embedding_dim,
                                        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg_embedding))
            ])

            # Rating model with Dropout & L2 Regularization
            self.rating_model = tf.keras.Sequential()
            for _ in range(num_layers):
                self.rating_model.add(tf.keras.layers.Dense(units_per_layer, activation='relu',
                                                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dense)))
                self.rating_model.add(tf.keras.layers.Dropout(dropout_rate))  # Add dropout here

            self.rating_model.add(tf.keras.layers.Dense(1))  # Output layer

            # Retrieval Task
            self.retrieval_task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=item_ids.batch(128).map(self.item_model)
                )
            )

            # Rating Task
            self.rating_task = tfrs.tasks.Ranking(
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

            # Loss weights
            self.rating_weight = rating_weight
            self.retrieval_weight = retrieval_weight

        def call(self, features):
            user_embeddings = self.user_model(features["user_id"])
            item_embeddings = self.item_model(features["item_id"])
            concatenated_embeddings = tf.concat([user_embeddings, item_embeddings], axis=1)

            return (
                user_embeddings,
                item_embeddings,
                self.rating_model(concatenated_embeddings),
            )

        def compute_loss(self, features, training=False):
            rating_labels = features.pop('weighted_rating')
            user_embeddings, item_embeddings, rating_predictions = self(features)

            retrieval_loss = self.retrieval_task(user_embeddings, item_embeddings)
            rating_loss = self.rating_task(labels=rating_labels, predictions=rating_predictions)

            return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)

    best_cf_model = CFModel(
        embedding_dim=112,
        num_layers=2,
        units_per_layer=128,
        dropout_rate=0.30000000000000004,
        l2_reg_dense=0.0010472536977192324,
        l2_reg_embedding= 0.0001880895482273795,
        rating_weight=0.4,
        retrieval_weight=0.6
    )
    best_cf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
    best_cf_model.load_weights('recommendation_system/cf/cf_weights/weights')
    return best_cf_model