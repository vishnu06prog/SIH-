"""
model.py
The Deep Neural Network (a small feed-forward network / multi-layer perceptron).

    Input (one value per preprocessed feature)
      -> Dense(32) + ReLU
      -> Dropout(0.3)
      -> Dense(16) + ReLU
      -> Dropout(0.3)
      -> Dense(1)  + Sigmoid   => probability of churn (0..1)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(n_features, learning_rate=0.001, dropout_rate=0.3):
    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(16, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


if __name__ == "__main__":
    # Quick look at the architecture and parameter count
    tf.keras.utils.set_random_seed(42)
    build_model(n_features=46).summary()
