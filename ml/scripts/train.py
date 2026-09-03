# Owner: S2
# Module: DS-CNN Training
# Input:  MFCC numpy arrays from augment.py
# Output: ml/models/naad_kws_float.h5
# Last validated: [date, commit]

import numpy as np
import tensorflow as tf
import os

os.makedirs("ml/models", exist_ok=True)

def build_ds_cnn(n_classes: int = 2) -> tf.keras.Model:
    """
    Depthwise Separable CNN for keyword spotting.
    Input shape: (49, 13, 1) — one MFCC matrix
    Output: softmax over n_classes (index 1 = keyword)

    Architecture:
      Initial Conv2D(64, 10x4) → learns low-level time-frequency patterns
      4x DS blocks: DepthwiseConv2D(3x3) + Conv2D(1x1, 64)
      BatchNorm + ReLU6 after every layer
      GlobalAveragePooling → Dropout(0.2) → Dense(n_classes, softmax)
    """
    inputs = tf.keras.Input(shape=(49, 13, 1), name="mfcc_input")

    # Initial conv: learn low-level time-frequency patterns
    x = tf.keras.layers.Conv2D(
        64, (10, 4), padding='same', use_bias=False, name="initial_conv"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_0")(x)
    x = tf.keras.layers.ReLU(6.0, name="relu_0")(x)

    # 4 DS blocks
    for i in range(4):
        # Depthwise: filter each channel independently
        x = tf.keras.layers.DepthwiseConv2D(
            (3, 3), padding='same', use_bias=False, name=f"dw_conv_{i}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_dw_{i}")(x)
        x = tf.keras.layers.ReLU(6.0, name=f"relu_dw_{i}")(x)

        # Pointwise: mix channels with 1x1 convolution
        x = tf.keras.layers.Conv2D(
            64, (1, 1), padding='same', use_bias=False, name=f"pw_conv_{i}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_pw_{i}")(x)
        x = tf.keras.layers.ReLU(6.0, name=f"relu_pw_{i}")(x)

    # Classifier head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name="classifier")(x)

    return tf.keras.Model(inputs, outputs, name="DS_CNN_KWS")


def train():
    # Load data (produced by augment.py)
    X_train = np.load("ml/data/train_X.npy")
    y_train = np.load("ml/data/train_y.npy")
    X_val   = np.load("ml/data/val_X.npy")
    y_val   = np.load("ml/data/val_y.npy")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Keyword ratio train: {y_train.mean():.2%}")

    model = build_ds_cnn(n_classes=2)
    model.summary()
    print(f"Total parameters: {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(
            "ml/models/naad_kws_best.h5", save_best_only=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    model.save("ml/models/naad_kws_float.h5")
    print("\n=== Training complete ===")
    print(f"Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    return model


if __name__ == "__main__":
    train()
