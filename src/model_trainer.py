import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)


def train_model(model, preprocess_input, train_images, train_masks, val_images=None, val_masks=None, epochs=30, batch_size=32, model_save_path="best_model.h5"):
    """Train the segmentation model."""
    # Preprocess the images
    train_images_preprocessed = preprocess_input(train_images)
    if val_images is not None and val_masks is not None:
        val_images_preprocessed = preprocess_input(val_images)

    # Callbacks for the training process
    callbacks = [
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(patience=10, monitor='val_loss', mode='min'),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, monitor='val_loss', mode='min')
    ]

    # Train the model
    if val_images is not None and val_masks is not None:
        history = model.fit(
            train_images_preprocessed,
            train_masks,
            validation_data=(val_images_preprocessed, val_masks),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
    else:
        history = model.fit(
            train_images_preprocessed,
            train_masks,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

    return model, history
