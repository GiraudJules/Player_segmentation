import numpy as np
import tensorflow as tf


def load_trained_model(model_path):
    """Load a trained segmentation model from the specified path."""
    model = tf.keras.models.load_model(model_path)
    return model

def predict_segmentation_masks(model, preprocess_input, images):
    """Predict segmentation masks for the provided images."""
    # Preprocess the images
    images_preprocessed = preprocess_input(images)

    # Make predictions
    predicted_masks = model.predict(images_preprocessed)
    return predicted_masks >= 0.5
