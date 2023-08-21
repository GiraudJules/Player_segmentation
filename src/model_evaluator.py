import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def evaluate_model(model, preprocess_input, test_images, test_masks):
    """Evaluate the segmentation model on the provided dataset."""
    # Preprocess the images
    test_images_preprocessed = preprocess_input(test_images)

    # Evaluate the model
    scores = model.evaluate(test_images_preprocessed, test_masks, verbose=1)
    return scores

def visualize_segmentation(test_image, true_mask, model, preprocess_input):
    """Visualize the segmentation result for a sample image."""
    # Preprocess the image and get the prediction
    test_image_preprocessed = preprocess_input(np.expand_dims(test_image, axis=0))
    predicted_mask = model.predict(test_image_preprocessed)
    predicted_mask = np.squeeze(predicted_mask) >= 0.5

    # Plot the original image, true mask, and predicted mask
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_image)
    axes[0].set_title("Original Image")

    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title("True Mask")

    axes[2].imshow(predicted_mask, cmap='gray')
    axes[2].set_title("Predicted Mask")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
