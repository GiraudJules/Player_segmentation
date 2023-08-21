import argparse

import numpy as np

from data_loader import (load_and_preprocess_data, load_dataset_annotations,
                         split_data)
from model_builder import build_segmentation_model
from model_evaluator import evaluate_model, visualize_segmentation
from model_predictor import load_trained_model, predict_segmentation_masks
from model_trainer import train_model


def main(args):
    # 1. Load and preprocess the dataset
    dataset_annotations, image_id_to_filename = load_dataset_annotations(args.annotation_path)
    images, masks = load_and_preprocess_data(dataset_annotations, image_id_to_filename, args.image_dir)
    train_images, val_images, test_images, train_masks, val_masks, test_masks = split_data(images, masks)

    # 2. Build and train the segmentation model
    model, preprocess_input = build_segmentation_model()
    model, history = train_model(model, preprocess_input, train_images, train_masks, val_images, val_masks, args.epochs, args.batch_size, args.model_save_path)

    # 3. Evaluate the model on the test set
    scores = evaluate_model(model, preprocess_input, test_images, test_masks)
    print(f"Test Loss: {scores[0]}, Test Accuracy: {scores[1]}")

    # 4. (Optional) Predict on new images
    if args.predict:
        # Assuming you have a function to load new images from a directory
        new_images = load_new_images(args.predict)
        predicted_masks = predict_segmentation_masks(model, preprocess_input, new_images)
        # You can save or further process the predicted masks here

    # 5. (Optional) Visualize some results
    if args.visualize:
        sample_image = test_images[0]
        sample_true_mask = test_masks[0]
        visualize_segmentation(sample_image, sample_true_mask, model, preprocess_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a segmentation model.")

    parser.add_argument('--annotation_path', type=str, default='path_to_annotations.json', help='Path to dataset annotations.')
    parser.add_argument('--image_dir', type=str, default='path_to_image_directory', help='Directory containing images.')
    parser.add_argument('--model_save_path', type=str, default='best_model.h5', help='Path to save the best model during training.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--predict', type=str, help='Directory containing new images for prediction.')
    parser.add_argument('--visualize', action='store_true', help='Flag to enable visualization of results.')

    args = parser.parse_args()
    main(args)
