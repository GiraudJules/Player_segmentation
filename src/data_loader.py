import json

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset_annotations(annotation_path):
    """Load dataset annotations and create a mapping between image IDs and filenames."""
    with open(annotation_path, 'r') as f:
        dataset_annotations = json.load(f)

    image_id_to_filename = {}
    for image in dataset_annotations['images']:
        image_id_to_filename[image['id']] = image['file_name']

    return dataset_annotations, image_id_to_filename


def load_and_preprocess_data(dataset_annotations, image_id_to_filename, image_dir, resized_image_size=256):
    """Load and preprocess images and masks from the dataset."""
    n_images = len(dataset_annotations['images'])
    images = np.zeros((n_images, resized_image_size, resized_image_size, 3), dtype=np.uint8)
    masks = np.zeros((n_images, resized_image_size, resized_image_size), dtype=bool)

    for image_id, image_filename in image_id_to_filename.items():
        # Load and resize the image
        image_path = f"{image_dir}/{image_filename}"
        actual_img = cv2.imread(image_path)
        actual_img = cv2.cvtColor(actual_img, cv2.COLOR_BGR2RGB)
        actual_img = cv2.resize(actual_img, (resized_image_size, resized_image_size))
        images[image_id - 1] = actual_img

        # Load and resize the mask
        for annotation in dataset_annotations['annotations']:
            if annotation['image_id'] == image_id:
                segmentation = annotation['segmentation']
                actual_mask = imantics.Polygons(segmentation).mask(*input_image_size).array
                actual_mask = cv2.resize(actual_mask.astype(float), (resized_image_size, resized_image_size)) >= 0.5
                masks[image_id - 1] = masks[image_id - 1] | actual_mask

    return images, masks


def split_data(images, masks, test_size=0.2, val_size=0.2, random_state=42):
    """Split the dataset into training, validation, and test sets."""
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(images, masks, test_size=test_size, random_state=random_state)
    train_images, val_images, train_masks, val_masks = train_test_split(train_val_images, train_val_masks, test_size=val_size, random_state=random_state)

    return train_images, val_images, test_images, train_masks, val_masks, test_masks
