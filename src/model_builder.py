import segmentation_models as sm
import tensorflow as tf


def build_segmentation_model(input_shape=(256, 256, 3), backbone='resnet34', classes=1, activation='sigmoid'):
    """Build and compile the segmentation model."""
    # Choose the model backbone and get the preprocessing function
    preprocess_input = sm.get_preprocessing(backbone)

    # Build the model
    model = sm.Unet(backbone, input_shape=input_shape, classes=classes, activation=activation)

    # Compile the model (you can modify the optimizer, loss, and metrics as needed)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model, preprocess_input
