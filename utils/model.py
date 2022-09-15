import json
from pathlib import Path
from typing import Optional, Tuple

import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


CUSTOM_OBJECTS = {
    "categorical_crossentropy": sm.losses.categorical_crossentropy,
    "categorical_crossentropy_plus_dice_loss": sm.losses.cce_dice_loss,
    "focal_loss_plus_dice_loss": sm.losses.categorical_focal_dice_loss,
    "focal_loss": sm.losses.categorical_focal_loss,
    "f1-score": sm.metrics.f1_score,
    "iou_score": sm.metrics.iou_score,
}

METRICS = [
    sm.metrics.f1_score,
    sm.metrics.iou_score
]


def get_model_input_shape(model: tf.keras.Model) -> tuple:
    """Return the input shape of a Keras model object in the format (height, width, channels).

    Args:
        model (tf.keras.Model): The model to get the input shape from.

    Returns:
        tuple: The model's input shape in the format `(height, width, channels)`.
    """
    _, height, width, channels = model.input_shape
    return (height, width, channels)


def replace_model_input_shape(model: tf.keras.Model, new_input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Replace the model input shape with the new given input shape.

    Args:
        model (tf.keras.Model): The model to have its input shape replaced.
        new_input_shape (Tuple[int, int]): The new input shape in the format `(height, width, channels)`.

    Returns:
        tf.keras.Model: The model with the updated input shape.
    """
    if not new_input_shape:
        raise TypeError("Argument `new_input_shape` is required in function `replace_model_input_shape`.")

    model_weights = model.get_weights()
    model_json = json.loads(model.to_json())

    model_json["config"]["layers"][0]["config"]["batch_input_shape"] = [None, *new_input_shape]

    updated_model = tf.keras.models.model_from_json(json.dumps(model_json))
    updated_model.set_weights(model_weights)
    return updated_model


def load_model(
    model_path: str,
    input_shape: Tuple[int, int] = None,
    loss_function: Optional[sm.losses.Loss] = sm.losses.cce_dice_loss,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = Adam(learning_rate=1e-5),
    compile: Optional[bool] = True) -> tf.keras.Model:
    """Load a Keras model.

    Args:
        model_path (str): The path to the model file.
        input_shape (Tuple[int, int], optional): The input shape the loaded model should have. If not `None`, the function `update_model_input_shape` gets called. Defaults to None.
        loss_function (sm.losses.Loss, optional): The loss function of the model. Defaults to sm.losses.cce_dice_loss.
        optimizer (tf.keras.optimizers.Optimizer, optional): The optimizer of the model. Defaults to Adam(learning_rate=1e-5).
        compile (bool, optional): If false, does not compile the loaded model before returning it. Defaults to True.

    Raises:
        FileNotFoundError: If the model file is not found.

    Returns:
        tf.keras.Model: The loaded model.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"The model file was not found at `{model_path}`.")

    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    if input_shape:
        if input_shape != get_model_input_shape(model):
            model = replace_model_input_shape(model, input_shape)

    if compile:
        model.compile(optimizer=optimizer, loss=loss_function, metrics=[METRICS])

    return model
