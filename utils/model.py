import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

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
    input_shape: Tuple[int, int, int] = None,
    loss_function: Optional[sm.losses.Loss] = sm.losses.cce_dice_loss,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = Adam(learning_rate=1e-5),
    compile: Optional[bool] = True,
    use_bias_layer: Optional[bool] = False) -> tf.keras.Model:
    """Load a Keras model.

    Args:
        model_path (str): The path to the model file.
        input_shape (Tuple[int, int, int], optional): The input shape the loaded model should have in format `(HEIGHT, WIDTH, CHANNELS)`. If not `None`, the function `update_model_input_shape` gets called. Defaults to None.
        loss_function (sm.losses.Loss, optional): The loss function of the model. Defaults to sm.losses.cce_dice_loss.
        optimizer (tf.keras.optimizers.Optimizer, optional): The optimizer of the model. Defaults to Adam(learning_rate=1e-5).
        compile (bool, optional): If false, does not compile the loaded model before returning it. Defaults to True.
        use_bias_layer (bool, optional): If true, adds a `PapBias` layer to the model. Defaults to False.

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

    if use_bias_layer:
        x = PapBias()(model.layers[-1].output)
        model = tf.keras.Model(inputs=model.input, outputs=x)

    if compile:
        model.compile(optimizer=optimizer, loss=loss_function, metrics=[METRICS])

    return model


class PapBias(tf.keras.layers.Layer):
    """Bias the probabilities of the convolution output of a model.

    This layer adds a bias to the convolution output layer of a model by increasing the probabilities of classes 4 through 7 and setting the probabilities of classes 0 through 3 to 0 where the cluster and cytoplasm masks are 1.
    """
    def __init__(self):
        super(PapBias, self).__init__()

    def call(self, batch):

        def process(prediction):
            prediction_identity = tf.identity(prediction)
            unstacked = tf.unstack(prediction, axis=-1)

            cluster = prediction_identity[:, :, 1] * 127
            cluster = tf.cast(cluster, tf.uint8)

            cytoplasm = prediction_identity[:, :, 2] * 127
            cytoplasm = tf.cast(cytoplasm, tf.uint8)

            def find_contours(mask):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                return contours

            def draw_contours(mask, contours):
                contours = [contour.numpy() if type(contour) != np.ndarray else contour for contour in contours]
                mask = cv2.drawContours(mask, contours, contourIdx=-1, color=1, thickness=cv2.FILLED)
                return mask

            cluster_contours = tf.numpy_function(find_contours, inp=[cluster], Tout=[tf.int32])

            # Check if there are any contours
            if len(cluster_contours) > 0:
                cluster_mask = tf.numpy_function(draw_contours, inp=[tf.zeros(cluster.shape, dtype=tf.uint8), cluster_contours], Tout=tf.uint8)

                # Set class 0 probabilities to 0 where cluster and cytoplasm masks are 1
                unstacked[0] = tf.where(tf.equal(cluster_mask, 1), 0.0, unstacked[0])

            cytoplasm_contours = tf.numpy_function(find_contours, inp=[cytoplasm], Tout=[tf.int32])

            if len(cytoplasm_contours) > 0:
                cytoplasm_mask = tf.numpy_function(draw_contours, inp=[tf.zeros(cytoplasm.shape, dtype=tf.uint8), cytoplasm_contours], Tout=tf.uint8)

                # Set class 0 probabilities to 0 where cluster and cytoplasm masks are 1
                unstacked[0] = tf.where(tf.equal(cytoplasm_mask, 1), 0.0, unstacked[0])

            # Increase the probabilities of classes 4 through 7
            delta = tf.constant(0.001, dtype=tf.float32)

            # Sum the delta to the probabilities of classes 4 through 7
            unstacked[4] = tf.add(unstacked[4], delta)
            unstacked[5] = tf.add(unstacked[5], delta)
            unstacked[6] = tf.add(unstacked[6], delta)

            prediction = tf.stack(unstacked, axis=-1)

            return prediction

        batch = tf.map_fn(process, batch)

        return batch
