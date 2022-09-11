import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

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


def make_model(
    backbone: str,
    decoder: str,
    input_shape: Tuple[int],
    classes: int,
    learning_rate: float,
    loss_function: str,
    metrics: List[str],
    model_name: Optional[str] = None) -> tf.keras.Model:
    """Make a TensorFlow model on top of Keras using `Segmentation models` library.

    The complete list of supported backbones and decoders is available at Segmentation models: https://github.com/qubvel/segmentation_models. 

    Args:
        backbone (str): The backbone of the model.
        decoder (str): The decoder of the model.
        input_shape (Tuple[int]): The input shape of the model in the format `(HEIGHT, WIDTH, CHANNELS)`.
        classes (int): The number of classes the model must predict.
        learning_rate (float): The learning rate to use to train the model.
        loss_function (str): The loss function to be used by the model. Should be a callable or a string supported by TensorFlow.
        metrics (List[str]): A list of metrics to evaluate the model.
        model_name (Optional[str], optional): A name to be added to the model. Defaults to None.

    Returns:
        tf.keras.Model: The compiled model.
    """
    if decoder == "U-Net":
        model = sm.Unet(
            backbone_name=backbone,
            input_shape=input_shape,
            classes=classes,
            activation="softmax" if classes > 1 else "sigmoid",
            encoder_weights="imagenet",
            encoder_freeze=False,
            decoder_block_type="transpose",
            decoder_filters=(512, 256, 128, 64, 32),
            decoder_use_batchnorm=True
        )
    elif decoder == "Linknet":
        model = sm.Linknet(
            backbone_name=backbone,
            input_shape=input_shape,
            classes=classes,
            activation="softmax" if classes > 1 else "sigmoid",
            encoder_weights="imagenet",
            encoder_freeze=False,
            decoder_filters=(None, None, None, None, 16),
            decoder_use_batchnorm=True,
            decoder_block_type="transpose"
        )
    elif decoder == "FPN":
        model = sm.FPN(
            backbone_name=backbone,
            input_shape=input_shape,
            classes=classes,
            activation="softmax" if classes > 1 else "sigmoid",
            encoder_weights="imagenet",
            encoder_freeze=False,
            pyramid_block_filters=256,
            pyramid_use_batchnorm=True,
            pyramid_aggregation="concat",
            pyramid_dropout=None
        )
    elif decoder == "PSPNet":
        model = sm.PSPNet(
            backbone_name=backbone,
            input_shape=input_shape,
            classes=classes,
            activation="softmax" if classes > 1 else "sigmoid",
            encoder_weights="imagenet",
            encoder_freeze=False,
            downsample_factor=8,
            psp_conv_filters=512,
            psp_pooling_type="avg",
            psp_use_batchnorm=True,
            psp_dropout=None,
        )

    if model_name:
        model._name = model_name
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=metrics)
    return model


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


def load_models(
    model_path: str,
    input_shape: Tuple[int, int] = None,
    loss_function: Optional[sm.losses.Loss] = sm.losses.cce_dice_loss,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = Adam(learning_rate=1e-5),
    compile: Optional[bool] = True) -> Union[List[tf.keras.Model], List[str]]:
    """Loads a list of models under a directory.

    Args:
        model_path (str): The path to a directory containing one or more `tf.keras.Model` files. If path is a file, it must be a `.h5` model file.
        input_shape (Tuple[int, int], optional): The input shape the loaded model should have. If not `None`, the function `update_model_input_shape` gets called. Defaults to None.
        loss_function (sm.losses.Loss, optional): The loss function of the model. Defaults to sm.losses.cce_dice_loss.
        optimizer (tf.keras.optimizers.Optimizer, optional): The optimizer of the model. Defaults to Adam(learning_rate=1e-5).
        compile (bool, optional): If false, does not compile the loaded model before returning it. Defaults to True.

    Raises:
        RuntimeError: [description]

    Returns:
        List[tf.keras.Model]: [description]
    """

    model_path = Path(model_path)
    if model_path.is_dir():
        models = [str(path) for path in model_path.glob("*.h5")]
    elif model_path.is_file() and model_path.suffix == ".h5":
        models = [str(model_path)]
    else:
        raise RuntimeError(f"The value of `model_path` is not a model path nor a directory\ containing one or more models:\
             `{str(model_path)}`.")

    for i, model in enumerate(models):
        models[i] = load_model(
            model_path=model,
            input_shape=input_shape,
            loss_function=loss_function,
            optimizer=optimizer,
            compile=compile)

    return models
