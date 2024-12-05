import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import imgviz
import numpy as np
import segmentation_models as sm
import tensorflow as tf


MSKG_VERSION = "v16"
ROOT_PATH = str(Path(__file__).parent.parent.resolve())
MODEL_PATH = Path(ROOT_PATH).joinpath("Papanicolaou_DenseNet-169_LinkNet_TSS_544x960x3.h5")
DEFAULT_MODEL_INPUT_SHAPE = (544, 960, 3)


def collapse_probabilities(
    prediction: Union[np.ndarray, tf.Tensor],
    pixel_intensity: Optional[int] = 127) -> Union[np.ndarray, tf.Tensor]:
    """Converts the Softmax probability of each each pixel class to the class with the highest probability.

    Args:
        prediction (Union[np.ndarray, tf.Tensor]): A prediction in the format `(HEIGHT, WIDTH, CLASSES)`.
        pixel_intensity (Optional[int], optional): The intensity each pixel class will be assigned. Defaults to 127.

    Returns:
        Union[np.ndarray, tf.Tensor]: The prediction with the collapsed probabilities into the classes.
    """
    classes = prediction.shape[-1]
    for i in range(classes):
        prediction[:, :, i] = np.where(
            np.logical_and.reduce(
                np.array([prediction[:, :, i] > prediction[:, :, j] for j in range(classes) if j != i])), pixel_intensity, 0)

    return prediction.astype(np.uint8)


def reset_class_values(mask: np.ndarray) -> np.ndarray:
    """Reset the mask values corresponding to classes to start from zero.

    Args:
        mask (np.ndarray): The mask to have its values reset.

    Returns:
        np.ndarray: The reset mask.
    """
    reset_image = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i in range(mask.shape[-1]):
        reset_image[:, :][mask[:, :, i] > 0] = i

    return reset_image


def get_color_map(colormap: Optional[str] = "agnor"):
    """Provides the default color map for the specified number of classes.

    The number of classes refers to the number of dimensions outputted by the model.

    Args:
        colormap (str): What color map to use. Pass `agnor` for a color map for 3 classes plus background, or pass `papanicolaou` for a color map for 7 classes plus background. Defaults to `agnor`.
    """
    if colormap == "agnor":
        # AgNOR
        color_map = [
            [130, 130, 130], # Gray         _background_
            [255, 128,   0], # Orange       citoplasma
            [  0,   0, 255], # Blue         AgNOR
            [128,   0,  64], # Purple       Satellite
        ]
    elif colormap == "papanicolaou":
        # Papanicolaou
        color_map = [
            [130, 130, 130], # Gray         _background_
            [ 78, 121, 167], # Blue         aglomerado
            [242, 142,  43], # Orange       citoplasma
            [ 44, 160,  44], # Green        escama
            [200,  82,   0], # Brown        superficial
            [ 23, 190, 207], # Turquoise    intermediaria
            [188, 189,  34], # Mustard      suspeita
            [148, 103, 189], # Purple       binucleacao
        ]
    else:
        return None

    return color_map


def color_classes(prediction: np.ndarray, colormap: Optional[str] = "agnor") -> np.ndarray:
    """Color a n-dimensional array of one-hot-encoded semantic segmentation image.

    Args:
        prediction (np.ndarray): The one-hot-encoded array image.
        colormap (str): What color map to use. Pass `agnor` for a color map for 3 classes plus background, or pass `papanicolaou` for a color map for 7 classes plus background. Defaults to `agnor`.

    Returns:
        np.ndarray: A RGB image with colored pixels per class.
    """
    # Check if the image encodes the number of classes.
    if len(prediction.shape) >= 3:

        # Extend color map if necessary.
        n_classes = prediction.shape[-1]
        if n_classes > 4:
            color_map = get_color_map(colormap="papanicolaou")
        else:
            color_map = get_color_map(colormap="agnor")

        if n_classes > len(color_map):
            color_map.extend(imgviz.label_colormap(n_label=n_classes))

        # Obtain color map before changing the array.
        class_maps = []
        for i in range(n_classes):
            class_maps.append(prediction[:, :, i] > 0)

        # Recolor classes.
        for i in range(n_classes):
            for j in range(3): # 3 color channels
                prediction[:, :, j] = np.where(class_maps[i], color_map[i][j], prediction[:, :, j])

        # Remove any extra channels so the array can be saved as an image.
        prediction = prediction[:, :, :3]

        return prediction
    else:
        class_values = np.unique(prediction)
        n_classes = len(class_values)

        color_map = get_color_map(colormap=colormap)

        # Extend color map if necessary.
        if n_classes > len(color_map):
            color_map.extend(imgviz.label_colormap(n_label=n_classes))

        # Obtain color map.
        class_maps = []
        for class_value in class_values:
            class_maps.append(prediction == class_value)

        colored = np.zeros(prediction.shape + (3,), dtype=np.uint8)

        # Recolor classes.
        for i, class_value in enumerate(class_values):
            for j in range(3): # 3 color channels
                colored[:, :, j] = np.where(class_maps[i], color_map[class_value][j], colored[:, :, j])

        return colored


def get_intersection(
    expected_contour: np.ndarray,
    predicted_contour: np.ndarray,
    shape: Tuple[int, int]) -> float:
    """Get the intersection value for the input contours.

    The function uses the Intersection Over Union (IoU) metric from the `Segmentation Models` library.

    Args:
        expected_contour (np.ndarray): The first contour.
        predicted_contour (np.ndarray): The second contour.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extracted, in the format `(HEIGHT, WIDTH)`.

    Returns:
        float: The intersection value in range [0, 1].
    """
    expected = np.zeros(shape, dtype=np.uint8)
    predicted = np.zeros(shape, dtype=np.uint8)

    expected = cv2.drawContours(expected, contours=[expected_contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
    predicted = cv2.drawContours(predicted, contours=[predicted_contour], contourIdx=-1, color=1, thickness=cv2.FILLED)

    expected = expected.reshape((1,) + expected.shape).astype(np.float32)
    predicted = predicted.reshape((1,) + predicted.shape).astype(np.float32)

    iou = sm.metrics.iou_score(expected, predicted).numpy()
    return iou


def pad_along_axis(array: np.ndarray, size: int, axis: int = 0, mode: Optional[str] = "constant"):
    """Pad an image along a specific axis.

    Args:
        array (np.ndarray): The image to be padded.
        size (int): The size the padded axis must have.
        axis (int, optional): Which axis to apply the padding. Defaults to 0.
        mode (str, optional): How to fill the padded pixels. Defaults to "constant".

    Returns:
        np.ndarray: The padded image.
    """
    pad_size = size - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    if mode == "constant":
        return np.pad(array, pad_width=npad, mode=mode, constant_values=0)
    else:
        return np.pad(array, pad_width=npad, mode=mode)


def get_labelme_shapes(annotation_path: str, shape_types: Optional[List[str]] = None) -> List[dict]:
    """Get all shapes of the given types from a `labelme` file.

    Args:
        annotation_path (str): The path to the .json `labelme` annotation file.
        shape_types (Optional[list], optional): List of shape types to return. Example: `["polygon", "circle"]. Defaults to None.

    Raises:
        FileNotFoundError: If the annotation file is not found.

    Returns:
        List[dict]: List containing the shapes in the `annotation_file` that are of any type in `shape_types`.
    """
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"The annotation file `{annotation_path}` was not found.")
    else:
        with open(str(annotation_path), "r", encoding="utf-8") as annotation_pointer:
            annotation_file = json.load(annotation_pointer)

        shapes = []
        for shape in annotation_file["shapes"]:
            if shape_types is not None:
                if shape["shape_type"] in shape_types:
                    shapes.append(shape)
            else:
                shapes.append(shape)
        return shapes


def get_labelme_points(annotation_path, shape_types: Optional[list] = None, reshape_for_opencv: Optional[bool] = False) -> List[List[int]]:
    """Get points from shapes from a `labelme` annotation file.

    Args:
        annotation_path (str): The path to the .json `labelme` annotation file.
        shape_types (Optional[list], optional): List of shape types to return. Example: `["polygon", "circle"]. Defaults to None.
        reshape (Optional[bool], optional): Whether or not to reshape points to be compatible with OpenCV. Defaults to False.

    Returns:
        List[List[int]]: List containing the points of the shapes in the `annotation_file` that are of any type in `shape_types`.
    """
    shapes = get_labelme_shapes(annotation_path, shape_types)
    points = []
    for shape in shapes:
        shape_points = shape["points"]
        shape_points = np.array(shape_points, dtype=np.int32)

        if reshape_for_opencv:
            shape_points = shape_points.reshape((shape_points.shape[0], 1, shape_points.shape[1]))
        if shape_types is not None:
            if shape["shape_type"] in shape_types:
                points.append(shape_points)
        else:
            points.append(shape_points)
    return points


def convert_bbox_to_contour(bbox: np.ndarray) -> np.array:
    """Converts a `labelme` bounding box points to a contour.

    Bounding boxes are represented with two points by `labelme`, the top left, and bottom right points.
    When checking if another contour is within a given `labelme` bounding box, OpenCV assumes the two points represent a line.
    This function convert the `labelme` bounding box into a proper contour that will be interpreted as a rectangle in OpenCV.

    Args:
        bbox (np.array): The bounding box points.

    Returns:
        np.array: The contour in the format of a polygon.
    """
    bbox.insert(1, [bbox[1][0], bbox[0][1]])
    bbox.insert(3, [bbox[0][0], bbox[2][1]])
    bbox = np.array(bbox, dtype=np.int32)
    return bbox


def get_object_classes(annotation_path):
    """Get the classes of the objects in an annotation file.

    Args:
        annotation_path (str): The path to the .json `labelme` annotation file.

    Raises:
        FileNotFoundError: If the annotation file is not found.

    Returns:
        List[str]: List containing the classes of the objects in the annotation file.
    """
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"The annotation file `{annotation_path}` was not found.")
    else:
        with open(str(annotation_path), "r") as annotation_pointer:
            annotation_file = json.load(annotation_pointer)

        classes = []
        for shape in annotation_file["shapes"]:
            classes.append(shape["label"])
        return classes


def get_mean_rgb_values(contour: np.ndarray, image: np.ndarray) -> List[Union[float, float, float]]:
    """Obtains the mean RGB values of a given contour in an image.

    Args:
        contour (np.ndarray): The contour that delimits the pixels that will be considered.
        image (np.ndarray): The image from the RGB values will be extracted.

    Returns:
        List[Union[float, float, float]]: List containing the average RGB value, per channel.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cv2.drawContours(mask, contours=[contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
    total = np.sum(mask)
    red = np.round(np.sum(mask * image[:, :, 0]) / total, 4)
    green = np.round(np.sum(mask * image[:, :, 1]) / total, 4)
    blue = np.round(np.sum(mask * image[:, :, 2]) / total, 4)
    return red, green, blue


def get_hash_file(path: str) -> str:
    """Generate and returns the SHA256 hash of a file.

    Args:
        path (str): Path to the file.

    Returns:
        str: The SHA256 hash of the file.
    """
    with open(path, "rb") as f:
        bytes = f.read()
        hash_file = hashlib.sha256(bytes).hexdigest()
    return hash_file


def open_with_labelme(path: str, wait: Optional[int] = 20) -> None:
    """Open `labeleme` on the informed path.

    Args:
        path (str): Path to open with `labelme`.
        wait (int): Time to wait after calling `labelme`.
    """
    logging.debug("Opening labelme")
    if Path("labelme.exe").is_file():
        subprocess.Popen([r"labelme.exe", str(path)])
        logging.debug("Labelme called")
        logging.debug(f"Waiting {wait} before resuming the program")
        time.sleep(wait)
    else:
        logging.debug("Labelme file not found")


def format_combobox_string(item: str, capitalization: Optional[str] = None) -> str:
    """Format combobox string elements.

    Args:
        item (str): The string to be formatted.
        capitalization (Optional[str], optional): How to capitalize the formatted string. Must be one of `title`, `upper`, or `lower`. Defaults to "title".

    Returns:
        str: The formatted string.
    """
    item = item.split(":")[1]
    item = item.strip()
    if capitalization is not None:
        if len(item) <= 3:
            item = item.upper()
        elif capitalization == "title":
            item = item.title()
        elif capitalization == "upper":
            item = item.upper()
        elif capitalization == "lower":
            item = item.lower()
    return item
