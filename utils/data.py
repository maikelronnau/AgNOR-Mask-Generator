from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from skimage.io import imread

from utils.contour_analysis import get_contours


SUPPORTED_IMAGE_TYPES = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]


def one_hot_encode(image: Union[np.ndarray, tf.Tensor], classes: int, as_numpy: Optional[bool] = False) -> tf.Tensor:
    """Converts an image to the one-hot-encode format.

    Args:
        image (Union[np.ndarray, tf.Tensor]): The image to be converted.
        classes (int): The number of classes in the image. Each class will be put in a dimension.
        as_numpy: (Optional[bool], optional): Whether to return the listed files as Numpy comparable objects.

    Raises:
        TypeError: If the type of `image` is not `np.ndarray` or `tf.Tensor`.

    Returns:
        tf.Tensor: The one-hot-encoded image.
    """
    if isinstance(image, np.ndarray):
        one_hot_encoded = np.zeros(image.shape[:2] + (classes,), dtype=np.uint8)
        for unique_value in enumerate(np.unique(image)):
            one_hot_encoded[:, :, unique_value][image == unique_value] = 1
        return one_hot_encoded
    elif isinstance(image, tf.Tensor):
        image = tf.cast(image, dtype=tf.int32)
        image = tf.one_hot(image, depth=classes, axis=2, dtype=tf.int32)
        image = tf.squeeze(image)
        image = tf.cast(image, dtype=tf.float32)

        if as_numpy:
            image = image.numpy()

        return image
    else:
        raise TypeError(f"Argument `image` should be `np.ndarray` or `tf.Tensor. Given `{type(image)}`.")


def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """Normalize image in range [0, 1].

    Args:
        image (tf.Tensor): The imput image.

    Returns:
        tf.Tensor: Normalized image.
    """
    image = tf.cast(image, dtype=tf.float32)

    min = tf.reduce_min(image)
    max = tf.reduce_max(image)

    image = (image - min) / (max - min)

    return image


def load_image(
    image_path: Union[str, tf.Tensor],
    shape: Tuple[int, int] = None,
    normalize: Optional[bool] = False,
    as_gray: Optional[bool] = False,
    as_numpy: Optional[bool] = False,
    return_original_shape: Optional[bool] = False) -> tf.Tensor:
    """Read an image from the storage media.

    Args:
        image_path (Union[str, tf.Tensor]): The path to the image file.
        shape (Tuple[int, int], optional): Shape the read image should have after reading it, in the format (HEIGHT, WIDTH). If `None`, the shape will not be changed. Defaults to None.
        normalize (Optional[bool], optional): Whether or not to put the image values between zero and one ([0,1]). Defaults to False.
        as_gray (Optional[bool], optional): Whether or not to read the image in grayscale. Defaults to False.
        as_numpy: (Optional[bool], optional): Whether to return the listed files as Numpy comparable objects.
        return_original_shape: (Optional[bool], optional): Whether to return the original shape of the image. Defaults to False.

    Raises:
        TypeError: If the image type is not supported.
        FileNotFoundError: If `image_path` does not point to a file.
        TypeError: If `image_path` is not a `str` or `tf.Tensor` object.

    Returns:
        tf.Tensor: The read image.
    """
    original_shape = None

    if isinstance(image_path, str):
        image_path = Path(image_path)

        if image_path.is_file():
            if image_path.suffix not in SUPPORTED_IMAGE_TYPES:
                raise TypeError(f"Image type `{image_path.suffix}` not supported.\
                    The supported types are `{', '.join(SUPPORTED_IMAGE_TYPES)}`.")
            else:
                channels = 1 if as_gray else 3

                if image_path.suffix in [".tif", ".tiff"]:
                    image = imread(str(image_path), as_gray=as_gray)
                    original_shape = image.shape
                    image = tf.convert_to_tensor(image, dtype=tf.float32)
                elif image_path.suffix in [".bmp", ".BMP"]:
                        image = tf.io.read_file(str(image_path))
                        image = tf.image.decode_bmp(image, channels=channels)
                        original_shape = tuple(image.shape.as_list())
                else:
                    image = tf.io.read_file(str(image_path))
                    image = tf.image.decode_png(image, channels=channels)
                    original_shape = tuple(image.shape.as_list())

                if shape:
                    if shape != image.shape[:2]:
                        if "mask" in image_path.stem:
                            image = tf.image.resize(image, shape, method="nearest")
                        else:
                            image = tf.image.resize(image, shape, method="bilinear")

                if normalize:
                    image = normalize_image(image)

                if as_numpy:
                    image = image.numpy()

                    if as_gray:
                        image = image[:, :, 0]

                if return_original_shape:
                    return image, original_shape
                else:
                    return image
        else:
            raise FileNotFoundError(f"The file `{image_path}` was not found.")
    elif isinstance(image_path, tf.Tensor):
        channels = 1 if as_gray else 3
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=channels)

        original_shape = tuple(image.shape.as_list())

        if shape:
            if shape != image.shape[:2]:
                image = tf.image.resize(image, shape, method="nearest")

        if normalize:
            image = tf.cast(image, dtype=tf.float32)
            image = image / 255.

        if return_original_shape:
            return image, original_shape
        else:
            return image
    else:
        raise TypeError("Object `{image_path}` is not supported.")


def list_files(
    files_path: str,
    as_numpy: Optional[bool] = False,
    file_types: Optional[list] = SUPPORTED_IMAGE_TYPES,
    seed: Optional[int] = 1234) -> tf.raw_ops.ShuffleDataset:
    """Lists files under `files_path`.

    Args:
        files_path (str): The path to the directory containing the files to be listed.
        as_numpy: (Optional[bool], optional): Whether to return the listed files as Numpy comparable objects.
        seed (Optional[int], optional): A seed used to shuffle the listed files. Note: If listing images and segmentation masks, the same seed must be used. Defaults to 1234.

    Raises:
        RuntimeError: If no files are found under `files_path`.
        FileNotFoundError: If `files_path` is not a directory.

    Returns:
        tf.raw_ops.ShuffleDataset: The listed files.
    """
    if Path(files_path).is_dir():
        patterns = [str(Path(files_path).joinpath(f"*{image_type}")) for image_type in file_types]
        files_list = tf.data.Dataset.list_files(patterns, shuffle=True, seed=seed)

        if len(files_list) == 0:
            raise RuntimeError(f"No files were found at `{files_path}`.")

        if as_numpy:
            files_list = [file.numpy().decode("utf-8") for file in files_list]
            files_list.sort()

        return files_list
    else:
        raise FileNotFoundError(f"The directory `{files_path}` does not exist.")
