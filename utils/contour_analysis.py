import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import splev, splprep

from utils.utils import (color_classes, convert_bbox_to_contour,
                         get_intersection, get_labelme_points)


MEASUREMENT_COLUMNS = [
    "patient",
    "source_image",
    "id",
    "parent_id",
    "type",
    "pixel_count",
    "nucleus_ratio",
    "smallest_agnor_ratio",
    "greatest_agnor_ratio",
    "flag"
]

CLASSES = [
    "control",
    "leukoplakia",
    "carcinoma",
    "unknown"]

MAX_NUCLEUS_PIXEL_COUNT = 67000
MIN_NUCLEUS_PERCENT_PIXEL_COUNT = 0.02 # 1196

MAX_NOR_PIXEL_COUNT = 3521
MIN_NOR_PERCENT_PIXEL_COUNT = 0.0017 # 1

MAX_CONTOUR_PERCENT_DIFF = 5.0

# Nuclei
# - count: 3300
# - min: 1196
# - max: 66129
# - avg: 15783.546363636364
# - std: 6670.074556984292

# AgNORs
# - count: 12337
# - min: 2
# - max: 3521
# - avg: 92.36167625840966
# - std: 171.46624198587395


def smooth_contours(contours: List[np.ndarray], points: Optional[int] = 30) -> List[np.ndarray]:
    """Smooth a list of contours using a B-spline approximation.

    Args:
        contours (List[np.ndarray]): The contours to be smoothed.
        points (Optional[int], optional): The number of points the smoothed contour should have. Defaults to 30.

    Returns:
        List[np.ndarray]: The smoothed contours.
    """
    smoothened_contours = []
    for contour in contours:
        try:
            x, y = contour.T

            # Convert from numpy arrays to normal arrays
            x = x.tolist()[0]
            y = y.tolist()[0]

            # Find the B-spline representation of an N-dimensional curve.
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
            tck, u = splprep([x, y], u=None, s=1.0, per=1, k=1)

            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
            u_new = np.linspace(u.min(), u.max(), points)

            # Given the knots and coefficients of a B-spline representation, evaluate the value of the smoothing polynomial and its derivatives.
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
            x_new, y_new = splev(u_new, tck, der=0)

            # Convert it back to Numpy format for OpenCV to be able to display it
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
            smoothened_contours.append(np.asarray(res_array, dtype=np.int32))
        except Exception as e:
            print(f"The smoothing of a contour caused a failure: {e}")
    return smoothened_contours


def get_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Find the contours in a binary segmentation mask.

    Args:
        mask (np.ndarray): The segmentation mask.

    Returns:
        List[np.ndarray]: The list of contours found in the mask.
    """
    mask = mask.copy()
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_contour_pixel_count(contour: np.ndarray, shape: List[np.ndarray]) -> int:
    """Counts the number of pixels in a given contour.

    Args:
        contour (np.ndarray): The contour to be evaluated.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.

    Returns:
        int: The number of pixels in the contour.
    """
    image = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(image, contours=[contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
    return int(image.sum())


def dilate_contours(
    contours: List,
    structuring_element: Optional[int] = cv2.MORPH_ELLIPSE,
    kernel: Optional[Tuple[int, int]] = (3, 3),
    iterations: Optional[int] = 1,
    shape: Optional[Tuple[int, int]]= None) -> np.ndarray:
    """Dilate a list of contours using the specified morphological operator and kernel size.

    Args:
        contours (List): The list of contours to be dilated.
        structuring_element (Optional[int], optional): The morphological transform to apply. Defaults to `cv2.MORPH_ELLIPSE`.
        kernel (Optional[Tuple[int, int]], optional): The size of the kernel to be used. Defaults to `(3, 3)`.
        iterations (Optional[int], optional): Number of iterations to apply the dilatation. Defaults to 1.
        shape (Optional[Tuple[int, int]], optional): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.

    Returns:
        np.ndarray: _description_
    """
    if shape is not None:
        mask = np.zeros(shape, dtype=np.uint8)
    else:
        max_value = 0
        for contour in contours:
            max_value = contour.max() if contour.max() > max_value else max_value
        mask = np.zeros((max_value * 2, max_value * 2), dtype=np.uint8)

    mask = cv2.drawContours(mask, contours=contours, contourIdx=-1, color=[255, 255, 255], thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(structuring_element, kernel)
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    return mask


def discard_contours_by_size(
    contours: List[np.ndarray],
    shape: Tuple[int, int],
    max_pixel_count: Optional[int] = MAX_NUCLEUS_PIXEL_COUNT,
    min_relative_pixel_count: Optional[float] = MIN_NUCLEUS_PERCENT_PIXEL_COUNT) -> Union[List[np.ndarray], List[np.ndarray]]:
    """Discards contours smaller or bigger than the given thresholds.

    Args:
        contours (List[np.ndarray]): The contours to be evaluated.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.
        max_pixel_count (Optional[int], optional): The maximum number of pixels the contour must have. Defaults to `MAX_NUCLEUS_PIXEL_COUNT`.
        min_relative_pixel_count (Optional[float], optional): The minimal percent of pixels the contour must have in range `[0, 100]`. Defaults to `MIN_NUCLEUS_PERCENT_PIXEL_COUNT`.

    Returns:
        Union[List[np.ndarray], List[np.ndarray]]: The `kept` array contains the contours that are withing the size specification. The `discarded` array contains the contours that are not withing the size specification.
    """
    kept = []
    discarded = []
    for contour in contours:
        contour_pixel_count = get_contour_pixel_count(contour, shape=shape)
        min_pixel_count = int((min_relative_pixel_count * max_pixel_count) / 100)
        if min_pixel_count <= contour_pixel_count and contour_pixel_count <= max_pixel_count:
            kept.append(contour)
        else:
            discarded.append(contour)
    return kept, discarded


def discard_contours_without_contours(
    parent_contours: List[np.ndarray],
    child_contours: List[np.ndarray]) -> Union[List[np.ndarray], List[np.ndarray]]:
    """Discards contours that do not have other contours inside.

    Args:
        parent_contours (List[np.ndarray]): The list of contours to be considered as the parent contours.
        child_contours (List[np.ndarray]): The list of contours to be considered as child contours of the parent contours.

    Returns:
        Union[List[np.ndarray], List[np.ndarray]]: The `kept` array contains the contours that have other contours inside. The `discarded` array contains the contours that do not have other contours inside.
    """
    kept = []
    discarded = []
    for parent in parent_contours:
        keep_parent = False
        for child in child_contours:
            for child_point in child:
                if cv2.pointPolygonTest(parent, tuple(child_point[0]), False) >= 0:
                    keep_parent = True
        if keep_parent:
            kept.append(parent)
        else:
            discarded.append(parent)
    return kept, discarded


def discard_contours_outside_contours(
    parent_contours: List[np.ndarray],
    child_contours: List[np.ndarray]) -> Union[List[np.ndarray], List[np.ndarray]]:
    """Discards contours that are outside other contours.

    This function iterates over all child contours and checks if at least one point is inside of at least one parent contour.

    Args:
        parent_contours (List[np.ndarray]): The list of contours to be considered as the parent contours.
        child_contours (List[np.ndarray]): The list of contours to be considered as child contours of the parent contours.

    Returns:
        Union[List[np.ndarray], List[np.ndarray]]: The `kept` array contains the child contours that are within a parent contour. The `discarded` array contains the child contours that are not in any parent contour.
    """
    kept = []
    discarded = []
    for child in child_contours:
        keep_child = False
        for parent in parent_contours:
            for child_point in child:
                if cv2.pointPolygonTest(parent, tuple(child_point[0]), False) >= 0:
                    keep_child = True
                    break
            if keep_child:
                break
        if keep_child:
            kept.append(child)
        else:
            discarded.append(child)
    return kept, discarded


def discard_overlapping_deformed_contours(
    contours: List[np.ndarray],
    shape: Tuple[int, int],
    max_diff: Optional[float] = MAX_CONTOUR_PERCENT_DIFF) -> Union[List[np.ndarray], List[np.ndarray]]:
    """Discards contours overlapping with other and defformed contours.

    This function verifies if contours are overlapping with others by computing the difference in the number of pixels between the contour and the convex hull of that contour.
    If the difference exceeds the given threshold (`diff`), then the contour is discarded as it is likely overlapping with another one.
    This assumes the nature of the objects being segmented, which are nuclei and NORs in AgNOR-stained images, which tend to have circular shapes.
    Deformed contours caused by obstruction or fragmented segmentation also get discarded under the same criterion.

    Args:
        contours (List[np.ndarray]): The list of contours to be evaluated.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extrated, in the format `(HEIGHT, WIDTH)`.
        max_diff (Optional[int], optional): The maximum percentage difference between the contour pixel count and its convex hull pixel count. If the difference is over `max_diff`, then the contour is discarded. Defaults to `MAX_CONTOUR_PERCENT_DIFF`.

    Returns:
        Union[List[np.ndarray], List[np.ndarray]]: The `kept` array contains the contours that are not overlapping with other or are not deformed. The `discarded` array contains the overlapping and deforemed nuclei.
    """
    kept = []
    discarded = []
    for contour in contours:
        contour_pixel_count = get_contour_pixel_count(contour, shape)
        contour_convex_pixel_count = get_contour_pixel_count(cv2.convexHull(contour), shape)
        diff = ((contour_convex_pixel_count - contour_pixel_count) / ((contour_convex_pixel_count + contour_pixel_count) / 2)) * 100
        if diff <= max_diff:
            kept.append(contour)
        else:
            discarded.append(contour)
    return kept, discarded


def draw_contour_lines(image: np.ndarray, contours: List[np.ndarray], type: Optional[str] = "multiple") -> np.ndarray:
    """Draw the line of contours.

    Args:
        image (np.ndarray): The image to draw the contours on.
        contours (List[np.ndarray]): The list of the contours to be drawn.
        type (Optional[str]): The type of contours to draw. If `multiple`, draws the segmented contour, the convex hull contour and the overlaps between the segmented and convex contours. If `single`, draws only the segmented contour. Defaults to "multiple".

    Raises:
        ValueError: If `type` is not in [`multiple`, `single`].

    Returns:
        np.ndarray: The image with the contours drawn on it.
    """
    if type == "multiple":
        contour = np.zeros(image.shape, dtype=np.uint8)
        contour_convex = np.zeros(image.shape, dtype=np.uint8)

        cv2.drawContours(contour, contours=contours, contourIdx=-1, color=[1, 1, 1], thickness=1)
        cv2.drawContours(
            contour_convex, contours=[cv2.convexHull(discarded_contour) for discarded_contour in contours], contourIdx=-1, color=[1, 1, 1], thickness=1)

        diff = contour + contour_convex
        diff[diff < 2] = 0
        diff[diff == 2] = 1

        # Yellow = Smoothed contour
        cv2.drawContours(image, contours=contours, contourIdx=-1, color=[255, 255, 0], thickness=1)
        # Cyan = Convex hull of the smoothed contour
        cv2.drawContours(
            image, contours=[cv2.convexHull(discarded_contour) for discarded_contour in contours], contourIdx=-1, color=[0, 255, 255], thickness=1)

        # White = Smoothed contour equals to Convex hull of the smoothed contour
        image = np.where(diff > 0, [255, 255, 255], image)
    elif type == "single":
        image = cv2.drawContours(image, contours=contours, contourIdx=-1, color=[255, 255, 255], thickness=1)
    else:
        raise ValueError("Argument `type` must be either `multiple` or `single`.")

    return image.astype(np.uint8)


def analyze_contours(
    mask: Union[np.ndarray, tf.Tensor],
    smooth: Optional[bool] = False) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Analyze the segmented contours and updates the segmentation mask.

    Args:
        mask (Union[np.ndarray, tf.Tensor]): The mask containing the contours to be analyzed.
        smooth (Optional[bool], optional): Whether or not to smooth the contours.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: The first tuple contains the updated mask, and the nuclei and NORs contours. The second one contains the image with the discarded nuclei contours and AgNORs, and the discarded nuclei and NORs contours.
    """
    # Obtain and filter nuclei and NORs contours
    nuclei_contours = get_contours(mask[:, :, 1] + mask[:, :, 2])
    nuclei_contours, nuclei_size_discarded = discard_contours_by_size(nuclei_contours, shape=mask.shape[:2])

    nors_contours = get_contours(mask[:, :, 2])
    nors_contours, nors_size_discarded = discard_contours_by_size(
        nors_contours, shape=mask.shape[:2], max_pixel_count=MAX_NOR_PIXEL_COUNT, min_relative_pixel_count=MIN_NOR_PERCENT_PIXEL_COUNT)

    if smooth:
        nuclei_contours = smooth_contours(nuclei_contours, points=40)
        nors_contours = smooth_contours(nors_contours, 16)

    nuclei_with_nors, nuclei_without_nors = discard_contours_without_contours(nuclei_contours, nors_contours)
    nuclei_contours_adequate, nuclei_overlapping_deformed = discard_overlapping_deformed_contours(
        nuclei_with_nors, shape=mask.shape[:2])

    nors_in_adequate_nuclei, _ = discard_contours_outside_contours(nuclei_contours_adequate, nors_contours)
    nors_in_overlapping_deformed, _ = discard_contours_outside_contours(nuclei_overlapping_deformed, nors_contours)

    # Create a new mask with the filtered nuclei and NORs
    pixel_intensity = int(np.max(np.unique(mask)))
    background = np.ones(mask.shape[:2], dtype=np.uint8)
    nucleus = np.zeros(mask.shape[:2], dtype=np.uint8)
    nor = np.zeros(mask.shape[:2], dtype=np.uint8)

    cv2.drawContours(nucleus, contours=nuclei_contours_adequate, contourIdx=-1, color=pixel_intensity, thickness=cv2.FILLED)
    cv2.drawContours(nor, contours=nors_in_adequate_nuclei, contourIdx=-1, color=pixel_intensity, thickness=cv2.FILLED)

    nucleus = np.where(nor, 0, nucleus)
    background = np.where(np.logical_and(nucleus == 0, nor == 0), pixel_intensity, 0)
    updated_mask = np.stack([background, nucleus, nor], axis=2).astype(np.uint8)

    contour_detail = mask.copy()
    contour_detail = color_classes(contour_detail).copy()

    if len(nuclei_size_discarded) > 0:
        contour_detail = draw_contour_lines(contour_detail, nuclei_size_discarded, type="single")

    if len(nuclei_without_nors) > 0:
        contour_detail = draw_contour_lines(contour_detail, nuclei_without_nors, type="single")

    if len(nors_size_discarded) > 0:
        contour_detail = draw_contour_lines(contour_detail, nors_size_discarded, type="single")

    if len(nuclei_overlapping_deformed) > 0:
        contour_detail = draw_contour_lines(contour_detail, nuclei_overlapping_deformed)
    else:
        nuclei_overlapping_deformed, nors_in_overlapping_deformed = [], []
        if len(nuclei_size_discarded) == 0 and len(nuclei_without_nors) == 0:
            contour_detail = None

    return (updated_mask, nuclei_contours_adequate, nors_in_adequate_nuclei),\
        (contour_detail, nuclei_overlapping_deformed, nors_in_overlapping_deformed)


def get_contour_measurements(
    parent_contours: List[np.ndarray],
    child_contours: List[np.ndarray],
    shape: Tuple[int, int],
    mask_name: str,
    min_contour_size: Optional[int] = None,
    max_contour_size: Optional[int] = None,
    parent_type: Optional[str] = "nucleus",
    child_type: Optional[str] = "agnor",
    record_id: Optional[str] = "unknown",
    start_index: Optional[int] = 0,
    contours_flag: Optional[str] = "valid") -> Union[List[dict], List[dict], int, int]:
    """Calculate the number of pixels per contour and create a record for each of them.

    Args:
        parent_contours (List[np.ndarray]): The parent contours.
        child_contours (List[np.ndarray]): The child contours.
        shape (Tuple[int, int]): The dimensions of the image from where the contours were extracted, in the format `(HEIGHT, WIDTH)`.
        mask_name (str): The name of the mask from where the contours were extracted.
        min_contour_size (Optional[int], optional): Pixel count of the smallest `cluster`. Defaults to None.
        max_contour_size (Optional[int], optional): Pixel count of the greatest `cluster`. Defaults to None.
        parent_type (Optional[str], optional): The type of the parent contour. Defaults to "nucleus".
        child_type (Optional[str], optional): The type of the child contour. Usually one of `["cluster", "satellite"]`. Defaults to "cluster".
        record_id (Optional[str]): The unique ID of the record. Defaults to "unknown".
        start_index (Optional[int], optional): The index to start the parent contour ID assignment. Usually it will not be `0` when discarded records are being measure for record purposes. Defaults to 0.
        contours_flag (Optional[str], optional): A string value identifying the characteristic of the record. Usually it will be `valid`, but it can be `discarded` or anything else. Defaults to "valid".

    Returns:
        List[dict]: A list containing the contours and its measurements.
    """
    measurements = []

    for parent_id, parent_contour in enumerate(parent_contours, start=start_index):
        parent_pixel_count = get_contour_pixel_count(parent_contour, shape)
        parent_features = [record_id, mask_name, parent_id, None, parent_type, parent_pixel_count, None, None, None, contours_flag]
        measurements.append({ key: value for key, value in zip(MEASUREMENT_COLUMNS, parent_features) })

        contours_size = []

        child_id = 0
        for child_contour in child_contours:
            for child_point in child_contour:
                if cv2.pointPolygonTest(parent_contour, tuple(child_point[0]), False) >= 0:
                    child_pixel_count = get_contour_pixel_count(child_contour, shape)
                    contours_size.append(child_pixel_count)
                    child_features = [record_id, mask_name, child_id, parent_id, child_type, child_pixel_count]
                    measurements.append({ key: value for key, value in zip(MEASUREMENT_COLUMNS, child_features)})
                    child_id += 1
                    break

    if len (contours_size) > 0:
        min_contour_size = np.min(contours_size)
        max_contour_size = np.max(contours_size)

        for i, record in enumerate(measurements):
            if record["parent_id"] is not None:
                features = list(record.values())
                features.append(record["pixel_count"] / measurements[0]["pixel_count"])
                features.append(record["pixel_count"] / max_contour_size)
                features.append(record["pixel_count"] / min_contour_size)
                features.append(contours_flag)
                measurements[i] = {key: value for key, value in zip(MEASUREMENT_COLUMNS, features)}

    return measurements


def write_contour_measurements(
    measurements: List[dict],
    output_path: str,
    datetime: Optional[str] = time.strftime('%Y%m%d%H%M%S')) -> None:
    """Writes contour measurements to `.csv` files.

    Args:
        parent_measurements (List[dict]): The parent contours.
        output_path (str): The path where the files should be written to.
        datetime (Optional[str], optional): A date and time identification for when the files were generated. Defaults to time.strftime('%Y%m%d%H%M%S').
    """
    df = pd.DataFrame(measurements, columns=MEASUREMENT_COLUMNS)
    df["datetime"] = datetime

    output_file = Path(output_path).joinpath(f"nucleus_agnor_measurements.csv")

    if Path(output_file).is_file():
        df.to_csv(str(output_file), mode="a", header=False, index=False)
    else:
        df.to_csv(str(output_file), mode="w", header=True, index=False)


def classify_agnor(model_path: str, contours: List[np.ndarray]) -> List[np.ndarray]:
    """Loads a Scikit-Learn model and classify the input arrays in `clusters` and `satellites`.

    Args:
        model_path (str): Path to the model file.
        contours (List[np.ndarray]): Input array containing the features `agnor_pixel_count`, `nucleus_ratio`, `smallest_agnor_ratio`, `greatest_agnor_ratio`."

    Returns:
        List[np.ndarray]: Updated array with a column containing the predicted class of each element in the input array, where `0` corresponds to `cluster` and `1` to `satellite`.
    """
    if len(contours) == 0:
        return contours

    features_list = [
        "pixel_count",
        "nucleus_ratio",
        "smallest_agnor_ratio",
        "greatest_agnor_ratio"
    ]

    df = pd.DataFrame.from_dict(contours)
    features = df[df["type"].str.contains("agnor|cluster|satellite")][features_list].copy()

    classifier =joblib.load(model_path)
    predictions = classifier.predict(features)
    df.loc[df["type"].str.contains("agnor|cluster|satellite"), "type"] = predictions

    contours = list(df.T.to_dict().values())
    return contours


def discard_unboxed_contours(
    prediction: Union[np.ndarray, tf.Tensor],
    parent_contours: List[np.ndarray],
    child_contours: List[np.ndarray],
    annotation: str) -> Tuple[Union[np.ndarray, tf.Tensor], np.ndarray, np.ndarray]:
    """Zero pixels that are not contained by a bounding box.

    Args:
        prediction (Union[np.ndarray, tf.Tensor]): Predicted segmentation.
        parent_contours (List[np.ndarray]): The parent contours.
        child_contours (List[np.ndarray]): The child contours.
        annotation (str): Path to the `labelme` annotation file.

    Returns:
        Tuple[Union[np.ndarray, tf.Tensor], np.ndarray, np.ndarray]: Updated segmentation mask and contour arrays containing only objects within bounding boxes.
    """
    if prediction is not None:
        bboxes = get_labelme_points(annotation, shape_types=["rectangle"])

        # Convert bboxes into contours with four points.
        for i in range(len(bboxes)):
            bboxes[i] = convert_bbox_to_contour(bboxes[i].tolist())

        nuclei_contours_adequate, _ = discard_contours_outside_contours(bboxes, parent_contours)
        nuclei_contours_adequate_final = []
        for contour in nuclei_contours_adequate:
            for bbox in bboxes:
                iou = get_intersection(bbox, contour, shape=prediction.shape[:2])
                if iou > 0.2:
                    nuclei_contours_adequate_final.append(contour)
        parent_contours = nuclei_contours_adequate_final
        child_contours, _ = discard_contours_outside_contours(parent_contours, child_contours)
        
        # Create a new mask with the filtered nuclei and NORs
        pixel_intensity = int(np.max(np.unique(prediction)))
        background = np.ones(prediction.shape[:2], dtype=np.uint8)
        nucleus = np.zeros(prediction.shape[:2], dtype=np.uint8)
        nor = np.zeros(prediction.shape[:2], dtype=np.uint8)

        cv2.drawContours(nucleus, contours=parent_contours, contourIdx=-1, color=pixel_intensity, thickness=cv2.FILLED)
        cv2.drawContours(nor, contours=child_contours, contourIdx=-1, color=pixel_intensity, thickness=cv2.FILLED)

        nucleus = np.where(nor, 0, nucleus)
        background = np.where(np.logical_and(nucleus == 0, nor == 0), pixel_intensity, 0)
        prediction = np.stack([background, nucleus, nor], axis=2).astype(np.uint8)
    
    return prediction, parent_contours, child_contours
