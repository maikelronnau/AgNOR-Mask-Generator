import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from utils import contour_analysis
from utils.utils import (DECISION_TREE_MODEL_PATH, MSKG_VERSION, color_classes, convert_bbox_to_contour,
                         get_labelme_shapes)


def create_annotation(
    input_image: np.ndarray,
    prediction: np.ndarray,
    patient: str,
    annotation_directory: str,
    output_directory: str,
    source_image_path: str,
    original_image_shape: Tuple[int, int],
    hashfile: Optional[str] = None,
    classify_agnor: Optional[bool] = False,
    datetime: Optional[str] = None) -> None:
    """Save a `labelme` annotation from a segmented input image.

    Args:
        input_image (np.ndarray): The input image.
        prediction (np.ndarray): The segmented image.
        patient (str): The identification of the patient.
        annotation_directory (str): Path where to save the annotations.
        output_directory (str): Path where to save the segmentation measurements.
        source_image_path (str): Input image path.
        original_image_shape (Tuple[int, int]): Height and width of the input image.
        hashfile (Optional[str], optional): Hashfile of the input image. Defaults to None.
        classify_agnor (Optional[bool], optional): Whether or not to classify AgNORs into `cluster` and `satellite`. Defaults to False.
        datetime (Optional[str], optional): Date and time the annotation was generated. Defaults to None.
    """
    annotation_directory = Path(annotation_directory)
    output_directory = Path(output_directory)
    source_image_path = Path(source_image_path)
    overlay_directory = output_directory.joinpath("overlay")
    overlay_directory.mkdir(exist_ok=True)

    logging.debug(f"Saving image annotations from {source_image_path.name} annotations to {str(annotation_directory)}")
    height = original_image_shape[0]
    width = original_image_shape[1]

    annotation = {
        "version": "4.5.7",
        "mskg_version": MSKG_VERSION,
        "flags": {},
        "shapes": [],
        "imageHeight": height,
        "imageWidth": width,
        "patient": patient,
        "imagePath": source_image_path.name,
        "imageHash": hashfile,
        "dateTime": datetime,
        "imageData": None
    }

    logging.debug("Analyze contours")
    prediction, _ = contour_analysis.analyze_contours(mask=prediction, smooth=True)
    prediction, parent_contours, child_contours = prediction

    if classify_agnor:
        logging.debug("Add an extra channel to map 'satellites'")
        prediction = np.stack([
            prediction[:, :, 0],
            prediction[:, :, 1],
            prediction[:, :, 2],
            np.zeros(original_image_shape, dtype=np.uint8)
        ], axis=2)

    logging.debug("Obtain contour measurements and append shapes to annotation file")
    for i, parent_contour in enumerate(parent_contours):
        filtered_child_contour, _ = contour_analysis.discard_contours_outside_contours([parent_contour], child_contours)
        measurements = contour_analysis.get_contour_measurements(
            parent_contours=[parent_contour],
            child_contours=filtered_child_contour,
            shape=original_image_shape,
            mask_name=source_image_path.name,
            record_id=patient,
            start_index=i)

        if classify_agnor:
            measurements = contour_analysis.classify_agnor(DECISION_TREE_MODEL_PATH, measurements)
            # OpenCV's `drawContours` fails using array slices, so a new matrix must be created, drawn on and assigned to `predictions`.
            satellites = prediction[:, :, 3].copy()
            for classified_measurement, classified_contour in zip(measurements[1:], filtered_child_contour):
                if classified_measurement["type"] == "satellite":
                    cv2.drawContours(satellites, contours=[classified_contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
            prediction[:, :, 3] = satellites

        # Prepare and append nucleus shape
        points = []
        for point in parent_contour:
            points.append([int(value) for value in point[0]])

        shape = {
            "label": "nucleus",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        annotation["shapes"].append(shape)

        for measurement, contour in zip(measurements[1:], filtered_child_contour):
            points = []
            for point in contour:
                points.append([int(value) for value in point[0]])

            shape = {
                "label": measurement["type"],
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            annotation["shapes"].append(shape)

        contour_analysis.write_contour_measurements(
            measurements=measurements,
            output_path=output_directory,
            datetime=datetime)

    logging.debug("Write annotation file")
    annotation_path = str(annotation_directory.joinpath(f"{source_image_path.stem}.json"))
    with open(annotation_path, "w") as output_file:
        json.dump(annotation, output_file, indent=4)

    logging.debug("Copy original image to the annotation directory")
    filename = annotation_directory.joinpath(source_image_path.name)
    if not filename.is_file():
        shutil.copyfile(str(source_image_path), str(filename))

    prediction = color_classes(prediction)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    prediction[prediction == 130] = 0

    alpha = 0.8
    beta = 0.1
    gamma = 0.0
    overlay = cv2.addWeighted(input_image, alpha, prediction, beta, gamma)
    cv2.imwrite(str(overlay_directory.joinpath(f"{source_image_path.name}.jpg")), overlay)


def update_annotation(
    input_image: np.ndarray,
    prediction: np.ndarray,
    patient: str,
    annotation_directory: str,
    output_directory: str,
    source_image_path: str,
    annotation_path: str,
    original_image_shape: Tuple[int, int],
    hashfile: Optional[str] = None,
    classify_agnor: Optional[bool] = False,
    datetime: Optional[str] = None) -> None:
    """Update an existing annotation file considering bounding boxes.

    Args:
        input_image (np.ndarray): The input image.
        prediction (np.ndarray): The segmented image.
        patient (str): The identification of the patient.
        annotation_path (str): Path of the annotation file to be updated.
        output_directory (str): Path where to save the segmentation measurements.
        source_image_path (str): Input image path.
        original_image_shape (Tuple[int, int]): Height and width of the input image.
        hashfile (Optional[str], optional): Hashfile of the input image. Defaults to None.
        classify_agnor (Optional[bool], optional): Whether or not to classify AgNORs into `cluster` and `satellite`. Defaults to False.
        datetime (Optional[str], optional): Date and time the annotation was generated. Defaults to None.
    """
    annotation_directory = Path(annotation_directory)
    output_directory = Path(output_directory)
    source_image_path = Path(source_image_path)
    overlay_directory = output_directory.parent.joinpath("overlay")
    overlay_directory.mkdir(exist_ok=True)

    logging.debug(f"Updating annotations from {source_image_path.name}")

    with open(annotation_path, "r") as annotation_file:
        annotation = json.load(annotation_file)

    if "patient" in annotation.keys():
        patient = annotation["patient"]
    elif patient != "":
        annotation["patient"] = patient
    
    annotation["imageHash"] = hashfile

    logging.debug("Analyze contours")
    prediction, _ = contour_analysis.analyze_contours(mask=prediction, smooth=True)
    prediction, parent_contours, child_contours = prediction

    logging.debug("Remove contours outside bounding boxes")
    prediction = contour_analysis.discard_unboxed_contours(prediction, parent_contours, child_contours, annotation=annotation_path)
    prediction, parent_contours, child_contours = prediction

    # Reset shapes in the annotation file
    annotation["shapes"] = []
    annotation["imageData"] = None

    if classify_agnor:
        logging.debug("Add an extra channel to map 'satellites'")
        prediction = np.stack([
            prediction[:, :, 0],
            prediction[:, :, 1],
            prediction[:, :, 2],
            np.zeros(original_image_shape, dtype=np.uint8)
        ], axis=2)

    logging.debug("Obtain contour measurements and append shapes to annotation file")
    bounding_boxes_shapes = get_labelme_shapes(annotation_path=annotation_path, shape_types=["rectangle"])
    # Variable to enumerate nucleus during processing
    i = 0
    for rectangle in bounding_boxes_shapes:
        annotation["shapes"].append(rectangle)
        
        # Convert rectangle points so it can be used in OpenCV to filter other contours
        rectangle = convert_bbox_to_contour(rectangle["points"].copy())
        rectangle = rectangle.reshape((rectangle.shape[0], 1, rectangle.shape[1]))
        
        filtered_parent_contours, _ = contour_analysis.discard_contours_outside_contours([rectangle], parent_contours)
        for parent_contour in filtered_parent_contours:
            filtered_child_contour, _ = contour_analysis.discard_contours_outside_contours([parent_contour], child_contours)
            measurements = contour_analysis.get_contour_measurements(
                parent_contours=[parent_contour],
                child_contours=filtered_child_contour,
                shape=original_image_shape,
                mask_name=source_image_path.name,
                record_id=patient,
                start_index=i)

            if classify_agnor:
                measurements = contour_analysis.classify_agnor(DECISION_TREE_MODEL_PATH, measurements)
                # OpenCV's `drawContours` fails using array slices, so a new matrix must be created, drawn on and assigned to `predictions`.
                satellites = prediction[:, :, 3].copy()
                for classified_measurement, classified_contour in zip(measurements[1:], filtered_child_contour):
                    if classified_measurement["type"] == "satellite":
                        cv2.drawContours(satellites, contours=[classified_contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
                prediction[:, :, 3] = satellites

            # Prepare and append nucleus shape
            points = []
            for point in parent_contour:
                points.append([int(value) for value in point[0]])

            shape = {
                "label": "nucleus",
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            annotation["shapes"].append(shape)

            for measurement, contour in zip(measurements[1:], filtered_child_contour):
                points = []
                for point in contour:
                    points.append([int(value) for value in point[0]])

                shape = {
                    "label": measurement["type"],
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                annotation["shapes"].append(shape)

            contour_analysis.write_contour_measurements(
                measurements=measurements,
                output_path=annotation_directory,
                datetime=datetime)

    logging.debug("Write annotation file")
    with open(annotation_path, "w") as output_file:
        json.dump(annotation, output_file, indent=4)

    prediction = color_classes(prediction)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    prediction[prediction == 130] = 0

    alpha = 0.8
    beta = 0.1
    gamma = 0.0
    overlay = cv2.addWeighted(input_image, alpha, prediction, beta, gamma)
    cv2.imwrite(str(overlay_directory.joinpath(f"{source_image_path.stem}.jpg")), overlay)
