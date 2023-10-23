import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from utils import contour_analysis
from utils.utils import (DECISION_TREE_MODEL_PATH, MSKG_VERSION, color_classes, convert_bbox_to_contour,
                         get_labelme_shapes, collapse_probabilities)


def get_segmentation_overlay(
    input_image: np.ndarray,
    prediction: np.ndarray,
    alpha: Optional[float] = 0.8,
    beta: Optional[float] = 0.4,
    gamma: Optional[float] = 0.0) -> np.ndarray:
    """Overlays the input image and the predicted segmentation.

    Args:
        input_image (np.ndarray): The input image.
        prediction (np.ndarray): The segmentation prediction for the input image.
        alpha (Optional[float], optional): Weight of the input image. Defaults to 0.8.
        beta (Optional[float], optional): Weight of the prediction. Defaults to 0.1.
        gamma (Optional[float], optional): Scalar added to each sum. Defaults to 0.0.

    Returns:
        np.ndarray: Overlaid segmentation.
    """
    overlay = cv2.addWeighted(input_image, alpha, prediction, beta, gamma)
    return overlay


def create_annotation(
    input_image: np.ndarray,
    prediction: np.ndarray,
    patient_record: str,
    patient: str,
    patient_group: str,
    annotation_directory: str,
    output_directory: str,
    source_image_path: str,
    original_image_shape: Tuple[int, int],
    hashfile: Optional[str] = None,
    use_bias: Optional[bool] = False,
    use_bias_layer: Optional[bool] = False,
    reclassify: Optional[bool] = False,
    remove_artifacts: Optional[bool] = False,
    exam_date: Optional[str] = "",
    exam_instance: Optional[str] = "",
    anatomical_site: Optional[str] = "",
    overlay: Optional[bool] = False,
    datetime: Optional[str] = None) -> None:
    """Save a `labelme` annotation from a segmented input image.

    Args:
        input_image (np.ndarray): The input image.
        prediction (np.ndarray): The segmented image.
        patient_record (str): Record of the patient.
        patient (str): The identification of the patient.
        patient_group (str): The group the patient belongs to.
        annotation_directory (str): Path where to save the annotations.
        output_directory (str): Path where to save the segmentation measurements.
        source_image_path (str): Input image path.
        original_image_shape (Tuple[int, int]): Height and width of the input image.
        hashfile (Optional[str], optional): Hashfile of the input image. Defaults to None.
        use_bias (Optional[bool], optional): Whether or not to use the bias. Defaults to False.
        use_bias_layer (Optional[bool], optional): Whether or not to use the bias layer. Defaults to False.
        reclassify (Optional[bool], optional): Whether or not to reclassify the segmented image. Defaults to False.
        remove_artifacts (Optional[bool], optional): Whether or not to remove artifacts from the segmented image. Defaults to False.
        exam_date (Optional[str], optional): The date the exam (brushing) ocurred. Defaults to "".
        exam_instance (Optional[str], optional): Instance of the exam. For example, `T0`, `T1`, `T2`, etc. Defaults to "".
        anatomical_site: (Optional[str], optional): The area of the mouth where the brushing was done. Defaults to "".
        overlay (Optional[bool], optional): Whether or not to save the segmentation overlay.
        datetime (Optional[str], optional): Date and time the annotation was generated. Defaults to None.
    """
    annotation_directory = Path(annotation_directory)
    output_directory = Path(output_directory)
    source_image_path = Path(source_image_path)

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
        "patient_record": patient_record,
        "patient": patient,
        "group": patient_group,
        "exam_date": exam_date,
        "exam_instance": exam_instance,
        "anatomical_site": anatomical_site,
        "imagePath": source_image_path.name,
        "imageHash": hashfile,
        "dateTime": datetime,
        "imageData": None
    }

    if use_bias:
        logging.debug("Adjust probability using bias")
        prediction = contour_analysis.adjust_probability(prediction=prediction.copy())

    logging.debug("Resize prediction to original image shape")
    prediction = cv2.resize(prediction, original_image_shape[::-1], interpolation=cv2.INTER_NEAREST)

    logging.debug("Collapse probabilities")
    prediction = collapse_probabilities(prediction=prediction, pixel_intensity=127)

    if reclassify:
        logging.debug("Reclassify segmentation objects")
        prediction = contour_analysis.reclassify_segmentation_objects(prediction)

    if remove_artifacts:
        logging.debug("Remove segmentation artifacts")
        prediction = contour_analysis.remove_segmentation_artifacts(prediction=prediction)

    logging.debug("Extract contours from prediction")
    clusters = contour_analysis.get_contours(prediction[:, :, 1])
    cytoplasms = contour_analysis.get_contours(prediction[:, :, 2])
    anucleates = contour_analysis.get_contours(prediction[:, :, 3])
    superficial = contour_analysis.get_contours(prediction[:, :, 4])
    intermediate = contour_analysis.get_contours(prediction[:, :, 5])
    suspected = contour_analysis.get_contours(prediction[:, :, 6])
    binucleate = contour_analysis.get_contours(prediction[:, :, 7])

    # Variable to enumerate clusters, cytoplasms and anucleates during processing
    cluster_cytoplasm_anucleate_counter = 0

    logging.debug("Obtain contour measurements and append shapes to annotation file")
    logging.debug("Process clusters")
    for cluster in clusters:
        superficial_nuclei, _ = contour_analysis.discard_contours_outside_contours([cluster], superficial)
        intermediate_nuclei, _ = contour_analysis.discard_contours_outside_contours([cluster], intermediate)
        suspect_nuclei, _ = contour_analysis.discard_contours_outside_contours([cluster], suspected)
        binucleate_nuclei, _ = contour_analysis.discard_contours_outside_contours([cluster], binucleate)

        nuclei_groups = [superficial_nuclei, intermediate_nuclei, suspect_nuclei, binucleate_nuclei]
        nuclei_groups = {
            "superficial": superficial_nuclei,
            "intermediate": intermediate_nuclei,
            "suspect": suspect_nuclei,
            "binucleate": binucleate_nuclei
        }

        parent_measurements, child_measurements = [], []
        nuclei_id_counter = 0
        for nuclei_group_name, nuclei_group in nuclei_groups.items():
            if len(nuclei_group) == 0:
                continue

            parent_measurements, child_measurements = contour_analysis.get_contour_measurements(
                parent_contours=[cluster],
                child_contours=nuclei_group,
                shape=original_image_shape,
                mask_name=source_image_path.name,
                parent_type="cluster",
                child_type=nuclei_group_name,
                record_id=patient_record,
                patient_name=patient,
                record_class=patient_group,
                exam_date=exam_date,
                exam_instance=exam_instance,
                anatomical_site=anatomical_site,
                start_index_parent=cluster_cytoplasm_anucleate_counter,
                start_index_child=nuclei_id_counter)

            nuclei_id_counter += len(nuclei_group)

            for measurement, contour in zip(child_measurements, nuclei_group):
                points = []
                for point in contour:
                    points.append([int(value) for value in point[0]])

                measurement["type"] = nuclei_group_name
                shape = {
                    "label": measurement["type"],
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                annotation["shapes"].append(shape)

            if len(child_measurements) > 0:
                contour_analysis.write_contour_measurements(
                    measurements=child_measurements,
                    filename="nuclei",
                    output_path=output_directory,
                    columns=contour_analysis.NUCLEI_COLUMNS,
                    datetime=datetime)

        # Prepare and append cluster shape
        points = []
        for point in cluster:
            points.append([int(value) for value in point[0]])

        shape = {
            "label": "cluster",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        annotation["shapes"].append(shape)

        contour_analysis.write_contour_measurements(
            measurements=parent_measurements,
            filename="cluster_cytoplasm_anucleate",
            output_path=output_directory,
            columns=contour_analysis.CLUSTER_CYTOPLAM_ANUCLEATE_COLUMNS,
            datetime=datetime)

        cluster_cytoplasm_anucleate_counter += 1

    for cytoplasm in cytoplasms:
        logging.debug("Process cytoplasms")
        superficial_nuclei, _ = contour_analysis.discard_contours_outside_contours([cytoplasm], superficial)
        intermediate_nuclei, _ = contour_analysis.discard_contours_outside_contours([cytoplasm], intermediate)
        suspect_nuclei, _ = contour_analysis.discard_contours_outside_contours([cytoplasm], suspected)
        binucleate_nuclei, _ = contour_analysis.discard_contours_outside_contours([cytoplasm], binucleate)

        nuclei_groups = [superficial_nuclei, intermediate_nuclei, suspect_nuclei, binucleate_nuclei]
        nuclei_groups = {
            "superficial": superficial_nuclei,
            "intermediate": intermediate_nuclei,
            "suspect": suspect_nuclei,
            "binucleate": binucleate_nuclei
        }

        parent_measurements, child_measurements = [], []
        nuclei_id_counter = 0
        for nuclei_group_name, nuclei_group in nuclei_groups.items():
            if len(nuclei_group) == 0:
                continue

            parent_measurements, child_measurements = contour_analysis.get_contour_measurements(
                parent_contours=[cytoplasm],
                child_contours=nuclei_group,
                shape=original_image_shape,
                mask_name=source_image_path.name,
                parent_type="cytoplasm",
                child_type=nuclei_group_name,
                record_id=patient_record,
                patient_name=patient,
                record_class=patient_group,
                exam_date=exam_date,
                exam_instance=exam_instance,
                anatomical_site=anatomical_site,
                start_index_parent=cluster_cytoplasm_anucleate_counter,
                start_index_child=nuclei_id_counter)

            cluster_cytoplasm_anucleate_counter
            nuclei_id_counter += len(nuclei_group)

            for measurement, contour in zip(child_measurements, nuclei_group):
                points = []
                for point in contour:
                    points.append([int(value) for value in point[0]])

                measurement["type"] = nuclei_group_name
                shape = {
                    "label": measurement["type"],
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                annotation["shapes"].append(shape)

            contour_analysis.write_contour_measurements(
                measurements=child_measurements,
                filename="nuclei",
                output_path=output_directory,
                columns=contour_analysis.NUCLEI_COLUMNS,
                datetime=datetime)

        # Prepare and append cytoplasm shape
        points = []
        for point in cytoplasm:
            points.append([int(value) for value in point[0]])

        shape = {
            "label": "cytoplasm",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        annotation["shapes"].append(shape)

        contour_analysis.write_contour_measurements(
            measurements=parent_measurements,
            filename="cluster_cytoplasm_anucleate",
            output_path=output_directory,
            columns=contour_analysis.CLUSTER_CYTOPLAM_ANUCLEATE_COLUMNS,
            datetime=datetime)

        cluster_cytoplasm_anucleate_counter += 1

    for anucleate in anucleates:
        logging.debug("Process anucleates")
        parent_measurements, child_measurements = contour_analysis.get_contour_measurements(
            parent_contours=[anucleate],
            child_contours=[],
            shape=original_image_shape,
            mask_name=source_image_path.name,
            parent_type="anucleate",
            child_type="",
            record_id=patient_record,
            patient_name=patient,
            record_class=patient_group,
            exam_date=exam_date,
            exam_instance=exam_instance,
            start_index_parent=cluster_cytoplasm_anucleate_counter)

        # Prepare and append anucleate shape
        points = []
        for point in anucleate:
            points.append([int(value) for value in point[0]])

        shape = {
            "label": "anucleate",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        annotation["shapes"].append(shape)

        contour_analysis.write_contour_measurements(
            measurements=parent_measurements,
            filename="cluster_cytoplasm_anucleate",
            output_path=output_directory,
            columns=contour_analysis.CLUSTER_CYTOPLAM_ANUCLEATE_COLUMNS,
            datetime=datetime)

        cluster_cytoplasm_anucleate_counter += 1

    logging.debug("Write annotation file")
    annotation_path = str(annotation_directory.joinpath(f"{source_image_path.stem}.json"))
    with open(annotation_path, "w") as output_file:
        json.dump(annotation, output_file, indent=4)

    logging.debug("Copy original image to the annotation directory")
    filename = annotation_directory.joinpath(source_image_path.name)
    if not filename.is_file():
        shutil.copyfile(str(source_image_path), str(filename))

    prediction = color_classes(prediction, colormap="papanicolaou")
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    # Zero background pixels so it does not mess with the overlay.
    prediction[prediction == 130] = 0

    if overlay:
        overlay_directory = output_directory.joinpath("overlay")
        overlay_directory.mkdir(exist_ok=True)
        overlay = get_segmentation_overlay(input_image, prediction)
        cv2.imwrite(str(overlay_directory.joinpath(f"{source_image_path.stem}.jpg")), overlay)





    # prediction, _ = contour_analysis.analyze_contours(mask=prediction, smooth=True)
    # prediction, parent_contours, child_contours = prediction

    # if classify_agnor:
    #     logging.debug("Add an extra channel to map 'satellites'")
    #     prediction = np.stack([
    #         prediction[:, :, 0],
    #         prediction[:, :, 1],
    #         prediction[:, :, 2],
    #         np.zeros(original_image_shape, dtype=np.uint8)
    #     ], axis=2)

    # logging.debug("Obtain contour measurements and append shapes to annotation file")
    # for i, parent_contour in enumerate(parent_contours):
    #     filtered_child_contour, _ = contour_analysis.discard_contours_outside_contours([parent_contour], child_contours)
        # parent_measurements, child_measurements = contour_analysis.get_contour_measurements(
    #         parent_contours=[parent_contour],
    #         child_contours=filtered_child_contour,
    #         shape=original_image_shape,
    #         mask_name=source_image_path.name,
    #         record_id=patient_record,
    #         patient_name=patient,
    #         record_class=patient_group,
    #         exam_date=exam_date,
    #         exam_instance=exam_instance,
    #         anatomical_site=anatomical_site,
    #         start_index=i)

    #     if classify_agnor:
    #         child_measurements = contour_analysis.classify_agnor(DECISION_TREE_MODEL_PATH, child_measurements)
    #         # OpenCV's `drawContours` fails using array slices, so a new matrix must be created, drawn on and assigned to `predictions`.
    #         satellites = prediction[:, :, 3].copy()
    #         for classified_measurement, classified_contour in zip(child_measurements, filtered_child_contour):
    #             if classified_measurement["type"] == "satellite":
    #                 cv2.drawContours(satellites, contours=[classified_contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
    #         prediction[:, :, 3] = satellites

    #     # Prepare and append nucleus shape
    #     points = []
    #     for point in parent_contour:
    #         points.append([int(value) for value in point[0]])

    #     shape = {
    #         "label": "nucleus",
    #         "points": points,
    #         "group_id": None,
    #         "shape_type": "polygon",
    #         "flags": {}
    #     }
    #     annotation["shapes"].append(shape)

    #     for measurement, contour in zip(child_measurements, filtered_child_contour):
    #         points = []
    #         for point in contour:
    #             points.append([int(value) for value in point[0]])

    #         shape = {
    #             "label": measurement["type"],
    #             "points": points,
    #             "group_id": None,
    #             "shape_type": "polygon",
    #             "flags": {}
    #         }
    #         annotation["shapes"].append(shape)

    #     contour_analysis.write_contour_measurements(
    #         parent_measurements=parent_measurements,
    #         child_measurements=child_measurements,
    #         output_path=output_directory,
    #         datetime=datetime)

    # logging.debug("Write annotation file")
    # annotation_path = str(annotation_directory.joinpath(f"{source_image_path.stem}.json"))
    # with open(annotation_path, "w") as output_file:
    #     json.dump(annotation, output_file, indent=4)

    # logging.debug("Copy original image to the annotation directory")
    # filename = annotation_directory.joinpath(source_image_path.name)
    # if not filename.is_file():
    #     shutil.copyfile(str(source_image_path), str(filename))

    # prediction = color_classes(prediction)
    # prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    # # Zero background pixels so it does not mess with the overlay.
    # prediction[prediction == 130] = 0

    # if overlay:
    #     overlay_directory = output_directory.joinpath("overlay")
    #     overlay_directory.mkdir(exist_ok=True)
    #     overlay = get_segmentation_overlay(input_image, prediction)
    #     cv2.imwrite(str(overlay_directory.joinpath(f"{source_image_path.stem}.jpg")), overlay)


def update_annotation(
    input_image: np.ndarray,
    prediction: np.ndarray,
    patient_record: str,
    patient: str,
    patient_group: str,
    annotation_directory: str,
    output_directory: str,
    source_image_path: str,
    annotation_path: str,
    original_image_shape: Tuple[int, int],
    hashfile: Optional[str] = None,
    classify_agnor: Optional[bool] = False,
    exam_date: Optional[str] = "",
    exam_instance: Optional[str] = "",
    anatomical_site: Optional[str] = "",
    overlay: Optional[bool] = False,
    datetime: Optional[str] = None) -> None:
    """Update an existing annotation file considering bounding boxes.

    Args:
        input_image (np.ndarray): The input image.
        prediction (np.ndarray): The segmented image.
        patient_record (str): Record of the patient.
        patient (str): The identification of the patient.
        patient_group (str): The group the patient belongs to.
        annotation_path (str): Path of the annotation file to be updated.
        output_directory (str): Path where to save the segmentation measurements.
        source_image_path (str): Input image path.
        original_image_shape (Tuple[int, int]): Height and width of the input image.
        hashfile (Optional[str], optional): Hashfile of the input image. Defaults to None.
        classify_agnor (Optional[bool], optional): Whether or not to classify AgNORs into `cluster` and `satellite`. Defaults to False.
        exam_date (Optional[str], optional): The date the exam (brushing) ocurred. Defaults to "".
        exam_instance (Optional[str], optional): Instance of the exam. For example, `T0`, `T1`, `T2`, etc. Defaults to "".
        anatomical_site: (Optional[str], optional): The area of the mouth where the brushing was done. Defaults to "".
        overlay (Optional[bool], optional): Whether or not to save the segmentation overlay.
        datetime (Optional[str], optional): Date and time the annotation was generated. Defaults to None.
    """
    annotation_directory = Path(annotation_directory)
    output_directory = Path(output_directory)
    source_image_path = Path(source_image_path)

    logging.debug(f"Updating annotations from {source_image_path.name}")

    with open(annotation_path, "r") as annotation_file:
        annotation = json.load(annotation_file)

    if patient_record != "":
        annotation["patient_record"] = patient_record
    elif "patient_record" in annotation.keys():
        patient_record = annotation["patient_record"]
    if patient != "":
        annotation["patient"] = patient
    elif "patient" in annotation.keys():
        patient = annotation["patient"]
    if exam_date != "":
        annotation["exam_date"] = exam_date
    elif "exam_date" in annotation.keys():
        exam_date = annotation["exam_date"]
    if exam_instance != "":
        annotation["exam_instance"] = exam_instance
    elif "exam_instance" in annotation.keys():
        exam_instance = annotation["exam_instance"]
    if anatomical_site != "":
        annotation["anatomical_site"] = anatomical_site
    elif "anatomical_site" in annotation.keys():
        anatomical_site = annotation["anatomical_site"]
    if patient_group != "":
        annotation["group"] = patient_group
    elif "group" in annotation.keys():
        patient_group = annotation["group"]

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
    # Prevent duplicate annotations by keeping a list of unseen objects
    unseen_contours = parent_contours
    for rectangle in bounding_boxes_shapes:
        rectangle["label"] = f"BoundingBox {i+1}"
        annotation["shapes"].append(rectangle)

        # Convert rectangle points so it can be used in OpenCV to filter other contours
        rectangle = convert_bbox_to_contour(rectangle["points"].copy())
        rectangle = rectangle.reshape((rectangle.shape[0], 1, rectangle.shape[1]))

        filtered_parent_contours, unseen_contours = contour_analysis.discard_contours_outside_contours([rectangle], unseen_contours)
        for parent_contour in filtered_parent_contours:
            filtered_child_contour, _ = contour_analysis.discard_contours_outside_contours([parent_contour], child_contours)
            parent_measurements, child_measurements = contour_analysis.get_contour_measurements(
                parent_contours=[parent_contour],
                child_contours=filtered_child_contour,
                shape=original_image_shape,
                mask_name=source_image_path.name,
                record_id=patient_record,
                patient_name=patient,
                record_class=patient_group,
                exam_date=exam_date,
                exam_instance=exam_instance,
                anatomical_site=anatomical_site,
                start_index=i)

            if classify_agnor:
                child_measurements = contour_analysis.classify_agnor(DECISION_TREE_MODEL_PATH, child_measurements)
                # OpenCV's `drawContours` fails using array slices, so a new matrix must be created, drawn on and assigned to `predictions`.
                satellites = prediction[:, :, 3].copy()
                for classified_measurement, classified_contour in zip(child_measurements, filtered_child_contour):
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

            for measurement, contour in zip(child_measurements, filtered_child_contour):
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
                parent_measurements=parent_measurements,
                child_measurements=child_measurements,
                output_path=output_directory,
                datetime=datetime)
            i += 1

    logging.debug("Write annotation file")
    annotation["last_updated"] = datetime
    with open(annotation_path, "w") as output_file:
        json.dump(annotation, output_file, indent=4)

    prediction = color_classes(prediction)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    # Zero background pixels so it does not mess with the overlay.
    prediction[prediction == 130] = 0

    if overlay:
        overlay_directory = output_directory.joinpath("overlay")
        overlay_directory.mkdir(exist_ok=True)
        overlay = get_segmentation_overlay(input_image, prediction)
        cv2.imwrite(str(overlay_directory.joinpath(f"{source_image_path.stem}.jpg")), overlay)
