import argparse
import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import PySimpleGUI as sg
import tensorflow as tf
from tqdm import tqdm

from utils import user_interface
from utils.annotation import create_annotation, update_annotation
from utils.contour_analysis import aggregate_measurements
from utils.data import list_files
from utils.model import load_model
from utils.utils import (DEFAULT_MODEL_INPUT_SHAPE, MODEL_PATH,
                         collapse_probabilities, get_hash_file,
                         open_with_labelme, pad_along_axis)


def main():
    parser = argparse.ArgumentParser(description=user_interface.PROGRAM_NAME)
    parser.add_argument(
        "--model",
        help="Path to the model to be used. Will replace the embedded model if specified.",
        default=None)

    parser.add_argument(
        "--gpu",
        help="Set which GPU to use. Pass '-1' to run on CPU.",
        default="0")

    parser.add_argument(
        "--input-directory",
        help="Input directory.",
        default="",
        required=False)

    parser.add_argument(
        "--patient",
        help="Patient name.",
        default="",
        required=False)

    parser.add_argument(
        "--patient-record",
        help="Patient record number.",
        default="",
        required=False)

    parser.add_argument(
        "--patient-group",
        help="Patient group.",
        default="",
        required=False)

    parser.add_argument(
        "--anatomical-site",
        help="Anatomical site.",
        default="",
        required=False)

    parser.add_argument(
        "--exam-date",
        help="Exam date.",
        default="",
        required=False)

    parser.add_argument(
        "--exam-instance",
        help="Exam instance.",
        default="",
        required=False)

    parser.add_argument(
        "--classify-agnor",
        help="Classify AgNORs.",
        default=False,
        action="store_true",
        required=False)

    parser.add_argument(
        "--bboxes",
        help="Use bounding boxes to restrict segmentation results.",
        default=False,
        action="store_true",
        required=False)

    parser.add_argument(
        "--overlay",
        help="Generate overlay of input images and segmentation.",
        default=False,
        action="store_true",
        required=False)

    parser.add_argument(
        "--database",
        help="Database file. A `.csv` to write the aggregate measurements to.",
        default="",
        required=False)
    
    parser.add_argument(
        "--console",
        help="Enable or disable console mode. If enabled, no GUI will be displayed.",
        default=False,
        action="store_true",
        required=False)

    parser.add_argument(
        "--debug",
        help="Enable or disable debug mode to log execution to a file.",
        default=False,
        action="store_true")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            filename="mskg.log",
            filemode="a",
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s")

    console_mode = args.console

    logging.debug(f"Program started at `{time.strftime('%Y%m%d%H%M%S')}`")

    logging.debug(f"Using GPU '{args.gpu}'")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not console_mode:
        window = user_interface.get_window()
        status = window["-STATUS-"]
        update_status = True

    if args.model is not None:
        if Path(args.model).is_file():
            logging.debug(f"Will load a custom model file: {args.model}")
            model_path = args.model
        else:
            logging.debug(f"A custom model file was specified by it was not found: {args.model}")
    else:
        logging.debug("No custom model file was specified. Will use the default embedded model")
        model_path = MODEL_PATH

    model = None

    # Prepare tensor
    height, width, channels = DEFAULT_MODEL_INPUT_SHAPE
    image_tensor = np.empty((1, height, width, channels))

    # UI loop
    while True:
        event = None
        if not console_mode:
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                logging.debug("The exit action was selected")
                break
            if event == "-CLOSE-":
                logging.debug("Close was pressed")
                break
            if event == "-MULTIPLE-PATIENTS-":
                if values["-MULTIPLE-PATIENTS-"]:
                    window["-PATIENT-"].update(disabled=True)
                    window["-PATIENT-"]("")
                    window["-OPEN-LABELME-"].update(disabled=True)
                else:
                    window["-PATIENT-"].update(disabled=False)
                    window["-OPEN-LABELME-"].update(disabled=False)

        # Folder name was filled in, make a list of files in the folder, OR console mode is True
        if event == "-OK-" or console_mode:
            if not console_mode:
                if values["-INPUT-DIRECTORY-"] == "":
                    logging.debug("OK was pressed without a directory being selected")
                    status.update("Please select a directory to start")
                    continue
                if values["-PATIENT-"] == "" and values["-PATIENT-RECORD-"] == "":
                    if not (values["-MULTIPLE-PATIENTS-"] or values["-USE-BOUNDING-BOXES-"]):
                        logging.debug("OK was pressed without patient and record number")
                        status.update("Please insert patient or record")
                        continue

                # Get user input from the interface
                patient = values["-PATIENT-"]
                patient_record = values["-PATIENT-RECORD-"]
                patient_group = values["-PATIENT-GROUP-"]

                anatomical_site = values["-ANATOMICAL-SITE-"]
                exam_date = values["-EXAM-DATE-"]
                exam_instance = values["-EXAM-INSTANCE-"]

                classify_agnor = values["-CLASSIFY-AGNOR-"]
                bboxes = values["-USE-BOUNDING-BOXES-"]
                overlay = values["-GENERATE-OVERLAY-"]
                multiple_patients = values["-MULTIPLE-PATIENTS-"]
                open_labelme = values["-OPEN-LABELME-"] if not multiple_patients else False

                base_directory = Path(values["-INPUT-DIRECTORY-"])
                database = values["-DATABASE-"]
            else:
                # Get user input from the command line
                patient = args.patient
                patient_record = args.patient_record
                patient_group = args.patient_group
                anatomical_site = args.anatomical_site
                exam_date = args.exam_date
                exam_instance = args.exam_instance
                classify_agnor = args.classify_agnor
                bboxes = args.bboxes
                overlay = args.overlay
                multiple_patients = False
                open_labelme = False
                base_directory = Path(args.input_directory)
                database = args.database

            if base_directory.is_dir():
                if multiple_patients:
                    directories = [directory for directory in base_directory.glob("*") if directory.is_dir()]
                else:
                    directories = [str(base_directory)]
            else:
                raise FileNotFoundError(f"The directory '{base_directory}' was not found.")

            for input_directory in directories:
                datetime = f"{time.strftime('%Y%m%d%H%M')}"
                # Bboxed annotations -> bboxed annotations
                if bboxes:
                    if input_directory is not None:
                        logging.debug("Bounding boxed images/annotations -> bounding boxed annotations")
                        logging.debug(f"Loading images from '{input_directory}'")
                        try:
                            images = list_files(input_directory, as_numpy=True)
                        except Exception:
                            images = []
                        logging.debug(f"Total of {len(images)} images found")
                        if len(images) == 0:
                            logging.debug("No images found!")
                            status.update("No images found!")
                            update_status = False
                            continue

                        logging.debug(f"Loading annotations from '{input_directory}'")
                        try:
                            annotations = list_files(input_directory, as_numpy=True, file_types=[".json"])
                        except Exception:
                            annotations = []
                        logging.debug(f"Total of {len(annotations)} annotations found")
                        if len(annotations) == 0:
                            logging.debug("No annotations found!")
                            status.update("No annotations found!")
                            update_status = False
                            continue

                        if len(images) != len(annotations):
                            message = f"Number of images and annotations does no match ({len(images)} images, {len(annotations)} annotations"
                            logging.warning(message)
                            if len(images) > len(annotations):
                                logging.warning("Number of images is higher than the number of annotations")
                            else:
                                logging.warning("Number of annotations is higher than the number of images")

                            logging.warning("Filtering image files to those an annotation file was found")
                            images_with_annotations = []
                            annotation_stems = [Path(annotation_path).stem for annotation_path in annotations]
                            annotations = []
                            for image_path in images:
                                image_stem = Path(image_path).stem
                                if image_stem in annotation_stems:
                                    images_with_annotations.append(image_path)
                                    annotations.append(Path(image_path).parent.joinpath(f"{image_stem}.json"))
                            images = images_with_annotations

                        output_directory = input_directory

                        if not console_mode:
                            status.update("Processing")
                            event, values = window.read(timeout=0)
                        else:
                            progress_bar = tqdm(total=len(images), desc=f"Patient {patient}", unit="image", leave=False)

                        # Load and process each image and annotation
                        logging.debug("Start processing images and annotations")
                        for i, (image_path, annotation_path) in enumerate(zip(images, annotations)):
                            logging.debug(f"Processing image {image_path} and annotation {annotation_path}")
                            if patient == "":
                                key = Path(input_directory).name
                            else:
                                key = patient
                            if not console_mode:
                                if not sg.OneLineProgressMeter("Progress", i + 1, len(images), key, orientation="h"):
                                    if not i + 1 == len(images):
                                        user_interface.clear_fields(window)
                                        status.update("Canceled by the user")
                                        open_labelme = False
                                        update_status = False
                                        break
                            else:
                                progress_bar.update(1)

                            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                            image_original = image.copy()
                            logging.debug(f"Shape {image.shape}")
                            original_shape = image.shape[:2]

                            image = tf.cast(image, dtype=tf.float32)
                            image = image / 255.
                            image = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB)

                            if image.shape[0] != height:
                                image = pad_along_axis(image, size=height, axis=0)
                            if image.shape[1] != width:
                                image = pad_along_axis(image, size=width, axis=1)

                            image_tensor[0, :, :, :] = image

                            if model is None:
                                model = load_model(str(model_path), input_shape=DEFAULT_MODEL_INPUT_SHAPE)

                            logging.debug("Predict")
                            prediction = model.predict_on_batch(image_tensor)[0]

                            if original_shape[0] != height:
                                prediction = prediction[:original_shape[0], :, :]
                            if original_shape[1] != width:
                                prediction = prediction[:, :original_shape[1], :]

                            prediction = collapse_probabilities(prediction, pixel_intensity=127)

                            hashfile = get_hash_file(image_path)

                            update_annotation(
                                input_image=image_original,
                                prediction=prediction,
                                patient_record=patient_record,
                                patient=patient,
                                anatomical_site=anatomical_site,
                                annotation_directory=str(input_directory),
                                output_directory=str(output_directory),
                                source_image_path=image_path,
                                annotation_path=annotation_path,
                                original_image_shape=original_shape,
                                hashfile=hashfile,
                                classify_agnor=classify_agnor,
                                patient_group=patient_group,
                                exam_date=exam_date,
                                exam_instance=exam_instance,
                                overlay=overlay,
                                datetime=datetime
                            )

                            tf.keras.backend.clear_session()
                            logging.debug(f"Done processing image {image_path}")

                        logging.debug(f"Aggregating measurements")
                        aggregation_result = aggregate_measurements(
                            nucleus_measurements=str(Path(output_directory).joinpath(f"nucleus_measurements_{datetime}.csv")),
                            agnor_measurements=str(Path(output_directory).joinpath(f"agnor_measurements_{datetime}.csv")),
                            remove_measurement_files=False,
                            database=database,
                            datetime=datetime)
                        logging.debug(f"Measurements aggregation complete")

                        if aggregation_result:
                            if Path(annotations[0]).is_file():
                                with open(annotations[0], "r") as annotation_file:
                                    annotation = json.load(annotation_file)

                                    if "patient" in annotation.keys():
                                        if "dateTime" in annotation.keys():
                                            filename = f"{annotation['dateTime']} - Aggregate measurements - {annotation['patient']}.csv"
                                            filename = Path(input_directory).joinpath(filename)
                                            if filename.is_file():
                                                os.remove(str(filename))
                                        elif "last_updated" in annotation.keys():
                                            filename = f"{annotation['last_updated']} - Aggregate measurements - {annotation['patient']}.csv"
                                            filename = Path(input_directory).joinpath(filename)
                                            if filename.is_file():
                                                os.remove(str(filename))

                        if open_labelme and not multiple_patients:
                            status.update("Opening labelme, please wait...")
                            open_with_labelme(str(input_directory))
                    else:
                        logging.debug("Input directory is `None`")
                        status.update("Select an input directory")
                        continue
                else:
                    # Images -> annotations
                    if patient is not None and patient != "" or multiple_patients:
                        if input_directory is not None and input_directory != "":
                            logging.debug("Images -> annotations")
                            logging.debug(f"Loading images from '{input_directory}'")
                            try:
                                images = list_files(input_directory, as_numpy=True)
                            except Exception:
                                images = []
                            logging.debug(f"Total of {len(images)} images found")
                            if len(images) == 0:
                                logging.debug("No images found!")
                                status.update("No images found!")
                                update_status = False
                                continue

                            logging.debug("Create output directories")
                            if multiple_patients:
                                output_directory = Path(input_directory).joinpath(f"{datetime} - {input_directory.name}")
                            else:
                                output_directory = Path(input_directory).joinpath(f"{datetime} - {patient}")
                            output_directory.mkdir(exist_ok=True)
                            logging.debug(f"Created '{str(output_directory)}' directory")
                            annotation_directory = output_directory
                            annotation_directory.mkdir(exist_ok=True)
                            logging.debug(f"Created '{str(annotation_directory)}' directory")

                            if not console_mode:
                                status.update("Processing")
                                event, values = window.read(timeout=0)
                            else:
                                progress_bar = tqdm(total=len(images), desc=f"Patient {patient}", unit="image", leave=False)

                            # Load and process each image
                            logging.debug("Start processing images")
                            for i, image_path in enumerate(images):
                                logging.debug(f"Processing image {image_path}")
                                if patient == "":
                                    key = input_directory.name
                                else:
                                    key = patient
                                if not console_mode:
                                    if not sg.OneLineProgressMeter("Progress", i + 1, len(images), key, orientation="h"):
                                        if not i + 1 == len(images):
                                            user_interface.clear_fields(window)
                                            status.update("Canceled by the user")
                                            open_labelme = False
                                            update_status = False
                                            break
                                else:
                                    progress_bar.update(1)

                                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                                image_original = image.copy()
                                logging.debug(f"Shape {image.shape}")
                                original_shape = image.shape[:2]

                                image = tf.cast(image, dtype=tf.float32)
                                image = image / 255.
                                image = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB)

                                if image.shape[0] != height:
                                    image = pad_along_axis(image, size=height, axis=0)
                                if image.shape[1] != width:
                                    image = pad_along_axis(image, size=width, axis=1)

                                image_tensor[0, :, :, :] = image

                                if model is None:
                                    model = load_model(str(model_path), input_shape=DEFAULT_MODEL_INPUT_SHAPE)

                                logging.debug("Predict")
                                prediction = model.predict_on_batch(image_tensor)[0]

                                if original_shape[0] != height:
                                    prediction = prediction[:original_shape[0], :, :]
                                if original_shape[1] != width:
                                    prediction = prediction[:, :original_shape[1], :]

                                prediction = collapse_probabilities(prediction, pixel_intensity=127)

                                hashfile = get_hash_file(image_path)

                                create_annotation(
                                    input_image=image_original,
                                    prediction=prediction,
                                    patient_record=patient_record,
                                    patient=patient,
                                    anatomical_site=anatomical_site,
                                    annotation_directory=str(annotation_directory),
                                    output_directory=str(output_directory),
                                    source_image_path=image_path,
                                    original_image_shape=original_shape,
                                    hashfile=hashfile,
                                    classify_agnor=classify_agnor,
                                    patient_group=patient_group,
                                    exam_date=exam_date,
                                    exam_instance=exam_instance,
                                    overlay=overlay,
                                    datetime=datetime
                                )

                                tf.keras.backend.clear_session()
                                logging.debug(f"Done processing image {image_path}")

                            logging.debug(f"Aggregating measurements")
                            aggregation_result = aggregate_measurements(
                                nucleus_measurements=str(output_directory.joinpath(f"nucleus_measurements_{datetime}.csv")),
                                agnor_measurements=str(output_directory.joinpath(f"agnor_measurements_{datetime}.csv")),
                                remove_measurement_files=True,
                                database=database,
                                datetime=datetime)
                            logging.debug(f"Measurements aggregation complete")

                            if not aggregation_result:
                                open_labelme = False
                                status.update("Could not generate aggregate measurement file")

                            if open_labelme and not multiple_patients:
                                status.update("Opening labelme, please wait...")
                                open_with_labelme(str(annotation_directory))
                        else:
                            message = "Input directory is `None` or empty"
                            logging.error(message)
                            raise ValueError(message)
                    else:
                        message = "Provide the information to continue"
                        logging.debug(message)
                        status.update(message)
                        continue

                logging.debug(f"Done processing directory '{input_directory}'")

            if open_labelme and multiple_patients:
                open_with_labelme(str(base_directory))
        else:
            update_status = False

        if not console_mode:
            if update_status:
                status.update("Done!")
                user_interface.clear_fields(window)
            else:
                update_status = True
        else:
            break
        
        logging.debug("Selected directory event end")

    if not console_mode:
        window.close()


if __name__ == "__main__":
    main()
