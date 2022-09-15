import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import PySimpleGUI as sg
import tensorflow as tf

from utils import user_interface
from utils.annotation import create_annotation, update_annotation
from utils.data import list_files
from utils.model import load_model
from utils.utils import (DEFAULT_MODEL_INPUT_SHAPE, MODEL_PATH, PROGRAM_NAME,
                         collapse_probabilities, get_hash_file,
                         open_with_labelme)


def main():
    parser = argparse.ArgumentParser(description=PROGRAM_NAME)
    parser.add_argument(
        "-d",
        "--debug",
        help="Enable or disable debug mode.",
        default=False,
        action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            filename="mskg.log",
            filemode="a",
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s")

    try:
        datetime = f"{time.strftime('%Y%m%d%H%M%S')}"
        logging.debug(f"Program started at `{datetime}`")

        window = user_interface.get_window()
        status = window["-STATUS-"]
        update_status = True

        model = load_model(str(MODEL_PATH), input_shape=DEFAULT_MODEL_INPUT_SHAPE)

        # Prepare tensor
        height, width, channels = DEFAULT_MODEL_INPUT_SHAPE
        image_tensor = np.empty((1, height, width, channels))

        # UI loop
        while True:
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                logging.debug("The exit action was selected")
                break
            if event == "-CLOSE-":
                logging.debug("Close was pressed")
                break
            if event == "-USE-BOUNDING-BOXES-":
                if values["-USE-BOUNDING-BOXES-"]:
                    window["-OUTPUT-DIRECTORY-"].update(disabled=True)
                else:
                    window["-OUTPUT-DIRECTORY-"].update(disabled=False)

            # Folder name was filled in, make a list of files in the folder
            if event == "-OK-":
                if values["-INPUT-DIRECTORY-"] == "":
                    logging.debug("OK was pressed without a directory being selected")
                    status.update("Status: select a directory")
                    continue

                patient = values["-PATIENT-"]
                input_directory = values["-INPUT-DIRECTORY-"]
                output_directory = values["-OUTPUT-DIRECTORY-"]
                bboxes = values["-USE-BOUNDING-BOXES-"]
                classify_agnor = values["-CLASSIFY-AGNOR-"]

                # Bboxed annotations -> bboxed annotations
                if bboxes:
                    if input_directory is not None:
                        logging.debug("Bounding boxed images/annotations -> bounding boxed annotations")
                        try:
                            logging.debug(f"Loading images from '{input_directory}'")
                            images = list_files(input_directory, as_numpy=True)
                            logging.debug(f"Total of {len(images)} images found")
                            if len(images) == 0:
                                status.update("Status: no images found!")
                                logging.debug("No images were found")
                                continue

                            logging.debug(f"Loading annotations from '{input_directory}'")
                            annotations = list_files(input_directory, as_numpy=True, file_types=[".json"])
                            logging.debug(f"Total of {len(annotations)} annotations found")
                            if len(annotations) == 0:
                                status.update("Status: no annotations found!")
                                logging.debug("No annotations were found")
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

                            status.update("Status: processing")
                            event, values = window.read(timeout=0)

                            # Load and process each image and annotation
                            logging.debug("Start processing images and annotations")
                            for i, (image_path, annotation_path) in enumerate(zip(images, annotations)):
                                logging.debug(f"Processing image {image_path} and annotation {annotation_path}")
                                if not sg.OneLineProgressMeter("Progress", i + 1, len(images), "key", orientation="h"):
                                    if not i + 1 == len(images):
                                        user_interface.clear_fields(window)
                                        status.update("Status: canceled by the user")
                                        update_status = False
                                        break

                                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                                image_original = image.copy()
                                logging.debug(f"Shape {image.shape}")
                                original_shape = image.shape[:2]

                                image = tf.cast(image, dtype=tf.float32)
                                image = image / 255.
                                image = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB)
                                image_tensor[0, :, :, :] = image

                                logging.debug("Predict")
                                prediction = model.predict_on_batch(image_tensor)[0]
                                prediction = collapse_probabilities(prediction, pixel_intensity=127)

                                hashfile = get_hash_file(image_path)

                                update_annotation(
                                    input_image=image_original,
                                    prediction=prediction,
                                    patient=patient,
                                    annotation_directory=str(input_directory),
                                    output_directory=str(output_directory),
                                    source_image_path=image_path,
                                    annotation_path=annotation_path,
                                    original_image_shape=original_shape,
                                    hashfile=hashfile,
                                    classify_agnor=classify_agnor,
                                    datetime=datetime
                                )

                                tf.keras.backend.clear_session()
                                logging.debug(f"Done processing image {image_path}")

                            if values["-OPEN-LABELME-"]:
                                status.update("Status: opening labelme, please wait...")
                                open_with_labelme(str(input_directory))
                        except Exception as e:
                            logging.error(f"{e.message}")
                    else:
                        logging.debug("Input directory is `None`")
                        status.update("Select an input directory")
                        continue
                else:
                    # Images -> annotations
                    if patient is not None and patient != "":
                        if input_directory is not None and input_directory != "":
                            if output_directory is not None:
                                logging.debug("Images -> annotations")
                                try:
                                    logging.debug(f"Loading images from '{input_directory}'")
                                    images = list_files(input_directory, as_numpy=True)
                                    logging.debug(f"Total of {len(images)} images found")
                                    if len(images) == 0:
                                        status.update("Status: no images found!")
                                        logging.debug("No images were found")
                                        continue

                                    logging.debug("Create output directories")
                                    output_directory = Path(f"{time.strftime('%Y-%m-%d-%Hh%Mm')} - {patient}")
                                    output_directory.mkdir(exist_ok=True)
                                    logging.debug(f"Created '{str(output_directory)}' directory")
                                    annotation_directory = output_directory.joinpath("annotations")
                                    annotation_directory.mkdir(exist_ok=True)
                                    logging.debug(f"Created '{str(annotation_directory)}' directory")

                                    status.update("Status: processing")
                                    event, values = window.read(timeout=0)

                                    # Load and process each image
                                    logging.debug("Start processing images")
                                    for i, image_path in enumerate(images):
                                        logging.debug(f"Processing image {image_path}")
                                        if not sg.OneLineProgressMeter("Progress", i + 1, len(images), "key", orientation="h"):
                                            if not i + 1 == len(images):
                                                user_interface.clear_fields(window)
                                                status.update("Status: canceled by the user")
                                                update_status = False
                                                break

                                        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                                        image_original = image.copy()
                                        logging.debug(f"Shape {image.shape}")
                                        original_shape = image.shape[:2]

                                        image = tf.cast(image, dtype=tf.float32)
                                        image = image / 255.
                                        image = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB)
                                        image_tensor[0, :, :, :] = image

                                        logging.debug("Predict")
                                        prediction = model.predict_on_batch(image_tensor)[0]
                                        prediction = collapse_probabilities(prediction, pixel_intensity=127)

                                        hashfile = get_hash_file(image_path)

                                        create_annotation(
                                            input_image=image_original,
                                            prediction=prediction,
                                            patient=patient,
                                            annotation_directory=str(annotation_directory),
                                            output_directory=str(output_directory),
                                            source_image_path=image_path,
                                            original_image_shape=original_shape,
                                            hashfile=hashfile,
                                            classify_agnor=classify_agnor,
                                            datetime=datetime
                                        )

                                        tf.keras.backend.clear_session()
                                        logging.debug(f"Done processing image {image_path}")

                                    if values["-OPEN-LABELME-"]:
                                        status.update("Status: opening labelme, please wait...")
                                        open_with_labelme(str(annotation_directory))

                                except Exception as e:
                                    logging.error(f"{e.message}")
                            else:
                                message = "Output directory is `None`"
                                logging.error(message)
                                raise ValueError(message)
                        else:
                            message = "Input directory is `None` or empty"
                            logging.error(message)
                            raise ValueError(message)
                    else:
                        message = "Provide the information to continue"
                        status.update(message)
                        logging.debug(message)
                        continue
            else:
                update_status = False

            if update_status:
                status.update("Status: done!")
                user_interface.clear_fields(window)
            else:
                update_status = True
            logging.debug("Selected directory event end")
        window.close()
    except Exception as e:
        logging.error(f"{e}")


if __name__ == "__main__":
    main()
