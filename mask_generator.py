import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import PySimpleGUI as sg
import tensorflow as tf

from utils import contour_analysis
from utils.data import list_files
from utils.model import load_model
from utils.utils import collapse_probabilities, color_classes, get_hash_file


PROGRAM_NAME = "AgNOR Mask Generator"
MSKG_VERSION = "v13"
MODEL_PATH = "AgNOR_e142_l0.0453_DenseNet-169_Linknet.h5"
DECISION_TREE_MODEL_PATH = "agnor_decision_tree_classifier.joblib"
DEFAULT_MODEL_INPUT_SHAPE = (1920, 2560, 3)

LABELME_CLASS_NAMES = (
    "_background_",
    "nucleus",
    "cluster",
    "satellite",
    "discarded_nucleus",
    "discarded_nor"
)
LABELME_CLASS_IDS = {
    "__ignore__": -1,
    "_background_": 0,
    "nucleus": 1,
    "cluster": 2,
    "satellite": 3,
    "discarded_nucleus": 4,
    "discarded_cluster": 5
}


def save_annotation(
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
    
    logging.debug(f"""Saving image annotations from {source_image_path.name} annotations to {str(annotation_directory)}""")
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

    logging.debug(f"""Analyze contours""")
    prediction, _ = contour_analysis.analyze_contours(mask=prediction, smooth=True)
    prediction, parent_contours, child_contours = prediction

    if classify_agnor:
        logging.debug("Add an extra channel to map `satellites`")
        prediction = np.stack([
            prediction[:, :, 0],
            prediction[:, :, 1],
            prediction[:, :, 2],
            np.zeros(original_image_shape, dtype=np.uint8) # Width and height reversed because of OpenCV.
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
        
    logging.debug(f"""Write annotation file""")
    annotation_path = str(annotation_directory.joinpath(f"{source_image_path.stem}.json"))
    with open(annotation_path, "w") as output_file:
        json.dump(annotation, output_file, indent=4)

    logging.debug(f"""Copy original image to the annotation directory""")
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
    cv2.imwrite(str(overlay_directory.joinpath(source_image_path.name)), overlay)


def clear_fields(window: sg.Window):
    """Clear the field in the UI.

    Args:
        window (sg.Window): The window handler.
    """
    window["-INPUT-DIRECTORY-"]("")
    window["-OUTPUT-DIRECTORY-"]("")
    window["-PATIENT-"]("")


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
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
        logging.debug(f"""Starting""")
        datetime = f"{time.strftime('%Y%m%d%H%M%S')}"
        logging.debug(datetime)
        logging.debug(f"""Setting theme""")

        sg.theme("DarkBlue")
        main_font = ("Arial", "10", "bold")
        secondary_font = ("Arial", "10")

        # Construct UI
        layout = [
            [
                sg.Text(f"{' ' * 14}Patient\t", text_color="white", font=main_font),
                sg.InputText(size=(30, 1), key="-PATIENT-")
            ],
            [
                sg.Text("Image Directory\t", text_color="white", font=main_font),
                sg.In(size=(70, 1), enable_events=True, key="-INPUT-DIRECTORY-"),
                sg.FolderBrowse()
            ],
            [
                sg.Text("Output Directory\t", text_color="white", font=main_font),
                sg.In(size=(70, 1), enable_events=True, key="-OUTPUT-DIRECTORY-"),
                sg.FolderBrowse()
            ],
            [
                sg.Text("Status: waiting" + " " * 30, text_color="white", key="-STATUS-", font=secondary_font),
            ],
            [
                sg.Cancel(size=(10, 1), pad=((502, 0), (10, 0)), key="-CANCEL-"),
                sg.Ok(size=(10, 1), pad=((10, 0), (10, 0)), font=main_font, key="-OK-")
            ]
        ]

        logging.debug(f"""Create window""")
        icon_path = "icon.ico"
        try:
            if Path(icon_path).is_file():
                logging.debug(f"""Load icon""")
                window = sg.Window(PROGRAM_NAME, layout, finalize=True, icon=icon_path)
            else:
                window = sg.Window(PROGRAM_NAME, layout, finalize=True)
        except Exception as e:
            logging.debug(f"Could not load icon.")
            window = sg.Window(PROGRAM_NAME, layout, finalize=True)
            logging.warning("Program will start without it.")

        status = window["-STATUS-"]
        update_status = True

        logging.info(f"""Load model""")
        try:
            if Path(MODEL_PATH).is_file():
                logging.debug(f"Loading model from '{MODEL_PATH}'")
                model = load_model(str(MODEL_PATH), input_shape=DEFAULT_MODEL_INPUT_SHAPE)
                logging.debug(f"Successfully loaded model '{MODEL_PATH}'")
            else:
                logging.error(f"Model file '{MODEL_PATH}' was not found.")
                raise Exception(f"Model file '{MODEL_PATH}' was not found.")
        except Exception as e:
            logging.error(f"""{e.message}""")
            raise Exception(f"Could not load '{MODEL_PATH}' model.")
        logging.debug(f"""Successfully loaded model.""")

        # Prepare tensor
        height, width, channels = DEFAULT_MODEL_INPUT_SHAPE
        image_tensor = np.empty((1, height, width, channels))

        # UI loop
        while True:
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                logging.debug(f"""The exit action was selected""")
                break
            if event == "-CANCEL-":
                logging.debug(f"""Cancel was pressed""")
                break

            # Folder name was filled in, make a list of files in the folder
            if event == "-OK-":
                if values["-INPUT-DIRECTORY-"] == "":
                    logging.debug(f"""OK was pressed without a directory being selected""")
                    status.update("Status: select a directory")
                    continue

                logging.debug(f"""Selected directory event start""")
                folder = values["-INPUT-DIRECTORY-"]
                logging.debug(f"""Loading images from '{folder}'""")
                try:
                    images = list_files(folder, as_numpy=True)
                    logging.debug(f"""Total of {len(images)} found""")
                except Exception as e:
                    logging.error(f"{e.message}")

                if len(images) == 0:
                    status.update("Status: no images found!")
                    logging.debug(f"""No images were found""")
                    continue

                patient = values["-PATIENT-"]

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
                logging.debug(f"""Start processing images""")
                for i, image_path in enumerate(images):
                    logging.debug(f"""Processing image {image_path}""")
                    if not sg.OneLineProgressMeter("Progress", i + 1, len(images), "key", orientation="h"):
                        if not i + 1 == len(images):
                            clear_fields(window)
                            status.update("Status: canceled by the user")
                            update_status = False
                            break

                    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    image_original = image.copy()
                    logging.debug(f"""Shape {image.shape}""")
                    original_shape = image.shape[:2]

                    image = tf.cast(image, dtype=tf.float32)
                    image = image / 255.
                    image = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB)
                    image_tensor[0, :, :, :] = image

                    logging.debug(f"""Predict""")
                    prediction = model.predict_on_batch(image_tensor)[0]

                    prediction = collapse_probabilities(prediction, pixel_intensity=127)

                    hashfile = get_hash_file(image_path)
                    save_annotation(
                        input_image=image_original,
                        prediction=prediction,
                        patient=patient,
                        annotation_directory=str(annotation_directory),
                        output_directory=str(output_directory),
                        source_image_path=image_path,
                        original_image_shape=original_shape,
                        hashfile=hashfile,
                        classify_agnor=True, # TODO: Update with argument coming from a checkbox in the UI.
                        datetime=datetime
                    )

                    tf.keras.backend.clear_session()
                    logging.debug(f"""Done processing image {image_path}""")
            else:
                update_status = False

            if update_status:
                status.update("Status: done!")
                clear_fields(window)
            else:
                update_status = True
            logging.debug(f"""Selected directory event end""")
        window.close()
    except Exception as e:
        logging.error(f"""{e}""")


if __name__ == "__main__":
    main()
