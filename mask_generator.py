import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import PySimpleGUI as sg
import tensorflow as tf

from utils import CUSTOM_OBJECTS, filter_contours, smooth_contours


def save_annotation(prediction, annotation_directory, name, original_shape, magnification):
    logging.info(f"""Saving image annotations from {Path(name).name} annotations to {str(annotation_directory)}""")
    width = original_shape[0]
    height = original_shape[1]

    prediction = cv2.resize(prediction, (width, height))
    prediction[:, :, 0] = np.where(
        np.logical_and(prediction[:, :, 0] > prediction[:, :, 1], prediction[:, :, 0] > prediction[:, :, 2]), 255, 0)
    prediction[:, :, 1] = np.where(
        np.logical_and(prediction[:, :, 1] > prediction[:, :, 0], prediction[:, :, 1] > prediction[:, :, 2]), 255, 0)
    prediction[:, :, 2] = np.where(
        np.logical_and(prediction[:, :, 2] > prediction[:, :, 0], prediction[:, :, 2] > prediction[:, :, 1]), 255, 0)

    nuclei_prediction = prediction[:, :, 1].astype(np.uint8)
    nors_prediction = prediction[:, :, 2].astype(np.uint8)

    annotation = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [],
        "imageHeight": height,
        "imageWidth": width,
        "magnification": magnification,
        "imagePath": os.path.basename(name),
        "imageData": None
    }

    logging.info(f"""Find nuclei contours""")
    # Find segmentation contours
    nuclei_polygons, _ = cv2.findContours(nuclei_prediction.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.info(f"""Filter nuclei contours""")
    nuclei_polygons = filter_contours(nuclei_polygons)
    logging.info(f"""Smooth nuclei contours""")
    nuclei_polygons = smooth_contours(nuclei_polygons)

    logging.info(f"""Find NORs contours""")
    nors_polygons, _ = cv2.findContours(nors_prediction.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logging.info(f"""Smooth NORs contours""")
    nors_polygons = smooth_contours(nors_polygons)

    logging.info(f"""Filter out nuclei without nors""")
    filtered_nuclei = []
    for nucleus in nuclei_polygons:
        keep_nucleus = False
        for nor in nors_polygons:
            if cv2.pointPolygonTest(nucleus, tuple(nor[0][0]), True) >= 0:
                keep_nucleus = True
        if keep_nucleus:
            filtered_nuclei.append(nucleus)

    logging.info(f"""Filter out NORs outside nuclei""")
    filtered_nors = []
    for nor in nors_polygons:
        keep_nor = False
        for nucleus in nuclei_polygons:
            if cv2.pointPolygonTest(nucleus, tuple(nor[0][0]), True) >= 0:
                keep_nor = True
        if keep_nor:
            filtered_nors.append(nor)

    logging.info(f"""Add nuclei shapes to annotation file""")
    for nucleus_points in filtered_nuclei:
        points = []
        for point in nucleus_points:
            points.append([int(value) for value in point[0]])

        shape = {
            "label": "nucleus",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        annotation["shapes"].append(shape)

    logging.info(f"""Add NORs shapes to annotation file""")
    for nors_points in filtered_nors:
        points = []
        for point in nors_points:
            points.append([int(value) for value in point[0]])

        shape = {
            "label": "nor",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        annotation["shapes"].append(shape)

    logging.info(f"""Write annotation file""")
    with open(os.path.join(annotation_directory, f'{os.path.splitext(os.path.basename(name))[0]}.json'), "w") as output_file:
        json.dump(annotation, output_file, indent=2)

    logging.info(f"""Copy original image to the annotation directory""")
    if Path(os.path.join(annotation_directory, os.path.basename(name))).is_file():
        filename = Path(os.path.join(annotation_directory, os.path.basename(name)))
        filename = filename.stem + f"_{np.random.randint(1, 1000)}" + filename.suffix
        shutil.copyfile(name, os.path.join(annotation_directory, filename))
    else:
        shutil.copyfile(name, os.path.join(annotation_directory, os.path.basename(name)))


def main():
    parser = argparse.ArgumentParser(description="Mask Generator")

    parser.add_argument(
        "-d",
        "--debug",
        help="Enable or disable debug mode.",
        default=False,
        action="store_true")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(filename="mskg.log", filemode="a", level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

    try:
        logging.info(f"""Starting""")
        logging.info(f"""Setting theme""")

        sg.theme("DarkBlue")
        main_font = ("Arial", "10", "bold")
        secondary_font = ("Arial", "10")

        # Consturct UI
        layout = [
            [
                sg.Text("Maginification", text_color="white", font=main_font),
                sg.InputText(size=(10, 1), key="-MAGNIFICATION-")
            ],
            [
                sg.Text("Image Folder ", text_color="white", font=main_font),
                sg.In(size=(50, 1), enable_events=True, key="-FOLDER-"),
                sg.FolderBrowse(),
            ],
            [
                sg.Text("Status: waiting" + " " * 30, text_color="white", key="-STATUS-", font=secondary_font),
            ]
        ]

        icon_paths = [icon_path for icon_path in Path(__file__).parent.rglob("icon.ico")]
        logging.info(f"Icons found: {icon_paths}")

        if icon_paths[0].is_file():
            icon = str(icon_paths[0])
            logging.info(f"Loading icon from '{icon}'")
        else:
            logging.warning("Did not find 'icon.ico'.")
            logging.warning("Program will start without it.")
            icon = None

        logging.info(f"""Create window""")
        try:
            if icon:
                window = sg.Window("Mask Generator", layout, finalize=True, icon=icon)
            else:
                window = sg.Window("Mask Generator", layout, finalize=True)
        except Exception as e:
            logging.info(f"Could not load found icon.")
            window = sg.Window("Mask Generator", layout, finalize=True)
            logging.warning("Program will start without it.")

        status = window["-STATUS-"]
        update_status = True

        # Prediction settings
        supported_types = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

        models_paths = [models_path for models_path in Path(__file__).parent.rglob("AgNOR_e030_l0.0782_vl0.2396.h5")]

        logging.info("Model(s) found:")
        logging.info(f"{models_paths}")

        if models_paths[0].is_file():
            model_path = str(models_paths[0])
            logging.info(f"Loading model from '{model_path}'")
        else:
            logging.error("Did not find a file corresponding to a model.")
            raise Exception(f"Could not load '{models_paths[0]}' model.")

        model = tf.keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS)
        logging.info(f"""Model loaded""")

        input_shape = model.input_shape[1:]
        logging.info(f"""Nuclei input shape: {input_shape}""")

        # Prepare tensor
        height, width, channels = input_shape
        image_tensor = np.empty((1, height, width, channels))

        logging.info(f"""List local files""")
        for file_name in [path for path in Path(__file__).parent.rglob("*.*")]:
            logging.info(f"""{str(file_name)}""")
        logging.info(f"""Finished listing local files""")

        # UI loop
        while True:
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                logging.info(f"""The exit action was selected""")
                break
            if values["-FOLDER-"] == "":
                logging.info(f"""Browse was used without a directory being selected""")
                continue
            if values["-MAGNIFICATION-"] == "":
                logging.info(f"""Browse was used without 'Magnification' being set""")
                status.update("Status: insert the magnification")
                continue

            # Folder name was filled in, make a list of files in the folder
            if event == "-FOLDER-":
                logging.info(f"""Selected directory event start""")
                folder = values["-FOLDER-"]
                logging.info(f"""Loading images from '{folder}'""")
                images = [path for path in Path(folder).rglob("*.*") if path.suffix.lower() in supported_types]

                if len(images) == 0:
                    status.update("Status: no images found!")
                    logging.info(f"""No images were found""")
                    continue

                try:
                    magnification = int(values["-MAGNIFICATION-"])
                except Exception as e:
                    logging.warning("Maginifcation information could not be converted to numberical values.")
                    logging.exception(e)
                    status.update("Status: review magnification")
                    continue

                logging.info(f"""Total of {len(images)} found""")
                for image in images:
                    logging.info(f"""{image}""")

                annotation_directory = f"{time.strftime('%Y-%m-%d-%Hh%Mm')}-proposed-annotations"
                logging.info(f"""Annotation will be saved at {annotation_directory}""")
                if not os.path.isdir(annotation_directory):
                    os.mkdir(annotation_directory)

                status.update("Status: processing")
                event, values = window.read(timeout=0)

                # Load and process each image
                logging.info(f"""Start processing images""")
                for i, image_path in enumerate(images):
                    logging.info(f"""Processing image {str(image_path)}""")
                    if not sg.OneLineProgressMeter("Progress", i + 1, len(images), "key", orientation="h"):
                        if not i + 1 == len(images):
                            window["-FOLDER-"]("")
                            window["-MAGNIFICATION-"]("")
                            status.update("Status: canceled by the user")
                            update_status = False
                            break

                    image_path = str(image_path)
                    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    image = tf.cast(image, dtype=tf.float32)
                    image = image / 255.

                    logging.info(f"""Shape {image.shape}""")
                    original_shape = image.shape[:2][::-1]

                    image = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (width, height))
                    image_tensor[0, :, :, :] = image

                    logging.info(f"""Predict""")
                    prediction = model.predict_on_batch(image_tensor)[0]

                    prediction[:, :, 0] = np.where(np.logical_and(prediction[:, :, 0] > prediction[:, :, 1], prediction[:, :, 0] > prediction[:, :, 2]), 127, 0)
                    prediction[:, :, 1] = np.where(np.logical_and(prediction[:, :, 1] > prediction[:, :, 0], prediction[:, :, 1] > prediction[:, :, 2]), 127, 0)
                    prediction[:, :, 2] = np.where(np.logical_and(prediction[:, :, 2] > prediction[:, :, 0], prediction[:, :, 2] > prediction[:, :, 1]), 127, 0)

                    save_annotation(prediction, annotation_directory, image_path, original_shape, magnification)
                    tf.keras.backend.clear_session()
                    logging.info(f"""Done processing image {str(image_path)}""")

            if update_status:
                status.update("Status: done!")
                window["-FOLDER-"]("")
                window["-MAGNIFICATION-"]("")
            else:
                update_status = True
            logging.info(f"""Selected directory event end""")
        window.close()
    except Exception as e:
        logging.error(f"""{e}""")


if __name__ == "__main__":
    main()
