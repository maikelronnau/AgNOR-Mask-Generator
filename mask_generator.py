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

from utils import (CUSTOM_OBJECTS, filter_contours_by_size,
                   filter_non_convex_nuclei, filter_nors_outside_nuclei,
                   filter_nuclei_without_nors, get_contour_pixel_count,
                   get_contours, get_hash_file,
                   get_number_of_nor_contour_points, smooth_contours)


MSKG_VERSION = "v12"


def save_annotation(prediction, annotation_directory, name, original_shape, id, magnification, hashfile=None, date_time=None):
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
        "mskg_version": MSKG_VERSION,
        "flags": {},
        "shapes": [],
        "imageHeight": height,
        "imageWidth": width,
        "id": id,
        "magnification": magnification,
        "imagePath": os.path.basename(name),
        "imageHash": hashfile,
        "dateTime": date_time,
        "imageData": None
    }

    logging.info(f"""Find nuclei contours""")
    # Find segmentation contours
    nuclei_polygons = get_contours(nuclei_prediction)
    logging.info(f"""Filter nuclei contours""")
    nuclei_polygons, _ = filter_contours_by_size(nuclei_polygons)
    logging.info(f"""Smooth nuclei contours""")
    nuclei_polygons = smooth_contours(nuclei_polygons, points=40)

    logging.info(f"""Find NORs contours""")
    nors_polygons = get_contours(nors_prediction)
    logging.info(f"""Smooth NORs contours""")
    for i in range(len(nors_polygons)):
        points = get_number_of_nor_contour_points(nors_polygons[i], shape=(height, width))
        smoothed_contour = smooth_contours([nors_polygons[i]], points=points)
        if len(smoothed_contour) == 1:
            nors_polygons[i] = smoothed_contour[0]

    logging.info(f"""Filter out nuclei without nors""")
    filtered_nuclei, _ = filter_nuclei_without_nors(nuclei_polygons, nors_polygons)
    # TODO: Verify whether this function should be called or not.
    # logging.info(f"""Filter out deformed nuclei""")
    # filtered_nuclei, _ = filter_non_convex_nuclei(filtered_nuclei, (height, width))

    logging.info(f"""Filter out NORs outside nuclei""")
    filtered_nors, _ = filter_nors_outside_nuclei(filtered_nuclei, nors_polygons)

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
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description="Mask Generator")

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
        logging.info(f"""Starting""")
        date_time = f"{time.strftime('%Y%m%d%H%M%S')}"
        logging.info(date_time)
        logging.info(f"""Setting theme""")

        sg.theme("DarkBlue")
        main_font = ("Arial", "10", "bold")
        secondary_font = ("Arial", "10")

        # Construct UI
        layout = [
            [
                sg.Text(f"{' ' * 19}ID\t", text_color="white", font=main_font),
                sg.InputText(size=(30, 1), key="-ID-")
            ],
            [
                sg.Text("Magnification\t", text_color="white", font=main_font),
                sg.InputText(size=(30, 1), key="-MAGNIFICATION-")
            ],
            [
                sg.Text("Image Folder\t", text_color="white", font=main_font),
                sg.In(size=(70, 1), enable_events=True, key="-FOLDER-"),
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

        models_paths = [models_path for models_path in Path(__file__).parent.rglob("AgNOR_e087_l0.0273_vl0.1332.h5")]

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

        # UI loop
        while True:
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                logging.info(f"""The exit action was selected""")
                break
            if event == "-CANCEL-":
                logging.info(f"""Cancel was pressed""")
                break

            # Folder name was filled in, make a list of files in the folder
            if event == "-OK-":
                if values["-MAGNIFICATION-"] == "":
                    logging.info(f"""Ok was pressed without 'Magnification' being set""")
                    status.update("Status: insert the magnification")
                    continue
                if values["-FOLDER-"] == "":
                    logging.info(f"""OK was pressed without a directory being selected""")
                    status.update("Status: select a directory")
                    continue

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
                    logging.warning("Magnification information could not be converted to numerical values.")
                    logging.exception(e)
                    status.update("Status: review magnification")
                    continue

                id = values["-ID-"]

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
                            window["-ID-"]("")
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

                    hashfile = get_hash_file(image_path)
                    save_annotation(prediction, annotation_directory, image_path, original_shape, id, magnification, hashfile, date_time)
                    tf.keras.backend.clear_session()
                    logging.info(f"""Done processing image {str(image_path)}""")
            else:
                update_status = False

            if update_status:
                status.update("Status: done!")
                window["-FOLDER-"]("")
                window["-ID-"]("")
                window["-MAGNIFICATION-"]("")
            else:
                update_status = True
            logging.info(f"""Selected directory event end""")
        window.close()
    except Exception as e:
        logging.error(f"""{e}""")


if __name__ == "__main__":
    main()
