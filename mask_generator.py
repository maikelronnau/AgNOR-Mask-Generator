import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import PySimpleGUI as sg
from tensorflow import keras

from utils import dice_coef, dice_coef_loss, filter_contours, smooth_contours


def save_annotation(nuclei_prediction, nors_prediction, annotation_directory, name, original_shape):
    logging.info(f"""Saving image annotations from {Path(name).name} annotations to {str(annotation_directory)}""")
    width = original_shape[0]
    height = original_shape[1]

    nuclei_prediction = cv2.resize(nuclei_prediction, (width, height))
    nuclei_prediction[nuclei_prediction < 0.5] = 0
    nuclei_prediction[nuclei_prediction >= 0.5] = 255

    nors_prediction = cv2.resize(nors_prediction, (width, height))
    nors_prediction[nors_prediction < 0.5] = 0
    nors_prediction[nors_prediction >= 0.5] = 255

    annotation = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(name),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
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
        # Consturct UI
        sg.theme("DarkBlue")
        layout = [
            [
                sg.Text("Image Folder", text_color="white"),
                sg.In(size=(50, 1), enable_events=True, key="-FOLDER-"),
                sg.FolderBrowse(),
            ],
            [
                sg.Text("Status: waiting" + " " * 30, text_color="white", key="-STATUS-"),
            ]
        ]

        # if "icon.ico" in glob.glob("icon.ico"):
        #     icon = os.path.join(".", "icon.ico")
        #     logging.info(f"""Loading icon from local directory""")
        # else:
        #     icon = os.path.join(sys._MEIPASS, "icon.ico")
        #     logging.info(f"""Loading icon from sys directory""")

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

        # Load models
        # if "nucleus.h5" in glob.glob("*.h5") and "nor.h5" in glob.glob("*.h5"):
        #     models_base_path = "."
        #     logging.info(f"""Loading models locally""")
        # else:
        #     models_base_path = sys._MEIPASS
        #     logging.info(f"""Loading models from sys""")

        nucleus_paths = [nucleus_path for nucleus_path in Path(__file__).parent.rglob("nucleus.h5")]
        nor_paths = [nor_path for nor_path in Path(__file__).parent.rglob("nor.h5")]

        logging.info("Models found:")
        logging.info(f"Nucleus: {nucleus_paths}")
        logging.info(f"NOR: {nor_paths}")

        if nucleus_paths[0].is_file():
            nucleus_model_path = str(nucleus_paths[0])
            logging.info(f"Loading nucleus model from '{nucleus_model_path}'")
        else:
            logging.error("Did not find 'nucleus.h5'.")
            raise Exception("Could not load 'nucleus.h5' model.")

        if nor_paths[0].is_file():
            nor_model_path = str(nor_paths[0])
            logging.info(f"Loading NOR model from '{nor_model_path}'")
        else:
            logging.error("Did not find 'nor.h5'.")
            raise Exception("Could not load 'nor.h5' model.")
        
        nuclei_model = keras.models.load_model(str(nucleus_model_path), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
        nors_model = keras.models.load_model(str(nor_model_path), custom_objects={"dice_coef_loss": dice_coef_loss, "dice_coef": dice_coef})
        logging.info(f"""Models loaded""")

        input_shape_nuclei = nuclei_model.input_shape[1:]
        logging.info(f"""Nuclei input shape: {input_shape_nuclei}""")
        input_shape_nors = nors_model.input_shape[1:]
        logging.info(f"""NOR input shape: {input_shape_nors}""")

        # Prepare tensor
        height_nuclei, width_nuclei, channels_nuclei = input_shape_nuclei
        height_nors, width_nors, channels_nors = input_shape_nors

        image_tensor_nuclei = np.empty((1, height_nuclei, width_nuclei, channels_nuclei))
        image_tensor_nors = np.empty((1, height_nors, width_nors, channels_nors))

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
                            status.update("Status: canceled by the user")
                            update_status = False
                            break

                    image_path = str(image_path)
                    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    logging.info(f"""Shape {image.shape}""")
                    original_shape = image.shape[:2][::-1]

                    image_nuclei = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB)
                    image_nuclei = cv2.resize(image_nuclei, (width_nuclei, height_nuclei))
                    image_tensor_nuclei[0, :, :, :] = image_nuclei

                    image_nors = cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB)
                    image_nors = cv2.resize(image_nors, (width_nors, height_nors))
                    image_tensor_nors[0, :, :, :] = image_nors

                    logging.info(f"""Predict nuclei""")
                    nuclei_prediction = nuclei_model.predict_on_batch(image_tensor_nuclei)
                    logging.info(f"""Predict NORs""")
                    nors_prediction = nors_model.predict_on_batch(image_tensor_nors)

                    save_annotation(nuclei_prediction[0], nors_prediction[0], annotation_directory, image_path, original_shape)
                    keras.backend.clear_session()
                    logging.info(f"""Done processing image {str(image_path)}""")

            if update_status:
                status.update("Status: done!")
                window["-FOLDER-"]("")
            else:
                update_status = True
            logging.info(f"""Selected directory event end""")
        window.close()
    except Exception as e:
        logging.error(f"""{e}""")


if __name__ == "__main__":
    main()