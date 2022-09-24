import logging
from pathlib import Path
from typing import List

import PySimpleGUI as sg


PROGRAM_NAME = "AgNOR Slide-Image Examiner"
PROGRAM_TITLE = "Select images to count nuclei and AgNOR"
TITLE_FONT = ("Arial", "14", "bold")
MAIN_FONT = ("Arial", "10", "bold")
SECONDARY_FONT = ("Arial", "10")

TOOLTIPS = {
    "patient": "Unique patient identifier. It can be the patient name or an ID/code.",
    "image_directory": "A directory containing images to be processed.",
    "inspect_with_labelme": "After processing, opens Labelme for inspection of the results.",
    "classify_agnor": "Classify AgNORs in clusters or satellites.",
    "bbox": "Restricts processing to nuclei within bounding boxes.",
    "overlay": "Generates an overlay of the input image and the predicted segmentation."
}


def clear_fields(window: sg.Window):
    """Clear the field in the UI.

    Args:
        window (sg.Window): The window handler.
    """
    window["-INPUT-DIRECTORY-"]("")
    window["-OUTPUT-DIRECTORY-"]("")
    window["-PATIENT-"]("")
    window["-OPEN-LABELME-"](False)
    window["-CLASSIFY-AGNOR-"](False)
    window["-USE-BOUNDING-BOXES-"](False)
    window["-GENERATE-OVERLAY-"](False)
    window["-OUTPUT-DIRECTORY-"].update(disabled=False)
    window["-PATIENT-"].update(disabled=False)


def get_layout() -> List[list]:
    """Provides the user interface layout.

    Returns:
        List[list]: List of layout element.
    """
    layout = [
        [
            sg.Text(PROGRAM_TITLE, text_color="white", font=TITLE_FONT),
        ],
        [
            sg.Text(f"{' ' * 14}Patient\t", text_color="white", font=MAIN_FONT, tooltip=TOOLTIPS["patient"]),
            sg.InputText(size=(70, 1), key="-PATIENT-", tooltip=TOOLTIPS["patient"])
        ],
        [
            sg.Text("Image Directory\t", text_color="white", font=MAIN_FONT, tooltip=TOOLTIPS["image_directory"]),
            sg.In(size=(70, 1), enable_events=True, key="-INPUT-DIRECTORY-", tooltip=TOOLTIPS["image_directory"]),
            sg.FolderBrowse()
        ],
        [
            sg.Text("Output Directory\t", text_color="white", font=MAIN_FONT),
            sg.In(size=(70, 1), enable_events=True, key="-OUTPUT-DIRECTORY-"),
            sg.FolderBrowse()
        ],
        [
            sg.Checkbox("Classify AgNOR", default=False, text_color="white", key="-CLASSIFY-AGNOR-", font=SECONDARY_FONT, tooltip=TOOLTIPS["classify_agnor"]),
            sg.Checkbox("Restrict processing to bounding boxes", default=False, enable_events=True, text_color="white", key="-USE-BOUNDING-BOXES-", font=SECONDARY_FONT, tooltip=TOOLTIPS["bbox"])
        ],
        [
            sg.Checkbox("Inspect results with Labelme", default=False, text_color="white", key="-OPEN-LABELME-", font=SECONDARY_FONT, tooltip=TOOLTIPS["inspect_with_labelme"]),
            sg.Checkbox("Generate segmentation overlay", default=False, text_color="white", key="-GENERATE-OVERLAY-", font=SECONDARY_FONT, tooltip=TOOLTIPS["overlay"])
        ],
        [
            sg.Text("Status: waiting" + " " * 30, text_color="white", key="-STATUS-", font=SECONDARY_FONT),
        ],
        [
            sg.Cancel("Close", size=(10, 1), pad=((502, 0), (10, 0)), key="-CLOSE-"),
            sg.Ok(size=(10, 1), pad=((10, 0), (10, 0)), font=MAIN_FONT, key="-OK-")
        ]
    ]
    return layout


def get_window() -> sg.Window:
    """Return user interface window handler.

    Returns:
        sg.Window: The user interface window handler.
    """
    sg.theme("DarkBlue")
    layout = get_layout()
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
    return window
