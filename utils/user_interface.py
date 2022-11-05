import logging
from pathlib import Path
from typing import List, Tuple

import PySimpleGUI as sg

from utils.utils import format_combobox_string


PROGRAM_NAME = "AgNOR Slide-Image Examiner"
PROGRAM_TITLE = "Select images to count nuclei and AgNOR"
TITLE_FONT = ("Arial", "14", "bold")
MAIN_FONT = ("Arial", "10", "bold")
SECONDARY_FONT = ("Arial", "10")
TERTIARY_FONT = ("Arial", "8", "italic")
CONFIG_FILE = "config.txt"

TOOLTIPS = {
    "patient": "Unique patient identifier. It can be the patient name or an ID/code.",
    "patient_record": "The record number of the patient.",
    "patient_group": "Group the patient belongs to. Note that if processing multiple patients at once, all of them will assigned the same group.",
    "anatomical_site": "The area of the mouth where the brushing was done.",
    "exam_date": "Date the brushing was done.",
    "exam_instance": "The 'time' of the exam. For example, 'T0', 'T1', 'T2', etc.",
    "image_directory": "A directory containing images to be processed.",
    "browse": "Opens a window that allows selecting a directory to process.",
    "classify_agnor": "Classify AgNORs in clusters or satellites.",
    "inspect_with_labelme": "After processing, opens Labelme for inspection of the results.",
    "bbox": "Restricts processing to nuclei within bounding boxes.",
    "overlay": "Generates an overlay of the input image and the predicted segmentation.",
    "multiple_patients": "Check this box if you selected a directory with multiple patients.",
}


def clear_fields(window: sg.Window):
    """Clear the field in the UI.

    Args:
        window (sg.Window): The window handler.
    """
    window["-PATIENT-"]("")
    window["-PATIENT-RECORD-"]("")
    window["-PATIENT-GROUP-"]("")
    window["-ANATOMICAL-SITE-"]("")
    window["-EXAM-DATE-"]("")
    window["-EXAM-INSTANCE-"]("")
    window["-INPUT-DIRECTORY-"]("")
    window["-CLASSIFY-AGNOR-"](False)
    window["-OPEN-LABELME-"](False)
    window["-USE-BOUNDING-BOXES-"](False)
    window["-GENERATE-OVERLAY-"](False)
    window["-MULTIPLE-PATIENTS-"](False)
    window["-PATIENT-"].update(disabled=False)
    window["-OPEN-LABELME-"].update(disabled=False)


def collapse(layout: List[sg.Element], key: str):
    """Helper function that creates a Column that can be later made hidden, thus appearing "collapsed".

    Args:
        layout (_type_): The layout for the section.
        key (_type_): Key used to make this section visible/invisible.

    Returns:
        sg.pin: A pinned column that can be placed directly into your layout
    """
    return sg.pin(sg.Column(layout, key=key))


def get_special_elements() -> Tuple[sg.Element]:
    """Gets the interface elements according to the configuration file (if available).

    If a valid config file is available, then then the elements returned will be of type `sg.Combo`, otherwise they will be of type `sg.In`.

    Returns:
        Tuple[sg.Element]: Tuple with the two special interface elements. The first corresponds to the patient groups, and the second to the anatomical sites.
    """
    patient_group = sg.In(size=(50, 1), key="-PATIENT-GROUP-", tooltip=TOOLTIPS["patient_group"])
    anatomical_site = sg.In(size=(50, 1), key="-ANATOMICAL-SITE-", tooltip=TOOLTIPS["anatomical_site"], pad=((5, 5), (6, 5)))

    if Path(CONFIG_FILE).is_file():
        with open(CONFIG_FILE, "r") as config_file:
            configs = config_file.readlines()

        patient_groups = []
        anatomical_sites = []
        for line in configs:
            if ":" in line:
                if line.upper().startswith("GROUP") or line.upper().startswith("GROUPO"):
                    group = format_combobox_string(line)
                    if len(group) > 0:
                        patient_groups.append(group)
                    continue
                if line.upper().startswith("SITE") or line.upper().startswith("SITIO"):
                    site = format_combobox_string(line)
                    if len(site) > 0:
                        anatomical_sites.append(site)
                    continue

        patient_groups = sorted(list(set(patient_groups)))
        anatomical_sites = sorted(list(set(anatomical_sites)))

        if len(patient_groups) > 0:
            patient_group = sg.Combo(patient_groups, size=(48, 1), key="-PATIENT-GROUP-", tooltip=TOOLTIPS["patient_group"])
        if len(anatomical_sites) > 0:
            anatomical_site = sg.Combo(anatomical_sites, size=(48, 1), key="-ANATOMICAL-SITE-", tooltip=TOOLTIPS["anatomical_site"])

    return patient_group, anatomical_site


def get_layout() -> List[list]:
    """Provides the user interface layout.

    Returns:
        List[list]: List of layout element.
    """
    patient_group, anatomical_site = get_special_elements()

    layout = [
        [
            sg.Text(PROGRAM_TITLE, text_color="white", font=TITLE_FONT, pad=((0, 0), (0, 15))),
        ],
        [
            sg.Text("Patient name\t", text_color="white", font=MAIN_FONT, tooltip=TOOLTIPS["patient"]),
            sg.InputText(size=(50, 1), key="-PATIENT-", tooltip=TOOLTIPS["patient"]),
            sg.Push(),

            sg.Text("Anatomical site\t", text_color="white", font=MAIN_FONT, key="-ANATOMICAL-SITE-TEXT-", tooltip=TOOLTIPS["anatomical_site"]),
            anatomical_site,
            # sg.Text("(optional)", text_color="white", font=TERTIARY_FONT),
            sg.Push(),
        ],
        [
            sg.Text("Patient record\t", text_color="white", font=MAIN_FONT, tooltip=TOOLTIPS["patient_record"]),
            sg.InputText(size=(50, 1), key="-PATIENT-RECORD-", tooltip=TOOLTIPS["patient_record"]),
            sg.Push(),

            sg.Text("Exam date\t", text_color="white", font=MAIN_FONT, key="-EXAM-DATE-TEXT-", tooltip=TOOLTIPS["exam_date"], pad=((91, 5), (5, 5))),
            sg.In(size=(50, 1), key="-EXAM-DATE-", tooltip=TOOLTIPS["exam_date"]),
            sg.CalendarButton("Select date", target="-EXAM-DATE-", format="%d/%m/%Y"),
            sg.Push(),
        ],
        [
            sg.Text("Patient group\t", text_color="white", font=MAIN_FONT, key="-PATIENT-GROUP-TEXT-", tooltip=TOOLTIPS["patient_group"]),
            patient_group,
            # sg.Text("(optional)", text_color="white", font=TERTIARY_FONT),
            sg.Push(),

            sg.Text("Exam instance\t", text_color="white", font=MAIN_FONT, key="-EXAM-INSTANCE-TEXT-", tooltip=TOOLTIPS["exam_instance"], pad=((6, 5), (5, 5))),
            sg.In(size=(50, 1), key="-EXAM-INSTANCE-", tooltip=TOOLTIPS["exam_instance"]),
            # sg.Text("(optional)", text_color="white", font=TERTIARY_FONT),
            sg.Push(),
        ],
        [
            sg.Text("Image Directory\t", text_color="white", font=MAIN_FONT, tooltip=TOOLTIPS["image_directory"], pad=((5, 0), (25, 0))),
            sg.In(size=(133, 1), enable_events=True, key="-INPUT-DIRECTORY-", tooltip=TOOLTIPS["image_directory"], pad=((9, 0), (25, 0))),
            sg.FolderBrowse(tooltip=TOOLTIPS["browse"], pad=((10, 0), (25, 0))),
            sg.Push(),
        ],
        [
            sg.Text("\n\nAdvanced options", text_color="white", font=MAIN_FONT)
        ],
        [
            sg.Checkbox("Classify AgNOR", default=False, text_color="white", key="-CLASSIFY-AGNOR-", font=SECONDARY_FONT, tooltip=TOOLTIPS["classify_agnor"]),
            sg.Checkbox("Inspect results with Labelme", default=False, text_color="white", key="-OPEN-LABELME-", font=SECONDARY_FONT, tooltip=TOOLTIPS["inspect_with_labelme"]),
            sg.Checkbox("Restrict processing to bounding boxes", default=False, text_color="white", enable_events=True, key="-USE-BOUNDING-BOXES-", font=SECONDARY_FONT, tooltip=TOOLTIPS["bbox"], visible=True),
            sg.Checkbox("Generate segmentation overlay", default=False, text_color="white", key="-GENERATE-OVERLAY-", font=SECONDARY_FONT, tooltip=TOOLTIPS["overlay"], visible=True),
            sg.Checkbox("Multiple patients per directory", default=False, text_color="white", enable_events=True, key="-MULTIPLE-PATIENTS-", font=SECONDARY_FONT, tooltip=TOOLTIPS["multiple_patients"], visible=True),
        ],
        [
            sg.Checkbox("Restrict processing to bounding boxes", default=False, text_color="white", enable_events=True, key="-USE-BOUNDING-BOXES-", font=SECONDARY_FONT, tooltip=TOOLTIPS["bbox"], visible=False),
            sg.Checkbox("Generate segmentation overlay", default=False, text_color="white", key="-GENERATE-OVERLAY-", font=SECONDARY_FONT, tooltip=TOOLTIPS["overlay"], visible=False)
        ],
        [
            sg.Checkbox("Multiple patients per directory", default=False, text_color="white", enable_events=True, key="-MULTIPLE-PATIENTS-", font=SECONDARY_FONT, tooltip=TOOLTIPS["multiple_patients"], visible=False),
        ],
        [
            sg.Text("", text_color="white", key="-STATUS-", font=SECONDARY_FONT, pad=((0, 0), (10, 0)))
        ],
        [
            sg.Cancel("Close", size=(10, 1), pad=((959, 0), (10, 0)), key="-CLOSE-"),
            sg.Ok("Start", size=(10, 1), pad=((10, 0), (10, 0)), font=MAIN_FONT, key="-OK-")
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
