import hashlib
import logging

import cv2
import numpy as np
import segmentation_models as sm
from scipy.interpolate import splev, splprep


CUSTOM_OBJECTS = {
    "categorical_crossentropy_plus_dice_loss": sm.losses.cce_dice_loss,
    "focal_loss_plus_dice_loss": sm.losses.categorical_focal_dice_loss,
    "f1-score": sm.metrics.f1_score,
    "iou_score": sm.metrics.iou_score,
}


def get_hash_file(path):
    with open(path, "rb") as f:
        bytes = f.read()
        hash_file = hashlib.sha256(bytes).hexdigest()
    return hash_file


def get_number_of_nor_contour_points(contour, shape):
    pixel_count = get_contour_pixel_count(contour, shape=shape)
    if pixel_count <= 100:
        return 8
    elif pixel_count <= 125:
        return 9
    elif pixel_count <= 150:
        return 10
    elif pixel_count <= 200:
        return 11
    elif pixel_count <= 250:
        return 13
    else:
        return 16


def smooth_contours(contours, points=30):
    smoothened_contours = []
    for contour in contours:
        try:
            x, y = contour.T

            # Convert from numpy arrays to normal arrays
            x = x.tolist()[0]
            y = y.tolist()[0]

            # Find the B-spline representation of an N-dimensional curve.
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
            tck, u = splprep([x, y], u=None, s=1.0, per=1, k=1)

            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
            u_new = np.linspace(u.min(), u.max(), points)

            # Given the knots and coefficients of a B-spline representation, evaluate the value of the smoothing polynomial and its derivatives.
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
            x_new, y_new = splev(u_new, tck, der=0)

            # Convert it back to Numpy format for OpenCV to be able to display it
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
            smoothened_contours.append(np.asarray(res_array, dtype=np.int32))
        except Exception as e:
            logging.warning("The smoothing of a contour caused a failure.")
            logging.exception(e)
    return smoothened_contours


def get_contour_pixel_count(contour, shape):
    image = np.zeros(shape)
    cv2.drawContours(image, contours=[contour], contourIdx=-1, color=1, thickness=cv2.FILLED)
    return int(image.sum())


def get_contours(binary_mask):
    binary_mask[binary_mask > 0] = 255
    binary_mask = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours_by_size(contours, min_area=5000, max_area=40000):
    filtered_contours = []
    discarded = []
    for contour in contours:
        try:
            contour_area = cv2.contourArea(contour)
            if min_area <= contour_area and contour_area <= max_area:
                filtered_contours.append(contour)
            else:
                discarded.append(contour)
        except:
            pass
    return filtered_contours, discarded


def filter_nuclei_without_nors(nuclei_polygons, nors_polygons):
    """Filter out nuclei without NORs."""
    filtered_nuclei = []
    discarded = []
    for nucleus in nuclei_polygons:
        keep_nucleus = False
        for nor in nors_polygons:
            for nor_point in nor:
                if cv2.pointPolygonTest(nucleus, tuple(nor_point[0]), False) >= 0:
                    keep_nucleus = True
        if keep_nucleus:
            filtered_nuclei.append(nucleus)
        else:
            discarded.append(nucleus)
    return filtered_nuclei, discarded


def filter_non_convex_nuclei(nuclei_polygons, shape):
    """Filter out non-convex enough nuclei."""
    convex_enough = []
    discarded = []
    for nucleus in nuclei_polygons:
        smoothed = get_contour_pixel_count(nucleus, shape)
        convex = get_contour_pixel_count(cv2.convexHull(nucleus), shape)
        if convex - smoothed < 1000:
            convex_enough.append(nucleus)
        else:
            discarded.append(nucleus)
    return convex_enough, discarded


def filter_nors_outside_nuclei(nuclei_polygons, nors_polygons):
    """Filter out NORs outside nuclei."""
    filtered_nors = []
    discarded = []
    for nor in nors_polygons:
        keep_nor = False
        for nucleus in nuclei_polygons:
            for nor_point in nor:
                if cv2.pointPolygonTest(nucleus, tuple(nor_point[0]), False) >= 0:
                    keep_nor = True
        if keep_nor:
            filtered_nors.append(nor)
        else:
            discarded.append(nor)
    return filtered_nors, discarded
