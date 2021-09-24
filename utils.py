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

def filter_contours(contours, min_area=5000, max_area=40000):
    filtered_contours = []
    for contour in contours:
        try:
            contour_area = cv2.contourArea(contour)
            if min_area <= contour_area and contour_area <= max_area:
                filtered_contours.append(contour)
        except:
            pass
    return filtered_contours


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
        except:
            pass
    return smoothened_contours
