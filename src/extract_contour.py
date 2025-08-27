from typing import Optional

import cv2
import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt

DOUBLE_LINE_JOIN_KERNEL = np.array([[2, 3, 2], [3, -24, 3], [2, 3, 2]])
DOUBLE_LINE_JOIN_BLUR_SIZE = 5
DOUBLE_LINE_JOIN_ERODE_SIZE = 3
BLOB_DETECTION_KERNEL = np.array([[2, 3, 2], [3, 0, 3], [2, 3, 2]])
BLOB_DETECTION_THRESHOLD = 4
AUTO_THRESHOLDS = np.arange(2.0, 5.1, 0.1)
AUTO_THRESHOLD_GAP = 0.5
AUTO_MASK_COEFFICIENT = 0.015
AUTO_EXTREMUM_DIFF = 0.01
AUTO_WEIGHTED_MEAN = 0.4


def _contour_mask(
    img_rgb: np.ndarray, hysteresis_params: tuple[int, int], detail_threshold: float
) -> np.ndarray:
    """
    Extract contour and fine details from an artwork.

    Args:
        img_rgb: Image in RGB format. Array of shape (h, w, 3).
        hysteresis_params: Parameters for OpenCV Canny algorithm.
        detail_threshold: Threshold value for contour extraction.

    Returns:
        Contour mask in grayscale format. Array of shape: (h, w).

    """
    # Extract edges
    edges = cv2.Canny(img_rgb, *hysteresis_params)

    # Filter with a custom kernel
    filtered = cv2.filter2D(edges / 255, -1, DOUBLE_LINE_JOIN_KERNEL)
    filtered = np.maximum(filtered, 0)

    # Blur
    blurred = cv2.GaussianBlur(filtered, [DOUBLE_LINE_JOIN_BLUR_SIZE] * 2, 0)

    # Threshold
    mask = ((blurred > detail_threshold) * 255).astype(np.uint8)

    # Erode mask
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, [DOUBLE_LINE_JOIN_ERODE_SIZE] * 2
    )
    eroded = cv2.erode(
        mask, erode_kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    )

    # Find blobs
    neighbors = cv2.filter2D(eroded / 255, -1, BLOB_DETECTION_KERNEL)
    blob_mask = (neighbors <= BLOB_DETECTION_THRESHOLD) & (eroded // 255).astype(bool)

    # eroded *= ~filtered
    return blob_mask.sum(), (eroded // 255).sum(), eroded


def extract_contour(
    img_rgb: np.ndarray,
    hysteresis_params: tuple[int, int],
    detail_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Extract contour and fine details from an artwork.

    Args:
        img_rgb: Image in RGB format. Array of shape (h, w, 3).
        hysteresis_params: Parameters for OpenCV Canny algorithm.
        detail_threshold: Threshold value for contour extraction.
            If set to None, optimal value is computed automatically.

    Returns:
        Contour mask in grayscale format. Array of shape: (h, w).

    """
    if detail_threshold is not None:
        return _contour_mask(img_rgb, hysteresis_params, detail_threshold)[-1]

    # Collect parameter values
    blob_parameter, mask_parameter, contour_masks = [], [], []
    for threshold in AUTO_THRESHOLDS:
        sum_blob, sum_mask, contour_mask = _contour_mask(
            img_rgb, hysteresis_params, threshold
        )
        blob_parameter.append(sum_blob)
        mask_parameter.append(sum_mask * AUTO_MASK_COEFFICIENT)
        contour_masks.append(contour_mask)

    n_test_points = len(AUTO_THRESHOLDS)

    # Find right most intersection of parameter curves
    for i in range(n_test_points - 1, 0, -1):
        if mask_parameter[i] >= blob_parameter[i]:
            cross_index = i
            break

    # Extremum finder
    extremum_diff = np.max(blob_parameter) * AUTO_EXTREMUM_DIFF

    def find_extremum(parameter, index_range, index_correction):
        prev = 0
        for i in index_range:
            # print(prev - parameter[i], extremum_diff)
            if prev - parameter[i] > extremum_diff:
                return i + index_correction
            prev = parameter[i]

    # Find right local maximum
    right_max_index = find_extremum(blob_parameter, range(n_test_points - 1, 0, -1), 1)

    # Find left local maximum
    left_max_index = find_extremum(blob_parameter, range(n_test_points), -1)
    # print(f"cross_index: {cross_index}, right_max_index: {right_max_index}, left_max_index: {left_max_index}")

    # Set operation mode
    if (
        cross_index < left_max_index
        and AUTO_THRESHOLDS[right_max_index] - AUTO_THRESHOLDS[left_max_index] >= AUTO_THRESHOLD_GAP
    ):
        mode = "leftmax->T localmin"
        local_min_range = range(left_max_index + 1, n_test_points)
        index_correction = -1
    elif (
        blob_parameter[right_max_index] > blob_parameter[left_max_index]
        and AUTO_THRESHOLDS[right_max_index] - AUTO_THRESHOLDS[left_max_index] >= AUTO_THRESHOLD_GAP
    ):
        mode = "leftmax->localmin->T<-rightmax"
        local_min_range = range(left_max_index + 1, n_test_points)
        index_correction = -1
    else:
        mode = "localmin->T<-leftmax"
        local_min_range = range(left_max_index - 1, 0, -1)
        index_correction = 1

    # print(mode, left_max_index, right_max_index)
    # plt.figure()
    # plt.plot(AUTO_THRESHOLDS, blob_parameter, label="blob_parameter")
    # plt.plot(AUTO_THRESHOLDS, mask_parameter, label="mask_parameter")
    # plt.legend()
    # plt.show()

    # Find local minimum
    prev = blob_parameter[left_max_index]
    for i in local_min_range:
        if blob_parameter[i] >= prev:
            local_min_index = i + index_correction
            break
        prev = blob_parameter[i]
    else:
        local_min_index = 0

    if mode == "leftmax->T localmin":
        optimal_index = local_min_index
    else:
        target_blob_parameter = (
            blob_parameter[left_max_index] + blob_parameter[local_min_index]
        ) * AUTO_WEIGHTED_MEAN
        if mode == "leftmax->localmin->T<-rightmax":
            search_slice = slice(local_min_index, right_max_index + 1)
        else:
            search_slice = slice(local_min_index, left_max_index + 1)
        optimal_index = (
            np.argmin(
                np.abs(np.array(blob_parameter[search_slice]) - target_blob_parameter)
            )
            + local_min_index
        )

    print("Optimal detail threshold: %g" % AUTO_THRESHOLDS[optimal_index])
    return contour_masks[optimal_index]


# def extract_contour(
#     img: np.ndarray, hysteresis_params: tuple[int, int], detail_threshold: float
# ) -> np.ndarray:
#     """
#     Extract contour and fine details from an artwork.
#
#     Args:
#         img_rgb: Image in RGBA format. Array of shape (h, w, 4).
#         hysteresis_params: Parameters for OpenCV Canny algorithm.
#         detail_threshold: Threshold value for contour art extraction.
#
#     Returns:
#         Contours in RGB format. Array of shape: (h, w, 3).
#
#     """
#     # Alpha blend image with a white background producing an RGB image
#     img_rgb, alpha = np.split(img, [-1], axis=-1)
#     img_rgb = (alpha / 255) * img_rgb + (255 - alpha)
#
#     # Extract contour art mask
#     contour_mask = contour_art_mask(
#         (img_rgb + 0.5).clip(0, 255).astype(np.uint8),
#         hysteresis_params,
#         detail_threshold,
#     )[..., None]
#
#     # Apply contour mask
#     contour = img * (contour_mask // 255)
#     return contour
