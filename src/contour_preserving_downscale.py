from typing import Optional

import cv2
import numpy as np

from .extract_contour import extract_contour
from .utils import padded_view_as_windows

RGB_INTENSITIES = np.array([0.299, 0.587, 0.114])


def preserve_contour(
    img: np.ndarray,
    target_width: int,
    dilate_factor: float = 0.5,
    precise_contour: bool = False,
    hysteresis_params: tuple[int, int] = (100, 200),
    detail_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Preserve subpixel details before artwork downscaling.

    Algorithm steps:
    - Edges are extracted from the image, filtered with
      a custom kernel and thresholded to extract contour
      and fine details.
    - Contours are dilated to make the contour `scale_factor`
      pixels wide in the original image.
    - Original image is fitted with dilated contour.

    Args:
        img: Image in RGBA format. Array of shape: (h, w, 4).
        target_width: Target image width.
        dilate_factor: TODO
        precise_contour: TODO
        hysteresis_params: Parameters for OpenCV Canny algorithm.
        detail_threshold: Threshold value for contour extraction.
            If set to None, an optimal value is computed automatically.

    Returns:
        Downscaled image in RGBA format. Array of shape (h / scale_factor, w / scale_factor, 4).

    """
    # Compute scale factor
    scale_factor = int(img.shape[1] / target_width + 0.5)

    # Alpha blend image with a white background producing an RGB image
    img_rgb, alpha = np.split(img, [-1], axis=-1)
    img_rgb = (alpha / 255) * img_rgb + (255 - alpha)

    # Extract contour mask
    contour_mask = extract_contour(
        (img_rgb + 0.5).clip(0, 255).astype(np.uint8),
        hysteresis_params,
        detail_threshold,
    )

    # Apply contour mask
    contour = img * (contour_mask[..., None] // 255)
    contour_rgb = img_rgb * (contour_mask[..., None] // 255) + (
        255 - contour_mask[..., None]
    )

    # Create windows view into nearest neighbors for each pixel
    dilate_size = max(int(scale_factor * dilate_factor + 0.5), 1)
    # dilate_size += (dilate_size + 1) % 2
    contour_windows = padded_view_as_windows(contour, (dilate_size, dilate_size))

    # Same for contour converted to grayscale
    contour_mono = np.sqrt(contour_rgb**2 @ RGB_INTENSITIES)
    contour_windows_mono = padded_view_as_windows(
        contour_mono, (dilate_size, dilate_size), value=255
    )

    # Same for contour mask itself
    contour_mask_windows = padded_view_as_windows(
        contour_mask, (dilate_size, dilate_size)
    )

    # Find index of the darkest pixel among nearest neighbors
    disk_kernel = 1 - cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
    ).reshape(-1)
    index_min = np.argmin(contour_windows_mono + 255 * disk_kernel, axis=-1)
    j, k = np.indices(index_min.shape)

    # Dilate contour with pixels from these indices
    dilated_contour = contour_windows[j, k, index_min, :]
    if precise_contour:
        dilated_contour = contour + dilated_contour * (
            1 - contour_mask[..., None] // 255
        )

    # Insert dilated contour in the original image
    dilated_contour_mask = contour_mask_windows[j, k, index_min, None]
    contour_img = img * (1 - dilated_contour_mask // 255) + dilated_contour * (
        dilated_contour_mask // 255
    )

    # kernel = cv2.getGaussianKernel(dilate_size, 0)
    # kernel = np.outer(kernel, kernel)
    # sigma = 0.3 * ((dilate_size - 1) * 0.5 - 1) + 0.8
    # # print(kernel)
    # # dilated_contour = (contour_windows * kernel.reshape(1, 1, -1, 1)).sum(axis=2)
    # dilated_contour = cv2.GaussianBlur(contour.astype(float), (dilate_size, dilate_size), 0)
    # contour_neighbors = np.sum(contour_mask_windows * kernel.reshape(1, 1, -1), axis=2, keepdims=True)
    # dilated_contour_mask = (contour_neighbors >= kernel[int((dilate_size - 1) // 2 - sigma + 0.5), 0] * 255)
    # contour_neighbors += ~dilated_contour_mask
    # dilated_contour = dilated_contour / contour_neighbors * 255

    # contourd_img = (img * (1 - dilated_contour_mask) + dilated_contour * (
    #     dilated_contour_mask
    # ) + 0.5).clip(0, 255).astype(np.uint8)

    return contour_img
