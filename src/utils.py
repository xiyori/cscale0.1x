import cv2
import numpy as np
from skimage.util.shape import view_as_windows


def padded_view_as_windows(
    img: np.ndarray, kernel_size: tuple[int, int], value: int = 0
) -> np.ndarray:
    """
    View as windows to perform convolution or other operations.
    Pad the image beforehand so that viewed dimensions don't
    change.

    Args:
        img: Image array of shape (h, w, *).
        kernel_size: Kernel size (kw, kh).
        value: Fill value for padding. Default 0.

    Returns:
        Array of shape (h, w, kw * kh, *).

    """
    kernel_size = (*kernel_size, *img.shape[2:])
    final_shape = (*img.shape[:2], -1, *img.shape[2:])

    # Pad image
    img_padded = cv2.copyMakeBorder(
        img,
        kernel_size[1] // 2,
        kernel_size[1] // 2 - ((kernel_size[1] % 2) == 0),
        kernel_size[0] // 2,
        kernel_size[0] // 2 - ((kernel_size[0] % 2) == 0),
        cv2.BORDER_CONSTANT,
        value,
    )
    return view_as_windows(img_padded, kernel_size).reshape(final_shape)
