import argparse
import os.path

import argcomplete
import cv2
import numpy as np
from PIL import Image

from src.contour_preserving_downscale import preserve_contour
from src.extract_contour import extract_contour
from src.getch import getch

MIN_SCALE_FACTOR = 2.5
MAX_SCALE_FACTOR = 15
OPTIMAL_SCALE_FACTOR = 10

INTER_MODES = {
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
    "bicubic": cv2.INTER_CUBIC,
    "linear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def parse_args():
    def int_pair(s: str) -> tuple[int, int]:
        pair = s.split(",")
        return int(pair[0]), int(pair[1])

    parser = argparse.ArgumentParser(
        description="Betterscale is a detail preserving artwork downscaler."
    )
    parser.add_argument("filename", type=str, help="Path to image.")
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        help="Downscaling factor. Mutually exclusive " "with other size options.",
    )
    parser.add_argument(
        "-d",
        "--dim",
        metavar="SIZE",
        type=int,
        help="Average target image dimension. "
        "Final image will be of area `SIZE^2`. "
        "Mutually exclusive with other size options.",
    )
    parser.add_argument(
        "-w",
        "--width",
        metavar="WIDTH",
        type=int,
        help="Target image width. Must be smaller than "
        "input image width. "
        "Mutually exclusive with other size options.",
    )
    parser.add_argument(
        "--height",
        metavar="HEIGHT",
        type=int,
        help="Target image height. Must be smaller than "
        "input image height. "
        "Mutually exclusive with other size options.",
    )
    parser.add_argument(
        "-f",
        "--dilate_factor",
        metavar="FACTOR",
        type=float,
        default=0.5,
        help="Contour dilation factor (default: %(default)s). "
        "Larger values result in more pronounced contour.",
    )
    parser.add_argument(
        "-p",
        "--precise_contour",
        action="store_true",
        help="Produce a more accurate contour. Decreases sharpness.",
    )
    parser.add_argument(
        "-i",
        "--interpolation",
        metavar="MODE",
        type=str,
        choices=list(INTER_MODES.keys()),
        default="area",
        help="Set interpolation mode (default: %(default)s). "
        "Available modes are: " + ", ".join(list(INTER_MODES.keys())) + ".",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Detail threshold parameter (default: auto).",
    )
    parser.add_argument(
        "--hysteresis",
        metavar="T1,T2",
        type=int_pair,
        default="100,200",
        help="Parameters for OpenCV Canny algorithm. " "(default: %(default)s).",
    )
    parser.add_argument(
        "-n",
        "--no_preserve_contour",
        action="store_true",
        help="Turn off contour dilation for testing purposes. "
        "Results in a simple interpolation with a selected interpolation mode.",
    )
    parser.add_argument(
        "--no_downscale",
        action="store_true",
        help="Turn off downscaling for testing purposes. "
        "An original image with preserved contour will be produced.",
    )

    # Enable tab completion
    argcomplete.autocomplete(parser)

    # Parse the command-line arguments
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        img = Image.open(args.filename).convert(mode="RGBA")
    except FileNotFoundError:
        print("No such file or insufficient permissions, exiting...")
        exit(0)

    if (
        sum(
            [
                args.scale is not None,
                args.dim is not None,
                args.width is not None,
                args.height is not None,
            ]
        )
        > 1
    ):
        print("Cannot set multiple size options at the same time, exiting...")
        exit(0)

    target_width = None
    target_height = None

    if args.scale is not None:
        target_width = int(img.width / args.scale + 0.5)
    if args.dim is not None:
        scale_factor = np.sqrt(img.width * img.height // args.dim**2)
        target_width = int(img.width / scale_factor + 0.5)
    elif args.width is not None:
        target_width = args.width
    elif args.height is not None:
        target_height = args.height

    # Compute dimensions
    if target_height is None:
        target_height = int(img.height * target_width / img.width + 0.5)
    else:
        target_width = int(img.width * target_height / img.height + 0.5)
    scale_factor = img.width / target_width
    print("Scale factor: %.3f" % scale_factor)

    if scale_factor <= 1:
        print("Target width cannot be larger than input width, exiting...")
        exit(0)
    elif scale_factor <= MIN_SCALE_FACTOR:
        print(
            f"Warning! Scale factor below {MIN_SCALE_FACTOR} is not recommended.\n"
            "Results may be identical to simple interpolation."
        )

    img = np.asarray(img)

    # Scale factor is too large, prescale
    if scale_factor >= MAX_SCALE_FACTOR:
        print("Large scale factor, prescaling...")
        img = cv2.resize(
            img,
            (target_width * OPTIMAL_SCALE_FACTOR, target_height * OPTIMAL_SCALE_FACTOR),
            interpolation=cv2.INTER_AREA,
        )

    # Dilate contour
    if not args.no_preserve_contour:
        img = preserve_contour(
            img,
            target_width,
            args.dilate_factor,
            args.precise_contour,
            args.hysteresis,
            args.threshold,
        )

    # Interpolate
    if not args.no_downscale:
        img = cv2.resize(
            img,
            (target_width, target_height),
            interpolation=INTER_MODES[args.interpolation],
        )

    basename = os.path.splitext(args.filename)[0]
    savename = f"{basename}{'' if args.no_downscale else f'_{args.interpolation}'}{'' if args.no_preserve_contour else '_cscale0x'}.png"

    final_img = Image.fromarray(img)
    final_img.save(savename)
    print(f"Success! Result saved to {savename}")


if __name__ == "__main__":
    main()
