import os

import cv2
import numpy as np


def imgs_to_video(
    save_path: str,
    img_dir: str = None,
    img_arr: np.ndarray = None,
    height: int = None,
    width: int = None,
    out_fps: int = 15,
    numbered_imgs: bool = False,
    verbose: bool = False,
):

    """
    Converts a directory of images or an array / list of image data to a video.

    Parameters
    ----------
    save_path : str
        Path (.mp4) to the output video file.
    img_dir : str
        Path to the directory containing the images.
    imgs_arr : np.ndarray or list
        Array containing the image data. If multi-dimensional array, it should be of shape (T x H x W) or (T x H x W x C).
    height : int
        Height of the images.
    width : int
        Width of the images.
    out_fps : int
        Output video frame rate.
    numbered_imgs : bool
        Whether the image file names are a number sequence.
    verbose : bool
        Whether to print progress.
    """

    assert (
        img_dir is not None or img_arr is not None
    ), "Either img_dir or img_arr must be provided"

    if img_dir is not None:

        img_list = sorted(os.listdir(img_dir))

        if height is None or width is None:
            sample_img = cv2.imread(os.path.join(img_dir, img_list[0]))
            height, width = sample_img.shape[:2]

        extension = img_list[0][-4:]

    else:
        height, width = img_arr[0].shape[:2]
        is_grayscale = len(img_arr[0].shape) == 2

    out_vid = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
    )

    for i, name in sorted(enumerate(os.listdir(img_dir))):

        if img_arr is not None:
            img = img_arr[i]

            if is_grayscale:
                img = img.astype(np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        else:
            if numbered_imgs:
                img_path = os.path.join(img_dir, str(i) + extension)
            else:
                img_path = os.path.join(img_dir, name)

            img = cv2.imread(img_path)

            if verbose is True:
                print(f"Writing {img_path}")

        img = img.astype(np.uint8)
        out_vid.write(img)

    out_vid.release()
