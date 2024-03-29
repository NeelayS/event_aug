import os
import re
import subprocess
from typing import Tuple, Union

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
        Array containing the image data. If it's a list, it should contain as many images as number of frames wanted.
        If it's a multi-dimensional array, it should be of shape (T x H x W) or (T x H x W x C).
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

        if numbered_imgs is True:
            img_list = sorted(
                img_list, key=lambda x: int(re.findall(r"\d+", x)[0])
            )  # if len(re.findall(r'\d+', x)) > 0 else 0)

        n_images = len(img_list)

        if height is None or width is None:
            sample_img = cv2.imread(os.path.join(img_dir, img_list[0]))
            height, width = sample_img.shape[:2]

    else:
        n_images = img_arr.shape[0]
        height, width = img_arr[0].shape[:2]
        is_grayscale = len(img_arr[0].shape) == 2

    out_vid = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
    )
    print(f"Frame width: {width}, Frame height: {height}, FPS: {out_fps}")

    for i in range(n_images):

        if img_arr is not None:
            img = img_arr[i]

            if is_grayscale:
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        else:
            img_path = os.path.join(img_dir, img_list[i])

            img = cv2.imread(img_path)

            if verbose is True:
                print(f"Writing {img_path}")

        img = img.astype(np.uint8)
        out_vid.write(img)

    out_vid.release()


def array_to_video(arr: np.ndarray, save_path: str, fps: int):

    """
    Converts an array of image data to a video.

    Parameters
    ----------
    arr : np.ndarray
        Array containing the image data.
    save_path : str
        Path to the output video file.
    fps : int
        Output video frame rate.
    """

    imgs_to_video(save_path=save_path, img_arr=arr, out_fps=fps)


def video_to_array(
    video_path: str, grayscale: bool = False, return_fps: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:

    """
    Converts a video to an array of frames data.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    grayscale : bool
        Whether to convert the frames to grayscale.
    return_fps : bool
        Whether to return the video frame rate.

    Returns
    -------
    arr : np.ndarray
        Array of frames data.
    fps : int
        Video frame rate.
    """

    vid = cv2.VideoCapture(video_path)

    W, H = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(vid.get(cv2.CAP_PROP_FPS))

    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"No. of frames: {n_frames}, Frame width: {W}, Frame height: {H}")

    if grayscale:
        arr = np.zeros((n_frames, H, W), dtype=np.uint8)
    else:
        arr = np.zeros((n_frames, H, W, 3), dtype=np.uint8)

    i = 0
    while True:

        ret, frame = vid.read()

        if ret is False:
            break

        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = frame.astype(np.uint8)
        arr[i] = frame
        i += 1

    vid.release()

    if return_fps:
        return arr, fps

    return arr


def save_video_frames_diffs(
    video_path: str, save_path: str, out_fps: int = None, neg_diff: bool = True
):
    """
    Saves the differences in intensities between video frames as a video.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    save_path : str
        Path to save the output video.
    out_fps : int
        Output video frame rate.
    neg_diff : bool
        Whether to consider negative differences.
    """

    """
    Saves the pixelwise differences between consecutive frames of a video to a video.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    save_path : str
        Path to the output video file.
    out_fps : int
        Output video frame rate.
    neg_diff : bool
        Whether to consider negative differences.
    """

    vid = cv2.VideoCapture(video_path)

    W, H = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(vid.get(cv2.CAP_PROP_FPS))
    if out_fps is None:
        out_fps = fps

    out_vid = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

    prev_frame = np.ones((H, W)) * 255

    i = 0
    while True:

        ret, frame = vid.read()

        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32)

        delta = frame - prev_frame
        prev_frame = frame

        if neg_diff is True:
            delta = abs(delta)
        else:
            delta = np.maximum(delta, 0)

        delta = delta * 255 / np.max(delta)
        delta = np.uint8(delta)

        delta = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
        out_vid.write(delta)

        i += 1

    vid.release()
    out_vid.release()


def resize_video(video_path: str, save_path: str, size: Tuple[int]):
    """
    Resizes a video to a given size.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    save_path : str
        Path to the output video.
    size : Tuple[int]
        Size to resize the video to.
    """

    assert type(video_path) == str and video_path.endswith(
        ".mp4"
    ), "Input video must be in mp4 format"
    assert type(save_path) == str and save_path.endswith(
        ".mp4"
    ), "Output video must be in mp4 format"
    assert type(size) == tuple and len(size) == 2, "Size must be a tuple of length 2"

    vid = cv2.VideoCapture(video_path)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    out_vid = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    W, H = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original frame width: {W}, Original frame height: {H}")
    print(f"Resizing to {size[0]}x{size[1]}")

    while True:

        ret, frame = vid.read()

        if ret is False:
            break

        frame = cv2.resize(frame, size)
        out_vid.write(frame)

    vid.release()
    out_vid.release()


def download_from_youtube(
    urls: Union[Tuple[str], str],
    start_times: Union[Tuple[int], str],
    end_times: Union[Tuple[int], str],
    save_dir: str,
):
    """
    Downloads videos from YouTube.

    Parameters
    ----------
    urls : Union[Tuple[str], str]
        YouTube URLs.
    start_times : Union[Tuple[int], str]
        Start times to trim the videos.
    end_times : Union[Tuple[int], str]
        End times to trim the videos.
    save_dir : str
        Directory to save the videos to.
    """

    if isinstance(urls, str):
        urls = (urls,)

    if isinstance(start_times, int):
        start_times = (start_times,)
        start_times = [int(time) for time in start_times]

    if isinstance(end_times, int):
        end_times = (end_times,)
        end_times = [int(time) for time in end_times]

    assert (
        len(urls) == len(start_times) == len(end_times)
    ), "Length of urls, start_times and end_times must be equal"

    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(urls)):

        print(f"Downloading video {i + 1} of {len(urls)}")

        try:
            duration = end_times[i] - start_times[i]
            command = (
                f"ffmpeg $(youtube-dl -g '{urls[i]}' | sed 's/.*/-ss {start_times[i]} -i"
                f" &/') -t {duration} -c copy {save_dir}/{i}.mp4"
            )
            subprocess.call(command, shell=True)

        except:
            raise Exception(f"Error downloading video {i + 1} of {len(urls)}")
