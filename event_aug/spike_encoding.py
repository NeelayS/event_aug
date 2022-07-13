from typing import Union

import cv2
import numpy as np
import torch


def rate_code(
    data: Union[np.ndarray, torch.Tensor], n_steps: int = 1, gain: int = 1
) -> np.ndarray:

    """
    Converts array of continuous input data to spikes using rate coding.
    Input array is normalized to [0, 1] and the resulting elements are treated as
    probabilities for Bernoulli trials.

    Parameters
    ----------
    data: np.ndarray or torch.Tensor
        2D array of continuous input data. 3D array in case of time-series data (shape: T x H x W).
    n_steps: int
        Number of time-steps to perform rate coding for. Only used for 2D input data.
        In case of time series data, rate coding is performed for each time-step.
    gain: int
        Factor by which to scale normalized input features.

    Returns
    -------
    np.ndarray
        2D array of spikes. 3D array in case of time-series data.
    """

    data = data / np.linalg.norm(data)
    data = data * gain

    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()

    assert len(data.shape) == 2 or len(data.shape) == 3, (
        "Input must be a 2D array of probabilities or a 3D array in case of time series"
        " data"
    )

    if len(data.shape) == 2 and n_steps > 1:
        new_data = torch.clone(data)
        for _ in range(n_steps - 1):
            new_data = torch.stack((new_data, data))
        data = new_data

    spikes = torch.bernoulli(data)

    return spikes.numpy()


def delta_intensity_code_arr(
    arr: Union[np.ndarray, torch.Tensor],
    threshold: int,
    use_neg_delta: bool = True,
    exclude_start: bool = False,
) -> Union[np.ndarray, torch.Tensor]:

    """
    Converts a video data array to spikes. If the difference in the intensity of a pixel in
    consecutive frames is greater than the threshold, the pixel is assigned an event.

    Parameters
    ----------
    arr: np.ndarray or torch.Tensor
        Array containing the video data (frames). Should be of shape (T x H x W) or (T x H x W x C).
    threshold: int
        Threshold for the difference in intensities of pixels in consecutive frames
        for assigning an event.
    use_neg_delta: bool
        Whether to consider decreases in intensity as well along with increases for assigning events.
    exclude_start: bool
        Whether to not return the spikes for the first frame which will always be 0 for all pixels.

    Returns
    -------
    np.ndarray or torch.Tensor
        Array containing the spikes.
    """

    assert (
        len(arr.shape) == 3 or len(arr.shape) == 4
    ), "Input must be an array with shape (T x H x W) or (T x H x W x C)"

    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
        torch_inp_arr = True
    else:
        torch_inp_arr = False

    multi_channel = len(arr.shape) == 4

    T = arr.shape[0]
    H, W = arr.shape[1], arr.shape[2]

    spikes = np.zeros((T, H, W))

    for i in range(1, arr.shape[0]):

        prev_frame, curr_frame = np.float32(arr[i - 1]), np.float32(arr[i])

        if multi_channel:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

        intensity_delta = curr_frame - prev_frame
        if use_neg_delta:
            intensity_delta = abs(intensity_delta)

        spikes[i][intensity_delta > threshold] = 1

    if exclude_start:
        return spikes[1:]

    if torch_inp_arr is True:
        return torch.from_numpy(spikes)

    return spikes


def delta_intensity_code_file(
    video_path: str,
    save_path: str,
    threshold: int = 25,
    out_fps: int = None,
    use_neg_delta: bool = False,
) -> None:

    """
    Reads a video file, convert the video to spiking form and saves to a file. If the difference in the intensity of a pixel in
    consecutive frames is greater than the threshold, the pixel is assigned an event.

    Parameters
    ----------
    video_path: str
        Path to the input video file.
    save_path: str
        Path to save the output video file to.
    threshold: int
        Threshold for the difference in intensities of pixels in consecutive frames
        for assigning events.
    out_fps: int
        Output video frame rate.
    use_neg_delta: bool
        Whether to consider decreases in intensity as well along with increases for assigning events.

    """

    vid = cv2.VideoCapture(video_path)

    W, H = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(vid.get(cv2.CAP_PROP_FPS))
    if out_fps is None:
        out_fps = fps

    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"Frame width: {W}, Frame height: {H}, Input FPS: {fps}, Output FPS: {out_fps},"
        f" Number of frames: {n_frames}"
    )

    out_vid = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    prev_frame = np.ones((H, W)) * 255

    while True:

        ret, frame = vid.read()
        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32)

        delta = frame - prev_frame
        prev_frame = frame

        if use_neg_delta is True:
            delta = abs(delta)

        delta[delta > threshold] = 255
        delta[delta <= threshold] = 0

        delta = delta.astype(np.uint8)
        delta = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)

        out_vid.write(delta)

    vid.release()
    out_vid.release()


def delta_intensity_code_video(
    video_path: str = None,
    video_arr: Union[np.ndarray, torch.Tensor] = None,
    threshold: int = 25,
    use_neg_delta: bool = False,
    save_path: str = None,
    out_fps: int = None,
    exclude_start: bool = False,
) -> Union[np.ndarray, None]:

    """
    Converts a video to spiking form. If the difference in the intensity of a pixel in
    consecutive frames is greater than the threshold, the pixel is assigned an event.
    Works with either a video file or a video array.

    Parameters
    ----------
    video_path: str
        Path to the input video file.
    video_arr: np.ndarray or torch.Tensor
        Array containing the video data (frames). Should be of shape (T x H x W) or (T x H x W x C).
    threshold: int
        Threshold for the difference in intensities of pixels in consecutive frames
        for assigning events.
    use_neg_delta: bool
        Whether to consider decreases in intensity as well along with increases for assigning events.
    save_path: str
        Path to save the output video file to (required if input video is a file).
    out_fps: int
        Output video frame rate (required if input video is a file).
    exclude_start: bool
        Whether to not return the spikes for the first frame which will always be 0 for all pixels (required if input video is a file).
    """

    assert (
        video_path is not None or video_arr is not None
    ), "Either video_path or video_arr must be provided"

    if video_path is not None:
        assert (
            save_path is not None
        ), "save_path must be provided if video_path is provided"

        delta_intensity_code_file(
            video_path, save_path, threshold, out_fps, use_neg_delta
        )

    else:
        spikes = delta_intensity_code_arr(
            video_arr, threshold, use_neg_delta, exclude_start
        )
        return spikes
