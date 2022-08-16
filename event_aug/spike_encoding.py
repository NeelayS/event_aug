from typing import Union

import cv2
import numpy as np
import torch

from event_aug.utils import array_to_video


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

        new_data = torch.clone(data).unsqueeze(0)
        data = data.unsqueeze(0)

        for _ in range(n_steps - 1):
            new_data = torch.cat((new_data, data))
        data = new_data

    spikes = torch.bernoulli(data)

    return spikes.numpy()


def delta_intensity_code_arr(
    arr: Union[np.ndarray, torch.Tensor],
    threshold: int = 15,
    percent_threshold: int = 10,
    mode: str = "threshold",
    use_neg_delta: bool = True,
    exclude_start: bool = False,
    return_arr: bool = True,
    save_arr: bool = False,
    arr_save_path: str = None,
    save_video: bool = False,
    video_save_path: str = None,
    video_fps: int = 25,
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
        for assigning events.
    percent_threshold: int
        Pixel-wise percentage threshold for the difference in intensities of pixels in consecutive frames
        for assigning events.
    mode: str
        Whether to use 'threshold' or 'percent_threshold'".
    use_neg_delta: bool
        Whether to consider decreases in intensity as well along with increases for assigning events.
    exclude_start: bool
        Whether to not return the spikes for the first frame which will always be 0 for all pixels.
    return_arr: bool
        Whether to return the spikes array.
    save_arr: bool
        Whether to save the spikes array to a .npy file.
    arr_save_path: str
        Name of the .npy file to save the spikes array to (excluding extension).
    save_video: bool
        Whether to save the spikes as a video (.mp4).
    video_save_path: str
        Name of the .mp4 file to save the spikes video to.
    video_fps: int
        Frames per second of the video.

    Returns
    -------
    np.ndarray or torch.Tensor
        Array containing the spikes.
    """

    assert (
        len(arr.shape) == 3 or len(arr.shape) == 4
    ), "Input must be an array with shape (T x H x W) or (T x H x W x C)"

    assert (
        threshold is not None or percent_threshold is not None
    ), "Either threshold or percent_threshold must be specified"

    mode = mode.lower()
    assert mode in [
        "threshold",
        "percent_threshold",
    ], "Mode must be either 'threshold' or 'percent_threshold'"

    if save_arr is True:
        assert (
            arr_save_path is not None
        ), "save_path must be specified if save_arr is True"
        assert arr_save_path.endswith(
            ".npy"
        ), "arr_save_path must be without and extension or end with .npy"

    if save_video is True:
        assert (
            video_save_path is not None
        ), "video_save_path must be specified if save_video is True"
        assert video_save_path.endswith(
            ".mp4"
        ), "video_save_path must be without and extension or end with .mp4"

    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
        torch_inp_arr = True
    else:
        torch_inp_arr = False

    multi_channel = len(arr.shape) == 4

    T = arr.shape[0]
    H, W = arr.shape[1], arr.shape[2]

    spikes = np.zeros((T, H, W), dtype="bool")

    for i in range(1, arr.shape[0]):

        prev_frame, curr_frame = np.float32(arr[i - 1]), np.float32(
            arr[i]
        )  # np.int16(arr[i - 1]), np.int16(arr[i])

        if multi_channel:
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

        intensity_delta = curr_frame - prev_frame
        if use_neg_delta:
            intensity_delta = abs(intensity_delta)

        if mode == "threshold":
            spikes[i][intensity_delta > threshold] = 1
        else:
            percent_frame_delta = prev_frame * percent_threshold / 100
            spikes[i][intensity_delta > percent_frame_delta] = 1

    if exclude_start:
        return spikes[1:]

    if save_arr is True:
        np.save(arr_save_path, spikes)

    if save_video is True:
        array_to_video(spikes.astype(np.uint8) * 255, video_save_path, fps=video_fps)

    if return_arr is True:
        if torch_inp_arr is True:
            return torch.from_numpy(spikes)

        return spikes


def delta_intensity_code_file(
    video_path: str,
    threshold: int = 15,
    percent_threshold: int = 10,
    mode: str = "threshold",
    out_fps: int = None,
    use_neg_delta: bool = True,
    exclude_start: bool = False,
    return_arr: bool = False,
    save_video: bool = True,
    video_save_path: str = None,
    save_arr: bool = False,
    arr_save_path: str = None,
) -> None:

    """
    Reads a video file, convert the video to spiking form and saves to a file. If the difference in the intensity of a pixel in
    consecutive frames is greater than the threshold, the pixel is assigned an event.

    Parameters
    ----------
    video_path: str
        Path to the input video file.
    threshold: int
        Threshold for the difference in intensities of pixels in consecutive frames
        for assigning events.
    percent_threshold: int
        Pixel-wise percentage threshold for the difference in intensities of pixels in consecutive frames
        for assigning events.
    mode: str
        Whether to use 'threshold' or 'percent_threshold'".
    out_fps: int
        Output video frame rate.
    use_neg_delta: bool
        Whether to consider decreases in intensity as well along with increases for assigning events.
    exclude_start: bool
        Whether to not return the spikes for the first frame which will always be 1 for all pixels.
    return_arr: bool
        Whether to return an array containing frame-wise spikes.
    save_video: bool
        Whether to save the frame-wise spikes as a video (.mp4).
    video_save_path: str
        Name of the .mp4 file to save the spikes video to.
    save_arr: bool
        Whether to save the spikes array to a .npy file.
    arr_save_path: str
        Name of the .npy file to save the spikes array to (excluding extension).
    """

    if save_video is True:
        assert (
            video_save_path is not None
        ), "video_save_path must be specified if save_video is True"
        assert video_save_path.endswith(".mp4"), "Output video file must be .mp4"

    if save_arr is True:
        assert (
            arr_save_path is not None
        ), "save_path must be specified if save_arr is True"
        assert arr_save_path.endswith(".npy"), "Output spikes file must be .npy"

    assert (
        threshold is not None or percent_threshold is not None
    ), "Either threshold or percent_threshold must be specified"

    mode = mode.lower()
    assert mode in [
        "threshold",
        "percent_threshold",
    ], "Mode must be either 'threshold' or 'percent_threshold'"

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

    if return_arr is True or save_arr is True:
        spikes_arr = np.zeros((n_frames, H, W), dtype="bool")

    if save_video is True:
        out_vid = cv2.VideoWriter(
            video_save_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H)
        )

    prev_frame = np.ones((H, W), dtype=np.int16) * 255

    i = 0
    while True:

        ret, frame = vid.read()
        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.int16)  # frame.astype(np.float32)

        if exclude_start and i == 0:
            prev_frame = frame
            i += 1
            continue

        delta = frame - prev_frame
        prev_frame = frame

        if use_neg_delta is True:
            delta = abs(delta)

        if mode == "threshold":
            delta[delta > threshold] = 255
            delta[delta <= threshold] = 0
        else:
            percent_frame_delta = prev_frame * percent_threshold / 100
            delta[delta > percent_frame_delta] = 255
            delta[delta <= percent_frame_delta] = 0

        delta = delta.astype(np.uint8)

        if return_arr is True or save_arr is True:
            spikes_arr[i] = (delta == 255).astype("bool")

        if save_video is True:
            delta = cv2.cvtColor(delta, cv2.COLOR_GRAY2BGR)
            out_vid.write(delta)

        i += 1

    vid.release()

    if save_video is True:
        out_vid.release()

    if save_arr is True:
        np.save(arr_save_path, spikes_arr)

    if return_arr is True:
        return spikes_arr


def delta_intensity_code_video(
    video_path: str = None,
    video_arr: Union[np.ndarray, torch.Tensor] = None,
    threshold: int = 15,
    percent_threshold: int = 10,
    mode: str = "threshold",
    use_neg_delta: bool = True,
    out_fps: int = None,
    exclude_start: bool = True,
    return_arr: bool = False,
    save_video: bool = True,
    video_save_path: str = None,
    save_arr: bool = False,
    arr_save_path: str = None,
) -> Union[np.ndarray, torch.Tensor, None]:

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
    percent_threshold: int
        Pixel-wise percentage threshold for the difference in intensities of pixels in consecutive frames
        for assigning events.
    mode: str
        Whether to use 'threshold' or 'percent_threshold'".
    use_neg_delta: bool
        Whether to consider decreases in intensity as well along with increases for assigning events.
    out_fps: int
        Output video frame rate (required if input video is a file).
    exclude_start: bool
        Whether to not return the spikes for the first frame which will always be 0 for all pixels (required if input video is a file).
    return_arr: bool
        Whether to return an array containing frame-wise spikes.
    save_video: bool
        Whether to save the frame-wise spikes as a video (.mp4).
    video_save_path: str
        Name of the .mp4 file to save the spikes video to.
    save_arr: bool
        Whether to save the spikes array to a .npy file.
    arr_save_path: str
        Name of the .npy file to save the spikes array to (excluding extension).
    """

    assert (
        video_path is not None or video_arr is not None
    ), "Either video_path or video_arr must be provided"

    if video_path is not None:

        spikes = delta_intensity_code_file(
            video_path=video_path,
            threshold=threshold,
            percent_threshold=percent_threshold,
            mode=mode,
            out_fps=out_fps,
            use_neg_delta=use_neg_delta,
            exclude_start=exclude_start,
            return_arr=return_arr,
            save_video=save_video,
            video_save_path=video_save_path,
            save_arr=save_arr,
            arr_save_path=arr_save_path,
        )
        return spikes

    else:
        spikes = delta_intensity_code_arr(
            arr=video_arr,
            threshold=threshold,
            percent_threshold=percent_threshold,
            mode=mode,
            use_neg_delta=use_neg_delta,
            exclude_start=exclude_start,
            return_arr=return_arr,
            save_arr=save_arr,
            arr_save_path=arr_save_path,
            save_video=save_video,
            video_save_path=video_save_path,
            video_fps=out_fps,
        )
        return spikes
