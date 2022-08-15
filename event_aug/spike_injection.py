from typing import Tuple

import cv2
import h5py
import numpy as np

from event_aug.utils import video_to_array


def inject_event_spikes(
    event_data_path: str,
    save_path: str,
    spikes_video_path: str = None,
    spikes_arr: np.ndarray = None,
    resize_size: Tuple[int, int] = None,
    crop_size: Tuple[int, int] = None,
    fps: int = None,
    label: int = None,
    polarity: int = None,
    timestamp_keys: Tuple[str] = None,
    xy_keys: Tuple[str] = None,
    label_keys: Tuple[str] = None,
    polarity_keys: Tuple[str] = None,
    verbose=False,
):

    """
    Injects specified event spikes into existing event recordings data to serve as augmentation.

    Parameters
    ----------
    event_data_path : str
        Path to the .h5 file containing the event recordings data.
    save_path : str
        Path to save the augmented event data with spikes injected to.
    spikes_video_path : str
        Path to the video containing the event spikes data as frames.
    spikes_arr : np.ndarray
        Array containing the event spikes data as frames.
    resize_size : Tuple[int, int]
        If specified, the size to reshape the event spikes frames to.
    crop_size : Tuple[int, int]
        If specified, the size to crop the event spikes frames to.
    fps : int
        Frame rate to use for the event spikes to be injected.
    label : int
        Label to assign for the event spikes to be injected.
    polarity : int
        Polarity to assign for the event spikes to be injected.
    timestamp_keys : Tuple[str]
        Keys to use to index the event timestamp data.
    xy_keys : Tuple[str]
        Keys to use to index the event xy coordinates data.
    label_keys : Tuple[str]
        Keys to use to index the event label data.
    polarity_keys : Tuple[str]
        Keys to use to index the event polarity data.
    verbose : bool
        Whether to print progress messages.
    """

    assert event_data_path.endswith(".h5"), "Event data path must poin to a .h5 file"

    with h5py.File(event_data_path, "r") as f:

        keys = list(f.keys())

        if timestamp_keys is None:
            timestamp_keys = [key for key in keys if key.startswith("timestamp")]
            assert len(timestamp_keys) != 0, "No timestamp data found in event data"

        if xy_keys is None:
            xy_keys = [key for key in keys if key.startswith("xy")]
            assert len(xy_keys) != 0, "No xy coordinate data found in event data"

        if label_keys is None:
            label_keys = [key for key in keys if key.startswith("label")]
            assert len(label_keys) != 0, "No label data found in event data"

        if polarity_keys is None:
            polarity_keys = [key for key in keys if key.startswith("polarity")]
            assert len(polarity_keys) != 0, "No polarity data found in event data"

        timestamps = {
            timestamp_key: f[timestamp_key][:].copy() for timestamp_key in timestamp_keys
        }
        xy_coords = {xy_key: f[xy_key][:].copy() for xy_key in xy_keys}
        labels = {label_key: f[label_key][:].copy() for label_key in label_keys}
        polarities = {
            polarity_key: f[polarity_key][:].copy() for polarity_key in polarity_keys
        }

    if label is None:
        label = 0
        print("\nNo label provided. Using label '0' for all event spikes to be injected")

    if polarity is None:
        polarity = 0
        print(
            "\nNo polarity provided. Using polarity '0' for all event spikes to be"
            " injected\n"
        )

    assert (spikes_video_path is not None) or (
        spikes_arr is not None
    ), "Either spikes_video_path or spikes_arr must be specified"

    if spikes_video_path is not None:
        assert spikes_video_path.endswith(
            ".mp4"
        ), "Spikes video path must point to a .mp4 file"
        spikes_arr, vid_fps = video_to_array(
            spikes_video_path, grayscale=True, return_fps=True
        )

        if fps is None:
            fps = vid_fps
            print(
                "No frame rate provided for injection. Using the video's frame rate:"
                f" {fps}"
            )

    else:
        assert spikes_arr.ndim == 3, (
            "Spikes array must of shape (n_frames, height, width), where each 2D frame"
            " contains the event spikes data as 0s or 1s"
        )
        assert (
            fps is not None
        ), "fps must be specified if a spikes array is given as input"

    total_events_injected = 0

    for n_frame in range(spikes_arr.shape[0]):

        if verbose is True:
            print(f"\nProcessing frame {n_frame} of the event spikes video/array")

        timestep = n_frame / fps
        timestep = round(timestep * 1e6)

        spikes_frame = spikes_arr[n_frame]

        if (np.unique(spikes_frame) > 1).any():
            spikes_frame = np.round(spikes_frame / 255).astype(np.uint8)

        if resize_size is not None:
            spikes_frame = cv2.resize(spikes_frame, resize_size)

        if crop_size is not None:
            frame_size = spikes_frame.shape
            assert (
                frame_size[0] > crop_size[0] and frame_size[1] > crop_size[1]
            ), "Crop size must be smaller than the frame size"

            start_x = frame_size[1] // 2 - crop_size[1] // 2
            start_y = frame_size[0] // 2 - crop_size[0] // 2

            spikes_frame = spikes_frame[
                start_y : start_y + crop_size[0], start_x : start_x + crop_size[1]
            ]

        spike_coords = np.argwhere(spikes_frame == 1)
        total_events_injected += len(spike_coords)

        if verbose is True:
            print(
                f"Injecting event spikes found at {len(spike_coords)} locations in the"
                " frame"
            )

        insert_ids = {key: None for key in timestamps.keys()}

        for timestamp_key in timestamps.keys():
            insert_ids[timestamp_key] = np.searchsorted(
                timestamps[timestamp_key], timestep
            )

        for (
            timestamp_key,
            xy_key,
            label_key,
            polarity_key,
        ) in zip(timestamps.keys(), xy_coords.keys(), labels.keys(), polarities.keys()):

            insertion_timesteps = [timestep for _ in range(len(spike_coords))]
            timestamps[timestamp_key] = np.insert(
                timestamps[timestamp_key], insert_ids[timestamp_key], insertion_timesteps
            )

            xy_coords[xy_key] = np.insert(
                xy_coords[xy_key], insert_ids[timestamp_key], spike_coords, axis=0
            )

            insertion_labels = [label for _ in range(len(spike_coords))]
            labels[label_key] = np.insert(
                labels[label_key], insert_ids[timestamp_key], insertion_labels
            )

            insertion_polarities = [polarity for _ in range(len(spike_coords))]
            polarities[polarity_key] = np.insert(
                polarities[polarity_key], insert_ids[timestamp_key], insertion_polarities
            )

    print(f"\nInjected {total_events_injected} events into the event data\n")

    if verbose is True:
        print(f"Saving event data with specified event spikes injected to {save_path}\n")

    with h5py.File(save_path, "w") as f:
        for timestamp_key, xy_key, label_key, polarity_key in zip(
            timestamps.keys(), xy_coords.keys(), labels.keys(), polarities.keys()
        ):
            f.create_dataset(timestamp_key, data=timestamps[timestamp_key])
            f.create_dataset(xy_key, data=xy_coords[xy_key])
            f.create_dataset(label_key, data=labels[label_key])
            f.create_dataset(polarity_key, data=polarities[polarity_key])
