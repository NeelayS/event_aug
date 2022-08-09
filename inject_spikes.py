import h5py
import numpy as np

from event_aug.utils import video_to_array


def inject_event_spikes(
    event_data_path: str,
    save_path: str,
    spikes_video_path: str = None,
    spikes_arr: np.ndarray = None,
    fps: int = None,
    label: int = None,
    polarity: int = None,
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
    fps : int
        Frame rate to use for the event spikes to be injected.
    label : int
        Label to assign for the event spikes to be injected.
    polarity : int
        Polarity to assign for the event spikes to be injected.
    verbose : bool
        Whether to print progress messages.
    """

    assert event_data_path.endswith(".h5"), "Event data path must poin to a .h5 file"

    with h5py.File(event_data_path, "r") as f:

        keys = list(f.keys())

        timestamp_keys = [key for key in keys if key.startswith("timestamp")]
        xy_keys = [key for key in keys if key.startswith("xy")]
        label_keys = [key for key in keys if key.startswith("label")]
        polarity_keys = [key for key in keys if key.startswith("polarity")]

        assert len(timestamp_keys) != 0, "No timestamp data found in event data"
        assert len(xy_keys) != 0, "No xy coordinate data found in event data"
        assert len(label_keys) != 0, "No label data found in event data"
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
            print(
                "No label provided. Using label '0' for all event spikes to be injected"
            )

        if polarity is None:
            polarity = 0
            print(
                "No polarity provided. Using polarity '0' for all event spikes to be"
                " injected"
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
        assert (
            fps is not None
        ), "fps must be specified if a spikes array is given as input"

    if (np.unique(spikes_arr) != 0).any() or (np.unique(spikes_arr) != 1).any():
        spikes = np.round(spikes_arr / 255).astype(np.uint8)

    total_events_injected = 0

    for n_frame in range(spikes.shape[0]):

        if verbose is True:
            print(f"Processing frame {n_frame} of the event spikes video/array")

        timestep = n_frame / fps
        timestep = round(timestep * 1e6)

        spike_coords = np.argwhere(spikes[n_frame] == 1)
        total_events_injected += len(spike_coords)

        if verbose is True:
            print(
                f"Injecting event spikes found at {len(spike_coords)} locations in the"
                " frame\n"
            )

        for coord in spike_coords:

            for timestamp_key, xy_key, label_key, polarity_key, in zip(
                timestamps.keys(), xy_coords.keys(), labels.keys(), polarities.keys()
            ):

                insert_idx = np.where(timestamps[timestamp_key] <= timestep)[0][-1] + 1

                timestamps[timestamp_key] = np.insert(
                    timestamps[timestamp_key], insert_idx, timestep
                )
                xy_coords[xy_key] = np.insert(
                    xy_coords[xy_key], insert_idx, coord, axis=0
                )
                labels[label_key] = np.insert(labels[label_key], insert_idx, label)
                polarities[polarity_key] = np.insert(
                    polarities[polarity_key], insert_idx, polarity
                )

    print(f"Injected {total_events_injected} events into the event data\n")

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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Injects specified event spikes into existing event recordings data to serve"
            " as augmentation"
        )
    )

    parser.add_argument(
        "--event_data_path",
        type=str,
        required=True,
        help="Path to the .h5 file containing the event recordings data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the augmented event data with spikes injected to",
    )
    parser.add_argument(
        "--spikes_video_path",
        type=str,
        required=True,
        help="Path to the video containing the event spikes data as frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frame rate to use for the event spike frames to be injected",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=None,
        help="Label to assign for the event spikes to be injected",
    )
    parser.add_argument(
        "--polarity",
        type=int,
        default=None,
        help="Polarity to assign for the event spikes to be injected",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Prints additional information",
    )

    args = parser.parse_args()

    inject_event_spikes(
        event_data_path=args.event_data_path,
        save_path=args.save_path,
        spikes_video_path=args.spikes_video_path,
        fps=args.fps,
        label=args.label,
        polarity=args.polarity,
        verbose=args.verbose,
    )
