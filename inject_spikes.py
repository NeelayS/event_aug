if __name__ == "__main__":

    import argparse
    from event_aug.spike_injection import inject_event_spikes

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
        "--resize_size",
        type=int,
        nargs="+",
        default=None,
        help="Reshape size for event spike frames to be injected",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="+",
        default=None,
        help="Crop size for event spike frames to be injected",
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
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        verbose=args.verbose,
    )
