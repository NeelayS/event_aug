import json
from os.path import join

import cv2
import numpy as np
import stl
import tables

from event_aug.utils import array_to_video

# OpenCV colours

WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GRAY = (50, 50, 50)


def get_next_event(events_iter, camera):
    event = {}
    event[f"timestamp_{camera}"] = next(events_iter[f"timestamp_{camera}"])
    event[f"polarity_{camera}"] = next(events_iter[f"polarity_{camera}"])
    event[f"xy_undistorted_{camera}"] = next(events_iter[f"xy_undistorted_{camera}"])
    event[f"label_{camera}"] = next(events_iter[f"label_{camera}"])

    return event


def get_next_pose(poses_iter, n_cameras):
    pose = {}

    pose["timestamp"] = next(poses_iter["timestamp"])

    pose["rotation"] = {}
    for prop_name in poses_iter["rotation"].keys():
        pose["rotation"][prop_name] = next(poses_iter["rotation"][prop_name])

    for i in range(n_cameras):
        pose[f"camera_{i}_rotation"] = {}
        for prop_name in poses_iter[f"camera_{i}_rotation"].keys():
            pose[f"camera_{i}_rotation"][prop_name] = next(
                poses_iter[f"camera_{i}_rotation"][prop_name]
            )

    pose["translation"] = {}
    for prop_name in poses_iter["translation"].keys():
        pose["translation"][prop_name] = next(poses_iter["translation"][prop_name])

    for i in range(n_cameras):
        pose[f"camera_{i}_translation"] = {}
        for prop_name in poses_iter[f"camera_{i}_translation"].keys():
            pose[f"camera_{i}_translation"][prop_name] = next(
                poses_iter[f"camera_{i}_translation"][prop_name]
            )

    return pose


def projection(
    data_path,
    output_video_path,
    max_frames=500,
    fps=25,
    camera_height=260,
    camera_width=346,
    augmentation_label=2,
    n_cameras=1,
    distinguish_polarity=False,
):

    dvs_cam_height = [np.uint32(camera_height) for i in range(n_cameras)]
    dvs_cam_width = [np.uint32(camera_width) for i in range(n_cameras)]
    dvs_cam_origin_x_offset = [dvs_cam_width[i] / 2 for i in range(n_cameras)]
    dvs_cam_origin_y_offset = [dvs_cam_height[i] / 2 for i in range(n_cameras)]
    dvs_cam_nominal_f_len = [4.0 for i in range(n_cameras)]
    dvs_cam_pixel_mm = [1.8e-2 for i in range(n_cameras)]

    # Read recording info from JSON
    with open(join(data_path, "info.json"), "r") as info_json_file:
        info_json = json.load(info_json_file)

    ##################################################################

    # === READ PROPS DATA ===
    props_markers = {}  # contains the translation of each marker, relative to prop origin
    props_meshes = {}  # contains prop STL meshes (polygon, translation, vertex)
    props_labels = {}  # contains integer > 0 class labels of the props
    props_dilation = {}  # contains dilation kernels for the mask of each prop

    props_names = list(info_json["prop_marker_files"].keys())
    for prop_name in props_names:
        with open(
            join(data_path, info_json["prop_marker_files"][prop_name]), "r"
        ) as marker_file:
            markers = json.load(marker_file)

        props_markers[prop_name] = markers
        mesh = stl.mesh.Mesh.from_file(
            join(data_path, info_json["prop_mesh_files"][prop_name])
        ).vectors.transpose(0, 2, 1)

        props_meshes[prop_name] = mesh
        props_labels[prop_name] = info_json["prop_labels"][prop_name]
        props_dilation[prop_name] = np.ones((3, 3), "uint8")

    # Change prop mask dilation
    # props_dilation['kth_hammer'] = np.ones((4, 4), 'uint8')
    # props_dilation['kth_screwdriver'] = np.ones((4, 4), 'uint8')
    # props_dilation['kth_spanner'] = np.ones((4, 4), 'uint8')

    ##################################################################

    # === READ CALIBRATION FILES ===

    path_projection = join(data_path, info_json["projection_calibration_path"])

    # v_to_dvs_rotation_file = [
    #     f"{path_projection}/v_to_dv_{i}_rotation.npy" for i in range(n_cameras)
    # ]
    # v_to_dvs_rotation = [np.load(name) for name in v_to_dvs_rotation_file]

    # v_to_dvs_translation_file = [
    #     f"{path_projection}/v_to_dv_{i}_translation.npy" for i in range(n_cameras)
    # ]
    # v_to_dvs_translation = [np.load(name) for name in v_to_dvs_translation_file]

    v_to_dvs_f_len_scale_file = [
        f"{path_projection}/v_to_dv_{i}_focal_length_scale.npy" for i in range(n_cameras)
    ]
    v_to_dvs_f_len_scale = [np.load(name) for name in v_to_dvs_f_len_scale_file]
    v_to_dvs_f_len = [
        dvs_cam_nominal_f_len[i] * v_to_dvs_f_len_scale[i] for i in range(n_cameras)
    ]

    v_to_dvs_x_scale_file = [
        f"{path_projection}/v_to_dv_{i}_x_scale.npy" for i in range(n_cameras)
    ]
    v_to_dvs_x_scale = [np.load(name) for name in v_to_dvs_x_scale_file]

    ##################################################################

    # initialise temp memory
    event_pos = [
        np.zeros((dvs_cam_height[i], dvs_cam_width[i]), dtype="uint64")
        for i in range(n_cameras)
    ]
    event_neg = [
        np.zeros((dvs_cam_height[i], dvs_cam_width[i]), dtype="uint64")
        for i in range(n_cameras)
    ]
    event_image = [
        np.zeros((dvs_cam_height[i], dvs_cam_width[i], 3), dtype="uint8")
        for i in range(n_cameras)
    ]
    prop_masks = [
        {
            prop_name: np.empty((dvs_cam_height[i], dvs_cam_width[i]), dtype="uint8")
            for prop_name in props_names
        }
        for i in range(n_cameras)
    ]

    events_vid = np.zeros((1, dvs_cam_height[0], dvs_cam_width[0], 3), dtype="uint8")

    # load DVS event data
    events_file_name = join(data_path, "event_data/augmented_event.h5")
    events_file = tables.open_file(events_file_name, mode="r")
    events_iter = []
    for i in range(n_cameras):
        e_iter = {}
        e_iter[f"timestamp_{i}"] = events_file.root[f"timestamp_{i}"].iterrows()
        e_iter[f"polarity_{i}"] = events_file.root[f"polarity_{i}"].iterrows()
        e_iter[f"xy_undistorted_{i}"] = events_file.root[f"xy_undistorted_{i}"].iterrows()
        e_iter[f"label_{i}"] = events_file.root[f"label_{i}"].iterrows()
        events_iter.append(e_iter)

    event = [get_next_event(events_iter[i], i) for i in range(n_cameras)]

    # load Vicon pose data file
    poses_file_name = join(data_path, "event_data/pose.h5")
    poses_file = tables.open_file(poses_file_name, mode="r")
    poses_iter = {}

    timestamp = poses_file.root.timestamp
    poses_iter["timestamp"] = timestamp.iterrows()

    poses_iter["rotation"] = {}
    for i in range(n_cameras):
        poses_iter[f"camera_{i}_rotation"] = {}

    poses_iter["translation"] = {}
    for i in range(n_cameras):
        poses_iter[f"camera_{i}_translation"] = {}

    for prop_name in props_names:
        rotation = poses_file.root.props[prop_name].rotation
        poses_iter["rotation"][prop_name] = rotation.iterrows()
        for i in range(n_cameras):
            cam_rotation = poses_file.root.props[prop_name][f"camera_{i}_rotation"]
            poses_iter[f"camera_{i}_rotation"][prop_name] = cam_rotation.iterrows()

        translation = poses_file.root.props[prop_name].translation
        poses_iter["translation"][prop_name] = translation.iterrows()
        for i in range(n_cameras):
            cam_translation = poses_file.root.props[prop_name][f"camera_{i}_translation"]
            poses_iter[f"camera_{i}_translation"][prop_name] = cam_translation.iterrows()

    pose = get_next_pose(poses_iter, n_cameras)

    frames_count = 0

    done_event = [False for i in range(n_cameras)]
    while not all(done_event):

        try:
            pose_new = get_next_pose(poses_iter, n_cameras)
            pose_midway = pose["timestamp"] / 2 + pose_new["timestamp"] / 2
        except StopIteration:
            print("DEBUG: out of Vicon poses")
            break

        frames_count += 1
        if frames_count > max_frames:
            break

        print(f"Processing frame {frames_count}")

        for prop_name in props_names:

            # compute prop mask for each camera
            for i in range(n_cameras):
                prop_masks[i][prop_name].fill(0)

                mesh_to_dvs_rotation = pose[f"camera_{i}_rotation"][prop_name]
                mesh_to_dvs_translation = pose[f"camera_{i}_translation"][prop_name]

                if (
                    not np.isfinite(mesh_to_dvs_rotation).all()
                    or not np.isfinite(mesh_to_dvs_translation).all()
                ):
                    continue

                # transform to DVS camera space
                dvs_space_p = (
                    np.matmul(mesh_to_dvs_rotation, props_meshes[prop_name])
                    + mesh_to_dvs_translation
                )
                dvs_space_p[:, :2, :] *= 1 / dvs_space_p[:, np.newaxis, 2, :]
                dvs_space_p = dvs_space_p[:, :2, :]
                dvs_space_p *= v_to_dvs_f_len[i]
                dvs_space_p /= dvs_cam_pixel_mm[i]
                dvs_space_p *= v_to_dvs_x_scale[i]
                dvs_space_p += [
                    [dvs_cam_origin_x_offset[i]],
                    [dvs_cam_origin_y_offset[i]],
                ]
                dvs_space_p_int = np.rint(dvs_space_p).astype("int32")

                # transpose points for OpenCV
                dvs_space_p_int = dvs_space_p_int.transpose(0, 2, 1)

                # compute prop mask
                cv2.fillPoly(prop_masks[i][prop_name], dvs_space_p_int, 255)
                prop_masks[i][prop_name] = cv2.dilate(
                    prop_masks[i][prop_name], props_dilation[prop_name]
                )

        # process DVS events
        for i in range(n_cameras):
            if not done_event[i]:

                image = event_image[i]
                pos = event_pos[i]
                neg = event_neg[i]

                image.fill(0)
                pos.fill(0)
                neg.fill(0)

                while event[i][f"timestamp_{i}"] < pose_midway:
                    xy_int = np.rint(event[i][f"xy_undistorted_{i}"]).astype("int32")

                    # get event label
                    label = event[i][f"label_{i}"]

                    if label != augmentation_label:
                        if event[i][f"polarity_{i}"]:
                            pos[xy_int[1], xy_int[0]] += 1
                        else:
                            neg[xy_int[1], xy_int[0]] += 1

                    if distinguish_polarity:
                        if event[i][f"polarity_{i}"]:
                            image[xy_int[1], xy_int[0]] = BLUE
                        else:
                            image[xy_int[1], xy_int[0]] = GREEN
                    else:
                        image[xy_int[1], xy_int[0]] = WHITE

                    try:
                        event[i] = get_next_event(events_iter[i], i)
                    except StopIteration:
                        print(f"DEBUG: out of DVS {i} events")
                        done_event[i] = True
                        break

                # fill DVS event image with events, then mask it
                for prop_name in props_names:

                    mask = prop_masks[i][prop_name].astype("bool")
                    image[mask] = GRAY

                    if distinguish_polarity:
                        mask_neg = neg > pos
                        image[(mask_neg & mask)] = BLUE
                        mask_pos = pos > neg
                        image[(mask_pos & mask)] = YELLOW
                    else:
                        mask_pos_neg = neg.astype("bool") | pos.astype("bool")
                        image[(mask_pos_neg & mask)] = RED

                img = image.copy()
                img = np.expand_dims(img, axis=0)
                events_vid = np.vstack((events_vid, img))

        pose = pose_new

    array_to_video(events_vid, output_video_path, fps)

    events_file.close()
    poses_file.close()
