from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts
from tqdm import tqdm
from zimg import ZImg, ZImgInfo, ZImgRegion, ZVoxelCoordinate

from nii_2_precomputed import (
    Resolution,
    convert_to_tensorstore_scale_metadata,
    open_tensorstore,
)


def get_target_data_type(data_type: str) -> type[np.number]:
    convert_types = {
        "float64": np.float32,
        "float32": np.float32,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    }
    return convert_types.get(data_type, np.uint8)


def convert_a_scale(
    full_resolution_info: dict[str, Any],
    scale_index: int,
    channel_names: list[str],
    target_data_type: type[np.number],
    out_folder: Path,
    scaled_resolution: Resolution,
    image_scale: float,
    image_info: ZImgInfo,
    image_path: Path,
) -> None:
    scale = convert_to_tensorstore_scale_metadata(
        full_resolution_info["scales"][scale_index]
    )
    multiscale = {
        "data_type": full_resolution_info["data_type"],
        "num_channels": full_resolution_info["num_channels"],
        "type": full_resolution_info["type"],
    }
    stores = open_tensorstores(scale, channel_names, out_folder, multiscale)

    this_scale_resolution = Resolution(*scale["resolution"])
    x_ratio = round(this_scale_resolution.x / scaled_resolution.x / image_scale)
    y_ratio = round(this_scale_resolution.y / scaled_resolution.y / image_scale)
    z_ratio = round(this_scale_resolution.z / scaled_resolution.z)

    save_step = scale["chunk_size"][2] * 2
    step_ranges = [
        slice(i * save_step, min(image_info.depth, save_step * (i + 1)), 1)
        for i in range((image_info.depth + save_step - 1) // save_step)
    ]
    for a_range in tqdm(step_ranges):
        image = ZImg(
            str(image_path),
            region=ZImgRegion(
                ZVoxelCoordinate(0, 0, a_range.start, 0, 0),
                ZVoxelCoordinate(-1, -1, a_range.stop, len(channel_names), 1),
            ),
            scene=0,
            xRatio=x_ratio,
            yRatio=y_ratio,
            zRatio=z_ratio,
        )
        z_start = (a_range.start + z_ratio - 1) // z_ratio
        z_end = (a_range.stop + z_ratio - 1) // z_ratio
        for channel_index, channel_data in enumerate(image.data[0]):
            channel_data = convert_channel_data_type(channel_data, target_data_type)
            stores[channel_index][ts.d["channel"][channel_index]][
                ts.d["z"][z_start:z_end]
            ] = np.reshape(channel_data, channel_data.shape[::-1], order="F")


def open_tensorstores(
    scale: dict[str, Any],
    channel_names: list[str],
    out_folder: Path,
    multiscale: dict[str, Any],
) -> list[ts.TensorStore]:
    return [
        open_tensorstore(channel_name, out_folder, scale, multiscale)
        for channel_name in channel_names
    ]


def convert_channel_data_type(
    channel_data: np.ndarray, target_data_type: type[np.number]
) -> np.ndarray:
    if (channel_dtype := channel_data.dtype) == target_data_type:
        return channel_data
    if target_data_type == np.float32:
        return channel_data.astype(np.float32)
    if target_data_type == np.uint8:
        min_value = float(np.iinfo(channel_dtype).min)
        max_value = float(np.iinfo(channel_dtype).max)
        return (channel_data.astype(np.float64) - min_value) / (max_value - min_value)
    return channel_data
