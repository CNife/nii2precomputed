import copy
import json
import math
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts
from neuroglancer_scripts.dyadic_pyramid import fill_scales_for_dyadic_pyramid
from tqdm import tqdm
from zimg import ZImg, ZImgInfo, ZImgRegion, ZVoxelCoordinate

Resolution = namedtuple("Resolution", ["x", "y", "z"])


def convert_nii_to_precomputed(
    out_folder: Path | str,
    image_path: Path | str,
    url_path: str,
    image_scale: float = 1.0,
    resolution: Resolution | None = None,
) -> None:
    out_folder, image_path = Path(out_folder), Path(image_path)
    out_folder.mkdir(exist_ok=True)

    image_info = read_image_info(image_path)
    if resolution is None:
        resolution = Resolution(
            x=image_info.width,
            y=image_info.height,
            z=image_info.depth,
        )
    scaled_resolution = Resolution(
        x=resolution.x / image_scale, y=resolution.y / image_scale, z=resolution.z
    )
    source_data_type = image_info.dataTypeString()
    target_data_type = get_target_data_type(source_data_type)

    full_resolution_info = build_full_resolution_info(
        image_info, scaled_resolution, image_scale
    )
    channel_names = [f"channel_{i}" for i in range(image_info.numChannels)]
    for scale_index in range(len(full_resolution_info["scales"])):
        convert_a_scale(
            full_resolution_info,
            scale_index,
            channel_names,
            target_data_type,
            out_folder,
            scaled_resolution,
            image_scale,
            image_info,
            image_path,
        )

    base_json_dict = build_base_json_dict(
        image_info, channel_names, image_scale, resolution, target_data_type, url_path
    )
    with open(out_folder / "base.json", "w") as base_json_file:
        base_json = json.dumps(base_json_dict)
        print(f"base.json={base_json}")
        base_json_file.write(base_json)


def read_image_info(image_path: Path) -> ZImgInfo:
    # noinspection PyTypeChecker
    image_infos: list[ZImgInfo] = ZImg.readImgInfos(str(image_path))
    assert len(image_infos) == 1
    assert image_infos[0].numTimes == 1
    return image_infos[0]


def get_target_data_type(data_type: str) -> type[np.number]:
    convert_types = {
        "float64": np.float32,
        "float32": np.float32,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    }
    return convert_types.get(data_type, np.uint8)


def build_full_resolution_info(
    image_info: ZImgInfo, scaled_resolution: Resolution, image_scale: float
) -> dict[str, Any]:
    info = {
        "type": "image",
        "data_type": image_info.dataTypeString(),
        "num_channels": 1,
        "scales": [
            {
                "chunk_sizes": [],
                "encoding": "raw",
                "sharding": {
                    "@type": "neuroglancer_uint64_sharded_v1",
                    "hash": "identity",
                    "minishard_bits": 6,
                    "minishard_index_encoding": "gzip",
                    "data_encoding": "gzip",
                    "preshift_bits": 9,
                    "shard_bits": 15,
                },
                "resolution": list(scaled_resolution),
                "size": [
                    math.ceil(image_info.width * image_scale),
                    math.ceil(image_info.height * image_scale),
                    image_info.depth,
                ],
                "voxel_offset": [0, 0, 0],
            }
        ],
    }
    fill_scales_for_dyadic_pyramid(info)
    print(f"info={json.dumps(info)}")
    return info


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
    scale = convert_scale_dict(full_resolution_info["scales"][scale_index])
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


def convert_scale_dict(scale: dict[str, Any]) -> dict[str, Any]:
    new_scale = copy.deepcopy(scale)
    new_scale["chunk_size"] = new_scale["chunk_sizes"][0]
    del new_scale["chunk_sizes"]
    del new_scale["key"]
    return new_scale


def open_tensorstores(
    scale: dict[str, Any],
    channel_names: list[str],
    out_folder: Path,
    multiscale: dict[str, Any],
) -> list[ts.TensorStore]:
    def open_a_tensorstore(channel_name: str) -> ts.TensorStore:
        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": {
                "driver": "file",
                "path": str(out_folder),
            },
            "path": channel_name,
            "scale_metadata": scale,
            "multiscale_metadata": multiscale,
            "create": True,
            "delete_existing": True,
        }
        print(f"tensorstore spec = {json.dumps(spec)}")
        return ts.open(spec).result()

    return [open_a_tensorstore(channel_name) for channel_name in channel_names]


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


def build_base_json_dict(
    image_info: ZImgInfo,
    channel_names: list[str],
    image_scale: float,
    resolution: Resolution,
    target_data_type: type[np.number],
    url_path: str,
):
    base_dict = {
        "dimensions": {
            "x": [resolution.x / image_scale * 1e-9, "m"],
            "y": [resolution.y / image_scale * 1e-9, "m"],
            "z": [resolution.z * 1e-9, "m"],
        },
        "position": [
            image_info.width / 2.0 * image_scale,
            image_info.height / 2.0 * image_scale,
            image_info.depth / 2.0,
        ],
        # "crossSectionScale": 59.90389939556908,
        # "projectionOrientation": [
        #     -0.11555982381105423,
        #     -0.09716008603572845,
        #     0.4296676218509674,
        #     0.8902761340141296,
        # ],
        # "projectionScale": 50764.49878262887,
        # "selectedLayer": {
        #     "layer": "annotation",
        #     "visible": True,
        # },
        "layout": "4panel",
        "layers": [],
    }
    if issubclass(target_data_type, np.floating):
        image_data_range = [0, 0, 1.0]
    else:
        image_data_range = [0, np.iinfo(target_data_type).max]
    for i, channel_name in enumerate(channel_names):
        channel_color = image_info.channelColors[i]
        channel_layer = {
            "type": "image",
            "name": channel_name,
            "source": f"{url_path}/{channel_name}",
            "opacity": 1,
            "blend": "additive",
            "shaderControls": {
                "color": f"#{channel_color.r:02x}{channel_color.g:02x}{channel_color.b:02x}",
                "normalized": {
                    "range": image_data_range,
                },
            },
        }
        base_dict["layers"].append(channel_layer)
    return base_dict
