import copy
import json
import math
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts
from neuroglancer_scripts.dyadic_pyramid import fill_scales_for_dyadic_pyramid
from tqdm import tqdm, trange
from zimg import ZImg, ZImgInfo, ZImgRegion, ZVoxelCoordinate

from util import pretty_print_object

Resolution = namedtuple("Resolution", ["x", "y", "z"])
ImageSize = namedtuple("ImageSize", ["x", "y", "z"])


def convert_nii_to_precomputed(
    out_folder: Path | str,
    image_path: Path | str,
    url_path: str,
    resolution: Resolution,
    image_scale: float = 1.0,
) -> None:
    out_folder, image_path = Path(out_folder), Path(image_path)
    out_folder.mkdir(exist_ok=True)

    # 读取图像信息
    image_info = read_image_info(image_path)
    pretty_print_object(image_info, "image info")
    scaled_resolution = Resolution(
        x=resolution.x / image_scale,
        y=resolution.y / image_scale,
        z=float(resolution.z),
    )
    image_size = ImageSize(x=image_info.width, y=image_info.height, z=image_info.depth)
    scaled_image_size = ImageSize(
        x=math.ceil(image_size.x * image_scale),
        y=math.ceil(image_size.y * image_scale),
        z=image_size.z,
    )
    data_type_str = image_info.dataTypeString()
    data_type = get_data_type_from_str(data_type_str)

    # 构造多分辨率缩放信息
    full_resolution_info = build_full_resolution_info(
        data_type_str, scaled_resolution, scaled_image_size
    )
    pretty_print_object(full_resolution_info, "full resolution info")

    # 构造并写入neuroglancer的base.json
    base_json_dict = build_base_json_dict(
        image_info, scaled_resolution, scaled_image_size, data_type, url_path
    )
    pretty_print_object(base_json_dict, "base.json")
    with open(out_folder / "base.json", "w") as base_json_file:
        json.dump(base_json_dict, base_json_file, indent=2)

    # 用tensorstore转换图像为neuroglancer的precomputed格式文件
    convert_image(image_info, full_resolution_info, out_folder, image_path, resolution)
    for i in range(image_info.numChannels):
        with open(out_folder / f"channel_{i}" / "info") as channel_info_file:
            channel_info_data = json.load(channel_info_file)
            pretty_print_object(
                channel_info_data,
                f"precomputed channel info {i}/{image_info.numChannels}",
            )


def read_image_info(image_path: Path) -> ZImgInfo:
    # noinspection PyTypeChecker
    image_infos: list[ZImgInfo] = ZImg.readImgInfos(str(image_path))
    assert len(image_infos) == 1
    assert image_infos[0].numTimes == 1
    return image_infos[0]


def get_data_type_from_str(data_type_str: str) -> type[np.number]:
    match data_type_str:
        case "float32":
            return np.float32
        case "float64":
            return np.float64
        case "uint8":
            return np.uint8
        case "uint16":
            return np.uint16
        case "uint32":
            return np.uint32
        case "uint64":
            return np.uint64
        case _:
            raise ValueError("invalid data type")


def build_full_resolution_info(
    data_type_str: str, scaled_resolution: Resolution, scaled_image_size: ImageSize
) -> dict[str, Any]:
    info = {
        "type": "image",
        "data_type": data_type_str,
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
                "size": list(scaled_image_size),
                "voxel_offset": [0, 0, 0],
            }
        ],
    }
    fill_scales_for_dyadic_pyramid(info, target_chunk_size=64)
    return info


def build_base_json_dict(
    image_info: ZImgInfo,
    scaled_resolution: Resolution,
    scaled_image_size: ImageSize,
    data_type: type[np.number],
    url_path: str,
):
    base_dict = {
        "dimensions": {
            dimension: [dimension_resolution * 1e-9, "m"]
            for dimension, dimension_resolution in scaled_resolution._asdict().items()
        },
        "position": [dimension_size / 2.0 for dimension_size in scaled_image_size],
        "layout": "4panel",
        "layers": [],
    }
    if issubclass(data_type, np.floating):
        image_data_range = [0, 0, 1.0]
    else:
        image_data_range = [0, np.iinfo(data_type).max]
    for channel_index in range(image_info.numChannels):
        channel_name = f"channel_{channel_index}"
        channel_color = image_info.channelColors[channel_index]
        channel_layer = {
            "type": "image",
            "name": channel_name,
            "source": f"precomputed://{url_path}/{channel_name}",
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


def convert_image(
    image_info: ZImgInfo,
    full_resolution_info: dict[str, Any],
    out_folder: Path,
    image_path: Path,
    origin_resolution: Resolution,
) -> None:
    multiscale_metadata = {
        "data_type": full_resolution_info["data_type"],
        "num_channels": full_resolution_info["num_channels"],
        "type": full_resolution_info["type"],
    }

    for i, scale_info in enumerate(tqdm(full_resolution_info["scales"], desc="scales", leave=False)):
        scale_metadata = convert_to_tensorstore_scale_metadata(scale_info)
        output_stores = [
            open_tensorstore(
                f"channel_{i}", out_folder, scale_metadata, multiscale_metadata
            )
            for i in range(image_info.numChannels)
        ]
        pretty_print_object(output_stores, title=f"tensorstore config {i}/{len(full_resolution_info['scales'])}")
        scale_resolution = Resolution(*scale_info["resolution"])
        z_step_size = scale_metadata["chunk_size"][2] * 2
        for z_step_index in trange(
            (image_info.depth + z_step_size - 1) // z_step_size,
            desc="steps",
            leave=False,
        ):
            z_start, z_end = z_step_index * z_step_size, min(
                image_info.depth, (z_step_index + 1) * z_step_size
            )
            z_ratio = round(scale_resolution.z / origin_resolution.z)
            image_region = ZImg(
                str(image_path),
                region=ZImgRegion(
                    ZVoxelCoordinate(0, 0, z_start, 0, 0),
                    ZVoxelCoordinate(-1, -1, z_end, -1, 1),
                ),
                scene=0,
                xRatio=round(scale_resolution.x / origin_resolution.x),
                yRatio=round(scale_resolution.y / origin_resolution.y),
                zRatio=z_ratio,
            )
            target_z_start = (z_start + z_ratio - 1) // z_ratio
            target_z_end = (z_end + z_ratio - 1) // z_ratio
            for channel_data, output_store in zip(image_region.data[0], output_stores):
                output_store[
                    ts.d["channel", "z"][0, target_z_start:target_z_end]
                ] = np.reshape(channel_data, channel_data.shape[::-1], order="F")


def convert_to_tensorstore_scale_metadata(scale: dict[str, Any]) -> dict[str, Any]:
    new_scale = copy.deepcopy(scale)
    new_scale["chunk_size"] = new_scale["chunk_sizes"][0]
    del new_scale["chunk_sizes"]
    del new_scale["key"]
    return new_scale


def open_tensorstore(
    channel_name: str,
    out_folder: Path,
    scale_metadata: dict[str, Any],
    multiscale_metadata: dict[str, Any],
) -> ts.TensorStore:
    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {
            "driver": "file",
            "path": str(out_folder),
        },
        "path": channel_name,
        "scale_metadata": scale_metadata,
        "multiscale_metadata": multiscale_metadata,
        "create": True,
        "delete_existing": True,
    }
    return ts.open(spec).result()
