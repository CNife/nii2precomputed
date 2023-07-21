import copy
import json
from collections import namedtuple
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import tensorstore as ts
from numpy import ndarray
from util import dbg
from zimg import Dimension, VoxelSizeUnit, ZImg, ZImgInfo, col4

from vendor.neuroglancer_scripts_dyadic_pyramid import fill_scales_for_dyadic_pyramid

Resolution = namedtuple("Resolution", ["x", "y", "z"])
ImageSize = namedtuple("ImageSize", ["x", "y", "z"])
Color = namedtuple("Color", ["r", "g", "b", "a"])

PathLike: TypeAlias = Path | str


def convert_nii_to_precomputed(
    out_folder: Path | str,
    image_path: Path | str,
    url_path: str,
    resolution: Resolution,
) -> None:
    out_folder, image_path = Path(out_folder), Path(image_path)
    out_folder.mkdir(exist_ok=True)

    # 读取图像信息
    image_info = read_image_info(image_path)
    dbg(image_info, "image info")
    image_size = ImageSize(x=image_info.width, y=image_info.height, z=image_info.depth)
    data_type_str = image_info.dataTypeString()
    data_type = np.dtype(data_type_str)

    # 构造多分辨率缩放信息
    full_resolution_info = build_full_resolution_info(
        data_type_str, resolution, image_size
    )
    dbg(full_resolution_info, "full resolution info")

    # 构造并写入neuroglancer的base.json
    build_and_write_base_json(
        image_info.channelColors,
        resolution,
        image_size,
        data_type,
        url_path,
        out_folder,
    )

    # 用tensorstore转换图像为neuroglancer的precomputed格式文件
    convert_image(image_path, full_resolution_info, out_folder, resolution)


def read_image_info(image_path: PathLike | list[PathLike]) -> ZImgInfo:
    if isinstance(image_path, list):
        image_infos = ZImg.readImgInfos(
            [str(path) for path in image_path], Dimension.Z, False
        )
    else:
        image_infos = ZImg.readImgInfos(str(image_path))
    return image_infos[0]


def build_full_resolution_info(
    data_type_str: str, resolution: Resolution, image_size: ImageSize
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
                "resolution": list(resolution),
                "size": list(image_size),
                "voxel_offset": [0, 0, 0],
            }
        ],
    }
    fill_scales_for_dyadic_pyramid(info, target_chunk_size=64)
    return info


def build_and_write_base_json(
    channel_colors: list[Color | col4],
    resolution: Resolution,
    image_size: ImageSize,
    data_type: np.dtype,
    url_path: str,
    out_folder: Path,
) -> None:
    base_dict = {
        "dimensions": {
            dimension: [dimension_resolution * 1e-9, "m"]
            for dimension, dimension_resolution in resolution._asdict().items()
        },
        "position": [dimension_size / 2.0 for dimension_size in image_size],
        "layout": "4panel",
        "layers": [],
    }
    if data_type.kind == "f":
        image_data_range = [0, 1.0]
    else:
        image_data_range = [0, np.iinfo(data_type).max]
    for channel_index, channel_color in enumerate(channel_colors):
        channel_name = f"channel_{channel_index}"
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

    dbg(base_dict, "base.json")
    with open(out_folder / "base.json", "w") as base_json_file:
        json.dump(base_dict, base_json_file, indent=2)


def convert_image(
    image_path: Path | list[str],
    full_resolution_info: dict[str, Any],
    out_folder: Path,
    resolution: Resolution,
) -> None:
    multiscale_metadata = {
        "data_type": full_resolution_info["data_type"],
        "num_channels": full_resolution_info["num_channels"],
        "type": full_resolution_info["type"],
    }
    scales = full_resolution_info["scales"]

    for scale_info in scales:
        scaled_resolution = Resolution(*scale_info["resolution"])
        x_ratio = round(scaled_resolution.x / resolution.x)
        y_ratio = round(scaled_resolution.y / resolution.y)
        z_ratio = round(scaled_resolution.z / resolution.z)
        if isinstance(image_path, list):
            zimg_reader = ZImg(
                filenames=image_path,
                catDim=Dimension.Z,
                catScenes=False,
                xRatio=x_ratio,
                yRatio=y_ratio,
                zRatio=z_ratio,
            )
        else:
            zimg_reader = ZImg(
                filename=str(image_path), xRatio=x_ratio, yRatio=y_ratio, zRatio=z_ratio
            )
        image_data: ndarray = zimg_reader.data[0][0]
        image_data = convert_image_data(image_data)

        scale_metadata = convert_to_tensorstore_scale_metadata(scale_info)
        output_store = open_tensorstore(
            f"channel_0",
            out_folder,
            scale_metadata,
            multiscale_metadata,
        )
        output_store[ts.d["channel"][0]] = image_data.transpose()


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
        "open": True,
        "create": True,
    }
    return ts.open(spec).result()


def convert_image_data(image_data: ndarray) -> ndarray:
    if image_data.dtype in (np.float32, np.float64):
        data_max, data_min = image_data.max(), image_data.min()
        image_data = (image_data - data_min) / (data_max - data_min)
        return image_data.astype(np.float32)
    return image_data


def convert_color(origin_color: col4) -> Color:
    return Color(origin_color.r, origin_color.g, origin_color.b, origin_color.a)


unit_scales = {
    VoxelSizeUnit.nm: 1,
    VoxelSizeUnit.um: 1000,
    VoxelSizeUnit.mm: 1000000,
}


def get_resolution(image_info: ZImgInfo) -> Resolution:
    scale = unit_scales.get(image_info.voxelSizeUnit, 1000)
    return Resolution(
        x=image_info.voxelSizeX * scale,
        y=image_info.voxelSizeY * scale,
        z=image_info.voxelSizeZ * scale,
    )


def get_image_size(image_info: ZImgInfo) -> ImageSize:
    return ImageSize(x=image_info.width, y=image_info.height, z=image_info.depth)
