import math
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
from neuroglancer_scripts.dyadic_pyramid import fill_scales_for_dyadic_pyramid
from zimg import ZImg, ZImgInfo

Resolution = namedtuple("Resolution", ["x", "y", "z"])


def convert_nii_to_precomputed(
        out_folder: Path | str,
        image_path: Path | str,
        out_url_path: str,
        image_scale: float = 1.0,
        resolution: Resolution | None = None,
) -> None:
    out_folder, image_path = Path(out_folder), Path(image_path)
    out_folder.mkdir(exist_ok=True)

    zimg = ZImg()
    image_info = read_image_info(zimg, image_path)
    if resolution is None:
        resolution = Resolution(
            x=image_info.width,
            y=image_info.height,
            z=image_info.depth,
        )
    source_data_type = image_info.dataTypeString()
    target_data_type = get_target_data_type(source_data_type)

    full_resolutions_info = build_full_resolutions_info(
        image_info, resolution, image_scale
    )

    # TODO 构造并写入base.json
    write_base_json({})


def read_image_info(zimg: ZImg, image_path: Path) -> ZImgInfo:
    image_infos: list[ZImgInfo] = zimg.readImgInfos(str(image_path))
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


def build_full_resolutions_info(
        image_info: ZImgInfo, resolution: Resolution, image_scale: float
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
                "resolution": [
                    resolution.x / image_scale,
                    resolution.y / image_scale,
                    resolution.z,
                ],
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
    return info


def write_base_json(base_dict: dict[str, Any]) -> None:
    pass
