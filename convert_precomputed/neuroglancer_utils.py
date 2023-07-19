from dataclasses import astuple

import numpy as np
from zimg import col4

from convert_precomputed.config import DEFAULT_URL
from convert_precomputed.types import ImageResolution, ImageSize, JsonObject
from vendor.neuroglancer_scripts_dyadic_pyramid import fill_scales_for_dyadic_pyramid

DEFAULT_SHARDING_ARG: JsonObject = {
    "@type": "neuroglancer_uint64_sharded_v1",
    "hash": "identity",
    "minishard_bits": 6,
    "minishard_index_encoding": "gzip",
    "data_encoding": "gzip",
    "preshift_bits": 9,
    "shard_bits": 15,
}


def build_scales_dyadic_pyramid(
    resolution: ImageResolution, size: ImageSize
) -> list[JsonObject]:
    init_scale_info = {
        "encoding": "raw",
        "sharding": DEFAULT_SHARDING_ARG,
        "resolution": list(astuple(resolution)),
        "size": list(astuple(size)),
    }
    info_dict = {"scales": [init_scale_info]}
    fill_scales_for_dyadic_pyramid(info_dict, target_chunk_size=64)

    result = info_dict["scales"]
    for scale in result:
        del scale["key"]
    return result


def build_ng_base_json(
    channel_colors: list[col4],
    resolution: ImageResolution,
    size: ImageSize,
    dtype: np.dtype,
    url_path: str,
) -> JsonObject:
    return {
        "dimensions": {
            "x": [resolution.x * 1e-9, "m"],
            "y": [resolution.y * 1e-9, "m"],
            "z": [resolution.z * 1e-9, "m"],
        },
        "position": [dimension_size / 2 for dimension_size in astuple(size)],
        "layout": "4panel",
        "layer": [
            {
                "type": "image",
                "name": f"channel_{channel}",
                "source": f"precomputed://{DEFAULT_URL}/{url_path}/channel_{channel}",
                "opacity": 1,
                "blend": "additive",
                "shaderControls": {
                    "color": f"#{channel_color.r:02x}{channel_color.g:02x}{channel_color.b:02x}",
                    "normalized": {
                        "range": [0.0, 1.0]
                        if dtype.kind == "f"
                        else [np.iinfo(dtype).min, np.iinfo(dtype).max],
                    },
                },
            }
            for channel, channel_color in enumerate(channel_colors)
        ],
    }
