import math
from dataclasses import astuple
from pathlib import Path

import numpy as np
import tensorstore as ts

from convert_to_precomputed.types import ImageResolution, ImageSize, JsonObject, ResolutionRatio, TsScaleMetadata
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


def build_scales_dyadic_pyramid(resolution: ImageResolution, size: ImageSize) -> list[TsScaleMetadata]:
    init_scale_info = {
        "encoding": "raw",
        "sharding": DEFAULT_SHARDING_ARG,
        "resolution": list(astuple(resolution)),
        "size": list(astuple(size)),
    }
    info_dict = {"scales": [init_scale_info]}

    target_chunk_size = 64
    assert math.log2(target_chunk_size).is_integer()
    max_scales = round(math.log2(target_chunk_size)) + 1
    fill_scales_for_dyadic_pyramid(info_dict, target_chunk_size=target_chunk_size, max_scales=max_scales)
    return info_dict["scales"]


def build_multiscale_metadata(dtype: np.dtype, num_channels: int) -> JsonObject:
    return {"data_type": str(dtype), "num_channels": num_channels, "type": "image"}


def scale_resolution_ratio(scale_info: TsScaleMetadata, origin_resolution: ImageResolution) -> ResolutionRatio:
    resolution = scale_info["resolution"]
    return ResolutionRatio(
        x=round(resolution[0] / origin_resolution.x),
        y=round(resolution[1] / origin_resolution.y),
        z=round(resolution[2] / origin_resolution.z),
    )


def open_tensorstore_to_write(
    channel_name: str, output_directory: Path, scale: TsScaleMetadata, multi_scale_metadata: JsonObject
) -> ts.TensorStore:
    scale_metadata = {k: v for k, v in scale.items() if k != "chunk_sizes"}
    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file", "path": str(output_directory)},
        "path": channel_name,
        "scale_metadata": scale_metadata,
        "multiscale_metadata": multi_scale_metadata,
        "open": True,
        "create": True,
    }
    return ts.open(spec).result()