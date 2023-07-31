from dataclasses import astuple

import numpy as np
from zimg import col4

from convert_precomputed.config import URL
from convert_precomputed.types import ImageResolution, ImageSize, JsonObject


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
                "source": f"precomputed://{URL}/{url_path}/channel_{channel}",
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
