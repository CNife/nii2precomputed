import numpy as np

from convert_precomputed.types import ImageResolution, JsonObject, ResolutionRatio


def build_multiscale_metadata(dtype: np.dtype, num_channels: int) -> JsonObject:
    return {
        "data_type": str(dtype),
        "num_channels": num_channels,
        "type": "image",
    }


def scale_resolution_ratio(
    scale_info: JsonObject, origin_resolution: ImageResolution
) -> ResolutionRatio:
    resolution = scale_info["resolution"]
    return ResolutionRatio(
        x=round(resolution[0] / origin_resolution.x),
        y=round(resolution[1] / origin_resolution.y),
        z=round(resolution[2] / origin_resolution.z),
    )
