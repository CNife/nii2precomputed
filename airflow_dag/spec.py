from dataclasses import dataclass

from convert_to_precomputed.types import ImageResolution, ImageSize, JsonObject


@dataclass
class ImageChannel:
    name: str
    color: str


@dataclass
class ImageInfo:
    data_type: str
    resolution: ImageResolution
    size: ImageSize
    channels: list[ImageChannel]


@dataclass
class ConvertSpec:
    image_path: str | list[str]
    output_directory: str
    resolution: list[float]
    scale_indexes: list[int]
    write_block_size: int
    base_path: str
    base_url: str

    image_info: ImageInfo
    scales: list[JsonObject]
    multiscale: JsonObject
    base_json: JsonObject
