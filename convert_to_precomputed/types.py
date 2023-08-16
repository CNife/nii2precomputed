from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypeVar

from pydantic import BaseModel, Field

T: TypeVar = TypeVar("T")

OsPath: TypeAlias = Path | str


@dataclass
class ImageResolution:
    x: float
    y: float
    z: float


@dataclass
class ImageSize:
    x: int
    y: int
    z: int


@dataclass
class ResolutionRatio(ImageSize):
    pass


@dataclass
class DimensionRange:
    start: int
    end: int


@dataclass
class ImageRegion:
    x: DimensionRange
    y: DimensionRange
    z: DimensionRange


JsonString: TypeAlias = str
JsonNumber: TypeAlias = int | float
JsonNull: TypeAlias = type(None)
JsonBoolean: TypeAlias = bool
JsonArray: TypeAlias = list["Json"]
JsonObject: TypeAlias = dict[JsonString, "Json"]
Json: TypeAlias = JsonString | JsonNumber | JsonNull | JsonBoolean | JsonArray | JsonObject

TsScaleMetadata: TypeAlias = JsonObject


class ResolutionPM(BaseModel):
    x: float
    y: float
    z: float


class SizePM(BaseModel):
    x: int
    y: int
    z: int


class ScaleRatioPM(SizePM):
    pass


class Sharding(BaseModel):
    type: str = Field(alias="@type")
    preshift_bits: int
    hash: str
    minishard_bits: int
    shard_bits: int
    minishard_index_encoding: str = "raw"
    data_encoding: str = "raw"


class ScaleMetadata(BaseModel):
    encoding: str
    sharding: Sharding
    resolution: list[float]
    size: list[int]
    chunk_sizes: list[int]


class MultiscaleMetadata(BaseModel):
    data_type: str
    num_channels: int
    type: str


class ConvertSpec(BaseModel):
    image_path: str
    output_directory: str
    resolution: ResolutionPM
    write_block_size: int
    size: SizePM
    multiscale: MultiscaleMetadata
    scales: list[ScaleMetadata]
