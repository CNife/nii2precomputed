from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

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
Json: TypeAlias = (
    JsonString | JsonNumber | JsonNull | JsonBoolean | JsonArray | JsonObject
)

TsScaleMetadata: TypeAlias = JsonObject
