from pathlib import Path
from typing import Iterable

import numpy as np
from deprecation import deprecated
from numpy import ndarray
from zimg import Dimension, VoxelSizeUnit, ZImg, ZImgInfo, ZImgRegion, ZVoxelCoordinate

from convert_to_precomputed.types import (
    ImageRegion,
    ImageResolution,
    ImageSize,
    OsPath,
    ResolutionPM,
    ResolutionRatio,
    SizePM,
)


# noinspection PyTypeChecker
@deprecated(details="use read_image_info_v2")
def read_image_info(image_path: OsPath | Iterable[OsPath]) -> ZImgInfo:
    if isinstance(image_path, Iterable):
        image_paths = [str(path) for path in image_path]
        image_infos = ZImg.readImgInfos(image_paths, catDim=Dimension.Z, catScenes=True)
    else:
        image_infos = ZImg.readImgInfos(str(image_path))
    result: ZImgInfo = image_infos[0]
    return result


# noinspection PyTypeChecker
def read_image_info_v2(image_path: OsPath) -> ZImgInfo:
    image_path = Path(image_path)
    if image_path.is_dir():
        image_paths = [str(path) for path in image_path.iterdir() if path.is_file()]
        image_infos = ZImg.readImgInfos(image_paths, catDim=Dimension.Z, catScenes=True)
    elif image_path.is_file():
        image_infos = ZImg.readImgInfos(str(image_path))
    else:
        raise ValueError(f"invalid image path {image_path}")
    return image_infos[0]


def get_image_size(image_info: ZImgInfo) -> ImageSize:
    return ImageSize(x=image_info.width, y=image_info.height, z=image_info.depth)


def get_image_size_v2(image_info: ZImgInfo) -> SizePM:
    return SizePM(x=image_info.width, y=image_info.height, z=image_info.depth)


def get_image_resolution(image_info: ZImgInfo) -> ImageResolution:
    scale = _unit_scale(image_info.voxelSizeUnit)
    return ImageResolution(
        x=image_info.voxelSizeX * scale, y=image_info.voxelSizeY * scale, z=image_info.voxelSizeZ * scale
    )


def get_image_resolution_v2(image_info: ZImgInfo) -> ResolutionPM:
    scale = _unit_scale(image_info.voxelSizeUnit)
    return ResolutionPM(
        x=image_info.voxelSizeX * scale, y=image_info.voxelSizeY * scale, z=image_info.voxelSizeZ * scale
    )


def _unit_scale(unit: VoxelSizeUnit) -> int:
    match unit:
        case VoxelSizeUnit.nm:
            return 1
        case VoxelSizeUnit.um | VoxelSizeUnit.none:
            return 1000
        case VoxelSizeUnit.mm:
            return 1000 * 1000
        case _:
            raise ValueError("unknown VoxelSizeUnit")


def get_image_dtype(image_info: ZImgInfo) -> np.dtype:
    return np.dtype(image_info.dataTypeString())


def read_image_data(image_path: OsPath | Iterable[OsPath], region: ImageRegion, ratio: ResolutionRatio) -> ndarray:
    zimg_region = _region_2_zimg(region)
    zimg_ratio_dict = _ratio_2_dict(ratio)
    if isinstance(image_path, Iterable):
        zimg = ZImg(
            [str(path) for path in image_path],
            catDim=Dimension.Z,
            catScenes=True,
            region=zimg_region,
            **zimg_ratio_dict,
        )
    else:
        zimg = ZImg(str(image_path), region=zimg_region, **zimg_ratio_dict)
    zimg_data = zimg.data[0].copy(order="C")
    return zimg_data


def read_image_data_v2(
    image_path: OsPath,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    z_start: int,
    z_end: int,
    x_ratio: int,
    y_ratio: int,
    z_ratio: int,
) -> ndarray:
    image_path = Path(image_path)
    zimg_region = ZImgRegion(
        ZVoxelCoordinate(x_start, y_start, z_start, 0, 0), ZVoxelCoordinate(x_end, y_end, z_end, -1, -1)
    )
    if image_path.is_dir():
        image_paths = [str(path) for path in image_path.iterdir() if path.is_file()]
        zimg = ZImg(
            image_paths,
            catDim=Dimension.Z,
            catScenes=True,
            region=zimg_region,
            xRatio=x_ratio,
            yRatio=y_ratio,
            zRatio=z_ratio,
        )
    else:
        zimg = ZImg(str(image_path), region=zimg_region, xRatio=x_ratio, yRatio=y_ratio, zRatio=z_ratio)
    return zimg.data[0].copy(order="C")


def _region_2_zimg(region: ImageRegion) -> ZImgRegion:
    return ZImgRegion(
        ZVoxelCoordinate(region.x.start, region.y.start, region.z.start, 0, 0),
        ZVoxelCoordinate(region.x.end, region.y.end, region.z.end, -1, -1),
    )


def _ratio_2_dict(ratio: ResolutionRatio) -> dict[str, float]:
    return {"xRatio": ratio.x, "yRatio": ratio.y, "zRatio": ratio.z}
