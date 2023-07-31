from typing import Iterable

import numpy as np
from numpy import ndarray
from zimg import Dimension, VoxelSizeUnit, ZImg, ZImgInfo, ZImgRegion, ZVoxelCoordinate

from convert_precomputed.types import (
    ImageRegion,
    ImageResolution,
    ImageSize,
    OsPath,
    ResolutionRatio,
)


# noinspection PyTypeChecker
def read_image_info(image_path: OsPath | Iterable[OsPath]) -> ZImgInfo:
    if isinstance(image_path, Iterable):
        image_paths = [str(path) for path in image_path]
        image_infos = ZImg.readImgInfos(
            image_paths,
            catDim=Dimension.Z,
            catScenes=True,
        )
    else:
        image_infos = ZImg.readImgInfos(str(image_path))
    result: ZImgInfo = image_infos[0]
    return result


def get_image_size(image_info: ZImgInfo) -> ImageSize:
    return ImageSize(x=image_info.width, y=image_info.height, z=image_info.depth)


def get_image_resolution(image_info: ZImgInfo) -> ImageResolution:
    scale = _unit_scale(image_info.voxelSizeUnit)
    return ImageResolution(
        x=image_info.voxelSizeX * scale,
        y=image_info.voxelSizeY * scale,
        z=image_info.voxelSizeZ * scale,
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


def read_image_data(
    image_path: OsPath | Iterable[OsPath],
    region: ImageRegion,
    ratio: ResolutionRatio,
) -> ndarray:
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


def _region_2_zimg(region: ImageRegion) -> ZImgRegion:
    return ZImgRegion(
        ZVoxelCoordinate(region.x.start, region.y.start, region.z.start, 0, 0),
        ZVoxelCoordinate(region.x.end, region.y.end, region.z.end, -1, -1),
    )


def _ratio_2_dict(ratio: ResolutionRatio) -> dict[str, float]:
    return {
        "xRatio": ratio.x,
        "yRatio": ratio.y,
        "zRatio": ratio.z,
    }
