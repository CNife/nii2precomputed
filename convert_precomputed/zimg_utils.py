from typing import Iterable

import numpy as np
from zimg import Dimension, VoxelSizeUnit, ZImg, ZImgInfo

from convert_precomputed.types import ImageResolution, ImageSize, OsPath


# noinspection PyTypeChecker
def read_image_info(image_path: OsPath | Iterable[OsPath]) -> ZImgInfo:
    if isinstance(image_path, OsPath):
        image_infos = ZImg.readImgInfos(str(image_path))
    else:
        image_paths = [str(path) for path in image_path]
        image_infos = ZImg.readImgInfos(
            image_paths,
            catDim=Dimension.Z,
            catScenes=True,
        )
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


def get_image_dtype(image_info: ZImgInfo) -> np.dtype:
    return np.dtype(image_info.dataTypeString())


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
