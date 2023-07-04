import math
from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts
from numpy import ndarray
from rich.progress import track
from skimage.transform import downscale_local_mean, resize
from skimage.util import img_as_float, img_as_ubyte, img_as_uint
from zimg import ZImg, ZImgInfo

from nii_2_precomputed import (
    ImageSize,
    Resolution,
    build_and_write_base_json,
    build_full_resolution_info,
    convert_image_data,
    convert_to_tensorstore_scale_metadata,
    open_tensorstore,
    read_image_info,
)
from util import console, dbg, humanize_size


def compute_new_size(image_info: ZImgInfo, target_y: int) -> tuple[ImageSize, int]:
    origin_size = ImageSize(x=image_info.width, y=image_info.height, z=image_info.depth)
    dbg(origin_size, "origin image size")
    ratio = origin_size.y / target_y
    result = ImageSize(
        x=round(origin_size.x / ratio),
        y=target_y,
        z=round(origin_size.z / ratio),
    )
    dbg(result, "new image size")
    return result, math.floor(ratio)


def read_image_data(image_path: Path, read_ratio: int) -> ndarray:
    image_data_obj = ZImg(
        str(image_path), xRatio=read_ratio, yRatio=read_ratio, zRatio=read_ratio
    )
    image_data = image_data_obj.data[0][0]
    console.print(f"Image size in memory: {humanize_size(image_data.nbytes)}")
    return image_data.transpose().copy()


def convert_skimage_dtype(image: ndarray, target_dtype: np.dtype) -> ndarray:
    match target_dtype:
        case np.float32:
            return img_as_float(image).astype(np.float32)
        case np.float64:
            return img_as_float(image)
        case np.uint8:
            return img_as_ubyte(image)
        case np.uint16:
            return img_as_uint(image)
        case _:
            raise ValueError("invalid target_dtype")


def write_tensorstore(
    info: dict[str, Any], image_data: ndarray, out_folder: Path
) -> None:
    multiscale_metadata = {
        "data_type": info["data_type"],
        "num_channels": info["num_channels"],
        "type": info["type"],
    }
    origin_resolution = info["scales"][0]["resolution"][0]
    for scale in track(info["scales"]):
        scale = convert_to_tensorstore_scale_metadata(scale)
        output_store = open_tensorstore(
            "channel_0", out_folder, scale, multiscale_metadata
        )
        ratio = round(scale["resolution"][0] / origin_resolution)
        if ratio > 1:
            rescaled_image = convert_image_data(downscale_local_mean(image_data, ratio))
            rescaled_image = convert_skimage_dtype(rescaled_image, image_data.dtype)
        else:
            rescaled_image = image_data
        output_store[ts.d["channel"][0]] = rescaled_image


def main():
    image_path = Path(r"D:\EEG Data\nii\20230530\full16_100um_2009b_sym.nii.gz")
    out_folder = image_path.parent / image_path.stem
    out_folder.mkdir(parents=True, exist_ok=True)
    resolution = Resolution(1_250_000, 1_250_000, 1_250_000)

    image_info = read_image_info(image_path)
    new_size, read_ratio = compute_new_size(image_info, 175)
    info = build_full_resolution_info(image_info.dataTypeString(), resolution, new_size)
    dbg(info, "info")

    image_data = read_image_data(image_path, read_ratio)
    resized_image = resize(image_data, new_size, anti_aliasing=True)
    resized_image = convert_skimage_dtype(resized_image, image_data.dtype)

    build_and_write_base_json(
        image_info.channelColors,
        resolution,
        new_size,
        resized_image.dtype,
        "http://localhost:8080",
        out_folder,
    )

    write_tensorstore(info, resized_image, out_folder)


if __name__ == "__main__":
    main()
