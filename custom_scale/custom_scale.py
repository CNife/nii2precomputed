import json
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
    build_base_json_dict,
    build_full_resolution_info,
    convert_image_data,
    convert_to_tensorstore_scale_metadata,
    open_tensorstore,
    read_image_info,
)
from util import console, humanize_size, pretty_print_object


def compute_new_size(image_info: ZImgInfo, target_y: int) -> ImageSize:
    origin_size = ImageSize(x=image_info.width, y=image_info.height, z=image_info.depth)
    pretty_print_object(origin_size, "origin image size")
    ratio = origin_size.y / target_y
    result = ImageSize(
        x=round(origin_size.x / ratio),
        y=target_y,
        z=round(origin_size.z / ratio),
    )
    pretty_print_object(result, "new image size")
    return result


def read_image_data(image_path: Path) -> ndarray:
    image_data_obj = ZImg(str(image_path))
    image_data = image_data_obj.data[0][0]
    console.print(f"Image size in memory: {humanize_size(image_data.nbytes)}")
    return image_data.transpose().copy()


def convert_skimage_dtype(image: ndarray, target_dtype: np.dtype) -> ndarray:
    match target_dtype:
        case np.float32:
            return img_as_float(image).asdtype(np.float32)
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
    image_path = Path(
        # r"C:\Workspace\002_slice_0020_fused_tp_0_ch_0_cropped_v2n4corr.nii.gz"
        r"C:\Workspace\stitched_0004.nii.gz"
    )
    out_folder = image_path.parent / image_path.stem
    out_folder.mkdir(parents=True, exist_ok=True)
    resolution = Resolution(1_250_000, 1_250_000, 1_250_000)

    image_info = read_image_info(image_path)
    new_size = compute_new_size(image_info, 175)
    info = build_full_resolution_info(image_info.dataTypeString(), resolution, new_size)
    pretty_print_object(info, "info")

    image_data = read_image_data(image_path)
    resized_image = resize(image_data, new_size, anti_aliasing=True)
    resized_image = convert_skimage_dtype(resized_image, image_data.dtype)

    base_json_dict = build_base_json_dict(
        image_info, resolution, new_size, resized_image.dtype, "http://localhost:8080"
    )
    pretty_print_object(base_json_dict, "base.json")
    with open(out_folder / "base.json", "w") as base_json_file:
        json.dump(base_json_dict, base_json_file, indent=2)

    write_tensorstore(info, resized_image, out_folder)


if __name__ == "__main__":
    main()
