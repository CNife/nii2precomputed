"""沿Z轴方向拼接多个NII文件片段
步骤：
1. 分别读取多个NII文件的元信息，确认X，Y的尺寸接近，找出最合适的X，Y尺寸，计算各自的Z轴位置
2. 根据X，Y，以及拼接后的Z轴尺寸，计算多分辨率缩放信息和base.json
3. 分别读取NII文件，缩放图片尺寸，写入多分辨率缩放尺寸、多文件的precomputed数据
"""
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorstore as ts
import typer
from numpy import dtype, ndarray
from rich.progress import Progress
from zimg import ZImg, ZImgInfo

from nii_2_precomputed import (
    Color,
    ImageSize,
    Resolution,
    build_and_write_base_json,
    build_full_resolution_info,
    convert_color,
    convert_image_data, convert_to_tensorstore_scale_metadata, open_tensorstore, read_image_info,
)
from util import console, dbg, dbg_args


def main(
    image_files_dir: Path,
    resolution: int,
    output_base_dir: Path = Path(r"C:\Workspace\splice_nii"),
    base_url: str = "http://localhost:8080",
) -> None:
    dbg_args()

    image_files = list_files_by_name(image_files_dir)
    image_infos = {
        image_path: read_image_info(image_path) for image_path in image_files
    }
    dbg(
        {
            str(image_path): (image_info.width, image_info.height, image_info.depth)
            for image_path, image_info in image_infos.items()
        },
        "image sizes",
    )
    huge_image_size, data_type, channel_colors, image_z_offsets = splice_images_size(
        image_infos
    )
    dbg(image_z_offsets, "z offsets")

    resolution = Resolution(resolution, resolution, resolution)
    info_dict = build_full_resolution_info(str(data_type), resolution, huge_image_size)
    dbg(info_dict)

    output_base_dir.mkdir(parents=True, exist_ok=True)
    build_and_write_base_json(
        channel_colors,
        resolution,
        huge_image_size,
        data_type,
        base_url,
        output_base_dir,
    )

    convert_data(image_z_offsets, huge_image_size, output_base_dir, info_dict)


def list_files_by_name(image_files_dir: Path) -> list[Path]:
    dir_items = [
        item
        for item in image_files_dir.iterdir()
        if item.is_file() and item.name.endswith((".nii", ".nii.gz"))
    ]
    if not dir_items:
        raise ValueError(f"no nii image files in {image_files_dir}")

    dir_items.sort(key=lambda path: path.name)
    dbg(dir_items, "Data files")
    return dir_items


def splice_images_size(
    image_infos: dict[Path, ZImgInfo]
) -> tuple[ImageSize, dtype, list[Color], dict[Path, int]]:
    z_offset, image_z_offsets = 0, {}
    image_x, image_y, image_dtype, image_channel_color = None, None, None, None
    for image_path, image_info in image_infos.items():

        def check_size(wanted: int, actual: int, name: str):
            if wanted is None:
                return actual
            if is_close(wanted, actual):
                return min(wanted, actual)
            raise ValueError(f"{name} isn't close, {image_path=}, {wanted=}, {actual=}")

        def check_same_value(wanted, actual, name):
            if wanted is None or wanted == actual:
                return actual
            raise ValueError(
                f"{name} isn't the same, {image_path=}, {wanted=}, {actual=}"
            )

        image_x = check_size(image_x, image_info.width, "width")
        image_y = check_size(image_y, image_info.height, "height")
        image_dtype = check_same_value(
            image_dtype, dtype(image_info.dataTypeString()), "data type"
        )
        image_channel_color = check_same_value(
            image_channel_color,
            [convert_color(c) for c in image_info.channelColors],
            "channel colors",
        )

        image_z_offsets[image_path] = z_offset
        z_offset += image_info.depth
    return (
        ImageSize(image_x, image_y, z_offset),
        image_dtype,
        image_channel_color,
        image_z_offsets,
    )


def is_close(a: int, b: int, ratio: float = 0.1) -> bool:
    a, b = min(a, b), max(a, b)
    diff = b - a
    return diff / a < ratio


def convert_data(
    image_z_offsets: dict[Path, int],
    huge_image_size: ImageSize,
    output_base_dir: Path,
    info_dict: dict[str, Any],
) -> None:
    multiscale_metadata = {
        "data_type": info_dict["data_type"],
        "num_channels": info_dict["num_channels"],
        "type": info_dict["type"],
    }
    scales = info_dict["scales"]

    with Progress(console=console) as progress:
        scales_task = progress.add_task("scales", total=len(scales))
        images_task = progress.add_task("images", total=len(image_z_offsets))

        for scale in scales:
            scale_metadata = convert_to_tensorstore_scale_metadata(scale)
            output_store = open_tensorstore("channel_0", output_base_dir, scale_metadata, multiscale_metadata)
            for image_path, z_offset in image_z_offsets.items():
                zimg_obj = ZImg(str(image_path))
                data = convert_image_data(zimg_obj.data[0][0])
                z_size = data.shape[2]
                resized_data = resize_image(data, scale['size'][0], scale['size'][1])
                output_store[ts.d['channel'][0]][ts.d['z'][z_offset:z_offset + z_size]] = resized_data
                progress.advance(images_task)

            progress.advance(scales_task)
            progress.reset(images_task)


def resize_image(image: ndarray, target_x: int, target_y: int) -> ndarray:
    return np.array(
        [cv2.resize(single_image, (target_x, target_y), interpolation=cv2.INTER_AREA) for single_image in image],
        dtype=image.dtype
    )


if __name__ == "__main__":
    typer.run(main)
