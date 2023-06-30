"""沿Z轴方向拼接多个NII文件片段
步骤：
1. 分别读取多个NII文件的元信息，确认X，Y的尺寸相同，计算各自的Z轴位置
2. 根据X，Y，以及拼接后的Z轴尺寸，计算多分辨率缩放信息和base.json
3. 分别写入多分辨率缩放尺寸、多文件的precomputed数据
"""
from pathlib import Path

import typer
from numpy import dtype
from zimg import ZImgInfo

from nii_2_precomputed import (
    Color, ImageSize,
    Resolution,
    build_and_write_base_json, build_full_resolution_info,
    convert_color, read_image_info,
)
from util import dbg, dbg_args


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
    huge_image_size, data_type, channel_colors, image_z_offsets = splice_images_size(image_infos)
    dbg(image_z_offsets, "z offsets")

    resolution = Resolution(resolution, resolution, resolution)
    info_dict = build_full_resolution_info(str(data_type), resolution, huge_image_size)
    dbg(info_dict)

    output_base_dir.mkdir(parents=True, exist_ok=True)
    build_and_write_base_json(channel_colors, resolution, huge_image_size, data_type, base_url, output_base_dir)


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
            image_channel_color, [convert_color(c) for c in image_info.channelColors], "channel colors"
        )

        image_z_offsets[image_path] = z_offset
        z_offset += image_info.depth
    return ImageSize(image_x, image_y, z_offset), image_dtype, image_channel_color, image_z_offsets


def is_close(a: int, b: int, ratio: float = 0.1) -> bool:
    a, b = min(a, b), max(a, b)
    diff = b - a
    return diff / a < ratio


if __name__ == "__main__":
    typer.run(main)
