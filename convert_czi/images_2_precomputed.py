from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts
import typer
from numpy import ndarray
from zimg import Dimension, ZImg, ZImgRegion, ZVoxelCoordinate

from nii_2_precomputed import (
    ImageSize, Resolution, build_and_write_base_json,
    build_full_resolution_info,
    convert_image_data, convert_to_tensorstore_scale_metadata, get_image_size,
    get_resolution,
    open_tensorstore, read_image_info,
)
from util import console, dbg_args, humanize_size


def main(
    src_dir: Path,
    out_dir: Path,
    # batch_size: int,
    url_path: str = "http://localhost:8080",
) -> None:
    dbg_args()

    src_files = [str(file.absolute()) for file in src_dir.iterdir() if file.is_file()]
    src_files.sort()

    image_info = read_image_info(src_files)
    resolution = get_resolution(image_info)
    image_size = get_image_size(image_info)

    build_and_write_base_json(
        image_info.channelColors,
        resolution,
        image_size,
        np.dtype(image_info.dataTypeString()),
        url_path,
        out_dir,
    )

    info = build_full_resolution_info(
        image_info.dataTypeString(), resolution, image_size
    )
    # for i, files_chunk in enumerate(chunks(src_files, batch_size)):
    #     convert_image(
    #         files_chunk,
    #         info,
    #         out_dir,
    #         resolution,
    #         i * batch_size,
    #     )
    convert_to_precomputed(src_files, info, out_dir, resolution, image_size)


def convert_to_precomputed(
    image_paths: list[str],
    info: dict[str, Any],
    out_dir: Path,
    resolution: Resolution,
    image_size: ImageSize,
):
    multiscale_metadata = {
        "data_type": info["data_type"],
        "num_channels": info["num_channels"],
        "type": info["type"],
    }

    for scale in info["scales"]:
        scaled_resolution = Resolution(*scale["resolution"])
        x_ratio = round(scaled_resolution.x / resolution.x)
        y_ratio = round(scaled_resolution.y / resolution.y)
        z_ratio = round(scaled_resolution.z / resolution.z)
        batch_size = scale['chunk_sizes'][0][2] * 2
        z_ranges = [(i * batch_size, min(image_size.z, batch_size * (i + 1))) for i in
                    range((image_size.z + batch_size - 1) // batch_size)]
        for z_start, z_end in z_ranges:
            zimg_reader = ZImg(
                filenames=image_paths,
                catDim=Dimension.Z,
                catScenes=False,
                region=ZImgRegion(ZVoxelCoordinate(0, 0, z_start, 0, 0), ZVoxelCoordinate(-1, -1, z_end, -1, -1)),
                xRatio=x_ratio,
                yRatio=y_ratio,
                zRatio=z_ratio,
            )
            image_data: ndarray = zimg_reader.data[0][0]
            console.log(f"SCALE{(x_ratio, y_ratio, z_ratio)}: read [cyan]{humanize_size(image_data.nbytes)}[/] data")
            image_data = convert_image_data(image_data)

            scale_metadata = convert_to_tensorstore_scale_metadata(scale)
            target_z_start, target_z_end = (z_start + z_ratio - 1) // z_ratio, (z_end + z_ratio - 1) // z_ratio
            output_store = open_tensorstore(
                f"channel_0",
                out_dir,
                scale_metadata,
                multiscale_metadata,
            )
            output_store[ts.d['channel'][0]][ts.d['z'][target_z_start:target_z_end]] = image_data.transpose()
            console.log(
                f"SCALE{(x_ratio, y_ratio, z_ratio)}: write data [{z_start}:{z_end}) -> [{target_z_start}:{target_z_end})")


if __name__ == "__main__":
    typer.run(main)
