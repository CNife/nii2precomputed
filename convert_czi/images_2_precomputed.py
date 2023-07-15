import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts
import typer
from numpy import ndarray
from zimg import Dimension, ZImg

from nii_2_precomputed import (
    Resolution,
    build_and_write_base_json,
    build_full_resolution_info,
    convert_image_data,
    convert_to_tensorstore_scale_metadata,
    get_image_size,
    get_resolution,
    open_tensorstore,
    read_image_info,
)
from util import console, dbg_args, humanize_size, ranges


def main(
    src_dir: Path,
    out_dir: Path,
    start_z: int = 0,
    end_z: int = -1,
    start_scale_index: int = 0,
    write_batch_size_x: int = 5120,
    write_batch_size_y: int = 5120,
    url_path: str = "http://localhost:8080",
) -> None:
    dbg_args()

    src_files = [str(file.absolute()) for file in src_dir.iterdir() if file.is_file()]
    src_files.sort()
    if end_z < 0:
        end_z = len(src_files)

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
    with open(out_dir / "full_resolutions_info.json", "w") as f:
        json.dump(info, f, indent=2)
    multiscale_metadata = {
        "data_type": info["data_type"],
        "num_channels": info["num_channels"],
        "type": info["type"],
    }
    scales = info["scales"][start_scale_index:]

    convert_to_precomputed(
        src_files,
        start_z,
        end_z,
        multiscale_metadata,
        scales,
        out_dir,
        resolution,
        write_batch_size_x,
        write_batch_size_y,
    )

    console.log("DONE")


def convert_to_precomputed(
    image_paths: list[str],
    start_z: int,
    end_z: int,
    multiscale_metadata: dict[str, Any],
    scales: list[dict[str, Any]],
    out_dir: Path,
    resolution: Resolution,
    scaled_batch_size_x: int,
    scaled_batch_size_y: int,
) -> None:
    for scale in scales:
        scaled_resolution = Resolution(*scale["resolution"])
        x_ratio = round(scaled_resolution.x / resolution.x)
        y_ratio = round(scaled_resolution.y / resolution.y)
        z_ratio = round(scaled_resolution.z / resolution.z)

        scale_metadata = convert_to_tensorstore_scale_metadata(scale)
        output_store = open_tensorstore(
            f"channel_0",
            out_dir,
            scale_metadata,
            multiscale_metadata,
        )

        batch_size = scale["chunk_sizes"][0][2]
        assert start_z % batch_size == 0
        z_ranges = [
            (batch_size * i, min(end_z, batch_size * (i + 1)))
            for i in range(
                start_z // batch_size, (end_z + batch_size - 1) // batch_size
            )
        ]
        for z_start, z_end in z_ranges:
            with open(out_dir / "current_working_status.json", "w") as f:
                json.dump(
                    {"scale": scale, "z_start": z_start, "z_end": z_end}, f, indent=2
                )

            zimg_reader = ZImg(
                filenames=image_paths[z_start:z_end],
                catDim=Dimension.Z,
                catScenes=False,
                xRatio=x_ratio,
                yRatio=y_ratio,
                zRatio=z_ratio,
            )
            image_data: ndarray = zimg_reader.data[0][0].transpose()
            console.log(
                f"SCALE{(x_ratio, y_ratio, z_ratio)}: read data in z range {(z_start, z_end)} in memory: {humanize_size(image_data.nbytes)}"
            )
            image_data = convert_image_data(image_data)

            target_z_start = (z_start + z_ratio - 1) // z_ratio
            target_z_end = (z_end + z_ratio - 1) // z_ratio
            for target_x_start, target_x_end in ranges(
                0, image_data.shape[0], scaled_batch_size_x
            ):
                for target_y_start, target_y_end in ranges(
                    0, image_data.shape[1], scaled_batch_size_y
                ):
                    output_store[
                        ts.d["x", "y", "z", "channel"][
                            target_x_start:target_x_end,
                            target_y_start:target_y_end,
                            target_z_start:target_z_end,
                            0,
                        ]
                    ] = image_data[
                        target_x_start:target_x_end, target_y_start:target_y_end
                    ]
                    console.log(
                        f"SCALE{(x_ratio, y_ratio, z_ratio)}: write data from "
                        f"[{target_x_start}, {target_y_start}, {target_z_start}] to "
                        f"({target_x_end}, {target_y_end}, {target_z_end})"
                    )


if __name__ == "__main__":
    typer.run(main)
