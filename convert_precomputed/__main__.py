import itertools
import sys
from dataclasses import astuple
from pathlib import Path

import numpy as np
import tensorstore as ts
from loguru import logger
from numpy import ndarray
from typer import Argument, Option, Typer

from convert_precomputed.io_utils import check_output_directory, dump_json, list_dir
from convert_precomputed.log_utils import ChainedIndexProgress, log_time_usage
from convert_precomputed.neuroglancer_utils import build_ng_base_json
from convert_precomputed.rich_utils import print_args
from convert_precomputed.tensorstore_utils import (
    build_multiscale_metadata,
    build_scales_dyadic_pyramid,
    open_tensorstore_to_write,
    scale_resolution_ratio,
)
from convert_precomputed.types import (
    DimensionRange,
    ImageRegion,
    ImageResolution,
    JsonObject,
    TsScaleMetadata,
)
from convert_precomputed.zimg_utils import (
    get_image_dtype,
    get_image_resolution,
    get_image_size,
    read_image_data,
    read_image_info,
)

app = Typer(help="Convert images into Neuroglancer precomputed data")


@app.command(help="Convert single image to precomputed")
def single_image(
    image_path: Path = Argument(
        help="Image file path", exists=True, show_default=False
    ),
    output_directory: Path = Argument(help="Output directory", show_default=False),
    resolution: tuple[float, float, float] = Argument(
        help="resolution of x, y, z", min=0.0, default=(0.0, 0.0, 0.0)
    ),
    z_range: tuple[int, int] = Option(help="Z range, -1 means end", default=(0, -1)),
    write_block_size: int = Option(
        help="Block size when writing precomputed", default=512
    ),
) -> None:
    logger.info(f"Converting single image to precomputed: image_path={str(image_path)}, "
                f"output_directory={str(output_directory)}, {resolution=}, {z_range=}, {write_block_size=}")
    image_2_precomputed(
        image_path, output_directory, resolution, z_range, write_block_size
    )


@app.command(help="Convert multiple images on z dimension to precomputed")
def multiple_images(
    images_directory: Path = Argument(
        help="Image files directory",
        exists=True,
        dir_okay=True,
        file_okay=False,
        show_default=False,
    ),
    output_directory: Path = Argument(help="Output directory", show_default=False),
    resolution: tuple[float, float, float] = Argument(
        help="resolution of x, y, z", min=0.0, default=(0.0, 0.0, 0.0)
    ),
    z_range: tuple[int, int] = Option(help="Z range, -1 means end", default=(0, -1)),
    write_block_size: int = Option(
        help="Block size when writing precomputed", default=512
    ),
) -> None:
    logger.info(f"Converting multiple images to precomputed: images_directory={str(images_directory)}, "
                f"output_directory={str(output_directory)}, {resolution=}, {z_range=}, {write_block_size=}")
    image_2_precomputed(
        list_dir(images_directory),
        output_directory,
        resolution,
        z_range,
        write_block_size,
    )


@app.command(help="Show single image or multiple images info")
def show_image_info(path: Path = Argument(exists=True, show_default=False)) -> None:
    if path.is_dir():
        image_info = read_image_info(list_dir(path))
    else:
        image_info = read_image_info(path)
    print_args(path=str(path), image_info=str(image_info))


scale_progress = ChainedIndexProgress(None, "scale")
z_range_progress = ChainedIndexProgress(scale_progress, "z_range")
channel_progress = ChainedIndexProgress(z_range_progress, "channel")
xy_range_progress = ChainedIndexProgress(channel_progress, "xy_range")


def image_2_precomputed(
    image_path: Path | list[Path],
    output_directory: Path,
    resolution: tuple[float, float, float],
    z_range: tuple[int, int],
    write_block_size: int,
) -> None:
    url_path = check_output_directory(output_directory)
    logger.info(f"{url_path=}")

    image_info = read_image_info(image_path)
    logger.info(f"{image_info=}")

    if resolution == (0.0, 0.0, 0.0):
        resolution = get_image_resolution(image_info)
    else:
        resolution = ImageResolution(*resolution)
    logger.info(f"{resolution=}")

    size = get_image_size(image_info)
    data_type = get_image_dtype(image_info)
    z_start, z_end = z_range
    if z_end < 0:
        z_end = size.z
    logger.info(f"{z_start=}, {z_end=}")

    base_dict = build_ng_base_json(
        image_info.channelColors, resolution, size, data_type, url_path
    )
    dump_json(base_dict, output_directory / "base.json")
    logger.info(f"base_json_dict={base_dict}")
    logger.info(f"dump base.json to {str(output_directory / 'base.json')}")

    scales = build_scales_dyadic_pyramid(resolution, size)
    logger.info(f"{scales=}")
    multi_scale_metadata = build_multiscale_metadata(data_type, image_info.numChannels)
    scale_progress.count = len(scales)
    for scale in scales:
        scale_progress.next()
        convert_data(
            image_path,
            output_directory,
            resolution,
            DimensionRange(z_start, z_end),
            write_block_size,
            scale,
            multi_scale_metadata
        )


def convert_data(
    image_path: Path | list[Path],
    output_directory: Path,
    resolution: ImageResolution,
    z_range: DimensionRange,
    write_block_size: int,
    scale: TsScaleMetadata,
    multi_scale_metadata: JsonObject,
):
    ratio = scale_resolution_ratio(scale, resolution)
    read_z_size = scale["chunk_sizes"][2]
    assert z_range.start % read_z_size == 0
    read_z_ranges = calc_ranges(z_range.start, z_range.end, read_z_size)
    z_range_progress.count = len(read_z_ranges)

    for read_z_range in read_z_ranges:
        read_z_start, read_z_end = astuple(read_z_range)
        dump_json(
            {"read_z_start": read_z_start, "read_z_end": read_z_end},
            output_directory / "current_status.json",
        )
        z_range_progress.next()
        z_range_progress.description = f"{read_z_start}-{read_z_end}"

        with log_time_usage(f"{z_range_progress} read image data"):
            image_data = read_image_data(
                image_path,
                ImageRegion(
                    x=DimensionRange(0, -1),
                    y=DimensionRange(0, -1),
                    z=DimensionRange(read_z_start, read_z_end),
                ),
                ratio,
            )
        image_data = convert_image_data(image_data)
        channel_progress.count = len(image_data)

        for channel_index, channel_data in enumerate(image_data):
            channel_progress.next()
            write_tensorstore(
                channel_index,
                channel_data,
                DimensionRange(
                    (read_z_start + ratio.z - 1) // ratio.z,
                    (read_z_end + ratio.z - 1) // ratio.z,
                ),
                write_block_size,
                output_directory,
                scale,
                multi_scale_metadata,
            )


def write_tensorstore(
    channel_index: int,
    channel_data: ndarray,
    write_z_range: DimensionRange,
    write_block_size: int,
    output_directory: Path,
    scale: TsScaleMetadata,
    multi_scale_metadata: JsonObject,
):
    channel_name = f"channel_{channel_index}"
    channel_data = channel_data.transpose()
    ts_writer = open_tensorstore_to_write(
        channel_name, output_directory, scale, multi_scale_metadata
    )

    xy_ranges = list(itertools.product(calc_ranges(0, channel_data.shape[0], write_block_size),
                                       calc_ranges(0, channel_data.shape[1], write_block_size)))
    xy_range_progress.count = len(xy_ranges)
    for x_range, y_range in xy_ranges:
        xy_range_progress.next()
        xy_range_progress.description = f"({x_range.start},{y_range.start})-({x_range.end},{y_range.end})"
        write_range = ts.d["channel", "x", "y", "z"][
                      channel_index,
                      x_range.start: x_range.end,
                      y_range.start: y_range.end,
                      write_z_range.start: write_z_range.end,
                      ]
        with log_time_usage(f"{xy_range_progress} write data"):
            ts_writer[write_range] = channel_data[
                                     x_range.start: x_range.end,
                                     y_range.start: y_range.end,
                                     ]


def calc_ranges(start: int, end: int, step: int) -> list[DimensionRange]:
    return [
        DimensionRange(start, min(end, start + step))
        for start in range(start, end, step)
    ]


def convert_image_data(data: ndarray) -> ndarray:
    if data.dtype.kind != "f":
        return data
    data_max, data_min = np.max(data), np.min(data)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data.astype(np.float32)


@logger.catch
def main():
    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>|"
                      "<level>{level: <8}</level>|"
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>|"
                      "<level>{message}</level>")
    app()


if __name__ == "__main__":
    main()
