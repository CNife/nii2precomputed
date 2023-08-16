import itertools
import time
from contextlib import contextmanager
from dataclasses import astuple
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorstore as ts
from loguru import logger
from numpy import ndarray
from zimg import col4

from convert_to_precomputed.chained_progress import ChainedProgress
from convert_to_precomputed.io_utils import check_output_directory, dump_json
from convert_to_precomputed.tensorstore_utils import (
    build_multiscale_metadata,
    build_scales_dyadic_pyramid,
    open_tensorstore_to_write,
    scale_resolution_ratio,
)
from convert_to_precomputed.types import (
    DimensionRange,
    ImageResolution,
    ImageSize,
    JsonObject,
    ResolutionPM,
    TsScaleMetadata,
)
from convert_to_precomputed.zimg_utils import (
    get_image_dtype,
    get_image_resolution,
    get_image_size,
    read_image_data_v2,
    read_image_info_v2,
)

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>|"
    "<level>{level}</level>|"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>|"
    "<level>{message}</level>"
)


def image_2_precomputed(
    image_path: Path | list[Path],
    output_directory: Path,
    resolution: tuple[float, float, float],
    z_range: tuple[int, int],
    write_block_size: int,
    resume: bool,
    base_url: str,
    base_path: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = output_directory / "convert_to_precomputed.log"
    logger.add(log_path, format=LOG_FORMAT)
    scale_progress = load_work_progress(resume, output_directory)

    url_path = check_output_directory(output_directory, base_path)
    logger.info(f"{url_path=}")

    image_info = read_image_info_v2(image_path)
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

    base_dict = build_ng_base_json(image_info.channelColors, resolution, size, data_type, url_path, base_url)
    dump_json(base_dict, output_directory / "base.json")
    logger.info(f"base_json_dict={base_dict}")
    logger.info(f"dump base.json to {str(output_directory / 'base.json')}")

    scales = build_scales_dyadic_pyramid(resolution, size)
    logger.info(f"{scales=}")
    multi_scale_metadata = build_multiscale_metadata(data_type, image_info.numChannels)
    for scale in scale_progress.bind(scales):
        convert_single_scale(
            image_path,
            output_directory,
            resolution,
            DimensionRange(z_start, z_end),
            write_block_size,
            scale,
            multi_scale_metadata,
            scale_progress,
            True,
        )
    logger.info("DONE")


def build_ng_base_json(
    channel_colors: list[col4],
    resolution: ImageResolution,
    size: ImageSize,
    dtype: np.dtype,
    url_path: str,
    base_url: str,
) -> JsonObject:
    return {
        "dimensions": {
            "x": [resolution.x * 1e-9, "m"],
            "y": [resolution.y * 1e-9, "m"],
            "z": [resolution.z * 1e-9, "m"],
        },
        "position": [dimension_size / 2 for dimension_size in astuple(size)],
        "layout": "4panel",
        "layer": [
            {
                "type": "image",
                "name": f"channel_{channel}",
                "source": f"precomputed://{base_url}/{url_path}/channel_{channel}",
                "opacity": 1,
                "blend": "additive",
                "shaderControls": {
                    "color": f"#{channel_color.r:02x}{channel_color.g:02x}{channel_color.b:02x}",
                    "normalized": {
                        "range": [0.0, 1.0] if dtype.kind == "f" else [np.iinfo(dtype).min, np.iinfo(dtype).max]
                    },
                },
            }
            for channel, channel_color in enumerate(channel_colors)
        ],
    }


def convert_single_scale(
    image_path: Path,
    output_directory: Path,
    resolution: ImageResolution | ResolutionPM,
    z_range: DimensionRange,
    write_block_size: int,
    scale: TsScaleMetadata,
    multi_scale_metadata: JsonObject,
    scale_progress: ChainedProgress | None,
    write_status_json: bool,
):
    ratio = scale_resolution_ratio(scale, resolution)
    read_z_size = scale["chunk_sizes"][2]
    assert z_range.start % read_z_size == 0
    read_z_ranges = calc_ranges(z_range.start, z_range.end, read_z_size)
    if scale_progress is None:
        z_range_progress = ChainedProgress("z_range", None)
    else:
        z_range_progress = scale_progress.get_or_add("z_range")
    for read_z_range in z_range_progress.bind(read_z_ranges, lambda zr: f"{zr.start}-{zr.end}"):
        read_z_start, read_z_end = astuple(read_z_range)

        with log_time_usage(f"{z_range_progress} read image data"):
            image_data = read_image_data_v2(
                image_path, 0, -1, 0, -1, read_z_start, read_z_end, ratio.x, ratio.y, ratio.z
            )
        image_data = convert_image_data(image_data)

        channel_progress = z_range_progress.get_or_add("channel")
        for channel_index, channel_data in channel_progress.bind(list(enumerate(image_data))):
            write_tensorstore(
                channel_index,
                channel_data,
                (read_z_start + ratio.z - 1) // ratio.z,
                (read_z_end + ratio.z - 1) // ratio.z,
                write_block_size,
                output_directory,
                scale,
                multi_scale_metadata,
                channel_progress,
                write_status_json=write_status_json,
            )


def write_tensorstore(
    channel_index: int,
    channel_data: ndarray,
    write_z_start: int,
    write_z_end: int,
    write_block_size: int,
    output_directory: Path,
    scale: TsScaleMetadata,
    multi_scale_metadata: JsonObject,
    channel_progress: ChainedProgress,
    write_status_json: bool,
):
    channel_name = f"channel_{channel_index}"
    channel_data = channel_data.transpose()
    ts_writer = open_tensorstore_to_write(channel_name, output_directory, scale, multi_scale_metadata)

    xy_range_progress = channel_progress.get_or_add("xy_range")
    for x_range, y_range in xy_range_progress.bind(
        list(
            itertools.product(
                calc_ranges(0, channel_data.shape[0], write_block_size),
                calc_ranges(0, channel_data.shape[1], write_block_size),
            )
        ),
        lambda xyr: f"({xyr[0].start},{xyr[1].start})-({xyr[0].end},{xyr[1].end})",
    ):
        write_range = ts.d["channel", "x", "y", "z"][
            channel_index, x_range.start : x_range.end, y_range.start : y_range.end, write_z_start:write_z_end
        ]
        if write_status_json:
            xy_range_progress.save(output_directory / "work_status.json")
        with log_time_usage(f"{xy_range_progress} write data"):
            ts_writer[write_range] = channel_data[x_range.start : x_range.end, y_range.start : y_range.end]


def load_work_progress(resume: bool, output_directory: Path) -> ChainedProgress:
    if resume and (work_status_path := output_directory / "work_status.json").exists():
        return ChainedProgress.load(work_status_path)
    return ChainedProgress("scale", None)


def calc_ranges(start: int, end: int, step: int) -> list[DimensionRange]:
    return [DimensionRange(start, min(end, start + step)) for start in range(start, end, step)]


def convert_image_data(data: ndarray) -> ndarray:
    if data.dtype.kind != "f":
        return data
    data_max, data_min = np.max(data), np.min(data)
    normalized_data = (data - data_min) / (data_max - data_min)
    return normalized_data.astype(np.float32)


@contextmanager
def log_time_usage(description: str) -> Iterable[None]:
    start_time = time.perf_counter_ns()
    try:
        yield
    finally:
        time_diff_ns = time.perf_counter_ns() - start_time
        used_time = timedelta(microseconds=time_diff_ns / 1000)
        logger.info(f"[used {str(used_time)}] {description}")
