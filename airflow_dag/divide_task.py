from pathlib import Path

import numpy as np
import typer

from airflow_dag.spec import ConvertSpec, ImageChannel, ImageInfo
from convert_to_precomputed.convert import build_ng_base_json
from convert_to_precomputed.io_utils import check_output_directory, dump_json, list_dir
from convert_to_precomputed.tensorstore_utils import build_multiscale_metadata, build_scales_dyadic_pyramid
from convert_to_precomputed.zimg_utils import get_image_resolution, get_image_size, read_image_info


def main(
    image_path: str,
    output_directory: str,
    resolution: str,
    scale_indexes: str,
    write_block_size: str,
    base_path: str,
    base_url: str,
) -> None:
    image_path, output_directory = Path(image_path), Path(output_directory)
    if image_path.is_dir():
        image_path = list_dir(image_path)
    if resolution:
        resolution = [float(s) for s in resolution.split(",")]
    scale_indexes = [int(s) for s in scale_indexes.split(",")]
    write_block_size = int(write_block_size)
    base_path = Path(base_path)
    url_path = check_output_directory(output_directory, base_path)

    zimg_info = read_image_info(image_path)
    image_info = ImageInfo(
        data_type=zimg_info.dataTypeString(),
        resolution=get_image_resolution(zimg_info) if not isinstance(resolution, list) else resolution,
        size=get_image_size(zimg_info),
        channels=[
            ImageChannel(name, f"#{color.r:02x}{color.g:02x}{color.b:02x}")
            for name, color in zip(zimg_info.channelNames, zimg_info.channelColors)
        ],
    )
    scales = build_scales_dyadic_pyramid(image_info.resolution, image_info.size)
    multiscale_metadata = build_multiscale_metadata(np.dtype(image_info.data_type), len(image_info.channels))
    base_json = build_ng_base_json(
        zimg_info.channelColors,
        image_info.resolution,
        image_info.size,
        np.dtype(image_info.data_type),
        base_url,
        url_path,
    )
    dump_json(base_json, output_directory / "base.json")

    spec = ConvertSpec(
        image_path=str(image_path) if isinstance(image_path, Path) else [str(p) for p in image_path],
        output_directory=str(output_directory),
        resolution=resolution,
        scale_indexes=scale_indexes,
        write_block_size=write_block_size,
        base_path=str(base_path),
        base_url=base_url,
        image_info=image_info,
        scales=scales,
        multiscale=multiscale_metadata,
        base_json=base_json,
    )
    dump_json(spec, output_directory / "spec.json")


if __name__ == "__main__":
    typer.run(main)
