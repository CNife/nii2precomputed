import os
import sys
from pathlib import Path

from loguru import logger
from typer import Argument, Option, Typer

from convert_to_precomputed.convert import LOG_FORMAT, build_ng_base_json, convert_single_scale, image_2_precomputed
from convert_to_precomputed.io_utils import check_output_directory, dump_json, list_dir
from convert_to_precomputed.tensorstore_utils import build_multiscale_metadata_v2, build_scales_dyadic_pyramid
from convert_to_precomputed.types import ConvertSpec, DimensionRange, ImageResolution, ResolutionPM, ScaleMetadata
from convert_to_precomputed.zimg_utils import (
    get_image_dtype,
    get_image_resolution,
    get_image_resolution_v2,
    get_image_size,
    get_image_size_v2,
    read_image_info,
    read_image_info_v2,
)

URL: str = str(os.environ.get("URL", "http://10.11.40.170:2000"))
BASE_PATH: Path = Path(os.environ.get("BASE_PATH", "/zjbs-data/share"))

app = Typer(help="Convert images into Neuroglancer precomputed data")


@app.command(help="Convert multiple images on z dimension to precomputed")
def convert(
    image_path: Path = Argument(
        help="Image file or files directory", exists=True, dir_okay=True, file_okay=True, show_default=False
    ),
    output_directory: Path = Argument(help="Output directory", show_default=False),
    resolution: tuple[float, float, float] = Argument(help="resolution of x, y, z", min=0.0, default=(0.0, 0.0, 0.0)),
    z_range: tuple[int, int] = Option(help="Z range, -1 means end", default=(0, -1)),
    write_block_size: int = Option(help="Block size when writing precomputed", default=512),
    resume: bool = Option(help="Resume from output_directory/work_progress.json", default=True),
    base_url: str = Option(help="Base url in base.json", default="http://10.11.40.170:2000"),
    base_path: Path = Option(help="Base path, must be parent of output directory", default=Path("/zjbs-data/share")),
) -> None:
    logger.info(
        f"Converting image to precomputed: "
        f"image_path={str(image_path)},output_directory={str(output_directory)},{resolution=},{z_range=},"
        f"{write_block_size=},{resume=}"
    )
    image_2_precomputed(
        image_path, output_directory, resolution, z_range, write_block_size, resume, base_url, base_path
    )


@app.command(help="Show single image or multiple images info")
def show_info(path: Path = Argument(exists=True, show_default=False)) -> None:
    if path.is_dir():
        image_info = read_image_info(list_dir(path))
    else:
        image_info = read_image_info(path)
    logger.info(f"path={str(path)}")
    logger.info(f"{image_info=}")


@app.command(help="Generate base.json for image")
def gen_base_json(
    image_path: Path = Argument(
        help="Image file or files directory", exists=True, dir_okay=True, file_okay=True, show_default=False
    ),
    output_directory: Path = Argument(help="Output directory", show_default=False),
    resolution: str = Argument(help="resolution of x, y, z", default="0.0,0.0,0.0"),
    base_path: Path = Option(help="Base path, must be parent of output directory", default=Path("/zjbs-data/share")),
    base_url: str = Option(help="Base url in base.json", default="http://10.11.40.170:2000"),
) -> None:
    image_info = read_image_info_v2(image_path)
    logger.info(f"{image_info=}")

    resolution = [float(r) for r in resolution.split(",")]
    if resolution == [0.0, 0.0, 0.0]:
        resolution = get_image_resolution(image_info)
    else:
        resolution = ImageResolution(*resolution)
    logger.info(f"{resolution=}")

    size = get_image_size(image_info)
    data_type = get_image_dtype(image_info)
    url_path = check_output_directory(output_directory, base_path)

    base_dict = build_ng_base_json(image_info.channelColors, resolution, size, data_type, url_path, base_url)
    output_directory.mkdir(parents=True, exist_ok=True)
    dump_json(base_dict, output_directory / "base.json")


@app.command(help="Generate specification for image")
def gen_spec(
    image_path: Path = Argument(
        help="Image file or files directory", exists=True, dir_okay=True, file_okay=True, show_default=False
    ),
    output_directory: Path = Argument(help="Output directory", show_default=False),
    resolution: str = Argument(help="resolution of x, y, z", default="0.0,0.0,0.0"),
    write_block_size: int = Option(help="Block size when writing precomputed", default=512),
) -> None:
    image_info = read_image_info_v2(image_path)
    resolution = [float(r) for r in resolution.split(",")]
    if resolution == [0.0, 0.0, 0.0]:
        resolution = get_image_resolution_v2(image_info)
    else:
        resolution = ResolutionPM(x=resolution[0], y=resolution[1], z=resolution[2])

    scales = [
        ScaleMetadata.model_validate(scale)
        for scale in build_scales_dyadic_pyramid(resolution, get_image_size(image_info))
    ]
    multiscale_metadata = build_multiscale_metadata_v2(image_info.dataTypeString(), image_info.numChannels)
    spec = ConvertSpec(
        image_path=str(image_path),
        output_directory=str(output_directory),
        resolution=resolution,
        size=get_image_size_v2(image_info),
        write_block_size=write_block_size,
        multiscale=multiscale_metadata,
        scales=scales,
    )
    logger.info(f"{spec=}")
    spec_json = spec.model_dump_json(indent=2, by_alias=True)
    (output_directory / "spec.json").write_text(spec_json)


@app.command(help="Convert single scale for image")
def convert_scale(
    spec_path: Path = Argument(
        help="Specification for converting image", exists=True, file_okay=True, dir_okay=False, show_default=False
    ),
    scale_index: int = Argument(help="The scale to be converted in spec file", show_default=False),
) -> None:
    spec = ConvertSpec.model_validate_json(spec_path.read_text())
    convert_single_scale(
        image_path=Path(spec.image_path),
        output_directory=Path(spec.output_directory),
        resolution=spec.resolution,
        z_range=DimensionRange(start=0, end=spec.size.z),
        write_block_size=spec.write_block_size,
        scale=spec.scales[scale_index].model_dump(by_alias=True),
        multi_scale_metadata=spec.multiscale.model_dump(),
        scale_progress=None,
        write_status_json=False,
    )


@logger.catch
def main():
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT)
    app()


if __name__ == "__main__":
    main()
