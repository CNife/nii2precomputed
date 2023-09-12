from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import tensorstore as ts

from convert_to_precomputed.io_utils import list_dir
from convert_to_precomputed.tensorstore_utils import (
    build_multiscale_metadata,
    build_scales_dyadic_pyramid,
    open_tensorstore_to_write,
    scale_resolution_ratio,
)
from convert_to_precomputed.my_types import DimensionRange, ImageRegion
from convert_to_precomputed.zimg_utils import (
    get_image_dtype,
    get_image_resolution,
    get_image_size,
    read_image_data,
    read_image_info,
)


def read_image(img, scale, resolution):
    print("start read image")
    data = read_image_data(
        img,
        ImageRegion(DimensionRange(0, -1), DimensionRange(0, -1), DimensionRange(0, -1)),
        scale_resolution_ratio(scale, resolution),
    )[0]
    print("end read image")
    return data


def write_tensorstore(ts_writer, data):
    print("start write image")
    ts_writer[ts.d["channel"][0]] = data.transpose()
    print("end write image")


def main():
    output_directory = Path(__file__).with_name("test-ts")
    output_directory.mkdir(parents=True, exist_ok=True)
    images = list_dir(Path(r"D:\WorkData\nii\20230608"))
    image_info = read_image_info(images)
    image_resolution = get_image_resolution(image_info)
    image_size = get_image_size(image_info)
    image_data_type = get_image_dtype(image_info)

    scales = build_scales_dyadic_pyramid(image_resolution, image_size)
    multiscale_metadata = build_multiscale_metadata(image_data_type, image_info.numChannels)

    with ProcessPoolExecutor() as executor:
        ts_writers = executor.map(
            open_tensorstore_to_write,
            repeat(f"channel_0"),
            repeat(output_directory),
            scales,
            repeat(multiscale_metadata),
        )
        datas = executor.map(read_image, repeat(images), scales, repeat(image_resolution))
        list(executor.map(write_tensorstore, ts_writers, datas))


if __name__ == "__main__":
    main()
    print("DONE")
