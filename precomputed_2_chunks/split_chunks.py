import itertools
from typing import Iterable

import typer
from rich.progress import track

from precomputed_2_chunks import convert_single_chunk
from precomputed_2_chunks.convert_single_chunk import open_tensorstore_to_read


def main(
    src_url: str, out_path: str, out_file_type: str = "tiff", block_size: int = 512
) -> None:
    ts = open_tensorstore_to_read(src_url)

    # 简单起见，只处理只有一个channel的数据
    assert ts.shape[3] == 1
    x_max, y_max, z_max = ts.shape[:3]

    block_ranges = list(itertools.product(
        chunks(x_max, block_size), chunks(y_max, block_size), chunks(z_max, block_size)
    ))
    for index, ranges in track(
        enumerate(block_ranges), description="Converting", total=len(block_ranges)
    ):
        x_range, y_range, z_range = ranges
        convert_single_chunk.main(
            src_url,
            out_path,
            str(index).zfill(len(str(len(block_ranges)))),
            out_file_type,
            x_start=x_range[0],
            x_end=x_range[1],
            y_start=y_range[0],
            y_end=y_range[1],
            z_start=z_range[0],
            z_end=z_range[1],
        )


def chunks(end: int, step: int) -> Iterable[tuple[int, int]]:
    for start in range(0, end, step):
        yield start, min(end, start + step)


if __name__ == "__main__":
    typer.run(main)
