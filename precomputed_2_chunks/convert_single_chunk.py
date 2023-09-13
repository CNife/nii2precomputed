import os.path

import numpy as np
import tensorstore as ts
from tensorstore import TensorStore
from zimg import ZImg


def main(
    src_url: str,
    out_path: str,
    out_basename: str,
    out_file_type: str,
    block_size: int,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
    z_start: int,
    z_end: int,
) -> str:
    # 打开TensorStore
    dataset = open_tensorstore_to_read(src_url)

    # 读取指定区域数据
    region = ts.d["channel", "x", "y", "z"][:1, x_start:x_end, y_start:y_end, z_start:z_end]
    region_data = dataset[region].read().result()

    # 填充到指定block_size大小
    target_shape = (block_size,) * 3 + (1,)
    zimg_region_data = np.zeros(target_shape, dtype=region_data.dtype, order="C")
    zimg_region_data[
        : region_data.shape[3], : region_data.shape[2], : region_data.shape[1], : region_data.shape[0]
    ] = region_data.transpose()

    # 写入结果
    result_zimg = ZImg(zimg_region_data)
    result_name = f"{out_basename}_{x_start}-{x_end}_{y_start}-{y_end}_{z_start}-{z_end}.{out_file_type}"
    result_path = os.path.join(out_path, result_name)
    result_zimg.save(result_path)

    return result_path


def open_tensorstore_to_read(src_url: str) -> TensorStore:
    ts_spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "http", "base_url": src_url},
        "scale_index": 0,
    }
    dataset = ts.open(ts_spec, read=True, write=False, create=False).result()
    return dataset


if __name__ == "__main__":
    import typer

    typer.run(main)
