import os.path

import tensorstore as ts
from tensorstore import TensorStore
from zimg import ZImg


def main(
    src_url: str,
    out_path: str,
    out_basename: str,
    out_file_type: str,
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
    region = ts.d["channel", "x", "y", "z"][
             0, x_start:x_end, y_start:y_end, z_start:z_end
             ]
    region_data = dataset[region].read().result()

    # 写入结果
    zimg_region_data = (
        region_data.reshape((1,) + region_data.shape).transpose().copy(order="C")
    )
    result_zimg = ZImg(zimg_region_data)
    result_name = f"{out_basename}-{x_start}_{x_end}-{y_start}_{y_end}-{z_start}_{z_end}.{out_file_type}"
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
