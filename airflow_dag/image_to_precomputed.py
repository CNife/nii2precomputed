"""
常用图像转换为 precomputed 的 airflow dag

参数
image_path: str, 图像路径
output_directory: str, 输出路径
resolution: float[3], 分辨率，默认从图像中读取，图像中没有的话按1um处理
scale_indexes: list[int], 分辨率缩放索引，默认所有分辨率
write_block_size: int, 一次性写入的块的大小，默认 512
base_path: str, 主路径，必须是输出路径的父目录，用于 base.json 中 source 字段
base_url: str, base.json 中 source 字段的基础部分

步骤
1. 检查图像和输出路径
2. 读取图像信息，
3. 生成 base.json
4. 分割任务
5. 子任务读取并写入数据
"""
