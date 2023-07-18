# 按固定大小区块分割 precomputed 文件

## 设置环境

```shell
conda create -n workspace python=3.10
conda activate workspace
conda install -c fenglab zimg
conda install typer rich
pip install tensorstore

```

## 功能

* `convert_single_chunk.py`：转换单个区块
* `split_chunks.py`：读取 precomputed 文件信息，自动划分区块，调用 `convert_single_chunk.py`
