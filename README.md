# PySIFNE

SIFNE 核心微管丝重建与分析的 Python 实现，支持 CPU 和 Numba CUDA，无需 MATLAB。

## 安装

```sh
git clone https://github.com/angela-nuaa/pysifne.git
cd pysifne
conda env create -f environment.yml
conda activate pysifne
```

已有 Conda 环境可直接运行 `python -m pip install -e ".[accel]"`。
CUDA 需要 NVIDIA GPU、驱动及兼容的 CUDA 运行环境。

## 使用

```python
from pysifne import analyze, export_matlab_compatibility

result = analyze("image.tif")
print(result.analysis_info)
export_matlab_compatibility(result, "output")
```

`analyze` 自动选择 backend；`analyze_mt(..., backend="reference")` 可显式选择 CPU，
`backend="cuda"` 选择 GPU。长度默认以像素计，物理长度使用 `result.lengths_um(pixel_size_um=...)`。

覆盖 LFT/OFT、分割、骨架、片段配对、微管丝重建、排序和基础统计。
不包含 GUI、Excel、完整曲率分析及最终交点合并；`NewCrPts.mat` 为分组前交点。
Linux CPU 与冻结 MATLAB 参考可能存在浮点末位差异。

测试：`python -m pytest -q tests`。

基于 [SIFNE](https://github.com/qiqi-lu/fluoresfm/tree/main/methods/SIFNE)，
原作者 Zhen Zhang、Pakorn Kanchanawong。[BSD-2-Clause](LICENSE)。
