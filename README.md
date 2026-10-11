# PySIFNE

不依赖 MATLAB 的 SIFNE 核心微管丝重建与分析实现，提供可读的 CPU reference 路径、
自行编写的 Numba CUDA Kernel，以及 MATLAB 冻结样例回归测试。

## 安装

```powershell
git clone https://github.com/angela-nuaa/pysifne.git
cd pysifne
conda env create -f environment.yml
conda activate pysifne
python -m pytest -q tests
```

在已有 Conda 环境内，也可运行 `python -m pip install -e ".[accel,test]"`。
CUDA 路径需要可用的 NVIDIA 驱动和 Numba CUDA 运行环境；安装 Python 依赖本身不保证
CUDA 可用。仅使用 CPU reference 时可安装 `python -m pip install -e ".[test]"`。

## 使用

```python
from pysifne import SIFNEConfig, analyze, analyze_mt, export_matlab_compatibility

# 自动选择可用的 CUDA backend，否则使用 CPU reference。
result = analyze("image.tif", config=SIFNEConfig())
print(result.analysis_info)
export_matlab_compatibility(result, "output")

# 显式选择 backend，便于记录和复现。
result = analyze_mt("image.tif", backend="reference")
# result = analyze_mt("image.tif", backend="cuda", device=0)
```

`AnalysisInfo` 保存方向、像素路径长度、端到端距离和中心坐标。
物理长度必须显式提供真实标定，例如 `result.lengths_um(pixel_size_um=0.0313)`。
内部历史 grouping 参数 `internal_grouping_reference_pixel_size_um` 不是输入图像标定。

## 功能范围

已实现：图像预处理、LFT/OFT、分割和骨架细化、交点清理、fragment 提取、
端点登记、双向配对、复合丝分组、路径排序、基础 `AnalysisInfo` 和核心 `.mat` 导出。

尚未覆盖原版全部 SIFNE：`GenFragmentInfo`、`GenLinkageInfo`、最终 overlap-collapse
`NewCrPts`、完整曲率分析、Excel 导出、GUI 和人工校正。
兼容导出中的 `NewCrPts.mat` 保存分组前交点，不能作为原版最终交点结果使用。
目前只实现 `faithful` mode，`robust` mode 尚未实现。

## CUDA 和验证边界

GPU 部分使用 `numba.cuda.jit(fastmath=False)` 自定义 Kernel；LFT/OFT、邻域操作、
细化、距离矩阵和排序中的候选搜索可在 GPU 上执行。顺序敏感部分仍在 CPU/Numba 上
执行，PyTorch 的 CPU 双精度 `atan` 用于与冻结 MATLAB 数值保持兼容。

冻结标识为 `0.2.0 / faithful-matlab-r2023b-cuda-exact`。
`data/derived/reference-freezing/` 包含原始冻结样例；源码及样例 SHA-256 见
[拆分清单](docs/extraction_manifest.json)。算法文件在拆分时逐字节保留。
精确相等的验收范围是已验证的 Windows MATLAB R2023b 样例和 CUDA 配置；
Linux CPU reference 存在已记录的浮点末位差异，不能宣称所有平台、任意输入都逐位相同。

历史报告记载 RTX 4090 上一张 `1004×1004` 图像的 warm 中位耗时为 1.2825 秒。
报告中的约 22.3× 相对历史 CPU 基线，其基线没有充分证据确认为 MATLAB + MEX；
本仓库不将该数字作为相对原版 MATLAB 的性能承诺。详见[验证说明](docs/validation.md)。

## 作为 Git submodule

仓库根目录就是 `pysifne` package，可直接挂到消费仓库的 `pysifne/`，保留
`import pysifne` 路径：

```powershell
git submodule add https://github.com/angela-nuaa/pysifne.git pysifne
git submodule update --init --recursive
conda run -n pysifne python -m pip install -e "./pysifne[accel]"
```

消费仓库用 gitlink 固定具体 commit；升级时先验证新 commit，再提交 gitlink 更新。
PyPI 发布不属于本次拆分范围。

## 来源与许可

源自 [SIFNE](https://github.com/qiqi-lu/fluoresfm/tree/main/methods/SIFNE)，
原作者 Zhen Zhang、Pakorn Kanchanawong；从 FluoResFM 中独立拆分。
许可证与来源说明见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
