# 来源与许可

PySIFNE 将 SIFNE 的核心 filament 分析链迁移到 Python，并提供 Numba CUDA 加速。
原始 SIFNE 开发者为 Zhen Zhang、Pakorn Kanchanawong，原始版权为 2016 年。
本仓库保留原始 BSD 2-Clause 许可证，见 `LICENSE`。

参考源码来自 [FluoResFM 中的 SIFNE](https://github.com/qiqi-lu/fluoresfm/tree/main/methods/SIFNE)。
冻结样例中的 MATLAB 文件和数值输出用于验证，不作为 Python 运行依赖。
NumPy、SciPy、scikit-image、tifffile、Numba 和 PyTorch 按各自许可证分发，
本仓库不重新分发这些库的源码或二进制。
