# 验证记录与结论边界

## 历史验收

冻结报告记录日期为 2026-09-15 至 2026-09-16，版本为
`0.2.0 / faithful-matlab-r2023b-cuda-exact / freeze-20260916`。

- Windows 本机历史全仓库回归：102 passed、46 skipped。
- 远端 RTX 4090：6 个从 TIFF 启动的冻结样例、14 个核心阶段逐元素 EXACT。
- Linux CPU reference 扩展回归：139 passed、1 skipped、11 failed。
  报告将首个差异定位到 OFT_Img，误差约为 1e-15–1e-13。

这些是拆分前的历史记录，不是此次重新执行的结果，也不能外推为任意输入或任意平台
都逐位相同。原始 reference 的 provenance 保留在各样例的 manifest 中。
Windows CI 执行完整 CPU 测试；Linux CI 只执行 API、LFT/OFT 容差测试和分割 smoke
测试，未将已知 Linux CPU exact 差异改成通过。CUDA 需要单独在具备 GPU 的环境验收。

## 性能

历史报告中的输入为 `1004×1004` 的 Cell_001.tif，RTX 4090 GPU0 预热后运行 12 次，
CUDA pipeline 中位耗时 1.2825 秒。历史 CPU 基线约 28.55 秒，因此报告写为约 22.3×。
该基线的实现身份、硬件及原始计时记录不足以确认为原版 MATLAB + MEX；不得将此
作为相对 MATLAB 的整体加速倍数。后续需要同输入、同参数、同功能范围的配对计时。

## 2026-10-11 独立拆分

算法模块未改动；20 个模块的 SHA-256 与冻结样例校验见 extraction_manifest.json。
复制测试中仅调整 pipeline 源码检查的文件路径，以适应根目录 package 布局。
本次验证结果：

- `cv` / Windows / Python 3.12.13：独立仓库完整测试 109 passed、41 skipped，
  373.45 秒；该环境 Numba CUDA 不可用。
- `fluoresfm`（原名 `napari-fluoresfm`）/ RTX 3060 Laptop GPU：加速定向测试
  49 passed，63.33 秒；主仓库 API 测试 3 passed。
- `fluoresfm` CUDA backend：1 个核心样例和 5 个独立 TIFF 样例的阶段输出全部 EXACT。
  结果见 [CUDA 验证记录](cuda_validation.json)。这些包含 JIT 和校验开销的耗时不是性能基准。
- editable 安装、sdist/wheel 构建、隔离目录中的 wheel 导入均通过；Git index 中的
  20 个算法模块仍与拆分前 SHA-256 一致，302 个冻结样例逐字节一致。

`fluoresfm` 的 Python 为 3.12.13，NumPy 2.4.6、SciPy 1.18.0、scikit-image 0.26.0、
tifffile 2026.7.14、Numba 0.66.0、PyTorch 2.6.0+cu124，与消费仓库核心配置一致。
详细验证配置见 [validation_record.json](validation_record.json)。

环境改名后，napari pytest 插件中的旧路径缓存曾阻止 pytest 启动；本次测试设置
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，仅隔离无关插件，同时将 `NUMBA_CACHE_DIR` 指向
本实验独立目录，避免两个不同版本的 Numba 共享缓存。环境内插件未修改。
