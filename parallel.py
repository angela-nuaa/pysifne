"""跨图像并行的 PySIFNE 执行辅助。

这里的并行边界是独立输入图像之间，而不是单张图内部的 MATLAB-faithful
循环。因此不会改变单张图的 column-major 遍历、tie-breaking 或 fragment
顺序。对于 reference backend，默认使用进程池以绕过 Python GIL；CUDA
backend 保持单 worker，因为 Windows WDDM 下多个进程共享同一 GPU 会增加
上下文和显存风险。
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np

from .config import SIFNEConfig


ExecutorKind = Literal["serial", "thread", "process"]


@dataclass(frozen=True)
class CPUParallelPlan:
    """一次批处理的可审计 CPU 并行计划。"""

    executor: ExecutorKind
    workers: int
    task_count: int
    cpu_count: int
    available_memory_bytes: int | None
    estimated_task_bytes: int
    memory_worker_limit: int | None
    requested_workers: int | None


def _available_memory_bytes() -> int | None:
    """读取可用内存；没有 psutil 时返回 ``None``，不改变算法路径。"""

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # pragma: no cover - depends on optional environment
        return None


def _image_shape(item: Any) -> tuple[int, ...] | None:
    """尽量只读取输入元数据，避免为规划阶段复制整张图。"""

    if isinstance(item, (str, Path)):
        try:
            import tifffile

            with tifffile.TiffFile(item) as tif:
                return tuple(int(v) for v in tif.series[0].shape)
        except Exception:
            return None
    try:
        return tuple(int(v) for v in np.asarray(item).shape)
    except Exception:
        return None


def _estimated_task_bytes(items: Sequence[Any]) -> int:
    """估算单个 pipeline worker 的峰值内存。

    估算只用于选择并发度，不参与任何数值计算。系数偏保守，覆盖输入、
    OFT/LFT 中间数组以及后续 skeleton/fragment 数组。
    """

    pixels: list[int] = []
    for item in items:
        shape = _image_shape(item)
        if shape is None:
            continue
        if len(shape) >= 2:
            pixels.append(int(shape[-1]) * int(shape[-2]))
    representative = max(pixels, default=1024 * 1024)
    # 256 MiB base + 512 bytes/pixel: conservative for the faithful path.
    return max(256 * 1024**2, 256 * 1024**2 + representative * 512)


def plan_cpu_parallelism(
    items: Sequence[Any],
    *,
    workers: int | None = None,
    executor: Literal["thread", "process"] = "process",
    reserve_cores: int = 1,
    memory_fraction: float = 0.75,
    max_workers_cap: int = 8,
) -> CPUParallelPlan:
    """根据 CPU、可用内存和输入尺寸生成动态并行计划。

    ``workers`` 是上限而非绕过内存保护的强制值。传入 ``1`` 可以显式恢复
    串行行为；``None`` 使用动态默认值。
    """

    values = list(items)
    task_count = len(values)
    if executor not in {"thread", "process"}:
        raise ValueError("executor must be 'thread' or 'process'")
    if workers is not None and (isinstance(workers, bool) or int(workers) != workers):
        raise ValueError(f"workers must be a positive integer or None, got {workers!r}")
    if workers is not None and int(workers) < 1:
        raise ValueError(f"workers must be positive, got {workers!r}")
    if isinstance(reserve_cores, bool) or int(reserve_cores) != reserve_cores:
        raise ValueError("reserve_cores must be an integer")
    if int(reserve_cores) < 0:
        raise ValueError("reserve_cores must be non-negative")
    if not np.isfinite(float(memory_fraction)) or not 0.0 < float(memory_fraction) <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    if isinstance(max_workers_cap, bool) or int(max_workers_cap) < 1:
        raise ValueError("max_workers_cap must be positive")

    cpu_count = max(1, int(os.cpu_count() or 1))
    cpu_limit = max(1, cpu_count - int(reserve_cores))
    task_bytes = _estimated_task_bytes(values)
    available = _available_memory_bytes()
    if available is None:
        memory_limit = None
    else:
        memory_limit = max(1, int((available * float(memory_fraction)) // task_bytes))

    upper = min(cpu_limit, int(max_workers_cap), task_count or 1)
    if memory_limit is not None:
        upper = min(upper, memory_limit)
    selected = upper if workers is None else min(int(workers), upper)
    selected = max(1, selected)
    kind: ExecutorKind = "serial" if task_count <= 1 or selected <= 1 else executor
    if task_count == 0:
        selected = 0
        kind = "serial"
    return CPUParallelPlan(
        executor=kind,
        workers=selected,
        task_count=task_count,
        cpu_count=cpu_count,
        available_memory_bytes=available,
        estimated_task_bytes=task_bytes,
        memory_worker_limit=memory_limit,
        requested_workers=None if workers is None else int(workers),
    )


def _analyze_one(payload: tuple[Any, SIFNEConfig, str, int, int, str]) -> Any:
    """ProcessPool 的顶层 worker；保持 Windows spawn 可导入。"""

    image_or_path, config, backend, device, threads_per_block, factor_mode = payload
    from .pipeline import analyze_mt

    return analyze_mt(
        image_or_path,
        config=config,
        backend=backend,
        device=device,
        threads_per_block=threads_per_block,
        factor_mode=factor_mode,
    )


def iter_analyze_mt(
    image_inputs: Iterable[Any],
    *,
    config: SIFNEConfig | None = None,
    backend: str = "reference",
    device: int = 0,
    threads_per_block: int = 256,
    factor_mode: str = "matlab_windows",
    workers: int | None = None,
    executor: Literal["thread", "process"] = "process",
    return_exceptions: bool = False,
) -> Iterator[Any]:
    """按输入顺序流式运行多个独立图像。

    只允许有限数量的 future 在飞行中，避免把所有大图和完整结果同时压入
    内存。即使 worker 完成顺序不同，yield 顺序仍与 ``image_inputs`` 一致。
    ``backend='cuda'`` 不在这里并行化，调用方应显式使用单 worker。
    """

    values = list(image_inputs)
    cfg = SIFNEConfig() if config is None else config
    if backend == "cuda" and workers is not None and int(workers) != 1:
        raise ValueError("CPU batch parallelism does not enable multi-process CUDA; use workers=1")
    if backend == "cuda":
        workers = 1
    plan = plan_cpu_parallelism(
        values,
        workers=workers,
        executor=executor,
    )
    if plan.executor == "serial":
        for item in values:
            try:
                yield _analyze_one((item, cfg, backend, int(device), int(threads_per_block), factor_mode))
            except Exception as exc:
                if not return_exceptions:
                    raise
                yield exc
        return

    executor_class = ProcessPoolExecutor if plan.executor == "process" else ThreadPoolExecutor
    payloads = [
        (item, cfg, backend, int(device), int(threads_per_block), factor_mode)
        for item in values
    ]
    with executor_class(max_workers=plan.workers) as pool:
        pending: dict[Any, int] = {}
        completed: dict[int, Any] = {}
        next_submit = 0
        next_yield = 0

        def submit_one(index: int) -> None:
            pending[pool.submit(_analyze_one, payloads[index])] = index

        max_pending = max(1, plan.workers * 2)
        while next_submit < len(payloads) and len(pending) < max_pending:
            submit_one(next_submit)
            next_submit += 1

        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    completed[index] = future.result()
                except Exception as exc:
                    if not return_exceptions:
                        for other in pending:
                            other.cancel()
                        raise
                    completed[index] = exc
                if next_submit < len(payloads):
                    submit_one(next_submit)
                    next_submit += 1
            while next_yield in completed:
                yield completed.pop(next_yield)
                next_yield += 1


def analyze_mt_batch(
    image_inputs: Iterable[Any],
    *,
    config: SIFNEConfig | None = None,
    backend: str = "reference",
    device: int = 0,
    threads_per_block: int = 256,
    factor_mode: str = "matlab_windows",
    workers: int | None = None,
    executor: Literal["thread", "process"] = "process",
    return_exceptions: bool = False,
) -> list[Any]:
    """批量执行并返回与输入同序的结果列表。"""

    return list(
        iter_analyze_mt(
            image_inputs,
            config=config,
            backend=backend,
            device=device,
            threads_per_block=threads_per_block,
            factor_mode=factor_mode,
            workers=workers,
            executor=executor,
            return_exceptions=return_exceptions,
        )
    )


__all__ = [
    "CPUParallelPlan",
    "analyze_mt_batch",
    "iter_analyze_mt",
    "plan_cpu_parallelism",
]
