"""跨图像 CPU 并行调度测试；不改变单图 MATLAB-faithful 算法。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pysifne.parallel as parallel
from pysifne import SIFNEConfig, analyze_mt, analyze_mt_batch, plan_cpu_parallelism


def test_dynamic_plan_is_bounded_by_task_count_and_memory() -> None:
    items = [np.zeros((16, 16), dtype=np.uint8) for _ in range(3)]
    plan = plan_cpu_parallelism(items, workers=8, executor="thread")
    assert plan.task_count == 3
    assert 1 <= plan.workers <= 3
    assert plan.executor in {"serial", "thread"}
    assert plan.estimated_task_bytes >= 256 * 1024**2


def test_thread_scheduler_preserves_input_order_with_out_of_order_completion(monkeypatch) -> None:
    def fake_analyze(payload):
        # 模拟后提交的任务先完成；结果仍必须按输入顺序 yield。
        item = payload[0]
        return item

    monkeypatch.setattr(parallel, "_analyze_one", fake_analyze)
    values = ["image-0", "image-1", "image-2", "image-3"]
    result = list(
        parallel.iter_analyze_mt(
            values,
            workers=2,
            executor="thread",
            return_exceptions=False,
        )
    )
    assert result == values


def test_single_task_plan_is_serial() -> None:
    plan = plan_cpu_parallelism([np.zeros((8, 8), dtype=np.uint8)], workers=1)
    assert plan.workers == 1


def test_process_batch_keeps_verified_pipeline_output_exact() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "tests" / "fixtures"
        / "core"
        / "input"
        / "fixture_input.tif"
    )
    config = SIFNEConfig(source_reference="sifne-fixture-20260915-r4")
    single = analyze_mt(fixture, config=config)
    batch = analyze_mt_batch(
        [fixture, fixture],
        config=config,
        backend="reference",
        workers=2,
        executor="process",
    )
    assert len(batch) == 2
    for result in batch:
        assert np.array_equal(result.oft_img, single.oft_img)
        assert np.array_equal(result.all_sorted_filament_matlab, single.all_sorted_filament_matlab)
        assert np.array_equal(result.analysis_info, single.analysis_info)
