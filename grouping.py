"""Readable reference implementation of SIFNE's composite grouping loop.

This module is intentionally a transcription of the active grouping block in
``func_analysis_mt.m``.  It is not a graph abstraction: the source mutates
``Lpts``, ``label_list``, ``NofLives`` and ``new_partner_list`` while walking
fragments in MATLAB's original loop order.  The Python implementation keeps
that order and keeps MATLAB-style fragment/tip/partner identifiers as values.

Coordinates at the Python API boundary are zero-based ``(row, column)``.
Fragment labels, global tip IDs and partner IDs remain one-based.  The
``*_matlab`` properties convert only the materialised coordinate rows back to
the MATLAB representation, leaving MATLAB's zero padding untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable, Iterable

import numpy as np

from .fragments import matlab_find


def _as_label_image(label_image: np.ndarray) -> np.ndarray:
    value = np.asarray(label_image)
    if value.ndim != 2:
        raise ValueError(f"label_image must be 2-D, got {value.shape}")
    return value


def _as_tip_table(all_tips: np.ndarray) -> np.ndarray:
    value = np.asarray(all_tips, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 8:
        raise ValueError(
            "all_tips must have shape (N, >=8): coordinates, label, "
            f"tip metadata and lives; got {value.shape}"
        )
    return value


def _matlab_find_rows(mask: np.ndarray) -> np.ndarray:
    """Return zero-based row indices in MATLAB ``find`` order."""

    return np.flatnonzero(np.asarray(mask).reshape(-1, order="F"))


def build_fragment_pixel_table(label_image: np.ndarray) -> np.ndarray:
    """Build the source's ``Lpts=[x,y,L(sub2ind(...))]`` table.

    The two coordinate columns are zero-based for Python use, while the label
    column is an unchanged one-based MATLAB fragment label.  ``matlab_find``
    supplies the required column-major traversal order.
    """

    labels = _as_label_image(label_image)
    coordinates_one = matlab_find(labels != 0)
    if coordinates_one.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    coordinates_zero = coordinates_one.astype(np.float64) - 1.0
    values = labels[
        coordinates_one[:, 0] - 1,
        coordinates_one[:, 1] - 1,
    ].astype(np.float64, copy=False)
    return np.column_stack((coordinates_zero, values))


def _rows_for_label(
    lpts: np.ndarray,
    label: Real,
    *,
    rows_by_label: dict[int, np.ndarray] | None = None,
    consumed_labels: set[int] | None = None,
) -> np.ndarray:
    label_int = int(float(label))
    if rows_by_label is not None:
        if consumed_labels is not None and label_int in consumed_labels:
            return np.empty((0, 2), dtype=np.float64)
        rows = rows_by_label.get(label_int)
        if rows is None:
            return np.empty((0, 2), dtype=np.float64)
        return rows.copy()
    rows = lpts[lpts[:, 2] == float(label), 0:2]
    return rows.copy()


def _nonzero_partner_ids(partner_row: np.ndarray) -> np.ndarray:
    """Remove zeros without changing the remaining MATLAB row order."""

    values = np.asarray(partner_row, dtype=np.float64).reshape(-1)
    return values[values != 0.0].copy()


def _tip_rows_for_label(label_list: np.ndarray, label: Real) -> np.ndarray:
    """MATLAB ``find(label_list==label)`` as zero-based row indices."""

    return np.flatnonzero(np.asarray(label_list) == float(label)).astype(np.int64)


def _as_coordinate_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return array.reshape(-1, 2)


@dataclass(frozen=True)
class GroupingTrace:
    """One composite produced by one outer ``for i=1:num`` iteration."""

    composite_id: int
    seed_fragment_id: int
    seed_tip_rows: np.ndarray
    fragment_membership: np.ndarray
    filament_pixels: np.ndarray
    connection_points: np.ndarray
    n_of_lives_before: np.ndarray
    n_of_lives_after: np.ndarray
    label_list_after: np.ndarray
    partner_list_after: np.ndarray
    lpts_labels_after: np.ndarray

    @property
    def filament_length(self) -> int:
        return int(self.filament_pixels.shape[0])

    @property
    def connection_count(self) -> int:
        return int(self.connection_points.shape[0])


@dataclass(frozen=True)
class GroupingResult:
    """Observable stage-07 arrays plus the mutable-loop diagnostics."""

    all_filament: np.ndarray
    all_connects: np.ndarray
    all_connects_shortlist: np.ndarray
    traces: tuple[GroupingTrace, ...]
    lpts_initial: np.ndarray
    lpts_final: np.ndarray
    n_of_lives_initial: np.ndarray
    n_of_lives_final: np.ndarray
    label_list_final: np.ndarray
    new_partner_list_initial: np.ndarray
    new_partner_list_final: np.ndarray
    all_tips_working: np.ndarray
    iteration_n_of_lives_before: np.ndarray
    iteration_n_of_lives_after: np.ndarray
    iteration_label_list_after: np.ndarray
    iteration_lpts_labels_after: np.ndarray
    unpaired_fragment_ids: tuple[int, ...] = ()

    @property
    def composite_count(self) -> int:
        return len(self.traces)

    @property
    def fragment_memberships(self) -> tuple[np.ndarray, ...]:
        return tuple(trace.fragment_membership.copy() for trace in self.traces)

    @staticmethod
    def _pad_matlab_coordinates(
        traces: Iterable[GroupingTrace], attribute: str
    ) -> np.ndarray:
        values = [np.asarray(getattr(trace, attribute), dtype=np.float64) for trace in traces]
        if not values:
            return np.empty((0, 0), dtype=np.float64)
        max_length = max(value.shape[0] for value in values)
        result = np.zeros((max_length, 2, len(values)), dtype=np.float64)
        for index, value in enumerate(values):
            if value.shape[0]:
                result[: value.shape[0], :, index] = value + 1.0
        if len(values) == 1:
            return result[:, :, 0]
        return result

    @property
    def all_filament_matlab(self) -> np.ndarray:
        """Stage-07 ``all_filament`` with one-based coordinates and zero pad."""

        return self._pad_matlab_coordinates(self.traces, "filament_pixels")

    @property
    def all_connects_matlab(self) -> np.ndarray:
        """Stage-07 ``all_connects`` with one-based coordinates and zero pad."""

        return self._pad_matlab_coordinates(self.traces, "connection_points")

    @property
    def all_connects_shortlist_matlab(self) -> np.ndarray:
        return self.all_connects_matlab.copy()

    @staticmethod
    def _matlab_lpts(value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64).copy()
        if result.shape[0]:
            result[:, 0:2] += 1.0
        return result

    @property
    def lpts_initial_matlab(self) -> np.ndarray:
        return self._matlab_lpts(self.lpts_initial)

    @property
    def lpts_final_matlab(self) -> np.ndarray:
        return self._matlab_lpts(self.lpts_final)


def _zero_references_to(
    partner_list: np.ndarray, tip_ids: np.ndarray | int
) -> None:
    """In-place MATLAB ``new_partner_list(new_partner_list==tip_ids)=0``."""

    ids = np.asarray(tip_ids, dtype=np.float64).reshape(-1)
    if ids.size == 0:
        return
    partner_list[np.isin(partner_list, ids)] = 0.0


def _select_other_tip(label_list: np.ndarray, current_label: Real, optimal_id: int) -> np.ndarray:
    """Transcribe ``find(label_list==current_label)`` and its two-tip rule."""

    rows = _tip_rows_for_label(label_list, current_label)
    if rows.size == 2:
        rows = rows[rows != int(optimal_id) - 1]
    return rows + 1  # MATLAB global tip IDs, including an empty vector.


def _advance_direction(
    *,
    direction_tip_id: int,
    prepend_connections: bool,
    all_tips: np.ndarray,
    lpts: np.ndarray,
    n_of_lives: np.ndarray,
    label_list: np.ndarray,
    partner_list: np.ndarray,
    l_list: list[float],
    filament_pixels: np.ndarray,
    connection_points: np.ndarray,
    rows_by_label: dict[int, np.ndarray] | None = None,
    consumed_labels: set[int] | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Run one of the source's two ``while 1+1==2`` loops.

    ``direction_tip_id`` and partner values are one-based IDs.  The returned
    ID can be a vector in malformed MATLAB inputs; ordinary SIFNE fragments
    have exactly one remaining tip and therefore take the scalar path.
    """

    current_tip_ids = np.asarray([direction_tip_id], dtype=np.int64)
    while True:
        # The active MATLAB path indexes a scalar tip.  The checked-in SIFNE
        # data always has one current tip; retain the first element if a
        # synthetic malformed input supplies more than one.
        current_tip_id = int(current_tip_ids.reshape(-1)[0])
        current_tip_row = current_tip_id - 1
        can_index = _nonzero_partner_ids(partner_list[current_tip_row, :])
        if can_index.size == 0:
            break
        first_candidate_id = int(can_index[0])
        first_candidate_row = first_candidate_id - 1
        if any(
            float(existing) == float(all_tips[first_candidate_row, 2])
            for existing in l_list
        ):
            break

        optimal_id = first_candidate_id
        optimal_row = optimal_id - 1
        current_label = float(all_tips[optimal_row, 2])
        candidate_label = float(label_list[optimal_row])
        candidate_pixels = _rows_for_label(
            lpts,
            candidate_label,
            rows_by_label=rows_by_label,
            consumed_labels=consumed_labels,
        )
        if candidate_pixels.shape[0]:
            filament_pixels = np.vstack((filament_pixels, candidate_pixels))
        else:
            # MATLAB concatenating an empty ``xx,yy`` contributes no rows but
            # still records the label in L_list.
            filament_pixels = filament_pixels.copy()
        l_list.append(candidate_label)

        candidate_point = all_tips[optimal_row, 0:2].reshape(1, 2)
        if prepend_connections:
            connection_points = np.vstack((candidate_point, connection_points))
        else:
            connection_points = np.vstack((connection_points, candidate_point))

        ll = float(n_of_lives[optimal_row])
        if n_of_lives[optimal_row] < 2.0:
            lpts[lpts[:, 2] == candidate_label, 2] = 0.0
            if consumed_labels is not None:
                consumed_labels.add(int(candidate_label))
            _zero_references_to(partner_list, current_tip_id)
            label_list[optimal_row] = 0.0
            _zero_references_to(partner_list, optimal_id)
        else:
            same_label_rows = label_list == current_label
            n_of_lives[same_label_rows] -= 1.0

        # The active source assigns through ``find(...)`` without a ``1``
        # selector, so duplicate occurrences of the chosen partner are all
        # removed in their original row positions.
        partner_list[current_tip_row, partner_list[current_tip_row, :] == float(optimal_id)] = 0.0

        current_tip_ids = _select_other_tip(label_list, current_label, optimal_id)
        if ll < 2.0:
            _zero_references_to(partner_list, current_tip_ids)
            if current_tip_ids.size:
                label_list[current_tip_ids - 1] = 0.0

        other_rows = current_tip_ids.astype(np.int64) - 1
        other_points = all_tips[other_rows, 0:2] if other_rows.size else np.empty((0, 2))
        if prepend_connections:
            connection_points = np.vstack((other_points, connection_points))
        else:
            connection_points = np.vstack((connection_points, other_points))

    return (
        int(current_tip_ids.reshape(-1)[0]) if current_tip_ids.size else 0,
        filament_pixels,
        connection_points,
        current_tip_ids,
    )


def trace_composite_filament(
    all_tips: np.ndarray,
    lpts: np.ndarray,
    n_of_lives: np.ndarray,
    partner_list: np.ndarray,
    fragment_id: int,
    *,
    rows_by_label: dict[int, np.ndarray] | None = None,
    consumed_labels: set[int] | None = None,
) -> tuple[GroupingTrace, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace one outer-loop fragment using the source's two directions.

    The mutable arrays are modified in place.  The return tuple contains the
    trace followed by ``label_list``, ``filament_pixels``, ``connection_points``
    and the remaining forward/backward tip IDs for diagnostics.
    """

    tips = _as_tip_table(all_tips)
    current_fragment = int(fragment_id)
    seed_rows = _tip_rows_for_label(tips[:, 2], current_fragment)
    if seed_rows.size != 2:
        raise ValueError(
            f"fragment {current_fragment} has {seed_rows.size} tips; "
            "the active MATLAB grouping code indexes exactly two"
        )

    label_list = tips[:, 2].copy()
    n_before = n_of_lives.copy()
    filament_pixels = _rows_for_label(
        lpts,
        current_fragment,
        rows_by_label=rows_by_label,
        consumed_labels=consumed_labels,
    )
    l_list: list[float] = [float(current_fragment)]
    tip1_id = int(seed_rows[0]) + 1
    tip2_id = int(seed_rows[1]) + 1
    connection_points = tips[tip1_id - 1, 0:2].reshape(1, 2).copy()

    if n_of_lives[tip1_id - 1] < 2.0:
        lpts[lpts[:, 2] == float(current_fragment), 2] = 0.0
        if consumed_labels is not None:
            consumed_labels.add(int(current_fragment))
        label_list[seed_rows] = 0.0
        _zero_references_to(partner_list, tip1_id)
        _zero_references_to(partner_list, tip2_id)
    else:
        n_of_lives[seed_rows[0]] -= 1.0
        n_of_lives[seed_rows[1]] -= 1.0

    _, filament_pixels, connection_points, _ = _advance_direction(
        direction_tip_id=tip1_id,
        prepend_connections=False,
        all_tips=tips,
        lpts=lpts,
        n_of_lives=n_of_lives,
        label_list=label_list,
        partner_list=partner_list,
        l_list=l_list,
        filament_pixels=filament_pixels,
        connection_points=connection_points,
        rows_by_label=rows_by_label,
        consumed_labels=consumed_labels,
    )

    # The source starts the reverse side with the second seed tip and prepends
    # both the seed point and every subsequently visited tip.
    connection_points = np.vstack((tips[tip2_id - 1, 0:2], connection_points))
    _, filament_pixels, connection_points, _ = _advance_direction(
        direction_tip_id=tip2_id,
        prepend_connections=True,
        all_tips=tips,
        lpts=lpts,
        n_of_lives=n_of_lives,
        label_list=label_list,
        partner_list=partner_list,
        l_list=l_list,
        filament_pixels=filament_pixels,
        connection_points=connection_points,
        rows_by_label=rows_by_label,
        consumed_labels=consumed_labels,
    )

    trace = GroupingTrace(
        composite_id=0,
        seed_fragment_id=current_fragment,
        seed_tip_rows=seed_rows.copy(),
        fragment_membership=np.asarray(l_list, dtype=np.float64),
        filament_pixels=filament_pixels.copy(),
        connection_points=connection_points.copy(),
        n_of_lives_before=n_before,
        n_of_lives_after=n_of_lives.copy(),
        label_list_after=label_list.copy(),
        partner_list_after=partner_list.copy(),
        lpts_labels_after=lpts[:, 2].copy(),
    )
    return trace, label_list, filament_pixels, connection_points, np.asarray(l_list)


def _pad_internal_arrays(traces: list[GroupingTrace], attribute: str) -> np.ndarray:
    if not traces:
        return np.empty((0, 0), dtype=np.float64)
    max_length = max(getattr(trace, attribute).shape[0] for trace in traces)
    result = np.zeros((max_length, 2, len(traces)), dtype=np.float64)
    for index, trace in enumerate(traces):
        value = np.asarray(getattr(trace, attribute), dtype=np.float64)
        if value.shape[0]:
            result[: value.shape[0], :, index] = value
    if len(traces) == 1:
        return result[:, :, 0]
    return result


def group_fragments(
    all_tips: np.ndarray,
    label_image: np.ndarray,
    *,
    num_fragments: int | None = None,
    trace_impl: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] | None = None,
    retain_diagnostics: bool = True,
    skip_unpaired_fragments: bool = False,
) -> GroupingResult:
    """Reproduce the active ``func_analysis_mt.m`` grouping block.

    ``all_tips`` must be the stage-06 table after bidirectional cleanup and
    after the active caller sets column 8 (Python column 7) to ``OverlapList``.
    Coordinates are zero-based; IDs and labels are MATLAB-style one-based.

    When ``skip_unpaired_fragments`` is enabled, a non-empty fragment with no
    registered tips is recorded as ungrouped and omitted from the composite
    grouping.  This is an explicit audit-mode repair for the source workflow's
    later ungrouped-fragment removal; fragments with one or more than two tips
    still raise the original invariant error.
    """

    tips = _as_tip_table(all_tips).copy()
    labels = _as_label_image(label_image)
    if num_fragments is None:
        nonzero = labels[labels != 0]
        num = int(np.max(nonzero)) if nonzero.size else 0
    else:
        numeric = float(num_fragments)
        if numeric < 0 or numeric != np.floor(numeric):
            raise ValueError("num_fragments must be a non-negative integer")
        num = int(numeric)

    # NofLives=all_tips(:,8); zeros become one; all_tips(:,8) is written back.
    n_of_lives = tips[:, 7].copy()
    n_of_lives[n_of_lives == 0.0] = 1.0
    tips[:, 7] = n_of_lives
    n_initial = n_of_lives.copy()
    partner_list = tips[:, 8:].copy()
    partner_initial = partner_list.copy()

    lpts = build_fragment_pixel_table(labels)
    lpts_initial = lpts.copy()
    # The readable path uses a stable row cache.  The optional accelerated
    # state machine owns its Lpts scan and therefore intentionally uses the
    # mutable table directly; this avoids a second cache with stale labels.
    rows_by_label: dict[int, np.ndarray] | None = None
    if trace_impl is None:
        rows_by_label_lists: dict[int, list[np.ndarray]] = {}
        for row in lpts:
            rows_by_label_lists.setdefault(int(row[2]), []).append(row[0:2].copy())
        rows_by_label = {
            label: np.asarray(rows, dtype=np.float64).reshape(-1, 2)
            for label, rows in rows_by_label_lists.items()
        }
    consumed_labels: set[int] = set()
    traces: list[GroupingTrace] = []
    unpaired_fragment_ids: list[int] = []
    if retain_diagnostics:
        iteration_n_before = np.zeros((tips.shape[0], num), dtype=np.float64)
        iteration_n_after = np.zeros((tips.shape[0], num), dtype=np.float64)
        iteration_label_after = np.zeros((tips.shape[0], num), dtype=np.float64)
        iteration_lpts_after = np.zeros((lpts.shape[0], num), dtype=np.float64)
    else:
        iteration_n_before = np.empty((0, 0), dtype=np.float64)
        iteration_n_after = np.empty((0, 0), dtype=np.float64)
        iteration_label_after = np.empty((0, 0), dtype=np.float64)
        iteration_lpts_after = np.empty((0, 0), dtype=np.float64)
    label_list_final = tips[:, 2].copy()

    for fragment_id in range(1, num + 1):
        # This reset is inside the outer MATLAB loop, not outside it.
        label_list = tips[:, 2].copy()
        label_list_final = label_list.copy()
        if retain_diagnostics:
            iteration_n_before[:, fragment_id - 1] = n_of_lives
        xx = _rows_for_label(
            lpts,
            fragment_id,
            rows_by_label=rows_by_label,
            consumed_labels=consumed_labels,
        )
        if xx.shape[0] == 0:
            if retain_diagnostics:
                iteration_n_after[:, fragment_id - 1] = n_of_lives
                iteration_label_after[:, fragment_id - 1] = label_list
                if lpts.shape[0]:
                    iteration_lpts_after[:, fragment_id - 1] = lpts[:, 2]
            continue

        seed_rows = _tip_rows_for_label(tips[:, 2], fragment_id)
        if seed_rows.size == 0 and skip_unpaired_fragments:
            unpaired_fragment_ids.append(int(fragment_id))
            if lpts.shape[0]:
                lpts[lpts[:, 2] == float(fragment_id), 2] = 0.0
            consumed_labels.add(int(fragment_id))
            if retain_diagnostics:
                iteration_n_after[:, fragment_id - 1] = n_of_lives
                iteration_label_after[:, fragment_id - 1] = label_list
                if lpts.shape[0]:
                    iteration_lpts_after[:, fragment_id - 1] = lpts[:, 2]
            continue
        if seed_rows.size != 2:
            raise ValueError(
                f"fragment {fragment_id} has {seed_rows.size} tips; "
                "the active MATLAB grouping code indexes exactly two"
            )

        # Keeping this call at the exact point where the MATLAB code begins a
        # non-empty fragment preserves all persistent mutation of
        # Lpts/NofLives/partner_list.  ``trace_impl`` is only a loop-level
        # accelerator; its inputs/outputs are the same observable state as the
        # readable trace_composite_filament implementation.
        if trace_impl is None:
            trace, trace_labels, _, _, _ = trace_composite_filament(
                tips,
                lpts,
                n_of_lives,
                partner_list,
                fragment_id,
                rows_by_label=rows_by_label,
                consumed_labels=consumed_labels,
            )
        else:
            (
                accelerated_filament,
                accelerated_connections,
                accelerated_membership,
                accelerated_seed_rows,
                accelerated_n_before,
                accelerated_n_after,
                accelerated_labels_after,
            ) = trace_impl(
                tips,
                lpts,
                n_of_lives,
                partner_list,
                fragment_id,
            )
            trace = GroupingTrace(
                composite_id=0,
                seed_fragment_id=int(fragment_id),
                seed_tip_rows=np.asarray(accelerated_seed_rows, dtype=np.int64),
                fragment_membership=np.asarray(accelerated_membership, dtype=np.float64),
                filament_pixels=np.asarray(accelerated_filament, dtype=np.float64),
                connection_points=np.asarray(accelerated_connections, dtype=np.float64),
                n_of_lives_before=(
                    np.asarray(accelerated_n_before, dtype=np.float64)
                    if retain_diagnostics
                    else np.empty(0, dtype=np.float64)
                ),
                n_of_lives_after=(
                    np.asarray(accelerated_n_after, dtype=np.float64)
                    if retain_diagnostics
                    else np.empty(0, dtype=np.float64)
                ),
                label_list_after=np.asarray(accelerated_labels_after, dtype=np.float64),
                partner_list_after=(
                    partner_list.copy()
                    if retain_diagnostics
                    else np.empty((0, 0), dtype=np.float64)
                ),
                lpts_labels_after=(
                    lpts[:, 2].copy()
                    if retain_diagnostics
                    else np.empty(0, dtype=np.float64)
                ),
            )
            trace_labels = trace.label_list_after
        if retain_diagnostics:
            iteration_n_after[:, fragment_id - 1] = n_of_lives
            iteration_label_after[:, fragment_id - 1] = trace_labels
        label_list_final = trace_labels.copy()
        if retain_diagnostics and lpts.shape[0]:
            iteration_lpts_after[:, fragment_id - 1] = lpts[:, 2]
        trace = GroupingTrace(
            composite_id=len(traces) + 1,
            seed_fragment_id=trace.seed_fragment_id,
            seed_tip_rows=trace.seed_tip_rows,
            fragment_membership=trace.fragment_membership,
            filament_pixels=trace.filament_pixels,
            connection_points=trace.connection_points,
            n_of_lives_before=(
                trace.n_of_lives_before
                if retain_diagnostics
                else np.empty(0, dtype=np.float64)
            ),
            n_of_lives_after=(
                trace.n_of_lives_after
                if retain_diagnostics
                else np.empty(0, dtype=np.float64)
            ),
            label_list_after=(
                trace_labels
                if retain_diagnostics
                else np.empty(0, dtype=np.float64)
            ),
            partner_list_after=(
                partner_list.copy()
                if retain_diagnostics
                else np.empty((0, 0), dtype=np.float64)
            ),
            lpts_labels_after=(
                lpts[:, 2].copy()
                if retain_diagnostics
                else np.empty(0, dtype=np.float64)
            ),
        )
        traces.append(trace)

    all_filament = _pad_internal_arrays(traces, "filament_pixels")
    all_connects = _pad_internal_arrays(traces, "connection_points")
    return GroupingResult(
        all_filament=all_filament,
        all_connects=all_connects,
        all_connects_shortlist=all_connects.copy(),
        traces=tuple(traces),
        lpts_initial=lpts_initial,
        lpts_final=lpts.copy(),
        n_of_lives_initial=n_initial,
        n_of_lives_final=n_of_lives.copy(),
        label_list_final=label_list_final,
        new_partner_list_initial=partner_initial,
        new_partner_list_final=partner_list.copy(),
        all_tips_working=tips,
        iteration_n_of_lives_before=iteration_n_before,
        iteration_n_of_lives_after=iteration_n_after,
        iteration_label_list_after=iteration_label_after,
        iteration_lpts_labels_after=iteration_lpts_after,
        unpaired_fragment_ids=tuple(unpaired_fragment_ids),
    )
