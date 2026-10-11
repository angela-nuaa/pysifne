"""Serial accelerator for the MATLAB-faithful composite grouping loop.

The grouping state machine is deliberately kept separate from the readable
implementation in :mod:`pysifne.grouping`.  This module contains no graph
algorithm and no reordering: it is a straight, serial translation of the
same mutations, using Numba only to remove Python interpreter overhead from
the already-defined loop order.
"""

from __future__ import annotations

import numpy as np

try:  # Numba is optional for the reference installation.
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without Numba.
    njit = None


if njit is not None:

    @njit(cache=True)
    def _zero_references_to_numba(partner_list: np.ndarray, tip_ids: np.ndarray) -> None:
        for row in range(partner_list.shape[0]):
            for column in range(partner_list.shape[1]):
                value = partner_list[row, column]
                for index in range(tip_ids.shape[0]):
                    if value == tip_ids[index]:
                        partner_list[row, column] = 0.0
                        break


    @njit(cache=True)
    def _append_connection_numba(
        connection_points: np.ndarray,
        connection_count: int,
        point: np.ndarray,
        prepend: bool,
    ) -> int:
        if prepend:
            for row in range(connection_count, 0, -1):
                connection_points[row, 0] = connection_points[row - 1, 0]
                connection_points[row, 1] = connection_points[row - 1, 1]
            connection_points[0, 0] = point[0]
            connection_points[0, 1] = point[1]
        else:
            connection_points[connection_count, 0] = point[0]
            connection_points[connection_count, 1] = point[1]
        return connection_count + 1


    @njit(cache=True)
    def _append_other_points_numba(
        connection_points: np.ndarray,
        connection_count: int,
        tips: np.ndarray,
        tip_ids: np.ndarray,
        prepend: bool,
    ) -> int:
        count = tip_ids.shape[0]
        if prepend:
            for row in range(connection_count - 1, -1, -1):
                connection_points[row + count, 0] = connection_points[row, 0]
                connection_points[row + count, 1] = connection_points[row, 1]
            for index in range(count):
                tip_row = int(tip_ids[index]) - 1
                connection_points[index, 0] = tips[tip_row, 0]
                connection_points[index, 1] = tips[tip_row, 1]
        else:
            for index in range(count):
                tip_row = int(tip_ids[index]) - 1
                connection_points[connection_count + index, 0] = tips[tip_row, 0]
                connection_points[connection_count + index, 1] = tips[tip_row, 1]
        return connection_count + count


    @njit(cache=True)
    def _advance_direction_numba(
        direction_tip_id: int,
        prepend_connections: bool,
        all_tips: np.ndarray,
        lpts: np.ndarray,
        n_of_lives: np.ndarray,
        label_list: np.ndarray,
        partner_list: np.ndarray,
        l_list: np.ndarray,
        l_count: int,
        filament_pixels: np.ndarray,
        filament_count: int,
        connection_points: np.ndarray,
        connection_count: int,
    ) -> tuple[int, int, int, int]:
        """Run one source ``while`` loop, preserving first-partner order."""

        current_tip_id = direction_tip_id
        tip_count = all_tips.shape[0]
        partner_count = partner_list.shape[1]
        while True:
            current_tip_row = current_tip_id - 1
            first_candidate_id = 0
            for column in range(partner_count):
                candidate = partner_list[current_tip_row, column]
                if candidate != 0.0:
                    first_candidate_id = int(candidate)
                    break
            if first_candidate_id == 0:
                break

            first_candidate_row = first_candidate_id - 1
            candidate_label_from_tips = all_tips[first_candidate_row, 2]
            duplicate = False
            for index in range(l_count):
                if l_list[index] == candidate_label_from_tips:
                    duplicate = True
                    break
            if duplicate:
                break

            optimal_id = first_candidate_id
            optimal_row = optimal_id - 1
            current_label = all_tips[optimal_row, 2]
            candidate_label = label_list[optimal_row]

            # ``xx=Lpts(Lpts(:,3)==candidate_label,:)`` retains the original
            # Lpts row order, which was created by MATLAB column-major find.
            for point_row in range(lpts.shape[0]):
                if lpts[point_row, 2] == candidate_label:
                    filament_pixels[filament_count, 0] = lpts[point_row, 0]
                    filament_pixels[filament_count, 1] = lpts[point_row, 1]
                    filament_count += 1
            l_list[l_count] = candidate_label
            l_count += 1

            connection_count = _append_connection_numba(
                connection_points,
                connection_count,
                all_tips[optimal_row, 0:2],
                prepend_connections,
            )

            lives = n_of_lives[optimal_row]
            if lives < 2.0:
                for point_row in range(lpts.shape[0]):
                    if lpts[point_row, 2] == candidate_label:
                        lpts[point_row, 2] = 0.0
                one_id = np.empty(2, dtype=np.float64)
                one_id[0] = float(current_tip_id)
                one_id[1] = float(optimal_id)
                _zero_references_to_numba(partner_list, one_id)
                label_list[optimal_row] = 0.0
            else:
                for tip_row in range(tip_count):
                    if label_list[tip_row] == current_label:
                        n_of_lives[tip_row] -= 1.0

            # MATLAB's assignment zeros all occurrences in this row.
            for column in range(partner_count):
                if partner_list[current_tip_row, column] == float(optimal_id):
                    partner_list[current_tip_row, column] = 0.0

            # ``find(label_list==current_label)`` is column-major over a
            # column vector, hence the increasing row order below.
            current_ids = np.empty(tip_count, dtype=np.int64)
            current_count = 0
            for tip_row in range(tip_count):
                if label_list[tip_row] == current_label:
                    current_ids[current_count] = tip_row + 1
                    current_count += 1

            if current_count == 2:
                kept = np.empty(2, dtype=np.int64)
                kept_count = 0
                for index in range(current_count):
                    if current_ids[index] != optimal_id:
                        kept[kept_count] = current_ids[index]
                        kept_count += 1
                current_ids = kept
                current_count = kept_count

            if lives < 2.0:
                _zero_references_to_numba(
                    partner_list,
                    current_ids[:current_count].astype(np.float64),
                )
                for index in range(current_count):
                    label_list[int(current_ids[index]) - 1] = 0.0

            connection_count = _append_other_points_numba(
                connection_points,
                connection_count,
                all_tips,
                current_ids[:current_count],
                prepend_connections,
            )
            if current_count == 0:
                break
            current_tip_id = int(current_ids[0])

        return current_tip_id, l_count, filament_count, connection_count


    @njit(cache=True)
    def _trace_composite_numba(
        all_tips: np.ndarray,
        lpts: np.ndarray,
        n_of_lives: np.ndarray,
        partner_list: np.ndarray,
        fragment_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        tip_count = all_tips.shape[0]
        point_count = lpts.shape[0]
        seed_rows = np.empty(2, dtype=np.int64)
        seed_count = 0
        for tip_row in range(tip_count):
            if all_tips[tip_row, 2] == float(fragment_id):
                if seed_count < 2:
                    seed_rows[seed_count] = tip_row
                seed_count += 1

        # The active MATLAB path requires exactly two tips per fragment.
        if seed_count != 2:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                seed_rows,
                n_of_lives.copy(),
                n_of_lives.copy(),
                all_tips[:, 2].copy(),
                0,
                0,
                0,
            )

        label_list = all_tips[:, 2].copy()
        n_before = n_of_lives.copy()
        # No composite can contain more source pixels than the source Lpts
        # table; duplicate protection prevents repeated fragment appends.
        filament_pixels = np.empty((point_count, 2), dtype=np.float64)
        l_list = np.empty(tip_count + 1, dtype=np.float64)
        l_count = 0
        filament_count = 0
        for point_row in range(point_count):
            if lpts[point_row, 2] == float(fragment_id):
                filament_pixels[filament_count, 0] = lpts[point_row, 0]
                filament_pixels[filament_count, 1] = lpts[point_row, 1]
                filament_count += 1
        l_list[l_count] = float(fragment_id)
        l_count += 1

        tip1_id = int(seed_rows[0]) + 1
        tip2_id = int(seed_rows[1]) + 1
        connection_points = np.empty((2 * tip_count + 2, 2), dtype=np.float64)
        connection_count = 0
        connection_count = _append_connection_numba(
            connection_points,
            connection_count,
            all_tips[tip1_id - 1, 0:2],
            False,
        )

        if n_of_lives[tip1_id - 1] < 2.0:
            for point_row in range(point_count):
                if lpts[point_row, 2] == float(fragment_id):
                    lpts[point_row, 2] = 0.0
            label_list[seed_rows[0]] = 0.0
            label_list[seed_rows[1]] = 0.0
            seed_ids = np.empty(2, dtype=np.float64)
            seed_ids[0] = float(tip1_id)
            seed_ids[1] = float(tip2_id)
            _zero_references_to_numba(partner_list, seed_ids)
        else:
            n_of_lives[seed_rows[0]] -= 1.0
            n_of_lives[seed_rows[1]] -= 1.0

        _, l_count, filament_count, connection_count = _advance_direction_numba(
            tip1_id,
            False,
            all_tips,
            lpts,
            n_of_lives,
            label_list,
            partner_list,
            l_list,
            l_count,
            filament_pixels,
            filament_count,
            connection_points,
            connection_count,
        )

        connection_count = _append_connection_numba(
            connection_points,
            connection_count,
            all_tips[tip2_id - 1, 0:2],
            True,
        )
        _, l_count, filament_count, connection_count = _advance_direction_numba(
            tip2_id,
            True,
            all_tips,
            lpts,
            n_of_lives,
            label_list,
            partner_list,
            l_list,
            l_count,
            filament_pixels,
            filament_count,
            connection_points,
            connection_count,
        )

        return (
            filament_pixels,
            connection_points,
            l_list,
            seed_rows,
            n_before,
            n_of_lives.copy(),
            label_list,
            l_count,
            filament_count,
            connection_count,
        )


def numba_trace_composite(
    all_tips: np.ndarray,
    lpts: np.ndarray,
    n_of_lives: np.ndarray,
    partner_list: np.ndarray,
    fragment_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trace one composite with the serial Numba state machine.

    The returned arrays are trimmed to the same observable lengths as the
    readable implementation.  ``lpts``, ``n_of_lives`` and ``partner_list``
    are mutated in place, matching :func:`grouping.trace_composite_filament`.
    """

    if njit is None:  # pragma: no cover - only for an optional dependency miss.
        raise ImportError("numba is required for numba_trace_composite")
    tips = np.asarray(all_tips, dtype=np.float64)
    points = np.asarray(lpts, dtype=np.float64)
    lives = np.asarray(n_of_lives, dtype=np.float64)
    partners = np.asarray(partner_list, dtype=np.float64)
    (
        filament,
        connections,
        membership,
        seed_rows,
        n_before,
        n_after,
        labels_after,
        membership_count,
        filament_count,
        connection_count,
    ) = _trace_composite_numba(tips, points, lives, partners, int(fragment_id))
    if membership_count == 0:
        raise ValueError(
            f"fragment {fragment_id} does not have exactly two active tips"
        )
    return (
        filament[:filament_count].copy(),
        connections[:connection_count].copy(),
        membership[:membership_count].copy(),
        seed_rows.copy(),
        n_before.copy(),
        n_after.copy(),
        labels_after.copy(),
    )


__all__ = ["numba_trace_composite"]
