from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from votekit.pref_profile.numpy_profile.profile import BLANK_RANKING_SENTINEL, NumpyRankProfile

if TYPE_CHECKING:
    from votekit.pref_profile import RankProfile


def left_compact_ballot_matrix(
    ballot_matrix: NDArray[np.integer],
    sentinel: np.integer | int = np.int8(-127),
) -> NDArray[np.integer]:
    """
    Stably shift non-sentinel entries in each row left and fill the remainder with the sentinel.
    """
    if ballot_matrix.ndim != 2:
        raise ValueError("Ballot matrix must be 2-dimensional.")

    compacted = np.full_like(ballot_matrix, fill_value=sentinel)
    keep_mask = ballot_matrix != sentinel
    pos = keep_mask.cumsum(axis=1) - 1
    row_idx, col_idx = np.nonzero(keep_mask)
    compacted[row_idx, pos[row_idx, col_idx]] = ballot_matrix[row_idx, col_idx]
    return compacted


def reindex_candidate_indices(
    ballot_matrix: NDArray[np.integer],
    old_candidate_to_index: dict[str, int],
    new_candidates: Sequence[str],
    sentinel: np.integer | int = BLANK_RANKING_SENTINEL,
) -> NDArray[np.int8]:
    """
    Reindex candidate ids in a ballot matrix to match a new candidate ordering.
    """
    reindex_lut = np.full(256, int(sentinel), dtype=np.int16)
    for new_index, candidate in enumerate(new_candidates):
        reindex_lut[old_candidate_to_index[candidate] + 128] = new_index

    return reindex_lut[ballot_matrix.astype(np.int16, copy=False) + 128].astype(np.int8, copy=False)


def _normalize_removed_candidate_indices(
    profile: NumpyRankProfile,
    removed: str | int | Sequence[str | int],
) -> NDArray[np.intp]:
    removed_entries = [removed] if isinstance(removed, (str, int, np.integer)) else list(removed)

    removed_indices: list[int] = []
    for entry in removed_entries:
        if isinstance(entry, str):
            try:
                removed_indices.append(profile.candidate_to_index[entry])
            except KeyError as exc:
                raise ValueError(f"Unknown candidate to remove: {entry}") from exc
        elif isinstance(entry, (int, np.integer)):
            removed_idx = int(entry)
            if removed_idx < 0 or removed_idx >= len(profile.candidates):
                raise ValueError(f"Removal index out of range: {removed_idx}")
            removed_indices.append(removed_idx)
        else:
            raise TypeError("Removed candidates must be candidate strings or integer indices.")

    return np.asarray(sorted(set(removed_indices)), dtype=np.intp)


def _normalize_removed_candidates_with_transfer_values(
    profile: NumpyRankProfile,
    removed: str | int | Sequence[str | int],
    transfer_values: float | Sequence[float],
) -> tuple[NDArray[np.intp], dict[int, float]]:
    removed_entries = [removed] if isinstance(removed, (str, int, np.integer)) else list(removed)
    transfer_entries = (
        [float(transfer_values)]
        if isinstance(transfer_values, (int, float, np.integer, np.floating))
        else [float(value) for value in transfer_values]
    )

    if len(removed_entries) != len(transfer_entries):
        raise ValueError("transfer_values must have the same length as removed.")

    removed_indices: list[int] = []
    transfer_value_by_index: dict[int, float] = {}

    for entry, transfer_value in zip(removed_entries, transfer_entries):
        if isinstance(entry, str):
            try:
                removed_idx = profile.candidate_to_index[entry]
            except KeyError as exc:
                raise ValueError(f"Unknown candidate to remove: {entry}") from exc
        elif isinstance(entry, (int, np.integer)):
            removed_idx = int(entry)
            if removed_idx < 0 or removed_idx >= len(profile.candidates):
                raise ValueError(f"Removal index out of range: {removed_idx}")
        else:
            raise TypeError("Removed candidates must be candidate strings or integer indices.")

        if removed_idx in transfer_value_by_index:
            raise ValueError("Removed candidates must be unique when transfer values are supplied.")

        removed_indices.append(removed_idx)
        transfer_value_by_index[removed_idx] = transfer_value

    return np.asarray(sorted(removed_indices), dtype=np.intp), transfer_value_by_index


def remove_and_condense_numpy_profile(
    profile: NumpyRankProfile,
    removed: str | int | Sequence[str | int],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> NumpyRankProfile:
    """
    Remove candidate(s) from a numpy-backed rank profile and left-compact each ballot row.
    """
    if not isinstance(profile, NumpyRankProfile):
        raise TypeError("Profile must be of type NumpyRankProfile.")

    removed_indices = _normalize_removed_candidate_indices(profile, removed)
    removed_set = {profile.candidates[idx] for idx in removed_indices.tolist()}

    ballot_matrix = profile.ballot_matrix.copy()
    if removed_indices.size:
        ballot_matrix[np.isin(ballot_matrix, removed_indices)] = BLANK_RANKING_SENTINEL

    compacted_ballot_matrix = left_compact_ballot_matrix(ballot_matrix, BLANK_RANKING_SENTINEL)
    remaining_candidates = tuple(c for c in profile.candidates if c not in removed_set)

    ballot_matrix = reindex_candidate_indices(
        compacted_ballot_matrix,
        profile.candidate_to_index,
        remaining_candidates,
        BLANK_RANKING_SENTINEL,
    )

    row_keep_mask = np.ones(len(ballot_matrix), dtype=bool)
    if remove_empty_ballots:
        row_keep_mask &= ~(ballot_matrix == BLANK_RANKING_SENTINEL).all(axis=1)
    if remove_zero_weight_ballots:
        row_keep_mask &= profile.wt_vec != 0

    metadata = dict(profile.metadata)
    metadata.setdefault("sentinel", int(BLANK_RANKING_SENTINEL))

    return NumpyRankProfile(
        ballot_matrix=ballot_matrix[row_keep_mask],
        wt_vec=profile.wt_vec[row_keep_mask].copy(),
        candidates=remaining_candidates,
        metadata=metadata,
    )


def remove_and_reweigh_and_condense(
    profile: NumpyRankProfile,
    removed: str | int | Sequence[str | int],
    transfer_values: float | Sequence[float],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> NumpyRankProfile:
    """
    Remove candidate(s), reweight ballots whose first preference was removed, and left-compact.
    """
    if not isinstance(profile, NumpyRankProfile):
        raise TypeError("Profile must be of type NumpyRankProfile.")

    removed_indices, transfer_value_by_index = _normalize_removed_candidates_with_transfer_values(
        profile,
        removed,
        transfer_values,
    )
    removed_set = {profile.candidates[idx] for idx in removed_indices.tolist()}

    ballot_matrix = profile.ballot_matrix.copy()
    updated_wt_vec = profile.wt_vec.copy()

    if removed_indices.size:
        first_preferences = ballot_matrix[:, 0]
        for removed_idx, transfer_value in transfer_value_by_index.items():
            updated_wt_vec[first_preferences == removed_idx] *= transfer_value
        ballot_matrix[np.isin(ballot_matrix, removed_indices)] = BLANK_RANKING_SENTINEL

    compacted_ballot_matrix = left_compact_ballot_matrix(ballot_matrix, BLANK_RANKING_SENTINEL)
    remaining_candidates = tuple(c for c in profile.candidates if c not in removed_set)

    ballot_matrix = reindex_candidate_indices(
        compacted_ballot_matrix,
        profile.candidate_to_index,
        remaining_candidates,
        BLANK_RANKING_SENTINEL,
    )

    row_keep_mask = np.ones(len(ballot_matrix), dtype=bool)
    if remove_empty_ballots:
        row_keep_mask &= ~(ballot_matrix == BLANK_RANKING_SENTINEL).all(axis=1)
    if remove_zero_weight_ballots:
        row_keep_mask &= updated_wt_vec != 0

    metadata = dict(profile.metadata)
    metadata.setdefault("sentinel", int(BLANK_RANKING_SENTINEL))

    return NumpyRankProfile(
        ballot_matrix=ballot_matrix[row_keep_mask],
        wt_vec=updated_wt_vec[row_keep_mask],
        candidates=remaining_candidates,
        metadata=metadata,
    )


def numpy_profile_fpv(profile: NumpyRankProfile) -> dict[str, float]:
    """
    Compute first-place vote totals for a numpy-backed rank profile.
    """
    if not isinstance(profile, NumpyRankProfile):
        raise TypeError("Profile must be of type NumpyRankProfile.")

    if profile.ballot_matrix.size == 0:
        return {candidate: 0.0 for candidate in profile.candidates}

    first_preferences = profile.ballot_matrix[:, 0]
    valid_mask = first_preferences != BLANK_RANKING_SENTINEL

    if not valid_mask.any():
        return {candidate: 0.0 for candidate in profile.candidates}

    fpv_by_index = np.bincount(
        first_preferences[valid_mask],
        weights=profile.wt_vec[valid_mask],
        minlength=len(profile.candidates),
    )

    return {
        candidate: float(fpv_by_index[idx]) for idx, candidate in enumerate(profile.candidates)
    }


def numpy_profile_to_rank_profile(profile: NumpyRankProfile) -> RankProfile:
    """
    Convert a numpy-backed rank profile into a legacy RankProfile.
    """
    from votekit.pref_profile import RankProfile

    if not isinstance(profile, NumpyRankProfile):
        raise TypeError("Profile must be of type NumpyRankProfile.")

    ballot_matrix = profile.ballot_matrix
    n_rows, n_cols = ballot_matrix.shape

    lut: np.ndarray = np.empty(256, dtype=object)
    lut[:] = frozenset(["~"])
    for idx, candidate in enumerate(profile.candidates):
        lut[idx + 128] = frozenset([candidate])

    rankings = lut[ballot_matrix.astype(np.int16, copy=False) + 128]

    df = pd.DataFrame({f"Ranking_{i + 1}": rankings[:, i] for i in range(n_cols)})
    df.insert(0, "Ballot Index", np.arange(n_rows, dtype=int))
    df.set_index("Ballot Index", inplace=True)
    df["Voter Set"] = pd.Series([set() for _ in range(n_rows)], dtype=object, index=df.index)
    df["Weight"] = profile.wt_vec.astype(np.float64, copy=False)

    return RankProfile(
        max_ranking_length=profile.max_ranking_length,
        candidates=profile.candidates,
        df=df,
    )