from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from votekit.pref_profile.pref_profile import RankProfile


BLANK_RANKING_SENTINEL = np.int8(-127)


@dataclass(frozen=True, slots=True)
class NumpyRankProfile:
    ballot_matrix: NDArray[np.integer]
    wt_vec: NDArray[np.floating]
    candidates: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def candidate_to_index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.candidates)}

    @property
    def total_ballot_wt(self) -> float:
        return float(self.wt_vec.sum())

    @property
    def max_ranking_length(self) -> int:
        return int(self.ballot_matrix.shape[1])


def rank_profile_to_numpy_profile(profile: RankProfile) -> NumpyRankProfile:
    from votekit.pref_profile.pref_profile import RankProfile

    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be of type RankProfile.")

    ranking_columns = [c for c in profile.df.columns if c.startswith("Ranking")]
    if len(ranking_columns) > len(profile.candidates):
        ranking_columns = ranking_columns[: len(profile.candidates)]

    candidate_to_index = {frozenset([name]): i for i, name in enumerate(profile.candidates)}
    candidate_to_index[frozenset()] = int(BLANK_RANKING_SENTINEL)
    candidate_to_index[frozenset(["~"])] = int(BLANK_RANKING_SENTINEL)

    cells = profile.df[ranking_columns].to_numpy() if ranking_columns else np.empty(
        (len(profile.df), 0), dtype=object
    )

    def map_cell(cell: frozenset[str]) -> int:
        try:
            return candidate_to_index[cell]
        except KeyError as exc:
            raise TypeError(f"Found invalid entry: {cell}") from exc

    ballot_matrix = np.frompyfunc(map_cell, 1, 1)(cells).astype(np.int8, copy=False)
    wt_vec = profile.df["Weight"].to_numpy(dtype=np.float64)

    return NumpyRankProfile(
        ballot_matrix=ballot_matrix,
        wt_vec=wt_vec,
        candidates=profile.candidates,
        metadata={"sentinel": int(BLANK_RANKING_SENTINEL)},
    )