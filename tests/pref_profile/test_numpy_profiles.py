import pytest
import numpy as np

from votekit.ballot import RankBallot
from votekit.cleaning import remove_and_condense_rank_profile
from votekit.elections import FastSTV
from votekit.pref_profile import RankProfile, rank_profile_to_numpy_profile
from votekit.pref_profile.numpy_profile import (
    BLANK_RANKING_SENTINEL,
    numpy_profile_fpv,
    numpy_profile_to_rank_profile,
    remove_and_reweigh_and_condense,
    reindex_candidate_indices,
    remove_and_condense_numpy_profile,
)
from votekit.pref_profile.numpy_profile.utils import left_compact_ballot_matrix


profile_no_ties = RankProfile(
    ballots=(
        RankBallot(ranking=tuple(map(frozenset, [{"A"}, {"B"}])), weight=1),
        RankBallot(ranking=tuple(map(frozenset, [{"A"}, {"B"}, {"C"}])), weight=1 / 2),
        RankBallot(ranking=tuple(map(frozenset, [{"C"}, {"B"}, {"A"}])), weight=3),
    )
)

profile_with_ties = RankProfile(
    ballots=(
        RankBallot(ranking=tuple(map(frozenset, [{"A", "B"}])), weight=1),
        RankBallot(ranking=tuple(map(frozenset, [{"A", "B", "C"}])), weight=1 / 2),
        RankBallot(ranking=tuple(map(frozenset, [{"A"}, {"C"}, {"B"}])), weight=3),
    )
)


def test_rank_profile_to_numpy_profile():
    array_profile = rank_profile_to_numpy_profile(profile_no_ties)

    assert array_profile.candidates == profile_no_ties.candidates
    assert array_profile.ballot_matrix.tolist() == [[0, 1, -127], [0, 1, 2], [2, 1, 0]]
    assert array_profile.wt_vec.tolist() == [1.0, 0.5, 3.0]


def test_rank_profile_array_profile_is_cached():
    assert profile_no_ties.array_profile is profile_no_ties.array_profile


def test_rank_profile_to_numpy_profile_rejects_ties():
    with pytest.raises(TypeError, match="Found invalid entry"):
        rank_profile_to_numpy_profile(profile_with_ties)


def _numpy_profile_to_ballots(array_profile) -> tuple[RankBallot, ...]:
    ballots = []
    for row, weight in zip(array_profile.ballot_matrix.tolist(), array_profile.wt_vec.tolist()):
        ranking = tuple(
            frozenset([array_profile.candidates[cand_idx]])
            for cand_idx in row
            if cand_idx != BLANK_RANKING_SENTINEL
        )
        ballots.append(RankBallot(ranking=ranking, weight=weight))

    return tuple(ballots)


def test_remove_and_condense_numpy_profile_matches_rank_profile_cleaner():
    array_profile = rank_profile_to_numpy_profile(profile_no_ties)

    cleaned_array_profile = remove_and_condense_numpy_profile(array_profile, "A")
    cleaned_profile = remove_and_condense_rank_profile("A", profile_no_ties)

    assert _numpy_profile_to_ballots(cleaned_array_profile) == cleaned_profile.ballots
    assert set(cleaned_array_profile.candidates) == set(cleaned_profile.candidates)


def test_remove_and_condense_numpy_profile_matches_rank_profile_cleaner_for_multiple_candidates():
    array_profile = rank_profile_to_numpy_profile(profile_no_ties)

    cleaned_array_profile = remove_and_condense_numpy_profile(array_profile, ["A", "B"])
    cleaned_profile = remove_and_condense_rank_profile(["A", "B"], profile_no_ties)

    assert _numpy_profile_to_ballots(cleaned_array_profile) == cleaned_profile.ballots
    assert set(cleaned_array_profile.candidates) == set(cleaned_profile.candidates)


def test_numpy_profile_to_rank_profile_roundtrip():
    array_profile = rank_profile_to_numpy_profile(profile_no_ties)

    roundtrip_profile = numpy_profile_to_rank_profile(array_profile)

    assert roundtrip_profile == profile_no_ties


def test_numpy_profile_to_rank_profile_matches_remove_and_condense_cleaner():
    array_profile = rank_profile_to_numpy_profile(profile_no_ties)

    cleaned_array_profile = remove_and_condense_numpy_profile(array_profile, "B")
    converted_profile = numpy_profile_to_rank_profile(cleaned_array_profile)
    cleaned_profile = remove_and_condense_rank_profile("B", profile_no_ties)

    assert converted_profile == cleaned_profile


def test_remove_and_condense_numpy_profile_accepts_candidate_indices():
    array_profile = rank_profile_to_numpy_profile(profile_no_ties)

    cleaned_by_index = remove_and_condense_numpy_profile(array_profile, [0, 1])
    cleaned_by_name = remove_and_condense_numpy_profile(array_profile, ["A", "B"])

    assert np.array_equal(cleaned_by_index.ballot_matrix, cleaned_by_name.ballot_matrix)
    assert np.array_equal(cleaned_by_index.wt_vec, cleaned_by_name.wt_vec)
    assert cleaned_by_index.candidates == cleaned_by_name.candidates
    assert cleaned_by_index.metadata == cleaned_by_name.metadata


def test_remove_and_reweigh_and_condense_reweights_only_removed_first_preferences():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=tuple(map(frozenset, [{"A"}, {"B"}])), weight=6),
            RankBallot(ranking=tuple(map(frozenset, [{"B"}, {"A"}])), weight=4),
            RankBallot(ranking=tuple(map(frozenset, [{"A"}])), weight=3),
        )
    )
    array_profile = rank_profile_to_numpy_profile(profile)

    reweighted_profile = remove_and_reweigh_and_condense(array_profile, ["A"], [0.5])

    assert reweighted_profile.candidates == ("B",)
    assert reweighted_profile.ballot_matrix.tolist() == [[0, BLANK_RANKING_SENTINEL], [0, BLANK_RANKING_SENTINEL]]
    assert reweighted_profile.wt_vec.tolist() == [3.0, 4.0]


def test_remove_and_reweigh_and_condense_matches_faststv_round_profile_for_winner_transfer():
    some_pf = RankProfile(
        ballots=(
            RankBallot(ranking=tuple(map(frozenset, [{"A"}, {"B"}])), weight=90),
            RankBallot(ranking=tuple(map(frozenset, [{"A"}])), weight=60),
            RankBallot(ranking=tuple(map(frozenset, [{"B"}])), weight=70),
            RankBallot(ranking=tuple(map(frozenset, [{"C"}])), weight=75),
            RankBallot(ranking=tuple(map(frozenset, [{"D"}])), weight=2),
        )
    )
    array_profile = rank_profile_to_numpy_profile(some_pf)
    elec = FastSTV(some_pf, 2, simultaneous=False)

    reweighted_profile = remove_and_reweigh_and_condense(array_profile, ["A"], [1 / 3])
    converted_profile = numpy_profile_to_rank_profile(reweighted_profile)

    assert converted_profile == elec.get_profile(1)


def test_numpy_profile_fpv_matches_expected_totals():
    array_profile = rank_profile_to_numpy_profile(profile_no_ties)

    assert numpy_profile_fpv(array_profile) == {"A": 1.5, "B": 0.0, "C": 3.0}


def test_reindex_candidate_indices_after_middle_candidate_removed():
    array_profile = rank_profile_to_numpy_profile(profile_no_ties)
    ballot_matrix = array_profile.ballot_matrix.copy()
    ballot_matrix[ballot_matrix == array_profile.candidate_to_index["B"]] = BLANK_RANKING_SENTINEL
    compacted = left_compact_ballot_matrix(ballot_matrix)

    reindexed = reindex_candidate_indices(
        compacted,
        array_profile.candidate_to_index,
        ("A", "C"),
    )

    assert reindexed.tolist() == [[0, BLANK_RANKING_SENTINEL, BLANK_RANKING_SENTINEL], [0, 1, BLANK_RANKING_SENTINEL], [1, 0, BLANK_RANKING_SENTINEL]]


def test_left_compact_ballot_matrix_condenses_internal_sentinels():
    ballot_matrix = np.array(
        [
            [0, BLANK_RANKING_SENTINEL, BLANK_RANKING_SENTINEL, 1, BLANK_RANKING_SENTINEL],
            [2, BLANK_RANKING_SENTINEL, BLANK_RANKING_SENTINEL, BLANK_RANKING_SENTINEL, BLANK_RANKING_SENTINEL],
            [BLANK_RANKING_SENTINEL] * 5,
        ],
        dtype=np.int8,
    )

    compacted = left_compact_ballot_matrix(ballot_matrix)

    assert compacted.tolist() == [
        [0, 1, BLANK_RANKING_SENTINEL, BLANK_RANKING_SENTINEL, BLANK_RANKING_SENTINEL],
        [2, BLANK_RANKING_SENTINEL, BLANK_RANKING_SENTINEL, BLANK_RANKING_SENTINEL, BLANK_RANKING_SENTINEL],
        [BLANK_RANKING_SENTINEL] * 5,
    ]