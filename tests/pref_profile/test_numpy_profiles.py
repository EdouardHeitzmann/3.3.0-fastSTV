import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile, rank_profile_to_numpy_profile


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