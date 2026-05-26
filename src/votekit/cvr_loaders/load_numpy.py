import csv
import io
import os
import urllib.request
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pandas.errors import DataError, EmptyDataError

from votekit.pref_profile.numpy_profile import (
    BLANK_RANKING_SENTINEL,
    NumpyRankProfile,
)


def load_numpy(
    fpath: Union[str, os.PathLike, Path],
) -> tuple[NumpyRankProfile, int, list[str], dict[str, str], str]:
    """
    Given a file path, loads cast vote record from the Scottish election csv format
    directly into a ``NumpyRankProfile``.

    Args:
        fpath (Union[str, os.PathLike, pathlib.Path]): Path to Scottish election csv file.
            Can be a url.

    Raises:
        FileNotFoundError: If fpath is invalid.
        EmptyDataError: If dataset is empty.
        DataError: If there is missing or incorrect metadata or candidate data.

    Returns:
        tuple:
            A tuple ``(NumpyRankProfile, seats, cand_list, cand_to_party, ward)``
            representing the election, the number of seats in the election, the candidate
            names, a dictionary mapping candidates to their party, and the ward.
    """

    def parse_csv_reader(reader: csv.reader) -> list[list[str]]:
        rows = [list(filter(None, row)) for row in reader]
        return [row for row in rows if row]

    fpath = str(fpath)

    if not os.path.isfile(fpath):
        with urllib.request.urlopen(fpath) as response:
            data = response.read().decode("utf-8")
        reader = csv.reader(io.StringIO(data))
        data = parse_csv_reader(reader)
    else:
        if os.path.getsize(fpath) == 0:
            raise EmptyDataError(f"CSV at {fpath} is empty.")
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = parse_csv_reader(reader)

    if len(data[0]) != 2:
        raise DataError(
            "The metadata in the first row should be number of candidates, seats."
        )

    try:
        cand_num = int(data[0][0])
        seats = int(data[0][1])
    except ValueError as exc:
        raise DataError("The first row metadata must contain integer candidate and seat counts.") from exc

    ward = data[-1][0]

    data_cand_num = len([r for r in data if "Candidate" in str(r[0])])
    if data_cand_num != cand_num:
        raise DataError(
            "Incorrect number of candidates in either first row metadata or"
            " in candidate list at end of csv file."
        )

    candidate_rows = data[len(data) - (cand_num + 1) : -1]
    for line in candidate_rows:
        if "Candidate" not in str(line[0]):
            raise DataError(
                f"The number of candidates on line 1 is {cand_num}, which does not match"
                f" the metadata."
            )

    cand_list = [line[1] for line in candidate_rows]
    cand_to_party = {line[1]: line[2] for line in candidate_rows}
    ballot_rows = data[1 : len(data) - (cand_num + 1)]

    ballot_df = pd.DataFrame(ballot_rows, dtype=object)

    if ballot_df.empty:
        ballot_matrix = np.empty((0, 0), dtype=np.int8)
        wt_vec = np.empty(0, dtype=np.float64)
    else:
        weights = pd.to_numeric(ballot_df.iloc[:, 0], errors="raise").to_numpy(dtype=np.float64)

        if ballot_df.shape[1] == 1:
            ballot_matrix = np.empty((len(ballot_df), 0), dtype=np.int8)
        else:
            ranking_df = ballot_df.iloc[:, 1:].replace("", np.nan)
            ranking_arr = ranking_df.to_numpy(dtype=np.float64, copy=True)
            valid_mask = ~np.isnan(ranking_arr)

            ranking_arr[valid_mask] -= 1
            invalid_mask = valid_mask & ((ranking_arr < 0) | (ranking_arr >= cand_num))
            if invalid_mask.any():
                bad_value = int(ranking_arr[invalid_mask][0] + 1)
                raise DataError(f"Ballot contains invalid candidate index {bad_value}.")

            ranking_arr[~valid_mask] = BLANK_RANKING_SENTINEL
            ballot_matrix = ranking_arr.astype(np.int8, copy=False)

        if ballot_matrix.size == 0:
            wt_vec = weights
        else:
            ballot_matrix, inverse = np.unique(ballot_matrix, axis=0, return_inverse=True)
            wt_vec = np.bincount(inverse, weights=weights, minlength=len(ballot_matrix)).astype(
                np.float64,
                copy=False,
            )

    profile = NumpyRankProfile(
        ballot_matrix=ballot_matrix,
        wt_vec=wt_vec,
        candidates=tuple(cand_list),
        metadata={
            "seats": seats,
            "cand_to_party": cand_to_party,
            "ward": ward,
            "source_format": "scottish_csv",
        },
    )

    return (profile, seats, cand_list, cand_to_party, ward)