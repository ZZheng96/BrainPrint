import numpy as np
import pandas as pd

from src.session_blind_candidate.run import (
    FORBIDDEN_COLUMNS,
    TEST_COLUMNS,
    TRAIN_COLUMNS,
    read_metadata,
)


def test_metadata_reading_uses_only_whitelisted_columns(tmp_path):
    frame = pd.DataFrame({
        "subject": [1, 2], "task": ["EC", "EO"], "session": ["session1", "session2"],
        "sample": ["a", "b"], "trial": [1, 2], "path": ["x", "y"],
    })
    path = tmp_path / "metadata.csv"
    frame.to_csv(path, index=False)
    train = read_metadata(path, TRAIN_COLUMNS)
    test = read_metadata(path, TEST_COLUMNS)
    assert list(train.columns) == TRAIN_COLUMNS
    assert list(test.columns) == TEST_COLUMNS
    assert not set(FORBIDDEN_COLUMNS).intersection(train.columns)
    assert not set(FORBIDDEN_COLUMNS).intersection(test.columns)


def test_session_changes_do_not_change_whitelisted_metadata(tmp_path):
    base = pd.DataFrame({"subject": [1, 2], "task": ["EC", "EO"], "sample": ["a", "b"]})
    altered = base.copy()
    altered["session"] = ["anything", "else"]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    base.to_csv(first, index=False)
    altered.to_csv(second, index=False)
    np.testing.assert_array_equal(
        read_metadata(first, TEST_COLUMNS).to_numpy(),
        read_metadata(second, TEST_COLUMNS).to_numpy(),
    )
