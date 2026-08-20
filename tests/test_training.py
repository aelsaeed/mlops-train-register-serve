from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from mlops_trs.constants import IRIS_FEATURE_NAMES
from mlops_trs.training import (
    DatasetValidationError,
    TrainConfig,
    _build_model_version,
    _dataset_sha256,
    _load_dataset,
    _validate_split,
)


def _config() -> TrainConfig:
    return TrainConfig(
        model_type="logreg",
        random_seed=42,
        test_size=0.2,
        n_estimators=10,
        max_depth=3,
        c_value=1.0,
        experiment_name="test-experiment",
        tracking_uri=None,
        artifact_location=None,
        dataset_path=None,
    )


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [5.1, 3.5, 1.4, 0.2, 0],
            [4.9, 3.0, 1.4, 0.2, 0],
            [7.0, 3.2, 4.7, 1.4, 1],
            [6.4, 3.2, 4.5, 1.5, 1],
            [6.3, 3.3, 6.0, 2.5, 2],
            [5.8, 2.7, 5.1, 1.9, 2],
        ],
        columns=[*IRIS_FEATURE_NAMES, "target"],
    )


def test_dataset_hash_and_model_version_are_deterministic_and_content_sensitive() -> None:
    frame = _valid_frame()
    features = frame.loc[:, list(IRIS_FEATURE_NAMES)]
    target = frame["target"]

    first_hash = _dataset_sha256(features, target)
    repeated_hash = _dataset_sha256(features.copy(), target.copy())
    changed_features = features.copy()
    changed_features.loc[0, "sepal_length"] = 9.9
    changed_hash = _dataset_sha256(changed_features, target)

    assert first_hash == repeated_hash
    assert first_hash != changed_hash
    assert len(first_hash) == 64
    assert _build_model_version(_config(), first_hash).startswith("v2-")
    assert _build_model_version(_config(), first_hash) == _build_model_version(
        _config(), repeated_hash
    )
    assert _build_model_version(_config(), first_hash) != _build_model_version(
        _config(), changed_hash
    )
    assert _build_model_version(replace(_config(), random_seed=7), first_hash) != (
        _build_model_version(_config(), first_hash)
    )
    runtime = {"python": "3.12.3", "scikit_learn": "1.9.0"}
    assert _build_model_version(
        _config(),
        first_hash,
        source_sha256="a" * 64,
        runtime_versions=runtime,
    ) != _build_model_version(
        _config(),
        first_hash,
        source_sha256="b" * 64,
        runtime_versions=runtime,
    )
    assert _build_model_version(
        _config(),
        first_hash,
        source_sha256="a" * 64,
        runtime_versions=runtime,
    ) != _build_model_version(
        _config(),
        first_hash,
        source_sha256="a" * 64,
        runtime_versions={**runtime, "scikit_learn": "1.10.0"},
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns=["petal_width"]), "missing columns"),
        (lambda frame: frame.assign(extra=1), "unexpected columns"),
        (lambda frame: frame.assign(sepal_length="invalid"), "must be numeric"),
        (lambda frame: frame.assign(petal_width=float("nan")), "missing values"),
        (lambda frame: frame.assign(petal_width=float("inf")), "must be finite"),
        (lambda frame: frame.assign(sepal_length=-1), "greater than 0"),
        (lambda frame: frame.assign(target=0), "at least two classes"),
        (
            lambda frame: frame.assign(target=[0, 0, 1, 1, 2.5, 2.5]),
            "integer class IDs",
        ),
        (
            lambda frame: frame.assign(target=[0, 0, 1, 1, 1, 1]),
            "must contain Iris classes",
        ),
        (
            lambda frame: frame.assign(target=[0, 0, 1, 1, 1, 2]),
            "at least two rows",
        ),
    ],
)
def test_invalid_csv_data_fails_with_actionable_error(
    tmp_path: Path,
    mutate: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    dataset_path = tmp_path / "invalid.csv"
    mutate(_valid_frame()).to_csv(dataset_path, index=False)

    with pytest.raises(DatasetValidationError, match=message):
        _load_dataset(str(dataset_path))


def test_missing_dataset_and_invalid_split_fail_before_training(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="Dataset not found"):
        _load_dataset(str(tmp_path / "missing.csv"))

    _, target = _load_dataset(None)
    with pytest.raises(DatasetValidationError, match="test_size must be between"):
        _validate_split(replace(_config(), test_size=1.0), target)
    with pytest.raises(DatasetValidationError, match="each contain at least one"):
        _validate_split(replace(_config(), test_size=0.01), target)
