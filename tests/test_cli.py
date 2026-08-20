from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mlops_trs.cli as cli
from mlops_trs.registry import RegistryConfig
from mlops_trs.training import TrainConfig


def test_write_json_creates_parent_directory_and_prints_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "nested" / "result.json"
    payload: dict[str, object] = {"status": "ok", "version": 2}

    cli._write_json(payload, str(output_path))

    assert json.loads(output_path.read_text()) == payload
    assert json.loads(capsys.readouterr().out) == payload


def test_train_cli_maps_arguments_and_keeps_outputs_under_requested_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    output_path = tmp_path / "metadata" / "training.json"
    model_output_dir = tmp_path / "models"
    dataset_path = tmp_path / "iris.csv"

    def fake_train(config: TrainConfig) -> dict[str, object]:
        captured["config"] = config
        return {"run_id": "run-123", "model_version": "v2-test"}

    monkeypatch.setattr(cli, "train_model", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlops-train",
            "--model-type",
            "rf",
            "--random-seed",
            "9",
            "--test-size",
            "0.25",
            "--n-estimators",
            "7",
            "--max-depth",
            "2",
            "--c-value",
            "0.5",
            "--experiment-name",
            "cli-test",
            "--tracking-uri",
            "sqlite:///cli.db",
            "--artifact-location",
            "file:///artifacts",
            "--dataset-path",
            str(dataset_path),
            "--model-output-dir",
            str(model_output_dir),
            "--output",
            str(output_path),
        ],
    )

    cli.train_main()

    assert captured["config"] == TrainConfig(
        model_type="rf",
        random_seed=9,
        test_size=0.25,
        n_estimators=7,
        max_depth=2,
        c_value=0.5,
        experiment_name="cli-test",
        tracking_uri="sqlite:///cli.db",
        artifact_location="file:///artifacts",
        dataset_path=str(dataset_path),
        output_dir=str(model_output_dir),
    )
    assert json.loads(output_path.read_text()) == {
        "run_id": "run-123",
        "model_version": "v2-test",
    }


def test_register_cli_reads_run_id_and_writes_candidate_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_file = tmp_path / "training.json"
    output_path = tmp_path / "registered.json"
    run_file.write_text(json.dumps({"run_id": "run-from-file"}))
    captured: dict[str, RegistryConfig] = {}

    def fake_register(config: RegistryConfig) -> SimpleNamespace:
        captured["config"] = config
        return SimpleNamespace(name="iris-cli", version="3", run_id=config.run_id)

    monkeypatch.setattr(cli, "register_best_run", fake_register)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlops-register",
            "--model-name",
            "iris-cli",
            "--experiment-name",
            "cli-test",
            "--tracking-uri",
            "sqlite:///cli.db",
            "--artifact-location",
            "file:///artifacts",
            "--run-file",
            str(run_file),
            "--output",
            str(output_path),
        ],
    )

    cli.register_main()

    assert captured["config"].run_id == "run-from-file"
    assert json.loads(output_path.read_text()) == {
        "name": "iris-cli",
        "version": "3",
        "run_id": "run-from-file",
        "alias": "candidate",
        "model_uri": "models:/iris-cli@candidate",
    }


def test_promote_cli_reads_version_and_writes_quality_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version_file = tmp_path / "registered.json"
    output_path = tmp_path / "promoted.json"
    version_file.write_text(json.dumps({"version": "3"}))
    captured: dict[str, object] = {}

    def fake_promote(
        model_name: str,
        version: str,
        tracking_uri: str | None,
        minimum_accuracy: float,
    ) -> SimpleNamespace:
        captured.update(
            model_name=model_name,
            version=version,
            tracking_uri=tracking_uri,
            minimum_accuracy=minimum_accuracy,
        )
        return SimpleNamespace(
            model_version=SimpleNamespace(name=model_name, version=version),
            alias="champion",
            accuracy=0.91,
            minimum_accuracy=minimum_accuracy,
        )

    monkeypatch.setattr(cli, "promote_model", fake_promote)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlops-promote",
            "--model-name",
            "iris-cli",
            "--version-file",
            str(version_file),
            "--tracking-uri",
            "sqlite:///cli.db",
            "--minimum-accuracy",
            "0.9",
            "--output",
            str(output_path),
        ],
    )

    cli.promote_main()

    assert captured == {
        "model_name": "iris-cli",
        "version": "3",
        "tracking_uri": "sqlite:///cli.db",
        "minimum_accuracy": 0.9,
    }
    assert json.loads(output_path.read_text())["quality_gate"] == {
        "metric": "accuracy",
        "value": 0.91,
        "minimum": 0.9,
        "status": "passed",
    }


def test_serve_cli_forwards_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(app: str, *, host: str, port: int) -> None:
        captured.update(app=app, host=host, port=port)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mlops-serve", "--host", "127.0.0.1", "--port", "8123"],
    )

    cli.serve_main()

    assert captured == {
        "app": "mlops_trs.api:app",
        "host": "127.0.0.1",
        "port": 8123,
    }
