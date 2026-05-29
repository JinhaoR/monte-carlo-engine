from __future__ import annotations

from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

OUTPUT_SCHEMA_VERSION = 1


class ModelLike(Protocol):
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

def local_timestamp() -> str:
    """
    Return a local timezone timestamp for manifests and run parameters.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")

def json_default(value: Any) -> Any:
    """
    Convert common Python/NumPy/path objects into JSON safe values.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )

def safe_output_token(value: Any, default: str = "simulation") -> str:
    """
    Convert a model name or label into a safe filename token.
    """
    token = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value).strip().lower(),
    )
    token = token.strip("._-")
    return token or default


def model_output_prefix(model: ModelLike) -> str:
    """
    Choose the filename prefix for a model.
    """
    metadata = dict(model.metadata())
    prefix = (
        metadata.get("output_prefix")
        or metadata.get("file_prefix")
        or metadata.get("model_name")
        or model.__class__.__name__
    )
    return safe_output_token(prefix)

def run_output_filename(output_prefix: str, L: int) -> str:
    """
    Build the filename for one system size.
    """
    return f"{safe_output_token(output_prefix)}_L{int(L)}.npz"

def _to_npz_value(value: Any) -> np.ndarray:
    """
    Convert a result value into something np.savez_compressed can store.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return np.asarray(value.item())
    if isinstance(value, (int, float, bool, complex)):
        return np.asarray(value)
    if isinstance(value, str):
        return np.asarray(value)
    return np.asarray(json.dumps(value, default=json_default))


def save_result_npz(
    out_path: str | Path,
    *,
    L: int,
    result: dict[str, Any],
    params: dict[str, Any],
) -> Path:
    """
    Save one simulation result as a compressed .npz file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_to_save: dict[str, Any] = {
        "schema_version": np.int32(OUTPUT_SCHEMA_VERSION),
        "L": np.int32(L),
        "params_json": np.asarray(
            json.dumps(params, default=json_default)
        ),
    }
    for key, value in result.items():
        arrays_to_save[key] = _to_npz_value(value)
    np.savez_compressed(out_path, **arrays_to_save)
    return out_path

def write_manifest(
    manifest_path: str | Path,
    manifest: dict[str, Any],
) -> Path:
    """
    Write manifest.json.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    return manifest_path

def start_experiment_output(
    *,
    model: ModelLike,
    output_dir: str | Path,
    L_values: list[int],
    temps: Any,
    ladder_diagnostics: dict[str, Any],
    parameters: dict[str, Any],
    output_prefix: str | None = None,
    started_at: str | None = None,
) -> tuple[Path, str, Path, dict[str, Any]]:
    """
    Create an output directory and initial manifest before simulations start.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_prefix is None:
        output_prefix = model_output_prefix(model)
    else:
        output_prefix = safe_output_token(output_prefix)
    if started_at is None:
        started_at = local_timestamp()
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "output_dir": str(output_dir.resolve()),
        "output_prefix": output_prefix,
        "started_at": started_at,
        "completed_at": None,
        "last_updated_at": started_at,
        "model_metadata": model.metadata(),
        "L_values": [int(L) for L in L_values],
        "temps": temps,
        "ladder_diagnostics": ladder_diagnostics,
        "parameters": parameters,
        "files": [],
        "runs": {},
    }
    write_manifest(manifest_path, manifest)
    print(f"Wrote initial manifest to {manifest_path}", flush=True)
    return output_dir, output_prefix, manifest_path, manifest

def save_l_output(
    *,
    output_dir: Path,
    output_prefix: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    model_metadata: dict[str, Any],
    parameters: dict[str, Any],
    L: int,
    result: dict[str, Any],
    started_at: str,
    completed_at: str,
) -> Path:
    """
    Save one L result immediately and update the manifest.
    """
    L_int = int(L)
    filename = run_output_filename(output_prefix, L_int)
    out_path = output_dir / filename
    params_for_this_L = {
        **parameters,
        "L": L_int,
        "model": model_metadata,
        "output_prefix": output_prefix,
        "experiment_started_at": manifest.get("started_at"),
        "L_started_at": started_at,
        "L_completed_at": completed_at,
    }
    save_result_npz(
        out_path,
        L=L_int,
        result=result,
        params=params_for_this_L,
    )
    files = [
        str(existing)
        for existing in manifest.get("files", [])
        if str(existing) != filename
    ]
    files.append(filename)
    manifest["files"] = files
    manifest.setdefault("runs", {})[str(L_int)] = {
        "L": L_int,
        "file": filename,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    manifest["last_updated_at"] = local_timestamp()
    write_manifest(manifest_path, manifest)
    print(f"Saved {out_path}", flush=True)
    print(f"Updated manifest after L={L_int}: {manifest_path}", flush=True)
    return out_path

def finish_experiment_output(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    completed_at: str | None = None,
) -> Path:
    """
    Mark the manifest as complete after all requested sizes finish.
    """
    if completed_at is None:
        completed_at = local_timestamp()
    manifest["completed_at"] = completed_at
    manifest["last_updated_at"] = completed_at
    write_manifest(manifest_path, manifest)
    print(f"Finalized manifest: {manifest_path}", flush=True)
    return manifest_path

def save_experiment_outputs(
    experiment: dict[str, Any],
    *,
    model: ModelLike,
    output_dir: str | Path,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    """
    Save all results from run_pt_experiment(...).
    This writes one .npz file per L and one manifest.json file.
    """
    results_by_L = experiment["results_by_L"]
    model_metadata = experiment.get("model_metadata", model.metadata())
    output_dir, output_prefix, manifest_path, manifest = start_experiment_output(
        model=model,
        output_dir=output_dir,
        output_prefix=output_prefix,
        L_values=experiment["L_values"],
        temps=experiment["temps"],
        ladder_diagnostics=experiment["ladder_diagnostics"],
        parameters=experiment["parameters"],
        started_at=experiment.get("started_at"),
    )
    run_times = experiment.get("runs", {})
    for L, result in results_by_L.items():
        L_int = int(L)
        timing = run_times.get(str(L_int), {})
        started_at = timing.get("started_at") or local_timestamp()
        completed_at = timing.get("completed_at") or started_at
        save_l_output(
            output_dir=output_dir,
            output_prefix=output_prefix,
            manifest_path=manifest_path,
            manifest=manifest,
            model_metadata=model_metadata,
            parameters=experiment["parameters"],
            L=L_int,
            result=result,
            started_at=started_at,
            completed_at=completed_at,
        )
    finish_experiment_output(
        manifest_path=manifest_path,
        manifest=manifest,
        completed_at=experiment.get("completed_at"),
    )
    return manifest
