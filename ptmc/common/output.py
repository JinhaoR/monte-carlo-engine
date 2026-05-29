from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

OUTPUT_SCHEMA_VERSION = 1


class ModelLike(Protocol):
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_prefix is None:
        output_prefix = model_output_prefix(model)
    else:
        output_prefix = safe_output_token(output_prefix)
    results_by_L = experiment["results_by_L"]
    manifest: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "output_dir": str(output_dir.resolve()),
        "output_prefix": output_prefix,
        "model_metadata": experiment.get("model_metadata", model.metadata()),
        "L_values": experiment["L_values"],
        "temps": experiment["temps"],
        "ladder_diagnostics": experiment["ladder_diagnostics"],
        "parameters": experiment["parameters"],
        "files": [],
    }
    for L, result in results_by_L.items():
        L_int = int(L)
        params_for_this_L = {
            **experiment["parameters"],
            "L": L_int,
            "model": experiment.get("model_metadata", model.metadata()),
            "output_prefix": output_prefix,
        }
        filename = run_output_filename(output_prefix, L_int)
        out_path = output_dir / filename
        save_result_npz(
            out_path,
            L=L_int,
            result=result,
            params=params_for_this_L,
        )
        manifest["files"].append(filename)
        print(f"Saved {out_path}", flush=True)
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(f"Wrote manifest to {manifest_path}", flush=True)
    return manifest
