"""
Compute absolute and relative L2 distance of checkpoints to a base model.

Supports two checkpoint formats:
- A safetensors directory (with model.safetensors.index.json or *.safetensors)
- A single FSDP PyTorch checkpoint file ending with .pt (state_dict-like)

The script discovers checkpoints under a root with subdirectories named
global_step_*, mirroring the discovery logic used by weight_dynamics_analysis.py
but extended to include .pt files.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open  # type: ignore
except Exception:
    safe_open = None  # Lazily required only when reading safetensors


# -----------------------------
# Checkpoint discovery (dirs with safetensors or .pt files)
# -----------------------------

GLOBAL_STEP_RE = re.compile(r"global_step_(\d+)$")
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def _dir_has_safetensors(dir_path: Path) -> bool:
    index_json = dir_path / "model.safetensors.index.json"
    if index_json.is_file():
        return True
    shards = list(dir_path.glob("*.safetensors"))
    return len(shards) > 0


def _dir_list_rank_pt(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.is_file() and re.match(r"model_world_size_\d+_rank_\d+\.pt$", p.name)])


def _dir_has_fsdp_shards(dir_path: Path) -> bool:
    return len(_dir_list_rank_pt(dir_path)) > 0


def _select_pt_file(dir_path: Path) -> Optional[Path]:
    """Heuristically pick one .pt file in the directory (non-recursive).

    Preference order (if multiple present): consolidated*.pt, model*.pt, *.pt (lexicographic).
    """
    pt_files = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix == ".pt"])
    if not pt_files:
        return None
    priority_names = [
        re.compile(r"consolidated.*\\.pt$"),
        re.compile(r"pytorch_model.*\\.pt$"),
        re.compile(r"model.*\\.pt$"),
    ]
    for pat in priority_names:
        candidates = [p for p in pt_files if pat.search(p.name)]
        if candidates:
            return candidates[0]
    return pt_files[0]


def discover_checkpoints_mixed(root: str) -> List[Dict[str, object]]:
    """Return checkpoints ordered by phase, supporting safetensors dirs or .pt files.

    Each item: { 'phase': int, 'path': str, 'format': 'safetensors_dir'|'pt_file' }
    For 'pt_file', 'path' points to the .pt file; for 'safetensors_dir', 'path' is the directory.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Root directory not found or not a directory: {root}")

    items: List[Dict[str, object]] = []
    for d in sorted(root_path.iterdir()):
        if not d.is_dir():
            continue
        m = GLOBAL_STEP_RE.search(d.name)
        m2 = CHECKPOINT_RE.search(d.name)
        if not m and not m2:
            continue
        phase = int((m or m2).group(1))

        # If there's an actor subdir (FSDP layout), operate inside it
        candidate = d / "actor" if (d / "actor").is_dir() else d

        # Prefer explicit fsdp shard groups, then safetensors, then single .pt
        if _dir_has_fsdp_shards(candidate):
            items.append({"phase": phase, "path": str(candidate.resolve()), "format": "fsdp_dir"})
            continue
        if _dir_has_safetensors(candidate):
            items.append({"phase": phase, "path": str(candidate.resolve()), "format": "safetensors_dir"})
            continue
        pt = _select_pt_file(candidate)
        if pt is not None:
            items.append({"phase": phase, "path": str(pt.resolve()), "format": "pt_file"})

    items.sort(key=lambda x: int(x["phase"]))
    if not items:
        raise RuntimeError(
            "No checkpoints discovered. Expected subdirs like 'global_step_123' with safetensors or .pt."
        )
    return items


def _format_checkpoint_table_mixed(rows: List[Dict[str, object]]) -> str:
    header = f"{'#':>3}  {'phase':>10}  {'format':>16}  path"
    out = [header, "-" * len(header)]
    for i, r in enumerate(rows):
        out.append(f"{i:>3}  {int(r['phase']):>10}  {str(r['format']):>16}  {r['path']}")
    return "\n".join(out)


# -----------------------------
# Model readers
# -----------------------------


class ModelReader:
    def list_keys(self) -> List[str]:
        raise NotImplementedError

    def get_tensor_np(self, key: str) -> Optional[np.ndarray]:
        raise NotImplementedError


class SafetensorsDirReader(ModelReader):
    def __init__(self, model_dir: str):
        self.dir_path = Path(model_dir).expanduser().resolve()
        if not self.dir_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self.weight_map: Dict[str, str] = self._build_weight_map(self.dir_path)

    @staticmethod
    def _build_weight_map(d: Path) -> Dict[str, str]:
        index_json = d / "model.safetensors.index.json"
        mapping: Dict[str, str] = {}
        if index_json.is_file():
            import json

            data = json.loads(index_json.read_text())
            for k, rel in data.get("weight_map", {}).items():
                mapping[k] = str((d / rel).resolve())
            return mapping
        shards = sorted(d.glob("*.safetensors"))
        if not shards:
            return {}
        if safe_open is None:
            raise RuntimeError("safetensors required to scan shards when no index.json present")
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                for k in f.keys():
                    mapping[k] = str(shard.resolve())
        return mapping

    def list_keys(self) -> List[str]:
        return sorted(list(self.weight_map.keys()))

    def get_tensor_np(self, key: str) -> Optional[np.ndarray]:
        shard_path = self.weight_map.get(key)
        if shard_path is None:
            return None
        if safe_open is None:
            raise RuntimeError("safetensors is required to load tensors")
        with safe_open(shard_path, framework="pt") as f:
            t = f.get_tensor(key)
        try:
            import torch  # type: ignore

            return t.to(dtype=torch.float32, device="cpu").numpy()
        except Exception:
            return t.detach().cpu().float().numpy()


class PtFileReader(ModelReader):
    def __init__(self, pt_file: str):
        self.file_path = Path(pt_file).expanduser().resolve()
        if not self.file_path.is_file():
            raise FileNotFoundError(f".pt file not found: {pt_file}")
        self._state: Dict[str, object] = self._load_state(self.file_path)

    @staticmethod
    def _load_state(pt_path: Path) -> Dict[str, object]:
        import torch  # type: ignore

        obj = torch.load(str(pt_path), map_location="cpu")
        # Try common containers
        candidates: List[Mapping[str, object]] = []
        if isinstance(obj, dict):
            # Direct state_dict-like
            if all(isinstance(k, str) for k in obj.keys()):
                # If values are tensors or arrays, likely a state dict
                candidates.append(obj)  # type: ignore[arg-type]
            # Nested common keys
            for k in ["state_dict", "model_state_dict", "module", "model", "ema_state_dict"]:
                v = obj.get(k)
                if isinstance(v, dict):
                    candidates.append(v)
        # Pick the first mapping that looks like param name -> tensor
        for cand in candidates:
            if not cand:
                continue
            sample_val = next(iter(cand.values()))
            if hasattr(sample_val, "shape") or hasattr(sample_val, "dtype"):
                return dict(cand)
        # Fallback: flatten one level if dict of dicts
        flat: Dict[str, object] = {}
        if isinstance(obj, dict):
            for outer_k, outer_v in obj.items():
                if isinstance(outer_v, dict):
                    for k, v in outer_v.items():
                        if isinstance(k, str):
                            flat[k] = v
        if flat:
            return flat
        raise RuntimeError(f"Could not extract a state_dict-like mapping from {pt_path}")

    def list_keys(self) -> List[str]:
        return sorted(list(self._state.keys()))

    def get_tensor_np(self, key: str) -> Optional[np.ndarray]:
        v = self._state.get(key)
        if v is None:
            return None
        try:
            import torch  # type: ignore

            if isinstance(v, torch.Tensor):
                return v.detach().cpu().float().numpy()
        except Exception:
            pass
        # NumPy array or unsupported type (attempt numpy conversion)
        if isinstance(v, np.ndarray):
            return v.astype(np.float32, copy=False)
        # Some checkpoints store tensors under {"_torch_load_untyped_storage": ...}; rely on torch above
        return None


class FsdpShardDirReader(ModelReader):
    """Reader that reconstructs full parameters from FSDP sharded rank_*.pt files.

    Assumptions:
    - Single FSDP mesh without TP (placements length 1). If TP is present, we error.
    - Per-rank files follow pattern model_world_size_{W}_rank_{r}.pt
    - Tensors are torch.distributed._tensor.DTensor; we use ._local_tensor and
      concatenate along placements[0].dim.
    - Non-DTensor values are treated as replicated/scalars and read from rank 0.
    """

    def __init__(self, dir_path: str):
        import torch  # type: ignore
        from torch.distributed._tensor import DTensor  # type: ignore

        self.dir_path = Path(dir_path).expanduser().resolve()
        if not self.dir_path.is_dir():
            raise FileNotFoundError(f"FSDP shard directory not found: {dir_path}")
        self.rank_files = _dir_list_rank_pt(self.dir_path)
        if not self.rank_files:
            raise RuntimeError(f"No rank .pt files found in {self.dir_path}")

        # Eager-load rank0 to enumerate keys and placements; maintain a cache for other ranks
        self._rank_state_cache: Dict[int, Dict[str, object]] = {}
        rank0_state = torch.load(str(self.rank_files[0]), map_location="cpu", weights_only=False)
        self._rank_state_cache[0] = rank0_state
        self.keys = sorted(list(rank0_state.keys()))
        # Inspect a pivot DTensor (if present) to infer world size and mesh
        pivot_tensor = None
        for k in self.keys:
            v = rank0_state[k]
            if hasattr(v, "__class__") and v.__class__.__name__ == "DTensor":
                pivot_tensor = v
                break
        if pivot_tensor is None:
            # Not DTensor: treat like replicated full state per rank. Use rank0 file via PtFileReader
            # but still expose as FsdpShardDirReader for uniform interface.
            self.rank0_reader = PtFileReader(str(self.rank_files[0]))
            self._dtensor = False
        else:
            self._dtensor = True
            device_mesh = pivot_tensor.device_mesh
            mesh_dim_names = device_mesh.mesh_dim_names
            if mesh_dim_names not in (("fsdp",),):
                raise NotImplementedError(f"Unsupported mesh_dim_names {mesh_dim_names}; only ('fsdp',) supported")
        self._rank0_state = rank0_state

    def list_keys(self) -> List[str]:
        return list(self.keys)

    def get_tensor_np(self, key: str) -> Optional[np.ndarray]:
        import torch  # type: ignore
        from torch.distributed._tensor import DTensor  # type: ignore

        if not self._dtensor:
            return self.rank0_reader.get_tensor_np(key)  # type: ignore[attr-defined]

        # Load local shards for this key across all ranks
        local_tensors: List[torch.Tensor] = []
        shard_dim: Optional[int] = None

        for i, rf in enumerate(self.rank_files):
            state = self._rank_state_cache.get(i)
            if state is None:
                state = torch.load(str(rf), map_location="cpu", weights_only=False)
                self._rank_state_cache[i] = state
            if key not in state:
                # Missing shard; skip
                continue
            v = state[key]
            if isinstance(v, DTensor):
                placements = tuple(v.placements)
                if len(placements) != 1:
                    raise NotImplementedError("FSDP+TP not supported (multiple placements)")
                if shard_dim is None:
                    shard_dim = int(placements[0].dim)
                else:
                    assert shard_dim == int(placements[0].dim)
                local_tensors.append(v._local_tensor.float().cpu())
            else:
                # replicated or non-DTensor param; read from rank0 only
                if i == 0:
                    if hasattr(v, "detach"):
                        return v.detach().cpu().float().numpy()
                    elif isinstance(v, np.ndarray):
                        return v.astype(np.float32, copy=False)
                # otherwise ignore
        if not local_tensors:
            return None
        if shard_dim is None:
            # Should not happen if we appended tensors
            return None
        merged = torch.cat(local_tensors, dim=shard_dim).contiguous()
        return merged.numpy()


def _make_reader(path: str) -> ModelReader:
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.suffix == ".pt":
        return PtFileReader(str(p))
    if p.is_dir():
        if _dir_has_fsdp_shards(p):
            return FsdpShardDirReader(str(p))
        if _dir_has_safetensors(p):
            return SafetensorsDirReader(str(p))
        # Allow base dir containing a single .pt file
        pt = _select_pt_file(p)
        if pt is not None:
            return PtFileReader(str(pt))
    raise FileNotFoundError(f"Unsupported checkpoint path: {path}")


# -----------------------------
# L2 computation
# -----------------------------


@dataclass
class L2Result:
    phase: int
    l2: float
    rel_l2: float
    num_params: int
    num_elems: int


def compute_l2_to_base(rows: List[Dict[str, object]], base_reader: ModelReader, eps: float = 1e-12) -> List[L2Result]:
    results: List[L2Result] = []
    base_keys = set(base_reader.list_keys())

    # Precompute base norms per key to avoid reloading safetensors repeatedly when base is safetensors
    base_cache: Dict[str, Tuple[int, float]] = {}  # key -> (num_elements, sum_sq)
    for k in base_keys:
        a = base_reader.get_tensor_np(k)
        if a is None:
            continue
        a = a.astype(np.float32, copy=False)
        base_cache[k] = (int(a.size), float(np.dot(a.ravel(), a.ravel())))

    total_base_sum_sq = sum(v for _, v in base_cache.values())
    total_base_elems = sum(n for n, _ in base_cache.values())

    for row in rows:
        phase = int(row["phase"])  # type: ignore[index]
        reader = _make_reader(str(row["path"]))
        ckpt_keys = set(reader.list_keys())
        shared = base_keys & ckpt_keys

        sum_sq_diff: float = 0.0
        num_params: int = 0
        num_elems: int = 0

        for k in shared:
            a_info = base_cache.get(k)
            a = None
            if a_info is None:
                a = base_reader.get_tensor_np(k)
                if a is None:
                    continue
                a = a.astype(np.float32, copy=False)
                a_info = (int(a.size), float(np.dot(a.ravel(), a.ravel())))
                base_cache[k] = a_info
            b = reader.get_tensor_np(k)
            if b is None:
                continue
            # shape check
            try:
                if a is None:
                    # Only load a if not already materialized above
                    a = base_reader.get_tensor_np(k)
                    if a is None:
                        continue
                    a = a.astype(np.float32, copy=False)
                b = b.astype(np.float32, copy=False)
                if a.shape != b.shape:
                    continue
                diff = (b - a).ravel()
                sum_sq_diff += float(np.dot(diff, diff))
                num_params += 1
                num_elems += int(a.size)
            except Exception:
                continue

        l2_abs = float(np.sqrt(sum_sq_diff))
        l2_rel = l2_abs / (float(np.sqrt(total_base_sum_sq)) + eps)
        results.append(L2Result(phase=phase, l2=l2_abs, rel_l2=l2_rel, num_params=num_params, num_elems=num_elems))

    # Sort results by phase just in case
    results.sort(key=lambda r: r.phase)
    return results


# -----------------------------
# CLI
# -----------------------------


def cli_main() -> None:
    p = argparse.ArgumentParser(description="Measure L2 and relative L2 distance to base across checkpoints.")
    p.add_argument("root", type=str, help="Path containing 'global_step_*' subdirectories")
    p.add_argument("--base", dest="base_path", type=str, required=True, help="Base model directory (safetensors) or .pt file")
    p.add_argument("--csv", type=str, default=None, help="Optional path to write CSV with columns: phase,l2,rel_l2,num_params,num_elems")
    p.add_argument("--quiet", action="store_true", help="Only print numbers: phase l2 rel_l2")
    p.add_argument("--verbose", action="store_true", help="Print discovered checkpoints table")
    args = p.parse_args()

    rows = discover_checkpoints_mixed(args.root)
    if args.verbose and not args.quiet:
        print(_format_checkpoint_table_mixed(rows))

    base_reader = _make_reader(args.base_path)
    results = compute_l2_to_base(rows, base_reader)

    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "l2", "rel_l2", "num_params", "num_elems"])
            for r in results:
                w.writerow([r.phase, f"{r.l2:.8e}", f"{r.rel_l2:.8e}", r.num_params, r.num_elems])

    if args.quiet:
        for r in results:
            print(f"{r.phase} {r.l2:.8e} {r.rel_l2:.8e}")
    else:
        header = f"{'phase':>10}  {'L2':>14}  {'rel_L2':>14}  {'#params':>8}  {'#elems':>12}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r.phase:>10}  {r.l2:>14.8e}  {r.rel_l2:>14.8e}  {r.num_params:>8}  {r.num_elems:>12}"
            )


if __name__ == "__main__":
    cli_main()


