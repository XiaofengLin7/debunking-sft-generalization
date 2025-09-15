"""
This script is used to analyze the weight dynamics of the model.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")  # headless by default
import matplotlib.pyplot as plt

try:
    from safetensors import safe_open  # type: ignore
except Exception:
    safe_open = None  # Loaded lazily only if needed when index.json is missing

# Step 1: Discover and order checkpoints (simple version)

GLOBAL_STEP_RE = re.compile(r"global_step_(\d+)$")


def discover_checkpoints(root: str) -> List[Dict[str, object]]:
    """Return a simple list of checkpoints ordered by phase ascending.

    Each item: { 'phase': int, 'path': str, 'has_index': bool, 'num_shards': int }
    A checkpoint is kept if it has model.safetensors.index.json or any *.safetensors.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Root directory not found or not a directory: {root}")

    items: List[Dict[str, object]] = []
    for d in sorted(root_path.iterdir()):
        if not d.is_dir():
            continue
        m = GLOBAL_STEP_RE.search(d.name)
        if not m:
            continue
        phase = int(m.group(1))
        index_json = d / "model.safetensors.index.json"
        shards = sorted(d.glob("*.safetensors"))
        if (not index_json.is_file()) and (len(shards) == 0):
            continue
        items.append(
            {
                "phase": phase,
                "path": str(d.resolve()),
                "has_index": index_json.is_file(),
                "num_shards": len(shards),
            }
        )

    items.sort(key=lambda x: int(x["phase"]))
    if not items:
        raise RuntimeError(
            "No checkpoints discovered. Expected subdirs like 'global_step_123' with safetensors."
        )
    return items


def _summarize_base_root(base_root: str) -> str:
    base_path = Path(base_root).expanduser().resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Base root not found or not a directory: {base_root}")
    index_json = base_path / "model.safetensors.index.json"
    shards = sorted(base_path.glob("*.safetensors"))
    has_index = index_json.is_file()
    num_shards = len(shards)
    return f"base: index={'y' if has_index else 'n'}, shards={num_shards}, path={base_path}"


def _format_checkpoint_table(rows: List[Dict[str, object]]) -> str:
    header = f"{'#':>3}  {'phase':>10}  {'index':>5}  {'shards':>6}  path"
    out = [header, "-" * len(header)]
    for i, r in enumerate(rows):
        out.append(
            f"{i:>3}  {int(r['phase']):>10}  {('y' if r['has_index'] else 'n'):>5}  {int(r['num_shards']):>6}  {r['path']}"
        )
    return "\n".join(out)


def cli_discover() -> None:
    p = argparse.ArgumentParser(description="List checkpoints under a run root (ordered by phase).")
    p.add_argument("root", type=str, help="Path containing 'global_step_*' subdirectories")
    p.add_argument("--quiet", action="store_true", help="Only print phases (space-separated)")
    p.add_argument("--base-root", type=str, default=None, help="Path to base model directory (safetensors)")
    p.add_argument("--verbose", action="store_true", help="Print detailed table and base summary")
    p.add_argument("--init-grid", action="store_true", help="Initialize heatmap buffers and print a short summary")
    p.add_argument("--compute", action="store_true", help="Compute pairwise relative Frobenius deltas and summarize")
    p.add_argument("--plot", action="store_true", help="Render stacked sub-rows heatmap (q,k,v,o[,+mlp] per layer)")
    p.add_argument("--out", type=str, default="heatmap_by_type.png", help="Output image path for --plot")
    p.add_argument("--include-mlp", action="store_true", help="Add aggregated MLP row per layer (gate/up/down)")
    args = p.parse_args()

    rows = discover_checkpoints(args.root)
    # Default to minimal output; use --verbose for details
    if args.init_grid:
        phases = [int(r["phase"]) for r in rows]
        layers, mapping = build_layer_type_mapping_with_fallback(args.base_root, rows[0]["path"])  # type: ignore[index]
        if args.include_mlp:
            add_mlp_to_mapping(rows[0]["path"], mapping)  # type: ignore[index]
        # Always include base column in the grid (labeled step 0)
        buffers = init_heatmap_buffers_from_mapping(phases, layers, include_mlp=args.include_mlp, include_base=True)
        print(f"grid rows={buffers.values.shape[0]} cols={buffers.values.shape[1]} layers={len(layers)} phases={len(phases)}")
    elif args.compute:
        if not args.base_root:
            raise SystemExit("--compute requires --base-root for mapping")
        phases = [int(r["phase"]) for r in rows]
        layers, mapping = build_layer_type_mapping_with_fallback(args.base_root, rows[0]["path"])  # type: ignore[index]
        if args.include_mlp:
            add_mlp_to_mapping(rows[0]["path"], mapping)  # type: ignore[index]
        buffers = init_heatmap_buffers_from_mapping(phases, layers, include_mlp=args.include_mlp, include_base=True)
        compute_pairwise_relative_frobenius(rows, mapping, buffers, base_dir=args.base_root)
        filled = int(np.isfinite(buffers.values).sum())
        total = int(buffers.values.size)
        meanv = float(np.nanmean(buffers.values)) if filled > 0 else float("nan")
        print(f"computed cells={filled}/{total} mean={meanv:.6f}")
    elif args.plot:
        if not args.base_root:
            raise SystemExit("--plot requires --base-root for mapping")
        phases = [int(r["phase"]) for r in rows]
        layers, mapping = build_layer_type_mapping_with_fallback(args.base_root, rows[0]["path"])  # type: ignore[index]
        if args.include_mlp:
            add_mlp_to_mapping(rows[0]["path"], mapping)  # type: ignore[index]
        buffers = init_heatmap_buffers_from_mapping(phases, layers, include_mlp=args.include_mlp, include_base=True)
        compute_pairwise_relative_frobenius(rows, mapping, buffers, base_dir=args.base_root)
        render_stacked_heatmap(buffers, out_path=args.out, include_mlp=args.include_mlp, include_base=True)
        print(f"saved {args.out}")
    else:
        if args.quiet or not args.verbose:
            print(" ".join(str(r["phase"]) for r in rows))
        else:
            if args.base_root:
                print(_summarize_base_root(args.base_root))
            print(_format_checkpoint_table(rows))


# -----------------------------
# Step 2: Build (layer, type) mapping from base model keys
# -----------------------------

LAYER_RE_DEFAULT = re.compile(r"model\.layers\.(\d+)\.")
TYPE_PATTERNS = [
    (re.compile(r"(self_)?attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$"), {
        "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"
    }),
    (re.compile(r"(self_)?attention\.(q_proj|k_proj|v_proj|o_proj)\.weight$"), {
        "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o"
    }),
    (re.compile(r"(self_)?attn\.(query|key|value|out(?:_proj)?)\.weight$"), {
        "query": "q", "key": "k", "value": "v", "out": "o", "out_proj": "o"
    }),
]

MLP_PATTERNS = [
    re.compile(r"mlp\.(gate_proj|up_proj|down_proj)\.weight$"),
]


def _list_parameter_keys(model_dir: str) -> List[str]:
    """List parameter keys in a safetensors directory without loading tensors.

    Prefer `model.safetensors.index.json` if present; otherwise iterate *.safetensors files.
    """
    d = Path(model_dir).expanduser().resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    index_json = d / "model.safetensors.index.json"
    if index_json.is_file():
        import json

        with index_json.open("r") as f:
            data = json.load(f)
        weight_map = data.get("weight_map", {})
        return sorted(list(weight_map.keys()))

    # Fallback: iterate shards and union keys
    keys: set = set()
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"No safetensors files found in: {model_dir}")
    if safe_open is None:
        raise RuntimeError("safetensors is required to list keys when index.json is missing")
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            keys.update(list(f.keys()))
    return sorted(list(keys))


def build_layer_type_mapping(
    model_dir: str,
    layer_re: re.Pattern = LAYER_RE_DEFAULT,
) -> Tuple[List[int], Dict[Tuple[int, str], List[str]]]:
    """Return (sorted_layers, mapping) where mapping[(layer, type)] -> param keys.

    Only attention q/k/v/o weights are included. Non-layer keys are ignored.
    """
    keys = _list_parameter_keys(model_dir)
    mapping: DefaultDict[Tuple[int, str], List[str]] = defaultdict(list)
    layers_found: set = set()

    for k in keys:
        m = layer_re.search(k)
        if not m:
            continue  # skip non-layer keys
        layer_idx = int(m.group(1))

        matched_type: str = ""
        for pat, trans in TYPE_PATTERNS:
            m2 = pat.search(k)
            if m2:
                raw = m2.group(2)
                matched_type = trans.get(raw, "")
                break
        if matched_type in {"q", "k", "v", "o"}:
            mapping[(layer_idx, matched_type)].append(k)
            layers_found.add(layer_idx)

    sorted_layers = sorted(list(layers_found))
    return sorted_layers, dict(mapping)


def add_mlp_to_mapping(example_dir: str, mapping: Dict[Tuple[int, str], List[str]]) -> None:
    """Augment mapping by adding aggregated 'mlp' buckets per layer.

    We scan keys in example_dir and add gate/up/down weights into a single
    (layer, 'mlp') entry to be aggregated size-weighted later.
    """
    keys = _list_parameter_keys(example_dir)
    for k in keys:
        m = LAYER_RE_DEFAULT.search(k)
        if not m:
            continue
        layer_idx = int(m.group(1))
        is_mlp = any(p.search(k) for p in MLP_PATTERNS)
        if is_mlp:
            mapping.setdefault((layer_idx, "mlp"), []).append(k)


def build_layer_type_mapping_with_fallback(base_dir: str | None, example_ckpt_dir: str | None) -> Tuple[List[int], Dict[Tuple[int, str], List[str]]]:
    """Try base first; if base is None or yields no layers, fall back to first checkpoint.

    This helps when the base model is provided but uses a different naming; we still
    attempt to infer from a concrete checkpoint index.
    """
    if base_dir:
        layers, mapping = build_layer_type_mapping(base_dir)
        if len(layers) > 0:
            return layers, mapping
    if example_ckpt_dir:
        layers, mapping = build_layer_type_mapping(example_ckpt_dir)
        return layers, mapping
    return [], {}


# -----------------------------
# Step 3: Pre-allocate heatmap buffers
# -----------------------------

TYPE_ORDER = ["q", "k", "v", "o"]


@dataclass
class HeatmapBuffers:
    phases: List[int]
    layers: List[int]
    row_labels: List[str]
    row_index: Dict[Tuple[int, str], int]
    values: np.ndarray  # float32, (rows, cols), initialized to NaN
    weighted_sums: np.ndarray  # float64 accumulators
    weight_totals: np.ndarray  # float64 accumulators
    include_base: bool = False


def init_heatmap_buffers_from_mapping(phases: List[int], layers: List[int], include_mlp: bool = False, include_base: bool = False) -> HeatmapBuffers:
    if len(phases) < 2:
        raise ValueError("Need at least two phases to form phase-to-phase columns")
    num_cols = (len(phases) - 1) + (1 if include_base else 0)
    row_labels: List[str] = []
    row_index: Dict[Tuple[int, str], int] = {}
    effective_types = TYPE_ORDER + (["mlp"] if include_mlp else [])
    for layer in layers:
        for t in effective_types:
            row_index[(layer, t)] = len(row_labels)
            row_labels.append(f"L{layer}.{t}")

    num_rows = len(row_labels)
    values = np.full((num_rows, num_cols), np.nan, dtype=np.float32)
    weighted_sums = np.zeros((num_rows, num_cols), dtype=np.float64)
    weight_totals = np.zeros((num_rows, num_cols), dtype=np.float64)
    return HeatmapBuffers(
        phases=list(phases),
        layers=list(layers),
        row_labels=row_labels,
        row_index=row_index,
        values=values,
        weighted_sums=weighted_sums,
        weight_totals=weight_totals,
        include_base=include_base,
    )


def _build_weight_map(model_dir: str) -> Dict[str, str]:
    d = Path(model_dir).expanduser().resolve()
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


def _load_tensor_np(shard_path: str, key: str) -> np.ndarray:
    if safe_open is None:
        raise RuntimeError("safetensors is required to load tensors")
    with safe_open(shard_path, framework="pt") as f:
        t = f.get_tensor(key)
    try:
        import torch  # type: ignore
        return t.to(dtype=torch.float32, device="cpu").numpy()
    except Exception:
        return t.detach().cpu().float().numpy()


def compute_pairwise_relative_frobenius(
    ordered_ckpts: List[Dict[str, object]],
    mapping: Dict[Tuple[int, str], List[str]],
    buffers: HeatmapBuffers,
    eps: float = 1e-12,
    base_dir: str | None = None,
) -> None:
    weight_maps: List[Dict[str, str]] = [_build_weight_map(str(row["path"])) for row in ordered_ckpts]
    col_offset = 0
    if base_dir is not None and buffers.include_base:
        wm_base = _build_weight_map(base_dir)
        wm_first = weight_maps[0]
        for (layer, t), keys in mapping.items():
            row_idx = buffers.row_index[(layer, t)]
            for key in keys:
                sp_prev = wm_base.get(key)
                sp_curr = wm_first.get(key)
                if not sp_prev or not sp_curr:
                    continue
                a = _load_tensor_np(sp_prev, key)
                b = _load_tensor_np(sp_curr, key)
                if a.shape != b.shape:
                    continue
                diff = b.astype(np.float32, copy=False) - a.astype(np.float32, copy=False)
                num = float(np.linalg.norm(diff))
                denom = float(np.linalg.norm(a)) + eps
                rf = num / denom
                n = float(a.size)
                buffers.weighted_sums[row_idx, 0] += rf * n
                buffers.weight_totals[row_idx, 0] += n
        col_offset = 1
    for col in range(len(ordered_ckpts) - 1):
        wm_prev = weight_maps[col]
        wm_curr = weight_maps[col + 1]
        for (layer, t), keys in mapping.items():
            row_idx = buffers.row_index[(layer, t)]
            for key in keys:
                sp_prev = wm_prev.get(key)
                sp_curr = wm_curr.get(key)
                if not sp_prev or not sp_curr:
                    continue
                a = _load_tensor_np(sp_prev, key)
                b = _load_tensor_np(sp_curr, key)
                if a.shape != b.shape:
                    continue
                diff = b.astype(np.float32, copy=False) - a.astype(np.float32, copy=False)
                num = float(np.linalg.norm(diff))
                denom = float(np.linalg.norm(a)) + eps
                rf = num / denom
                n = float(a.size)
                buffers.weighted_sums[row_idx, col + col_offset] += rf * n
                buffers.weight_totals[row_idx, col + col_offset] += n
    mask = buffers.weight_totals > 0
    buffers.values[mask] = (buffers.weighted_sums[mask] / buffers.weight_totals[mask]).astype(np.float32)


# -----------------------------
# Step 6: Plot stacked heatmap (Option A)
# -----------------------------

def _compute_clip(values: np.ndarray, lower_p: float = 1.0, upper_p: float = 99.0) -> tuple:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, lower_p))
    vmax = float(np.percentile(finite, upper_p))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    return vmin, vmax


def render_stacked_heatmap(buffers: HeatmapBuffers, out_path: str = "heatmap_by_type.png", include_mlp: bool = False, include_base: bool = False) -> None:
    data = buffers.values.copy()
    vmin, vmax = _compute_clip(data, 1.0, 99.0)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#dddddd")

    rows, cols = data.shape
    fig_w = max(6.0, min(18.0, 1.2 * cols))
    fig_h = max(6.0, min(24.0, 0.18 * rows))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    # Layer separators every group size
    group = 5 if include_mlp else 4
    for i in range(group, rows, group):
        ax.axhline(i - 0.5, color="white", linewidth=1.0, alpha=0.8)

    # Y labels: one per sub-row (L{layer}.{type})
    ax.set_yticks(range(rows))
    ax.set_yticklabels(buffers.row_labels, fontsize=7)

    # X labels: phase-to-phase arrows
    xticks = list(range(cols))
    if include_base and buffers.include_base:
        xlabels = [f"step0→{buffers.phases[0]}"] + [f"{buffers.phases[i]}→{buffers.phases[i+1]}" for i in range(len(buffers.phases)-1)]
    else:
        xlabels = [f"{buffers.phases[i]}→{buffers.phases[i+1]}" for i in range(len(buffers.phases)-1)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")

    title_suffix = "+mlp" if include_mlp else ""
    base_suffix = " + base" if (include_base and buffers.include_base) else ""
    ax.set_title(f"Relative Frobenius shift (k−1 → k): q/k/v/o{title_suffix}{base_suffix} per layer")
    ax.set_xlabel("Training phase pairs")
    ax.set_ylabel("Layer · type")
    fig.colorbar(im, ax=ax, shrink=0.85, label="shift magnitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    cli_discover()
