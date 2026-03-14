#!/usr/bin/env python3
"""
Visualize per-token predictions from a trained shadow controller checkpoint.

The script expects the per-token diagnostics dataset produced by
`script/collect_pertoken_diagnostics.py`. You can select a sequence by the
original dataset row index (`--row_index`) or directly by `--seq_id`.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from shadow_controller_dist_match import (
    FEATURE_SET_N_BINARY,
    apply_standardize,
    build_feature_bundle,
    build_feature_bundle_from_token_data,
    get_feature_matrix,
    hard_top_p_from_sorted_logits_torch,
    instantiate_controller,
    pick_device,
    prepare_feature_sets,
    split_by_seq_mod,
)

TEMP_COLOR_MIN = 0.0
TEMP_COLOR_MAX = 1.5


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("plot_shadow_controller_curve")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    smoothed: List[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
            denom = window
        else:
            denom = idx + 1
        smoothed.append(running / denom)
    return smoothed


def temp_to_hex(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))

    hue = 210.0 * (1.0 - t)
    saturation = 0.85
    lightness = 0.75
    c = (1.0 - abs(2.0 * lightness - 1.0)) * saturation
    h_prime = hue / 60.0
    x = c * (1.0 - abs(h_prime % 2.0 - 1.0))
    r1 = g1 = b1 = 0.0
    if 0.0 <= h_prime < 1.0:
        r1, g1, b1 = c, x, 0.0
    elif 1.0 <= h_prime < 2.0:
        r1, g1, b1 = x, c, 0.0
    elif 2.0 <= h_prime < 3.0:
        r1, g1, b1 = 0.0, c, x
    elif 3.0 <= h_prime < 4.0:
        r1, g1, b1 = 0.0, x, c
    elif 4.0 <= h_prime < 5.0:
        r1, g1, b1 = x, 0.0, c
    elif 5.0 <= h_prime <= 6.0:
        r1, g1, b1 = c, 0.0, x
    m = lightness - c / 2.0
    r = int(round((r1 + m) * 255))
    g = int(round((g1 + m) * 255))
    b = int(round((b1 + m) * 255))
    return f"#{r:02X}{g:02X}{b:02X}"


def display_token(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def infer_model_name_from_checkpoint_path(path: str) -> str:
    stem = Path(path).stem
    for suffix in ("_checkpoint", "_state_dict"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_checkpoint_metadata(
    checkpoint_path: str,
    model_name_override: Optional[str],
    logger: logging.Logger,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], bool]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw.get("metadata"), dict):
        return raw["state_dict"], dict(raw["metadata"]), False

    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(raw).__name__}.")

    state_dict = raw
    model_name = model_name_override or infer_model_name_from_checkpoint_path(checkpoint_path)
    manifest_path = os.path.join(os.path.dirname(checkpoint_path), "checkpoint_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        models = manifest.get("models") if isinstance(manifest, dict) else None
        if isinstance(models, dict) and isinstance(models.get(model_name), dict):
            metadata = dict(models[model_name])
            return state_dict, metadata, False

    logger.warning(
        "Checkpoint metadata not found for %s. Falling back to legacy reconstruction.",
        checkpoint_path,
    )
    return state_dict, {"model_name": model_name, "feature_set": model_name}, True


def resolve_sequence_info(
    sequences_ds: Any,
    *,
    row_index: Optional[int],
    seq_id: Optional[int],
) -> Dict[str, Any]:
    columns = [
        "seq_id",
        "dataset_index",
        "split",
        "model_id",
        "autodeco_ckpt",
        "prompt_len",
        "gen_len",
        "seq_len",
        "token_count",
    ]
    data = sequences_ds.select_columns(columns)[:]
    seq_ids = np.asarray(data["seq_id"], dtype=np.int64)

    if seq_id is not None:
        matches = np.flatnonzero(seq_ids == seq_id)
        if len(matches) == 0:
            raise ValueError(f"seq_id={seq_id} was not found in diagnostics dataset.")
        if len(matches) > 1:
            raise ValueError(f"seq_id={seq_id} matched multiple sequence rows.")
        idx = int(matches[0])
    elif row_index is not None:
        dataset_indices = np.asarray(data["dataset_index"], dtype=np.int64)
        matches = np.flatnonzero(dataset_indices == row_index)
        if len(matches) == 0:
            raise ValueError(f"row_index={row_index} was not found in diagnostics dataset sequences.")
        if len(matches) > 1:
            raise ValueError(
                f"row_index={row_index} matched multiple sequence rows; pass --seq_id to disambiguate."
            )
        idx = int(matches[0])
    else:
        idx = 0

    return {
        "seq_id": int(seq_ids[idx]),
        "dataset_index": int(data["dataset_index"][idx]),
        "split": str(data["split"][idx]),
        "model_id": str(data["model_id"][idx]),
        "autodeco_ckpt": str(data["autodeco_ckpt"][idx]),
        "prompt_len": int(data["prompt_len"][idx]),
        "gen_len": int(data["gen_len"][idx]),
        "seq_len": int(data["seq_len"][idx]),
        "token_count": int(data["token_count"][idx]),
    }


def load_sequence_token_data(tokens_ds: Any, seq_id: int) -> Dict[str, np.ndarray]:
    seq_ids = np.asarray(tokens_ds["seq_id"], dtype=np.int64)
    indices = np.flatnonzero(seq_ids == seq_id)
    if len(indices) == 0:
        raise ValueError(f"seq_id={seq_id} has no token rows.")

    cols = [
        "seq_id",
        "t",
        "token_id",
        "T_hat",
        "p_hat",
        "H_norm",
        "gap12",
        "p_max",
        "mass10",
        "mass50",
        "mass200",
        "expH",
        "is_boundary",
        "is_punct",
        "is_whitespace",
        "is_newline",
        "in_code_block",
        "topk_logits",
    ]
    subset = tokens_ds.select(indices.tolist()).select_columns(cols).with_format("numpy")
    data = subset[:]
    order = np.argsort(data["t"], kind="stable")
    return {key: np.asarray(value)[order] for key, value in data.items()}


def metadata_is_complete(metadata: Dict[str, Any]) -> bool:
    base_keys = {"feature_set", "profile_k", "dist_k", "hidden", "dropout", "quant_step", "tmin", "tmax", "pmin", "pmax"}
    if not base_keys.issubset(metadata):
        return False
    feature_set = str(metadata["feature_set"])
    if feature_set == "mean":
        return True
    return {"mu", "sd", "n_binary"}.issubset(metadata)


def derive_legacy_metadata(
    diagnostics_path: str,
    metadata: Dict[str, Any],
    logger: logging.Logger,
    profile_k: int,
    dist_k: int,
    val_mod: int,
    hidden: int,
    dropout: float,
    quant_step: float,
) -> Dict[str, Any]:
    feature_set = str(metadata["feature_set"])
    full_bundle = build_feature_bundle(diagnostics_path, profile_k=profile_k, dist_k=dist_k, logger=logger)
    tr_mask, _ = split_by_seq_mod(full_bundle.seq_id, val_mod)
    prepared = prepare_feature_sets(full_bundle, tr_mask)
    tr_idx = np.flatnonzero(tr_mask)
    if len(tr_idx) == 0:
        raise ValueError("Legacy checkpoint reconstruction failed because the training split is empty.")

    prepared_set = prepared[feature_set]
    return {
        "model_name": feature_set,
        "feature_set": feature_set,
        "hidden": int(hidden),
        "dropout": float(dropout),
        "profile_k": int(profile_k),
        "dist_k": int(dist_k),
        "quant_step": float(quant_step),
        "tmin": float(full_bundle.yT[tr_idx].min()),
        "tmax": float(full_bundle.yT[tr_idx].max()),
        "pmin": float(full_bundle.yp[tr_idx].min()),
        "pmax": float(full_bundle.yp[tr_idx].max()),
        "n_binary": int(prepared_set.n_binary),
        "mu": prepared_set.mu.tolist(),
        "sd": prepared_set.sd.tolist(),
        "feature_names": list(prepared_set.feature_names),
    }


def load_tokenizer(name_or_path: Optional[str], trust_remote_code: bool, logger: logging.Logger) -> Any | None:
    if not name_or_path:
        return None
    try:
        return AutoTokenizer.from_pretrained(
            name_or_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except Exception as exc:  # pragma: no cover - best-effort UX fallback
        logger.warning("Failed to load tokenizer from %s: %s", name_or_path, exc)
        return None


def decode_tokens(token_ids: List[int], tokenizer: Any | None) -> List[str]:
    if tokenizer is None:
        return [f"<tok:{token_id}>" for token_id in token_ids]
    return [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to shadow controller checkpoint or state_dict.")
    parser.add_argument(
        "--diagnostics_path",
        default=None,
        help="Path to diagnostics dataset saved with load_from_disk. Optional for new rich checkpoints that record source_path.",
    )
    parser.add_argument("--row_index", type=int, default=None, help="Original dataset row index stored in sequences.dataset_index.")
    parser.add_argument("--seq_id", type=int, default=None, help="Direct sequence id from the diagnostics dataset.")
    parser.add_argument("--tokenizer_name_or_path", default=None, help="Override tokenizer for decoding tokens in the HTML output.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smooth_window", type=int, default=0)
    parser.add_argument("--output_dir", default="figure/shadow_controller_trace")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--hide_targets", action="store_true", help="Do not overlay the AutoDeco target T_hat/p_hat curves.")
    parser.add_argument("--model_name", default=None, help="Only needed for legacy state_dict checkpoints without metadata.")
    parser.add_argument("--profile_k", type=int, default=64, help="Legacy fallback only.")
    parser.add_argument("--dist_k", type=int, default=200, help="Legacy fallback only.")
    parser.add_argument("--val_mod", type=int, default=10, help="Legacy fallback only.")
    parser.add_argument("--hidden", type=int, default=128, help="Legacy fallback only.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Legacy fallback only.")
    parser.add_argument("--quant_step", type=float, default=1.0 / 128.0, help="Legacy fallback only.")
    args = parser.parse_args()

    logger = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    state_dict, metadata, needs_legacy_rebuild = load_checkpoint_metadata(
        args.checkpoint_path,
        args.model_name,
        logger,
    )
    diagnostics_path = args.diagnostics_path or metadata.get("source_path")
    if not diagnostics_path:
        raise ValueError(
            "--diagnostics_path is required for legacy checkpoints without embedded source_path metadata."
        )
    if needs_legacy_rebuild or not metadata_is_complete(metadata):
        metadata = derive_legacy_metadata(
            diagnostics_path=diagnostics_path,
            metadata=metadata,
            logger=logger,
            profile_k=args.profile_k,
            dist_k=args.dist_k,
            val_mod=args.val_mod,
            hidden=args.hidden,
            dropout=args.dropout,
            quant_step=args.quant_step,
        )

    feature_set = str(metadata["feature_set"])
    profile_k = int(metadata["profile_k"])
    dist_k = int(metadata["dist_k"])
    quant_step = float(metadata.get("quant_step", 0.0))

    ds = load_from_disk(diagnostics_path)
    sequence_info = resolve_sequence_info(
        ds["sequences"],
        row_index=args.row_index,
        seq_id=args.seq_id,
    )
    token_data = load_sequence_token_data(ds["tokens"], sequence_info["seq_id"])
    seq_bundle = build_feature_bundle_from_token_data(
        token_data,
        profile_k=profile_k,
        dist_k=dist_k,
        logger=logger,
    )

    feature_matrix = get_feature_matrix(seq_bundle, feature_set)
    if feature_matrix is not None:
        mu = np.asarray(metadata["mu"], dtype=np.float32)
        sd = np.asarray(metadata["sd"], dtype=np.float32)
        n_binary = int(metadata.get("n_binary", FEATURE_SET_N_BINARY[feature_set]))
        feature_matrix = apply_standardize(feature_matrix, mu, sd, n_binary=n_binary)

    device = pick_device(args.device)
    model = instantiate_controller(
        feature_set=feature_set,
        feature_dim=0 if feature_matrix is None else feature_matrix.shape[1],
        hidden=int(metadata["hidden"]),
        dropout=float(metadata["dropout"]),
        tmin=float(metadata["tmin"]),
        tmax=float(metadata["tmax"]),
        pmin=float(metadata["pmin"]),
        pmax=float(metadata["pmax"]),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    logits_t = torch.from_numpy(seq_bundle.topk_logits).to(device=device, dtype=torch.float32)
    target_t = torch.from_numpy(seq_bundle.yT).to(device=device, dtype=torch.float32)
    target_p = torch.from_numpy(seq_bundle.yp).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if feature_matrix is None:
            pred_t, pred_p = model(None, n=logits_t.shape[0])
        else:
            feat_t = torch.from_numpy(feature_matrix).to(device=device, dtype=torch.float32)
            pred_t, pred_p = model(feat_t)
        if quant_step > 0:
            pred_t = torch.round(pred_t / quant_step) * quant_step
            pred_p = torch.round(pred_p / quant_step) * quant_step
        _, target_stats = hard_top_p_from_sorted_logits_torch(logits_t, target_t, target_p)
        _, pred_stats = hard_top_p_from_sorted_logits_torch(logits_t, pred_t, pred_p)

    positions = token_data["t"].astype(np.int64).tolist()
    token_ids = token_data["token_id"].astype(np.int64).tolist()
    pred_t_np = pred_t.detach().cpu().numpy()
    pred_p_np = pred_p.detach().cpu().numpy()
    target_t_np = seq_bundle.yT
    target_p_np = seq_bundle.yp
    pred_entropy = pred_stats["entropy"].detach().cpu().numpy()
    target_entropy = target_stats["entropy"].detach().cpu().numpy()
    pred_nucleus = pred_stats["nucleus_size"].detach().cpu().numpy()
    target_nucleus = target_stats["nucleus_size"].detach().cpu().numpy()
    pred_top1 = pred_stats["top1"].detach().cpu().numpy()
    target_top1 = target_stats["top1"].detach().cpu().numpy()

    tokenizer_name = (
        args.tokenizer_name_or_path
        or sequence_info.get("model_id")
        or sequence_info.get("autodeco_ckpt")
    )
    tokenizer = load_tokenizer(tokenizer_name, args.trust_remote_code, logger)
    token_texts = decode_tokens(token_ids, tokenizer)
    decoded = "".join(token_texts)

    pred_t_smoothed = moving_average(pred_t_np.tolist(), args.smooth_window) if args.smooth_window > 1 else None
    pred_p_smoothed = moving_average(pred_p_np.tolist(), args.smooth_window) if args.smooth_window > 1 else None
    target_t_smoothed = moving_average(target_t_np.tolist(), args.smooth_window) if args.smooth_window > 1 else None
    target_p_smoothed = moving_average(target_p_np.tolist(), args.smooth_window) if args.smooth_window > 1 else None

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax_t, ax_p = axes
    ax_t.plot(positions, pred_t_np, color="#2E3A59", linewidth=1.7, label="Predicted temperature")
    if not args.hide_targets:
        ax_t.plot(positions, target_t_np, color="#C8553D", linewidth=1.2, linestyle="--", label="Target T_hat")
    if pred_t_smoothed is not None:
        ax_t.plot(positions, pred_t_smoothed, color="#4C6A92", linewidth=2.0, alpha=0.85, label=f"Predicted T smooth ({args.smooth_window})")
    if not args.hide_targets and target_t_smoothed is not None:
        ax_t.plot(positions, target_t_smoothed, color="#D98F7C", linewidth=1.7, alpha=0.85, label=f"Target T smooth ({args.smooth_window})")
    ax_t.set_ylabel("Temperature")
    ax_t.set_title(
        f"Shadow controller trace ({feature_set}) | seq_id={sequence_info['seq_id']} | dataset_index={sequence_info['dataset_index']}"
    )
    ax_t.grid(alpha=0.2)
    ax_t.legend(loc="upper right")

    ax_p.plot(positions, pred_p_np, color="#44633F", linewidth=1.7, label="Predicted top-p")
    if not args.hide_targets:
        ax_p.plot(positions, target_p_np, color="#A35D2D", linewidth=1.2, linestyle="--", label="Target p_hat")
    if pred_p_smoothed is not None:
        ax_p.plot(positions, pred_p_smoothed, color="#6B8E65", linewidth=2.0, alpha=0.85, label=f"Predicted p smooth ({args.smooth_window})")
    if not args.hide_targets and target_p_smoothed is not None:
        ax_p.plot(positions, target_p_smoothed, color="#C88E69", linewidth=1.7, alpha=0.85, label=f"Target p smooth ({args.smooth_window})")
    ax_p.set_xlabel("Token position (t)")
    ax_p.set_ylabel("Top-p")
    ax_p.set_ylim(0.0, 1.05)
    ax_p.grid(alpha=0.2)
    ax_p.legend(loc="upper right")

    fig.tight_layout()
    output_png_path = os.path.join(args.output_dir, "shadow_trace.png")
    fig.savefig(output_png_path, dpi=160)

    output_json_path = os.path.join(args.output_dir, "shadow_trace.jsonl")
    with open(output_json_path, "w", encoding="utf-8") as f:
        for idx, token_id in enumerate(token_ids):
            entry = {
                "index": idx,
                "position": positions[idx],
                "token_id": int(token_id),
                "token_text": token_texts[idx],
                "pred_temperature": float(pred_t_np[idx]),
                "target_temperature": float(target_t_np[idx]),
                "pred_top_p": float(pred_p_np[idx]),
                "target_top_p": float(target_p_np[idx]),
                "pred_entropy": float(pred_entropy[idx]),
                "target_entropy": float(target_entropy[idx]),
                "pred_nucleus_size": float(pred_nucleus[idx]),
                "target_nucleus_size": float(target_nucleus[idx]),
                "pred_top1": float(pred_top1[idx]),
                "target_top1": float(target_top1[idx]),
            }
            if pred_t_smoothed is not None and pred_p_smoothed is not None:
                entry["pred_temperature_smoothed"] = float(pred_t_smoothed[idx])
                entry["pred_top_p_smoothed"] = float(pred_p_smoothed[idx])
            if target_t_smoothed is not None and target_p_smoothed is not None:
                entry["target_temperature_smoothed"] = float(target_t_smoothed[idx])
                entry["target_top_p_smoothed"] = float(target_p_smoothed[idx])
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    output_text_path = os.path.join(args.output_dir, "shadow_trace.txt")
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(decoded)

    token_spans_html: List[str] = []
    for idx, token_text in enumerate(token_texts):
        escaped = html.escape(token_text)
        color = temp_to_hex(float(pred_t_np[idx]), TEMP_COLOR_MIN, TEMP_COLOR_MAX)
        tip_lines = [
            f"t={positions[idx]}",
            f"pred_T={pred_t_np[idx]:.4f}",
            f"target_T={target_t_np[idx]:.4f}",
            f"pred_p={pred_p_np[idx]:.4f}",
            f"target_p={target_p_np[idx]:.4f}",
            f"pred_entropy={pred_entropy[idx]:.4f}",
            f"target_entropy={target_entropy[idx]:.4f}",
            f"pred_nucleus={pred_nucleus[idx]:.2f}",
            f"target_nucleus={target_nucleus[idx]:.2f}",
            f"pred_top1={pred_top1[idx]:.4f}",
            f"target_top1={target_top1[idx]:.4f}",
            display_token(token_text),
        ]
        title = "&#10;".join(html.escape(line, quote=True) for line in tip_lines)
        token_spans_html.append(
            f"<span class='tok' style='background-color: {color};' data-tip=\"{title}\">{escaped}</span>"
        )

    smoothing_note = f" Smoothed with window={args.smooth_window}." if args.smooth_window > 1 else ""
    output_html_path = os.path.join(args.output_dir, "shadow_trace.html")
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html>\n"
            "<html><head><meta charset='utf-8'><title>Shadow Trace</title>\n"
            "<style>\n"
            "body { font-family: Arial, sans-serif; }\n"
            ".token-box { white-space: pre-wrap; word-break: break-word; border: 1px solid #DDD; "
            "padding: 12px; border-radius: 8px; background: #FAFAFA; "
            "font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; }\n"
            ".tok { padding: 0 1px; border-radius: 3px; position: relative; }\n"
            ".tok:hover::after { content: attr(data-tip); position: absolute; left: 0; top: 1.2em; "
            "background: #1F2937; color: #F9FAFB; padding: 8px 10px; border-radius: 8px; font-size: 12px; "
            "line-height: 1.3; white-space: pre; box-shadow: 0 6px 18px rgba(0,0,0,0.25); z-index: 5; min-width: 220px; }\n"
            ".summary { font-size: 14px; color: #333; }\n"
            "</style></head>\n"
            "<body>\n"
            f"<h2>Shadow Controller Trace ({feature_set})</h2>\n"
            f"<p class='summary'>seq_id={sequence_info['seq_id']} | dataset_index={sequence_info['dataset_index']} | "
            f"split={html.escape(sequence_info['split'])} | token_count={len(token_ids)}.{smoothing_note}</p>\n"
            f"<p class='summary'>checkpoint={html.escape(args.checkpoint_path)}<br />"
            f"diagnostics_path={html.escape(diagnostics_path)}</p>\n"
            "<p class='summary'>Tokens are colored by predicted temperature. Hover for predicted vs target controller outputs and induced distribution stats.</p>\n"
            "<img src='shadow_trace.png' style='max-width: 100%; height: auto;' />\n"
            f"<div class='token-box'>{''.join(token_spans_html)}</div>\n"
            "</body></html>\n"
        )

    logger.info("Saved figure to %s", output_png_path)
    logger.info("Saved HTML trace to %s", output_html_path)
    logger.info("Saved token JSONL to %s", output_json_path)


if __name__ == "__main__":
    main()
