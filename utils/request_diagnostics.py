"""Visualization utilities for request-generation diagnostics.

The helpers in this module are designed to help inspect per-token metadata that is
returned by generation APIs (for example, temperatures and top-p values).
The output plot aligns these metadata values with full token positions by
including prompt tokens and marks contextual regions like `<think>` blocks or
code fences in the rendered token text.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import matplotlib.pyplot as plt


def _find_ranges(text: str, start_tag: str, end_tag: str) -> List[Tuple[int, int]]:
    """Find all content ranges between matching tag markers.

    Returns half-open `(start, end)` character spans of tag content in `text`.
    If the closing tag is missing for a start tag, the range extends to the end
    of `text`.
    """
    ranges: List[Tuple[int, int]] = []
    pos = 0
    while True:
        start = text.find(start_tag, pos)
        if start == -1:
            break
        content_start = start + len(start_tag)
        end = text.find(end_tag, content_start)
        if end == -1:
            ranges.append((content_start, len(text)))
            break
        ranges.append((content_start, end))
        pos = end + len(end_tag)
    return ranges


def _mask_from_ranges(spans: List[Tuple[int, int]], ranges: List[Tuple[int, int]]) -> List[bool]:
    """Mark token spans that overlap any input character range.

    The `spans` input is expected to be a list of half-open `(start, end)`
    offsets for each token in decoded text.
    """
    mask = []
    for start, end in spans:
        inside = False
        for r_start, r_end in ranges:
            if start < r_end and end > r_start:
                inside = True
                break
        mask.append(inside)
    return mask


def _mask_to_segments(mask: List[bool]) -> List[Tuple[int, int]]:
    """Group consecutive `True` entries into inclusive index segments."""
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, val in enumerate(mask):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _moving_average(values: List[float], window: int) -> List[float]:
    """Compute a causal moving average with a growing warm-up window."""
    if window <= 1:
        return list(values)
    smoothed: List[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
            denom = window
        else:
            denom = idx + 1
        smoothed.append(running_sum / denom)
    return smoothed


def plot_request_output_temps(
    request_output: Any,
    tokenizer: Any,
    *,
    output_index: int = 0,
    smooth_window: int = 1,
    show_top_p: bool = True,
    title: str | None = None,
    figsize: Tuple[float, float] = (14.0, 4.0),
    output_path: str | None = None,
) -> plt.Figure:
    """Plot token-level temperature (and optional top-p) trajectories.

    This function aligns generation metrics (`temperatures`, `top_ps`) with token
    positions after the prompt. Prompt tokens are included in the x-axis for
    context, with generated tokens mapped immediately after.

    Parameters
    ----------
    request_output:
        A request-like object exposing `.outputs` and `.prompt_token_ids`.
    tokenizer:
        Tokenizer with a `.decode([token_id], skip_special_tokens=False)` method.
    output_index:
        Which element in `request_output.outputs` to visualize.
    smooth_window:
        Window size passed to the moving-average smoother. A value of 1 disables
        smoothing.
    show_top_p:
        When `True`, overlay top-p on a secondary axis if `.top_ps` exists.
    title:
        Optional title for the generated figure.
    figsize:
        Figure size passed to `plt.subplots`.
    output_path:
        Optional path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The plotted figure object.

    Raises
    ------
    ValueError
        If required metadata is missing or mismatched in length.
    IndexError
        If `output_index` is out of range.
    """
    outputs = getattr(request_output, "outputs", None)
    if not outputs:
        raise ValueError("RequestOutput.outputs is empty.")
    if output_index < 0 or output_index >= len(outputs):
        raise IndexError(f"output_index={output_index} is out of range.")

    output = outputs[output_index]
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None) or []
    output_token_ids = list(getattr(output, "token_ids", []) or [])
    temperatures = getattr(output, "temperatures", None)
    top_ps = getattr(output, "top_ps", None)

    if temperatures is None:
        raise ValueError("RequestOutput.outputs[0].temperatures is None.")
    if len(temperatures) != len(output_token_ids):
        raise ValueError(
            "Mismatch between temperatures and output token ids: "
            f"{len(temperatures)} vs {len(output_token_ids)}."
        )
    if show_top_p and top_ps is not None and len(top_ps) != len(output_token_ids):
        raise ValueError(
            "Mismatch between top_ps and output token ids: "
            f"{len(top_ps)} vs {len(output_token_ids)}."
        )

    token_ids = list(prompt_token_ids) + output_token_ids
    prompt_len = len(prompt_token_ids)

    aligned_temps: List[float | None] = [None] * len(token_ids)
    aligned_smoothed: List[float | None] | None = None
    if smooth_window > 1:
        smoothed = _moving_average([float(t) for t in temperatures], smooth_window)
        aligned_smoothed = [None] * len(token_ids)
    aligned_top_p: List[float | None] | None = None
    if show_top_p and top_ps is not None:
        aligned_top_p = [None] * len(token_ids)

    for idx, temp in enumerate(temperatures):
        pos = prompt_len + idx
        if pos >= len(token_ids):
            break
        aligned_temps[pos] = float(temp)
        if aligned_smoothed is not None:
            aligned_smoothed[pos] = smoothed[idx]
        if aligned_top_p is not None and top_ps is not None:
            aligned_top_p[pos] = float(top_ps[idx])

    decoded = ""
    spans: List[Tuple[int, int]] = []
    for token_id in token_ids:
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        start = len(decoded)
        decoded += token_text
        spans.append((start, len(decoded)))

    think_ranges = _find_ranges(decoded, "<think>", "</think>")
    code_ranges = _find_ranges(decoded, "```", "```")
    think_mask = _mask_from_ranges(spans, think_ranges)
    code_mask = _mask_from_ranges(spans, code_ranges)

    fig, ax = plt.subplots(figsize=figsize)
    plot_temps = [t if t is not None else float("nan") for t in aligned_temps]
    ax.plot(
        range(len(plot_temps)),
        plot_temps,
        color="#2E3A59",
        linewidth=1.6,
        label="Predicted temperature",
    )
    if aligned_smoothed is not None:
        plot_smoothed = [t if t is not None else float("nan") for t in aligned_smoothed]
        ax.plot(
            range(len(plot_smoothed)),
            plot_smoothed,
            color="#E07A5F",
            linewidth=1.8,
            label=f"Smoothed (window={smooth_window})",
        )

    think_segments = _mask_to_segments(think_mask)
    code_segments = _mask_to_segments(code_mask)
    think_labeled = False
    code_labeled = False
    for start, end in think_segments:
        ax.axvspan(
            start,
            end,
            color="#CFE8FF",
            alpha=0.35,
            label="Inside <think>" if not think_labeled else None,
        )
        think_labeled = True
    for start, end in code_segments:
        ax.axvspan(
            start,
            end,
            color="#FFF2B2",
            alpha=0.35,
            label="Inside code" if not code_labeled else None,
        )
        code_labeled = True

    if prompt_len > 0:
        ax.axvline(prompt_len - 1, color="#888888", linestyle="--", linewidth=1, label="Prompt end")

    ax.set_xlabel("Token position")
    ax.set_ylabel("Predicted temperature")
    ax.set_xlim(0, max(len(token_ids) - 1, 1))
    ax.grid(alpha=0.2)

    ax2 = None
    if aligned_top_p is not None:
        ax2 = ax.twinx()
        plot_top_p = [t if t is not None else float("nan") for t in aligned_top_p]
        ax2.plot(
            range(len(plot_top_p)),
            plot_top_p,
            color="#6C8EBF",
            linewidth=1.4,
            alpha=0.85,
            label="Top-p",
        )
        ax2.set_ylabel("Top-p")
        ax2.set_ylim(0.0, 1.05)

    if title is None:
        request_id = getattr(request_output, "request_id", "unknown")
        title = f"RequestOutput temperature trace (request_id={request_id})"
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    ax.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160)
    return fig
