"""Visualization utilities for request-generation diagnostics.

The helpers in this module are designed to help inspect per-token metadata that is
returned by generation APIs (for example, temperatures and top-p values).
The output plot aligns these metadata values with full token positions by
including prompt tokens and marks contextual regions like `<think>` blocks or
code fences in the rendered token text.
"""

from __future__ import annotations

import html
import json
import os
from typing import Any, List, Tuple

import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.figure import Figure

TEMP_COLOR_MIN = 0.0
TEMP_COLOR_MAX = 1.5
TEMP_COLOR_GAMMA = 0.72
TEMP_COLOR_STOPS = [
    (0.00, "#24364F"),
    (0.18, "#3A5A80"),
    (0.34, "#4E79A7"),
    (0.52, "#9BB2C9"),
    (0.60, "#E7ECF2"),
    (0.72, "#D39A6A"),
    (0.86, "#C9785D"),
    (1.00, "#E15759"),
]
TEMP_TOKEN_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "request_temp_tokens",
    TEMP_COLOR_STOPS,
)


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


def _empty_float_series(length: int) -> List[float | None]:
    """Create a None-initialized float-or-None sequence."""
    return [None for _ in range(length)]


def _normalize_temp(value: float, vmin: float, vmax: float) -> float:
    """Normalize temperature values while expanding the low-temperature regime."""
    if vmax <= vmin:
        return 0.0
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return t ** TEMP_COLOR_GAMMA


def _temp_to_hex(value: float, vmin: float, vmax: float) -> str:
    """Map a scalar temperature to a paper-style color hex string."""
    return mcolors.to_hex(TEMP_TOKEN_CMAP(_normalize_temp(value, vmin, vmax))).upper()


def _temp_gradient_css(vmin: float, vmax: float, *, steps: int = 12) -> str:
    """Build a CSS gradient that matches the temperature token colormap."""
    if steps < 2:
        steps = 2
    gradient_parts: List[str] = []
    for idx in range(steps):
        frac = idx / float(steps - 1)
        value = vmin + frac * (vmax - vmin)
        gradient_parts.append(f"{_temp_to_hex(value, vmin, vmax)} {frac * 100.0:.1f}%")
    return ", ".join(gradient_parts)


def plot_request_output_temps(
    request_output: Any,
    tokenizer: Any,
    *,
    output_index: int = 0,
    smooth_window: int = 1,
    show_top_p: bool = True,
    title: str | None = None,
    figsize: Tuple[float, float] = (14.0, 4.0),
    output_dir: str | None = None,
) -> Figure:
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
    output_dir:
        Optional directory where figure and HTML are saved.
        Uses `temp_trace.png` and `temp_trace.html` as filenames.

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
        aligned_smoothed_values = _moving_average(
            [float(t) for t in temperatures],
            smooth_window,
        )
        aligned_smoothed = _empty_float_series(len(token_ids))
    else:
        aligned_smoothed_values = None
    aligned_top_p: List[float | None] | None = None
    if show_top_p and top_ps is not None:
        aligned_top_p = _empty_float_series(len(token_ids))

    for idx, temp in enumerate(temperatures):
        pos = prompt_len + idx
        if pos >= len(token_ids):
            break
        aligned_temps[pos] = float(temp)
        if aligned_smoothed is not None and aligned_smoothed_values is not None:
            aligned_smoothed[pos] = aligned_smoothed_values[idx]
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
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FCFCFD")
    plot_temps = [t if t is not None else float("nan") for t in aligned_temps]
    ax.plot(
        range(len(plot_temps)),
        plot_temps,
        color="#4E79A7",
        linewidth=1.8,
        label="Predicted temperature",
    )
    if aligned_smoothed is not None:
        plot_smoothed = [t if t is not None else float("nan") for t in aligned_smoothed]
        ax.plot(
            range(len(plot_smoothed)),
            plot_smoothed,
            color="#D39A6A",
            linewidth=2.0,
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
            color="#DCE8F5",
            alpha=0.45,
            label="Inside <think>" if not think_labeled else None,
        )
        think_labeled = True
    for start, end in code_segments:
        ax.axvspan(
            start,
            end,
            color="#F3E6BF",
            alpha=0.45,
            label="Inside code" if not code_labeled else None,
        )
        code_labeled = True

    if prompt_len > 0:
        ax.axvline(
            prompt_len - 1,
            color="#98A2B3",
            linestyle="--",
            linewidth=1.0,
            label="Prompt end",
        )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Predicted temperature")
    ax.set_xlim(0, max(len(token_ids) - 1, 1))
    ax.grid(axis="y", color="#E7ECF2", linewidth=0.7, alpha=0.9)
    ax.grid(axis="x", color="#F2F4F7", linewidth=0.45, alpha=0.55)
    for spine_name in ("top", "right", "left", "bottom"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("#98A2B3")
        ax.spines[spine_name].set_linewidth(0.9)

    ax2 = None
    if aligned_top_p is not None:
        ax2 = ax.twinx()
        ax2.set_facecolor("none")
        plot_top_p = [t if t is not None else float("nan") for t in aligned_top_p]
        ax2.plot(
            range(len(plot_top_p)),
            plot_top_p,
            color="#B07AA1",
            linewidth=1.4,
            alpha=0.9,
            label="Top-p",
        )
        ax2.set_ylabel("Top-p")
        ax2.set_ylim(0.0, 1.05)
        for spine_name in ("top", "right", "left", "bottom"):
            if spine_name in ax2.spines:
                ax2.spines[spine_name].set_color("#98A2B3")
                ax2.spines[spine_name].set_linewidth(0.9)

    if title is None:
        request_id = getattr(request_output, "request_id", "unknown")
        title = f"RequestOutput temperature trace (request_id={request_id})"
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#D0D5DD",
    )

    fig.tight_layout()
    img_data_uri = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_png_path = os.path.join(output_dir, "temp_trace.png")

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=160)
        buf.seek(0)
        img_data_uri = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        with open(output_png_path, 'wb') as f:
            f.write(base64.b64decode(img_data_uri.split(',')[1]))

        gen_min, gen_max = TEMP_COLOR_MIN, TEMP_COLOR_MAX
        temp_gradient = _temp_gradient_css(gen_min, gen_max)
        gradient_ticks = [
            gen_min,
            gen_min + 0.25 * (gen_max - gen_min),
            gen_min + 0.50 * (gen_max - gen_min),
            gen_min + 0.75 * (gen_max - gen_min),
            gen_max,
        ]

        output_text_path = os.path.join(output_dir, "temp_trace.txt")
        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(decoded)

        output_json_path = os.path.join(output_dir, "temp_trace.jsonl")
        with open(output_json_path, "w", encoding="utf-8") as f:
            for idx, (token_id, (start, end)) in enumerate(zip(token_ids, spans)):
                entry = {
                    "index": idx,
                    "token_id": token_id,
                    "token_text": decoded[start:end],
                    "temperature": aligned_temps[idx],
                    "in_think": bool(think_mask[idx]),
                    "in_code": bool(code_mask[idx]),
                }
                if smooth_window > 1:
                    entry["temperature_smoothed"] = aligned_smoothed[idx] if aligned_smoothed is not None else None
                if show_top_p and aligned_top_p is not None:
                    entry["top_p"] = aligned_top_p[idx]
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        token_spans_html: List[str] = []
        for idx in range(prompt_len, len(token_ids)):
            token_text = decoded[spans[idx][0]:spans[idx][1]]
            escaped = html.escape(token_text)
            temp_value = aligned_temps[idx]
            top_p_value = aligned_top_p[idx] if aligned_top_p is not None else None
            if temp_value is None:
                color = "#E0E0E0"
                tip_lines = ["temp=n/a (no previous token)"]
            else:
                color = _temp_to_hex(temp_value, gen_min, gen_max)
                tip_lines = [f"temp={temp_value:.4f}"]
                if smooth_window > 1 and aligned_smoothed is not None:
                    smoothed_value = aligned_smoothed[idx]
                    if smoothed_value is not None:
                        tip_lines.append(f"smooth={smoothed_value:.4f}")
                if show_top_p and top_p_value is not None:
                    tip_lines.append(f"top_p={top_p_value:.4f}")
            title_text = "&#10;".join(html.escape(line, quote=True) for line in tip_lines)
            token_display = escaped
            token_spans_html.append(
                f"<span class='tok' style='background-color: {color};' data-tip=\"{title_text}\">"
                f"{token_display}</span>"
            )

        generation_html = "".join(token_spans_html)
        output_html_path = os.path.join(output_dir, "temp_trace.html")
        prompt_len_display = max(prompt_len, 0)
        with open(output_html_path, "w", encoding="utf-8") as f:
            smoothing_note = ""
            if smooth_window > 1:
                smoothing_note = f" Smoothed with window={smooth_window}."
            f.write(
                "<!doctype html>\n"
                "<html><head><meta charset='utf-8'><title>Temp Trace</title>\n"
                "<style>\n"
                "body { font-family: Arial, sans-serif; color: #1F2937; background: #FFFFFF; }\n"
                ".panel { border: 1px solid #E5E7EB; border-radius: 10px; padding: 14px 16px; background: #FCFCFD; }\n"
                ".legend-row { display: flex; align-items: center; gap: 12px; margin: 10px 0 14px; }\n"
                ".gradient-bar { flex: 1 1 auto; min-width: 220px; height: 14px; border-radius: 999px; "
                f"background: linear-gradient(90deg, {temp_gradient}); border: 1px solid #D0D5DD; }}\n"
                ".gradient-labels { display: flex; justify-content: space-between; gap: 8px; "
                "font-size: 11px; color: #667085; margin-top: 6px; }\n"
                ".token-box { white-space: pre-wrap; word-break: break-word; "
                "border: 1px solid #DDD; padding: 12px; border-radius: 8px; "
                "background: #FAFAFA; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace; "
                "line-height: 1.55; font-size: 13px; }\n"
                ".tok { padding: 0 1px; border-radius: 3px; position: relative; box-decoration-break: clone; }\n"
                ".tok:hover::after { "
                "content: attr(data-tip); "
                "position: absolute; "
                "left: 0; top: 1.2em; "
                "background: #1F2937; color: #F9FAFB; "
                "padding: 8px 10px; border-radius: 8px; "
                "font-size: 12px; line-height: 1.3; "
                "white-space: pre; "
                "box-shadow: 0 6px 18px rgba(0,0,0,0.25); "
                "z-index: 5; min-width: 200px; }\n"
                ".legend { font-size: 12px; color: #444; }\n"
                "</style></head>\n"
                "<body>\n"
                f"<h2>Temperature Trace (request_id={getattr(request_output, 'request_id', 'unknown')})</h2>\n"
                "<p>Blue shading: &lt;think&gt; spans. Yellow shading: code blocks. "
                "Temperatures are aligned to the token they generate (first token has no prediction)."
                f"{smoothing_note}</p>\n"
                f"<img src='{img_data_uri}' style='max-width: 100%; height: auto;' />\n"
                f"<p>Prompt length: {prompt_len_display} tokens, total length: {len(token_ids)} tokens.</p>\n"
                "<h3>Generated Tokens (colored by predicted temperature)</h3>\n"
                "<div class='panel'>\n"
                "<div class='legend'>Lower temperatures now use a broader cool-color range so small low-end differences remain visible. "
                "Hover a token for exact values.</div>\n"
                "<div class='legend-row'>\n"
                f"<div class='gradient-bar' aria-label='temperature color scale from {gen_min:.4f} to {gen_max:.4f}'></div>\n"
                "</div>\n"
                "<div class='gradient-labels'>\n"
                f"<span>{gradient_ticks[0]:.2f}</span>\n"
                f"<span>{gradient_ticks[1]:.2f}</span>\n"
                f"<span>{gradient_ticks[2]:.2f}</span>\n"
                f"<span>{gradient_ticks[3]:.2f}</span>\n"
                f"<span>{gradient_ticks[4]:.2f}</span>\n"
                "</div>\n"
                f"<div class='token-box'>{generation_html}</div>\n"
                "</div>\n"
                "</body></html>\n"
            )
    return fig
