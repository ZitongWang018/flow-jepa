"""Plot final sparse336 Cube router use and load balance from real inference."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
INPUT = HERE / "route_probe.json"
PNG = HERE / "route_balance.png"
SVG = HERE / "route_balance.svg"
REPORT = HERE / "route_balance_stats.json"

INK = "#183247"
MUTED = "#5F7684"
LINE = "#D9E4E9"
RED = "#CA4551"
BLUE = "#2B7597"

data = json.loads(INPUT.read_text(encoding="utf-8"))
train = np.asarray(data["training_usage_ema"], dtype=float)
assert train.shape == (6, 4)
numerator = np.zeros((6, 4), dtype=float)
denominator = np.zeros(6, dtype=float)
by_euler_n = np.zeros((8, 4), dtype=float)
by_euler_d = np.zeros(8, dtype=float)
layer_euler = np.zeros((6, 8, 4), dtype=float)
for entry in data["inference_entries"]:
    layer = int(entry["block"])
    step = int(entry["euler_step"])
    n = int(entry["candidate_samples"])
    frequencies = np.asarray(entry["selection_frequency"], dtype=float)
    assert 0 <= layer < 6 and 0 <= step < 8 and frequencies.shape == (4,)
    numerator[layer] += n * frequencies
    denominator[layer] += n
    by_euler_n[step] += n * frequencies
    by_euler_d[step] += n
    layer_euler[layer, step] = frequencies
assert np.all(denominator > 0) and np.all(by_euler_d > 0)
inference = numerator / denominator[:, None]
by_euler = by_euler_n / by_euler_d[:, None]
assert np.allclose(inference.sum(axis=1), 1, atol=2e-5)
assert np.allclose(train.sum(axis=1), 1, atol=2e-5)


def entropy(matrix):
    values = np.clip(matrix, 1e-12, 1)
    return (-np.sum(values * np.log(values), axis=1) / math.log(4)).tolist()


training_span = np.ptp(train, axis=1)
inference_span = np.ptp(inference, axis=1)
worst = int(inference_span.argmax())
euler_span = np.ptp(by_euler, axis=1)
worst_euler = int(euler_span.argmax())
dominant = layer_euler.max(axis=2)
dominant_group = layer_euler.argmax(axis=2) + 1
peak_layer, peak_step = np.unravel_index(dominant.argmax(), dominant.shape)
stats = {
    "probe": "20 real Cube starts, independent CEM route, final epoch-20 checkpoint",
    "training_usage_ema": train.tolist(),
    "inference_selection_frequency": inference.tolist(),
    "inference_by_euler": by_euler.tolist(),
    "inference_by_layer_and_euler": layer_euler.tolist(),
    "dominant_group_by_layer_and_euler": dominant_group.tolist(),
    "dominant_group_frequency_by_layer_and_euler": dominant.tolist(),
    "inference_candidate_samples_by_layer": denominator.astype(int).tolist(),
    "training_normalized_entropy_by_layer": entropy(train),
    "inference_normalized_entropy_by_layer": entropy(inference),
    "training_max_minus_min_by_layer": training_span.tolist(),
    "inference_max_minus_min_by_layer": inference_span.tolist(),
    "inference_max_minus_min_by_euler": euler_span.tolist(),
    "inference_normalized_entropy_by_euler": entropy(by_euler),
    "worst_inference_layer": worst + 1,
    "worst_inference_euler_step": worst_euler + 1,
    "peak_conditional_layer": int(peak_layer + 1),
    "peak_conditional_euler_step": int(peak_step + 1),
    "peak_conditional_frequency": float(dominant.max()),
    "inference_min": float(inference.min()),
    "inference_max": float(inference.max()),
    "training_min": float(train.min()),
    "training_max": float(train.max()),
}
REPORT.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

fig = plt.figure(figsize=(19.2, 8.0), dpi=180, facecolor="white")
fig.text(.055, .947, "Dynamic-channel load in the completed Cube model",
         fontsize=21, weight="bold", color=INK)
fig.text(.055, .902,
         "6 predictor layers × 4 candidate groups (48 D each); one group selected + 144 shared D always active",
         fontsize=11.7, color=MUTED)
fig.text(.055, .86, "Cell = selection frequency  ·  balanced = 25% per dynamic group",
         fontsize=11.2, color=BLUE, weight="bold")

cmap = plt.get_cmap("RdBu_r")
norm = colors.TwoSlopeNorm(vmin=0, vcenter=.25, vmax=.5)
positions = ((.049, .245, .236, .51), (.342, .245, .236, .51))
titles = ("Training: final EMA", "Inference: by layer")
matrices = (train, inference)
for side, (position, title, matrix) in enumerate(zip(positions, titles, matrices)):
    ax = fig.add_axes(position)
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="none")
    ax.set_xticks(np.arange(4), ["D1", "D2", "D3", "D4"],
                  fontsize=11, color=INK)
    ax.set_yticks(np.arange(matrix.shape[0]),
                  [f"L{i}" for i in range(1, 7)],
                  fontsize=11, color=INK)
    ax.tick_params(length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for r in range(matrix.shape[0]):
        for c in range(4):
            val = matrix[r, c]
            ax.text(c, r, f"{100*val:.1f}%", ha="center", va="center",
                    color="white" if val < .10 or val > .40 else INK,
                    fontsize=11.3, weight="bold")
            ax.add_patch(Rectangle((c-.5, r-.5), 1, 1, fill=False,
                                   edgecolor="white", linewidth=2))
    if side == 1:
        ax.add_patch(Rectangle((-.51, worst-.51), 4.02, 1.02, fill=False,
                               edgecolor=RED, linewidth=2.4, clip_on=False))
    fig.text(position[0], .809, title, fontsize=15, color=INK, weight="bold")

# The layer-by-Euler panel reveals conditional routing hidden by aggregation.
ax_cond = fig.add_axes((.635, .245, .324, .51))
ax_cond.imshow(dominant, aspect="auto", cmap="OrRd",
               norm=colors.Normalize(vmin=.25, vmax=1), interpolation="none")
ax_cond.set_xticks(np.arange(8), [f"E{i}" for i in range(1, 9)],
                   fontsize=10.5, color=INK)
ax_cond.set_yticks(np.arange(6), [f"L{i}" for i in range(1, 7)],
                   fontsize=11, color=INK)
ax_cond.tick_params(length=0, pad=6)
for spine in ax_cond.spines.values():
    spine.set_visible(False)
for layer in range(6):
    for step in range(8):
        value = dominant[layer, step]
        ax_cond.text(step, layer, f"D{dominant_group[layer, step]}\n{100*value:.0f}%",
                     ha="center", va="center", fontsize=8.8,
                     color="white" if value > .72 else INK,
                     weight="bold")
        ax_cond.add_patch(Rectangle((step-.5, layer-.5), 1, 1,
                                    fill=False, edgecolor="white", linewidth=1.5))
ax_cond.add_patch(Rectangle((-.51, 5-.51), 8.02, 1.02, fill=False,
                            edgecolor=RED, linewidth=2.4, clip_on=False))
fig.text(.635, .809, "Inference: dominant group at each layer × Euler step",
         fontsize=14.5, color=INK, weight="bold")

fig.text(.052, .155,
         f"Train EMA: {100*train.min():.1f}–{100*train.max():.1f}%"
         f"   |   Inference by layer: {100*inference.min():.1f}–{100*inference.max():.1f}%"
         f"   |   L6 changes from D3 {100*layer_euler[5,0,2]:.1f}% (E1)"
         f" to D2 {100*layer_euler[5,7,1]:.1f}% (E8)",
         fontsize=11.5, color=INK, weight="bold")
fig.text(.052, .109,
         "Red outlines: largest per-layer spread and time-dependent L6. The 144 shared dimensions are always active (not plotted).",
         fontsize=10.5, color=MUTED)
fig.text(.052, .07,
         "Independent routing over 20 real Cube starts; CEM choices are correlated. The middle panel pools 8 Euler steps.",
         fontsize=10.5, color=MUTED)
fig.savefig(PNG, dpi=180, facecolor="white")
fig.savefig(SVG, facecolor="white")
plt.close(fig)
print(PNG)
print(SVG)
print(REPORT)

