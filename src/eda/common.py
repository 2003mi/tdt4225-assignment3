"""Printing and plotting helpers shared by all EDA modules."""
import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from ..paths import OUTPUT_DIR

PLOT_DIR = OUTPUT_DIR / "eda"

# Plot colours
BAR = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
GENDER = {1: ("Female", "#2a78d6"), 2: ("Male", "#eb6834"), 0: ("Unknown", "#a3a29c")}


def banner(title):
    print("\n\n" + "#" * 72)
    print(f"# {title}")
    print("#" * 72)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def subsection(title):
    print(f"\n--- {title} ---")


def show(data, headers="keys", **kwargs):
    """Pretty-print a DataFrame or a list of rows with tabulate."""
    if isinstance(data, pd.DataFrame):
        if headers == "keys":
            headers = list(data.columns)
        data = data.astype(object).values.tolist()
    # numpy numbers (e.g. np.int64) -> plain Python numbers, so tabulate formats
    # integers as integers instead of as floats like "4,912.00"
    rows = [[v.item() if isinstance(v, np.generic) else v for v in r] for r in data]
    # Columns holding text are printed as-is. Without this, tabulate 0.9 tries to
    # apply the "," number format to strings like "1" and crashes.
    n_cols = max((len(r) for r in rows), default=0)
    text_cols = [i for i in range(n_cols) if any(isinstance(r[i], str) for r in rows if i < len(r))]
    print(tabulate(rows, headers=headers, tablefmt="simple", floatfmt=",.2f",
                   intfmt=",", disable_numparse=text_cols, **kwargs))


def pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "-"


def short(value, n=45):
    """Value as text, cut to n characters; missing values become ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    return text if len(text) <= n else text[: n - 1] + "…"


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    })


def save_plot(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / name)
    plt.close(fig)
    print(f"  [plot saved] output/eda/{name}")


def hbar(labels, values, title, xlabel, name, fmt="{:,.0f}"):
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(labels) + 1.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color=BAR, height=0.72)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    xmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.01, yi, fmt.format(v), va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, xmax * 1.15)
    save_plot(fig, name)


def log_bins(values, n=40):
    lo, hi = max(values.min(), 1), max(values.max(), 2)
    return np.logspace(np.log10(lo), np.log10(hi), n)


def hist(ax, values, bins, title, xlabel, log_x=False):
    ax.hist(values, bins=bins, color=BAR, edgecolor="white", linewidth=0.6)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Movies")
