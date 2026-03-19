"""
Plotting utilities for experiment logs.

Two usage modes:
  1. Direct call from experiment code with in-memory data
  2. CLI: python plot_utils.py <log_path_or_dir>

Parses .log files produced by equiv_experiments.py and inv_regression.py
and generates publication-quality loss curve PNGs.
"""

import re
import os
import sys
import ast
import glob
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  Plotting style
# ──────────────────────────────────────────────────────────────

STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
    "figure.figsize": (8, 5),
}

COLORS = [
    "#2196F3",  # blue
    "#FF5722",  # red-orange
    "#4CAF50",  # green
    "#9C27B0",  # purple
    "#FF9800",  # orange
    "#00BCD4",  # cyan
    "#E91E63",  # pink
    "#795548",  # brown
    "#607D8B",  # blue-grey
]


def _apply_style():
    plt.rcParams.update(STYLE)


# ──────────────────────────────────────────────────────────────
#  Direct-call plotting (from experiment functions)
# ──────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses,
    val_losses_dict=None,
    title="Training Curves",
    save_path=None,
    show=True,
):
    """
    Plot train loss and optional per-structure validation losses.

    Args:
        train_losses: list of floats, one per epoch
        val_losses_dict: dict {structure_name: [loss_per_epoch]} or None
        title: plot title
        save_path: if given, save PNG here
        show: if True, call plt.show()
    """
    _apply_style()
    fig, ax = plt.subplots()

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color=COLORS[0], label="Train Loss", linewidth=2.2)

    if val_losses_dict:
        for i, (name, vals) in enumerate(val_losses_dict.items()):
            color = COLORS[(i + 1) % len(COLORS)]
            val_epochs = range(1, len(vals) + 1)
            ax.plot(val_epochs, vals, color=color, label=f"Val [{name}]", linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pretrain_finetune(
    pretrain_losses,
    finetune_train_losses,
    finetune_val_losses=None,
    title="Pretrain → Finetune",
    save_path=None,
    show=True,
):
    """
    Two-phase plot with a vertical divider at the pretrain/finetune boundary.

    Args:
        pretrain_losses: list of pretrain-phase epoch losses
        finetune_train_losses: list of finetune-phase train losses
        finetune_val_losses: list of finetune-phase val losses (optional)
        title: plot title
        save_path: optional save path
        show: if True, call plt.show()
    """
    _apply_style()
    fig, ax = plt.subplots()

    n_pre = len(pretrain_losses)
    n_ft = len(finetune_train_losses)

    # Pretrain
    pre_epochs = range(1, n_pre + 1)
    ax.plot(pre_epochs, pretrain_losses, color=COLORS[0], label="Pretrain Loss", linewidth=2.2)

    # Vertical divider
    ax.axvline(x=n_pre + 0.5, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(n_pre + 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else max(pretrain_losses),
            " finetune →", fontsize=8, color="grey", va="top")

    # Finetune
    ft_epochs = range(n_pre + 1, n_pre + n_ft + 1)
    ax.plot(ft_epochs, finetune_train_losses, color=COLORS[1], label="Finetune Train", linewidth=2.2)

    if finetune_val_losses:
        ft_val_epochs = range(n_pre + 1, n_pre + len(finetune_val_losses) + 1)
        ax.plot(ft_val_epochs, finetune_val_losses, color=COLORS[2], label="Finetune Val", linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Plot saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────
#  Log parsing
# ──────────────────────────────────────────────────────────────

# Regex patterns for the new logging format:
#   <timestamp> - <module> - INFO - <message>
# We match from the message portion only.

# equiv_experiments multitask:
#   "Epoch 1/40 - Train Loss: 0.1234"
_RE_EQUIV_TRAIN = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*-\s*Train Loss:\s*([\d.]+)"
)

# equiv_experiments multitask val (printed as a list at end of trial):
#   "Trial 1, Struct palindrome: [1.23, 0.98, ...]"
_RE_EQUIV_VAL_LIST = re.compile(
    r"Trial\s+\d+,\s*Struct\s+(\w+):\s*(\[[\d.,\s\-eE+]+\])"
)

# equiv_experiments pretrain:
#   "Pretrain Epoch 5: Loss = 0.1234"
_RE_PRETRAIN = re.compile(
    r"Pretrain Epoch\s+(\d+):\s*Loss\s*=\s*([\d.]+)"
)

# equiv_experiments finetune train:
#   "Trial 1, Epoch 5: Finetune Train Loss = 0.1234"
_RE_FT_TRAIN_EQUIV = re.compile(
    r"Trial\s+\d+,\s*Epoch\s+(\d+):\s*Finetune Train Loss\s*=\s*([\d.]+)"
)

# equiv_experiments finetune val:
#   "Trial 1, Epoch 5: Val Loss = 0.1234"
_RE_FT_VAL_EQUIV = re.compile(
    r"Trial\s+\d+,\s*Epoch\s+(\d+):\s*Val Loss\s*=\s*([\d.]+)"
)

# inv_regression multitask:
#   "Trial 1, Epoch 5: Train wL1 = 0.1234"
_RE_INV_TRAIN = re.compile(
    r"Trial\s+\d+,\s*Epoch\s+(\d+):\s*Train wL1\s*=\s*([\d.]+)"
)

# inv_regression pretrain:
#   "Pretrain Epoch 5: Scaled L1 = 0.1234"
_RE_INV_PRETRAIN = re.compile(
    r"Pretrain Epoch\s+(\d+):\s*Scaled L1\s*=\s*([\d.]+)"
)

# inv_regression finetune train:
#   "Trial 1, Epoch 5: Train L1 = 0.1234"
_RE_FT_TRAIN_INV = re.compile(
    r"Trial\s+\d+,\s*Epoch\s+(\d+):\s*Train L1\s*=\s*([\d.]+)"
)

# inv_regression finetune val:
#   "Finetune Epoch 5: Val L1 = 0.1234"
_RE_FT_VAL_INV = re.compile(
    r"Finetune Epoch\s+(\d+):\s*Val L1\s*=\s*([\d.]+)"
)

# inv_regression multitask val (printed as list):
#   "palindrome: [0.12, 0.08, ...]"
_RE_INV_VAL_LIST = re.compile(
    r"^\s*(\w+):\s*(\[[\d.,\s\-eE+]+\])\s*$"
)

# Final test loss lines:
#   "Test Loss: 0.1234"  or  "Test L1: 0.1234"
_RE_TEST_LOSS = re.compile(
    r"Test (?:Loss|L1):\s*([\d.]+)"
)


def parse_log_file(log_path):
    """
    Parse a .log file and extract metrics.

    Returns a dict with available keys:
        "train_losses": list[float]
        "val_losses":   dict[str, list[float]]  (structure → per-epoch)
        "pretrain_losses": list[float]
        "finetune_train_losses": list[float]
        "finetune_val_losses": list[float]
        "test_losses": list[float]
        "experiment_type": "multitask" | "pretrain" | "unknown"
    """
    with open(log_path, "r") as f:
        lines = f.readlines()

    data = {
        "train_losses": [],
        "val_losses": {},
        "pretrain_losses": [],
        "finetune_train_losses": [],
        "finetune_val_losses": [],
        "test_losses": [],
        "experiment_type": "unknown",
    }

    for line in lines:
        # Strip the logging prefix to get the message
        # Format: "2026-02-20 12:03:01,234 - __main__ - INFO - <message>"
        parts = line.split(" - ", 3)
        msg = parts[-1].strip() if len(parts) >= 4 else line.strip()

        # --- Multitask equiv train ---
        m = _RE_EQUIV_TRAIN.search(msg)
        if m:
            data["train_losses"].append(float(m.group(3)))
            data["experiment_type"] = "multitask"
            continue

        # --- Multitask inv train ---
        m = _RE_INV_TRAIN.search(msg)
        if m:
            data["train_losses"].append(float(m.group(2)))
            data["experiment_type"] = "multitask"
            continue

        # --- Multitask val list (equiv) ---
        m = _RE_EQUIV_VAL_LIST.search(msg)
        if m:
            struct = m.group(1)
            vals = ast.literal_eval(m.group(2))
            data["val_losses"][struct] = vals
            continue

        # --- Pretrain (equiv or inv) ---
        m = _RE_PRETRAIN.search(msg)
        if m:
            data["pretrain_losses"].append(float(m.group(2)))
            data["experiment_type"] = "pretrain"
            continue
        m = _RE_INV_PRETRAIN.search(msg)
        if m:
            data["pretrain_losses"].append(float(m.group(2)))
            data["experiment_type"] = "pretrain"
            continue

        # --- Finetune train ---
        m = _RE_FT_TRAIN_EQUIV.search(msg)
        if m:
            data["finetune_train_losses"].append(float(m.group(2)))
            continue
        m = _RE_FT_TRAIN_INV.search(msg)
        if m:
            data["finetune_train_losses"].append(float(m.group(2)))
            continue

        # --- Finetune val ---
        m = _RE_FT_VAL_EQUIV.search(msg)
        if m:
            data["finetune_val_losses"].append(float(m.group(2)))
            continue
        m = _RE_FT_VAL_INV.search(msg)
        if m:
            data["finetune_val_losses"].append(float(m.group(2)))
            continue

        # --- Test loss ---
        m = _RE_TEST_LOSS.search(msg)
        if m:
            data["test_losses"].append(float(m.group(1)))
            continue

    return data


def plot_from_log(log_path, save_dir=None, show=True):
    """
    Parse a log file and generate appropriate plots.

    Args:
        log_path: path to .log file
        save_dir: directory to save PNGs (defaults to same dir as log)
        show: if True, display plots interactively
    """
    log_path = str(log_path)
    data = parse_log_file(log_path)

    if save_dir is None:
        save_dir = os.path.dirname(log_path)
    base = os.path.splitext(os.path.basename(log_path))[0]

    if data["experiment_type"] == "pretrain" and data["pretrain_losses"]:
        save_path = os.path.join(save_dir, f"{base}_pretrain_finetune.png")
        plot_pretrain_finetune(
            pretrain_losses=data["pretrain_losses"],
            finetune_train_losses=data["finetune_train_losses"],
            finetune_val_losses=data["finetune_val_losses"] or None,
            title=f"Pretrain → Finetune\n({os.path.basename(log_path)})",
            save_path=save_path,
            show=show,
        )
    elif data["train_losses"]:
        save_path = os.path.join(save_dir, f"{base}_training_curves.png")
        plot_training_curves(
            train_losses=data["train_losses"],
            val_losses_dict=data["val_losses"] if data["val_losses"] else None,
            title=f"Training Curves\n({os.path.basename(log_path)})",
            save_path=save_path,
            show=show,
        )
    else:
        logger.warning(f"No plottable data found in {log_path}")
        return

    if data["test_losses"]:
        logger.info(f"Final test losses from {os.path.basename(log_path)}: {data['test_losses']}")


def plot_from_directory(log_dir, show=False):
    """
    Find all .log files in a directory (recursively) and plot each one.
    """
    log_files = glob.glob(os.path.join(log_dir, "**", "*.log"), recursive=True)
    if not log_files:
        logger.warning(f"No .log files found in {log_dir}")
        return

    logger.info(f"Found {len(log_files)} log file(s) in {log_dir}")
    for log_file in sorted(log_files):
        logger.info(f"Processing: {log_file}")
        plot_from_log(log_file, show=show)


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from experiment log files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_utils.py path/to/experiment.log
  python plot_utils.py path/to/logs/           # batch-process all .log files
  python plot_utils.py experiment.log --show    # also display interactively
        """,
    )
    parser.add_argument(
        "path",
        help="Path to a .log file or a directory containing .log files",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively (default: save only)",
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to save plots (default: same directory as log file)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if os.path.isdir(args.path):
        plot_from_directory(args.path, show=args.show)
    elif os.path.isfile(args.path):
        plot_from_log(args.path, save_dir=args.save_dir, show=args.show)
    else:
        logger.error(f"Path not found: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
