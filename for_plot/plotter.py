# TODO: maybe some improvements required
#  + refactor
#  + change parsing logic from input to argparse

# Added regex support starting at line 63.

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import glob
import re

try:
    import readline
except ImportError:
    try:
        import pyreadline3 as readline
    except ImportError:
        print("warning: pyreadline not installed, skipping readline")
        readline = None


def path_completer(text, state):
    expanded_text = os.path.expanduser(text)
    matches = glob.glob(expanded_text + "*")
    results = [m + "/" if os.path.isdir(os.path.expanduser(m)) else m for m in matches]
    return results[state] if state < len(results) else None


def list_completer_factory(valid_list):
    def completer(text, state):
        parts = text.split(",")
        curr = parts[-1].strip()
        before = ",".join(parts[:-1]) + ("," if len(parts) > 1 else "")
        matches = [c for c in valid_list if c.startswith(curr)]
        return before + matches[state] if state < len(matches) else None

    return completer


if readline:
    readline.set_completer_delims(" \t\n;")

    # for macOS
    if sys.platform == "darwin":
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

    readline.set_completer(path_completer)

readline.set_completer_delims(" \t\n;")
readline.parse_and_bind("tab: complete")
readline.set_completer(path_completer)


def plot_csv_data(file_path):
    try:
        full_path = os.path.abspath(os.path.expanduser(file_path))
        df = pd.read_csv(full_path)
        cols = list(df.columns)
        print(f"Available columns are: {cols}")
        if readline:
            readline.set_completer(list_completer_factory(cols))

        y_axis = input("\nSelext Y-axis column: ").strip()
        x_axes_input = input("SelectX-axis columns (comma separated): ")
        x_axes = []
        for pattern in [p.strip() for p in x_axes_input.split(",")]:
            try:
                matched = [c for c in cols if re.match(pattern, c)]
                if matched:
                    x_axes.extend(matched)
                else:
                    print(f"No columns matched '{pattern}'")
            except re.error:
                print(f"Invalid regex pattern '{pattern}'")

        plot_type = input("\nLine or Scatter plot? (l/s): ").strip().lower()
        export_pdf = input("Export to PDF? (y/n): ").strip().lower()

        num_plots = len(x_axes)
        fig, axes = plt.subplots(
            num_plots, 1, figsize=(10, 5 * num_plots), constrained_layout=True
        )
        if num_plots == 1:
            axes = [axes]

        for i, x_col in enumerate(x_axes):
            sorted_df = df.sort_values(by=x_col)

            if plot_type == "s":
                axes[i].scatter(sorted_df[x_col], sorted_df[y_axis], alpha=0.3, s=1)
            else:
                axes[i].plot(
                    sorted_df[x_col], sorted_df[y_axis], linewidth=0.8, alpha=0.8
                )

            axes[i].set_title(f"{y_axis} vs {x_col} (Sorted)")
            axes[i].set_xlabel(x_col)
            axes[i].set_ylabel(y_axis)
            axes[i].grid(True, alpha=0.3)

        if export_pdf == "y":
            out = input("Enter PDF name: ").strip() or "output.pdf"
            base_dir = Path(__file__).resolve().parent.parent
            save_dir = base_dir / "data"

            save_dir.mkdir(parents=True, exist_ok=True)

            final_path = save_dir / out

            with PdfPages(final_path) as pdf:
                pdf.savefig(fig)
            print(f"Saved at: {os.path.abspath(out)}")

        plt.show()

    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    path = input("CSV Path: ").strip()
    plot_csv_data(path)
