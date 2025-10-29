#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from io import StringIO


def main():
    # === Configure Matplotlib backend ===
    configure_matplotlib_backend()

    parser = argparse.ArgumentParser(
        description="Plot a radar (spider) chart for each configuration in a CSV file or inline string."
    )
    parser.add_argument("--csv", help="Path to the CSV file")
    parser.add_argument(
        "--string",
        help="CSV content provided directly as a string (overrides --csv)",
        default=None,
    )
    parser.add_argument(
        "--invert",
        help="Comma-separated list of variables to invert (e.g., 'time,overlap')",
        default="",
    )
    parser.add_argument(
        "--relative",
        help="Comma-separated list of variables to normalize relative to their min–max range (others assumed in [0,1]).",
        default="",
    )
    parser.add_argument(
        "--title",
        help="Title for the radar chart (default: 'Radar Chart of Configurations')",
        default="Radar Chart of Configurations",
    )
    parser.add_argument(
        "--dump-to",
        help="Path to save the chart image (e.g., 'chart.png' or 'out.pdf'). "
        "If omitted, the chart is shown interactively.",
        default=None,
    )

    args = parser.parse_args()

    # === Load data ===
    if args.string:
        #debug
        #print("📄 Using CSV content from --string argument.")
        df = pd.read_csv(StringIO(args.string.strip()), skipinitialspace=True)
    elif args.csv:
        df = pd.read_csv(args.csv, skipinitialspace=True)
    else:
        raise ValueError(
            "You must provide either --csv <file> or --string <csv_content>"
        )

    df.columns = df.columns.str.strip()
    #debug
    #print("DataFrame columns:", df.columns.tolist())
    #print("DataFrame columns:", df.columns.tolist())
    #for col in df.columns:
    #    print(f"Column '{col}' sample data:", df[col].head().tolist())

    if df.columns[0].lower() != "name":
        raise ValueError(
            "The first column must be 'name' (configuration label)."
        )

    labels = df.iloc[:, 0].astype(str)
    df_numeric = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    if df_numeric.empty:
        raise ValueError("No numeric columns found after the 'name' column.")

    # === Parse CLI options ===
    invert_list = [v.strip() for v in args.invert.split(",") if v.strip()]
    relative_list = [v.strip() for v in args.relative.split(",") if v.strip()]

    for var in invert_list + relative_list:
        if var not in df_numeric.columns:
            raise ValueError(
                f"Variable '{var}' not found in CSV numeric columns."
            )

    # === Normalization ===
    df_scaled = df_numeric.copy()
    min_vals, max_vals = {}, {}

    for col in df_numeric.columns:
        if col in relative_list:
            min_val, max_val = df_numeric[col].min(), df_numeric[col].max()
            if max_val == min_val:
                df_scaled[col] = 0.0
            else:
                df_scaled[col] = (df_numeric[col] - min_val) / (
                    max_val - min_val
                )
        else:
            min_val, max_val = 0, 1
        min_vals[col], max_vals[col] = min_val, max_val

    # === Handle inversion ===
    df_columns_lower = {col.lower(): col for col in df_scaled.columns}
    inverted_vars = set()

    for var in invert_list:
        var_lower = var.lower()
        if var_lower in df_columns_lower:
            col_name = df_columns_lower[var_lower]
            df_scaled[col_name] = 1 - df_scaled[col_name]
            inverted_vars.add(col_name)
            #debug
            #print(f"Inverted variable: {col_name}")
        else:
            print(
                f"Warning: '{var}' not found among numeric columns, skipping."
            )

    # === Prepare axis labels ===
    categories = []
    for col in df_numeric.columns:
        if col in relative_list:
            lo, hi = df_numeric[col].min(), df_numeric[col].max()
        else:
            lo, hi = 0, 1
        if col in inverted_vars:
            lo, hi = hi, lo
            if col in relative_list:
                label = f"{col} [{lo:.2f}, {hi:.2f}]\n (inv, rel)"
            else:
                label = f"{col} [{lo:.2f}, {hi:.2f}] (inv)"
        else:
            label = f"{col} [{lo:.2f}, {hi:.2f}]"
        categories.append(label.replace(".00", ""))

    # === Radar chart geometry ===
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # === Plot ===
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for name, row in zip(labels, df_scaled.iterrows()):
        values = row[1].tolist()
        values.append(values[0])  # close the polygon
        closed_angles = np.concatenate([angles, [angles[0]]])
        values = [max(v, 0.05) for v in values]  # avoid center clutter

        ax.plot(
            closed_angles,
            values,
            linewidth=1.5,
            marker=".",
            label=name,
            alpha=0.8,
        )
        ax.fill(closed_angles, values, alpha=0.05)

    # === Cosmetics ===
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=15)
    ax.xaxis.set_tick_params(pad=30)

    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="grey", size=8)

    ax.set_yticklabels([])
    ax.set_yticks([])  # remove tick marks

    # === Add real numeric value labels for relative *and inverted* variables ===
    tick_levels = np.linspace(0, 1, 5)
    for angle, col in zip(angles, df_numeric.columns):
        min_val, max_val = min_vals[col], max_vals[col]
        if col in inverted_vars:
            real_vals = np.linspace(max_val, min_val, len(tick_levels))
        else:
            real_vals = np.linspace(min_val, max_val, len(tick_levels))

        for r_norm, real in zip(tick_levels, real_vals):
            text = (
                ""
                if np.isclose(r_norm, 0.0)
                else f"{real:.2f}".rstrip("0").rstrip(".")
            )
            ax.text(
                angle,
                r_norm + 0.02,
                text,
                color="gray",
                fontsize=7,
                ha="center",
                va="center",
            )

    # === Final touches ===
    ax.set_rlabel_position(0)
    plt.ylim(0, 1)

    ax.set_title(args.title, pad=20, fontweight="bold", fontsize=17)
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()

    if args.dump_to:
        save_chart(fig, args.dump_to)
    else:
        plt.show()


def save_chart(fig, output_path):
    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    valid_exts = [".png", ".pdf", ".svg", ".jpg", ".jpeg"]
    if ext not in valid_exts:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Use one of: {', '.join(valid_exts)}"
        )
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=False,
        facecolor="white",
    )
    print(f"✅ Chart saved successfully to: {output_path}")


def configure_matplotlib_backend():
    """
    Configure Matplotlib backend automatically depending on the environment.
    Uses 'Agg' in headless or non-interactive contexts, 'TkAgg' otherwise.
    """
    import os
    import matplotlib

    # Detect headless / non-interactive mode
    headless = (
        not os.environ.get("DISPLAY")      # no display (typical for SSH/system calls)
        or os.environ.get("SSH_CONNECTION") # running over SSH
        or os.environ.get("SSH_TTY")        # SSH TTY session
        or not os.isatty(0)                 # launched non-interactively (e.g., from system())
    )

    if headless and os.name != "nt":
        matplotlib.use("Agg")
        print("💡 Headless mode detected — using 'Agg' backend (non-interactive).")


if __name__ == "__main__":
    main()
