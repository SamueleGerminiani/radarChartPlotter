# 🕸️ Radar Chart Plotter

A Python script to generate **radar (spider) charts** for comparing multiple configurations from a CSV file.

---

## 📦 Requirements

Before running, install the required Python dependencies:

```bash
./install_deps.sh
```

This installs:
- `pandas`
- `numpy`
- `matplotlib`

---

## 🚀 Usage

Run the script from the terminal:

```bash
./prc.py --csv data.csv [OPTIONS]
```

### Required argument

- `--csv <path>`  
  Path to the input CSV file.  
  The first column must be `name` (used as labels for each configuration), and the following columns must be numeric variables.

Example CSV:
```csv
name, MS, RS, FC, time
HARM, 0.1, 0.9, 1, 10
samples2ltl, 0.4, 0.6, 1, 20
Texada, 0.5, 0.6, 0.8, 100
Goldminer, 1, 0, 0.1, 1000
```

---

## ⚙️ Options

| Option | Description | Example |
|:--------|:-------------|:---------|
| `--invert` | Comma-separated list of variables to invert (e.g., high = bad). | `--invert time,RS` |
| `--relative` | Variables to normalize using their own [min, max] range. Others are assumed in [0,1]. | `--relative MS,RS` |
| `--title` | Custom title for the radar chart. | `--title "Performance Comparison"` |

---

## 🧭 Output

- The script displays a radar chart for all configurations.  
- Each line corresponds to one configuration (row in the CSV).  
- Inverted variables can optionally be shown in bold or colored (see code comments).  
- Relative variables display their actual numeric range on the chart.

---

## 🧩 Example Command

```bash
./radar_chart.py \
  --csv results.csv \
  --invert time \
  --relative MS,RS \
  --title "Tool Performance Overview"
```

---

## 🖼️ Notes

- The script automatically normalizes values to `[0,1]` for consistent visualization.  
- It prevents clutter near the center of the radar.  
- The plot legend shows each configuration name.  
- Works on Linux, macOS, and Windows (with Python 3.8+).

---

© 2025 — Radar Chart Plotter by Samuele Germiniani

