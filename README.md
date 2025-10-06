# Lorenz96 ML Surrogate Modeling

This repository explores the use of machine learning to approximate the dynamics of the Lorenz96 (L96) system — a simplified yet chaotic model originally developed to capture essential features of atmospheric behavior.

L96 is widely used as a testbed for studying chaos and predictability, making it an ideal stepping stone toward more complex domains such as weather forecasting, energy systems, and other nonlinear dynamical systems.

---

### 🔍 Project Overview

- **Goal:** Train an AI/ML model to approximate the Lorenz96 system’s chaotic behavior.

This work sets the foundation for future research into surrogate modeling and inverse modeling.

---

✨ Acknowledgments

This project is part of ongoing research in chaotic systems modeling and AI-driven surrogate models.

# Step 1: Project Setup

### 1. Clone Repository
```sh
git clone https://github.com/shawn-mcdonald-dev/lorenz96
cd lorenz96
```

### 2. Installing Dependencies
Create and activate conda env:
```sh
conda env create -f environment.yml
conda activate l96
```

# Step 2: Create raw datasets
```sh
# Example: 100 samples, deterministic RK4, moderate variability in F and x0
python make_l96_dataset.py \
  --outdir data/l96_N40_T20_S600 \
  --num-samples 100 \
  --jobs 4 \
  --N 40 --tmax 20 --steps 600 \
  --F-min 4 --F-max 16 \
  --ic-mode random --eps 2.0 \
  --method RK4FIXED
```

**Expected output:**
```sh
Wrote 100 samples and manifest: data/l96_N40_T20_S600/manifest.csv
```

# Step 3: Get metrics from dataset
```sh
python l96_metrics_cli.py --path data/l96_N40_T20_S600
```

**Expected output:**
```sh
Wrote 100 rows to data/l96_N40_T20_S600/targets_time_mean_energy.csv
```