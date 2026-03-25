# Bipolar HiPIMS Deposition Rates - Active Learning with SHAP Explanations

This repository contains the data, code, and analysis for the publication on active learning-guided optimization of bipolar HiPIMS deposition rates, with SHAP-based model interpretability.

## Zenodo

| Record | Concept DOI (latest version) | Version DOI (v1) |
|--------|------------------------------|-------------------|
| **Code & Analysis** (this repository) | [10.5281/zenodo.18495504](https://doi.org/10.5281/zenodo.18495504) | [10.5281/zenodo.18495505](https://doi.org/10.5281/zenodo.18495505) |
| **Raw Data** (compressed) | [10.5281/zenodo.18495401](https://doi.org/10.5281/zenodo.18495401) | [10.5281/zenodo.18495402](https://doi.org/10.5281/zenodo.18495402) |

> Use the **Concept DOI** to always link to the latest version.

## Repository Structure

```
.
├── Bipolar Datasets - Al and Ti - low and high PW/
│   ├── Datasets Used for Publication/       # Cleaned experimental datasets
│   │   ├── Al - 120 W  - short PW/
│   │   ├── Al - 200 W - high PW/
│   │   ├── Al - 250 W - duty cycle series/
│   │   ├── Ti - 120 W - short PW/
│   │   ├── Ti - 200 W - high PW/
│   │   └── Ti 250 W low duty cycle/
│   └── Raw Data .Json Files/               # Original unprocessed data
├── Pickles/                                  # Pre-computed model explanations
│   ├── Ipk-PW-PRR-posPulse/                # SHAP explanations per campaign
│   ├── Conditional Explainers/              # Conditional SHAP explanations
│   ├── LIME/                                # LIME explanations
│   ├── *_combined_dataset*.pkl              # Combined dataset explanations
│   └── global_dataset.pkl                   # Global (all data) explanation
├── Manually Measured Data/                  # Manually measured reference data
├── src/                                      # Python utilities
│   ├── shap_utils.py                        # SHAP plotting utilities
│   ├── plot_utils.py                        # General plotting
│   ├── hipims_bo_utils.py                   # Data preparation helpers
│   ├── campaignvisualizer.py                # Campaign visualization
│   ├── lime_utils.py                        # LIME utilities
│   └── xgb_training/                        # XGBoost model training
├── Main - Paper Figures.ipynb               # Generates all publication figures
├── Pickle SHAP datasets example.ipynb       # Original SHAP creation example
├── SHAP Object Creation - Documentation.ipynb  # Documented SHAP creation walkthrough
├── GPR Cross-Validation.ipynb               # GPR model cross-validation & metrics
├── LIME Analysis.ipynb                      # LIME-based model explanations
├── InitializeCampaign.py                    # BayBE Campaign initialization
├── shapexplainers.py                        # SHAP explainer wrappers
├── environment.yml                          # Conda environment specification
└── READ_ME.txt                              # Original brief README
```

## Dataset Description

Each dataset folder in `Datasets Used for Publication/` contains:
- **`Campaign.json`** — Serialized BayBE Campaign with the full search space definition and all measurements
- **`calibration.txt`** — Oscilloscope sampling window (us-window) and material-specific deposition rate calibration factor
- **`Logfile - Oscilloscope/`** — Individual oscilloscope waveform recordings (JSON) for peak current extraction
- **`Logfiles/`** — System logfiles from the experiment

### Process Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `PW (us)` | Negative pulse width | microseconds |
| `PRR (Hz)` | Pulse repetition rate | Hz |
| `pos. Delay (us)` | Delay before positive counter-pulse | microseconds |
| `pos. PW (us)` | Positive pulse width | microseconds |
| `pos. Setpoint (V)` | Positive pulse voltage setpoint | V |
| `Ipk (A)` | Peak current density (measured) | A/cm² |

**Target variable:** Deposition rate (`y1`) in Angstrom/s.

### Note on J_pk (Peak Current Density)

J_pk is **not** a controlled process parameter. It is a measured quantity that depends on the other process settings and the plasma conditions. J_pk was **not included in the active learning screening** (i.e., it was not part of the BayBE search space). It was measured post-hoc from oscilloscope waveforms and added to the SHAP analysis as an additional feature to understand its relationship with deposition rate. Its inclusion in the SHAP feature set is for interpretability purposes only.

## Known Dataset Biases and Limitations

### Correlated Parameter Space

The experimental campaigns were designed to scan specific regions of the pulse width (PW) and duty cycle parameter space. Because duty cycle is computed as `Duty Cycle = PRR * PW * 1e-6`, the ranges of PW and PRR scanned in each campaign are **not independent** — they create inherent correlations in the dataset.

Specifically:
- **Short PW campaigns** (e.g., 5-100 microseconds) tend to have higher PRR values to maintain reasonable duty cycles
- **High PW campaigns** (e.g., 100-500 microseconds) tend to have lower PRR values
- **Duty cycle series** explicitly vary duty cycle at fixed power, creating a direct PW-PRR coupling

These correlations create inhomegneities in the datastet, caused by design constraints (constant power control for example), not from the plasma physics. The SHAP explanations may therefore capture some of these design-induced correlations as apparent feature interactions. When interpreting SHAP dependence plots, users should be aware that strong PW-PRR interactions may partly reflect the correlated sampling rather than true physical dependencies. This is mentioned in the main text as well.

### Campaign-Specific Bounds

Each campaign operates within a specific power and pulse width regime. The parameter bounds differ between campaigns, which means:
- Combined datasets (Al_combined, Ti_combined, global) span a wider but non-uniformly sampled parameter space
- Edge regions of parameter space may be underrepresented, specifically areas with no pos. pulse (values set to 0)

## Data Preprocessing

**No preprocessing is applied.** Raw measurement values are used directly without normalization, standardization, or feature scaling.

**No outlier removal is performed.** All collected data points are included in the analysis. The active learning loop guides the sampling, and removing points would bias the surrogate model explanations.

The only data transformations are:
1. Deposition rate unit conversion: kAngstrom/s to Angstrom/s (factor 1000)
2. Material-specific density calibration applied to deposition rate
3. Peak current density normalization by the 3-inch target area (45.58 cm²)

## Notebooks

- **`Main - Paper Figures.ipynb`** — Generates all figures used in the publication from pre-computed SHAP explanation objects
- **`SHAP Object Creation - Documentation.ipynb`** — Detailed, documented walkthrough of how a single SHAP explanation is created from raw data
- **`GPR Cross-Validation.ipynb`** — Cross-validates the GPR surrogate models underlying the SHAP explanations and reports fit quality metrics
- **`LIME Analysis.ipynb`** — Alternative interpretability analysis using LIME (Local Interpretable Model-agnostic Explanations)
- **`Pickle SHAP datasets example.ipynb`** — Original working script for SHAP object creation

## Setup

1. Install the conda environment:
   ```
   conda env create -f environment.yml
   ```

2. Key dependencies include:
   - `baybe>=0.14.1` — Bayesian optimization framework with SHAP integration
   - `shap>=0.46.0` — SHAP explanations
   - `scikit-learn>=1.7.2` — Machine learning utilities
   - `xgboost>=3.0.5` — Gradient boosting (used in LIME analysis)
   - `torch>=2.6.0` — PyTorch backend for BoTorch/GPyTorch
   - `gpytorch>=1.14` — Gaussian Process models

3. Download the raw data from [Zenodo (doi.org/10.5281/zenodo.18495402)](https://doi.org/10.5281/zenodo.18495402) and place it in the repository root.
