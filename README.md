# HyConGT public main-model code

This repository contains the cleaned public implementation of the main Hydrologically Constrained Graph Temporal model used in the manuscript *Hydrologically constrained graph learning for reconstructing wastewater and heavy metal mass fluxes in engineered mine water networks*.

The release intentionally contains no mine monitoring data, engineering records, operational records, SWAT project files, raw-data parsers, trained checkpoints, or model outputs.

## Scope

The retained implementation contains only the main HyConGT flow-reconstruction model.

The model uses three directed multi-head GAT layers, a one-layer GRU, state-dependent bounded process parameters, dynamic engineering pathway corrections, previous-day model-reconstructed flow, temporal inertia for slowly varying engineering parameters, and the differentiable Source Retention Treatment Outfall water balance.

Baseline models, ablation experiments, SHAP analysis, figure-generation utilities, and heavy-metal concentration reconstruction are outside this main-model repository.

## Repository layout

```text
hycongt/
  data.py          prepared-data contract, paper splits, standardization
  model.py         directed GAT, GRU, parameter heads
  physics.py       differentiable SRTO water balance
  metrics.py       R2, NSE, KGE, RMSE, MAE
  training.py      chronological rollout, TBPTT, supervision, evaluation
train.py           main training entry point
DATA_SCHEMA.md     confidential prepared-input interface
```

## Data handling

The raw data are confidential and are not required to be stored in this repository. Prepare a single NPZ file inside the secure environment according to `DATA_SCHEMA.md`. The `.gitignore` excludes NPZ, CSV, Excel, NumPy arrays, checkpoints, and output directories.

The public code deliberately starts at the prepared-tensor boundary. This avoids publishing confidential file structures or undocumented site-specific raw-data heuristics while retaining the complete main-model computation.

## Training settings represented in the manuscript

The default model dimensions are 64 for both the graph and GRU hidden states. Each GAT layer uses four heads with dropout 0.10. The optimizer is AdamW with learning rate `1e-3` and weight decay `1e-4`. The SRTO balance uses three intraday routing iterations. The first-order inertia coefficient is exposed as `--slow-rho` because it is an implementation hyperparameter.

Dynamic graph-learning inputs are standardized using training-period statistics only. Static features are standardized across nodes within the site. The previous-day flow feature is generated from the previous model prediction and never replaced by observed flow.

For DCM, monthly cumulative discharge is the primary supervision signal and available true daily observations can contribute the daily loss. For JCPM, the optimization target uses a causal 7-day trailing moving average. Final JCPM test metrics use the original unsmoothed daily observations.

## Run

```bash
python -m pip install -r requirements.txt
python train.py --data /secure/path/dexing_prepared.npz --out /secure/path/outputs/dexing
python train.py --data /secure/path/jiama_prepared.npz --out /secure/path/outputs/jiama
```

Keep both prepared data and generated outputs outside the Git repository.

## Verification

```bash
python -m py_compile train.py hycongt/*.py tests/*.py
python -m pytest -q
```

The tests use synthetic data only. They check edge-allocation normalization, SRTO nonnegativity and storage bounds, parameter ranges, gradient propagation, and the chronological training and rollout path.

## Reproducibility boundary

This release is structurally aligned with the manuscript and Supplementary Texts S2 to S4. Exact numerical reproduction of the reported site metrics requires the confidential prepared inputs used in the study. Those data are intentionally not included.
