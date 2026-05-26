# HyConGAT: Hydrologically Constrained Graph Attention Temporal Network for Mine Wastewater Discharge Prediction

This repository provides the implementation code accompanying the manuscript:

## Overview

HyConGAT is a hybrid spatiotemporal framework for predicting daily wastewater discharge at permitted outfall nodes of open-pit mine drainage systems. The framework integrates:

- A **Source-Retention-Treatment-Outfall (SRTO) directed weighted graph** encoding engineered conveyance topology
- **SWAT-derived hydrological outputs** as physically constrained node features
- A **node-level process-based water balance** producing unit-specific simulated discharge inputs
- A coupled **GAT-GRU architecture** for spatiotemporal discharge prediction
- **SHAP attribution analysis** for node-level mechanistic interpretation

## Repository Structure

```
├── train_hycongat.py        # Model definition, training, and evaluation
├── node_water_balance.py    # Process-based node-level water balance (generates sim_outflow features)
├── model_params.py          # Physical parameters for the water balance model
├── data/
│   ├── node.csv            # Node engineering attributes and SRTO type indicators
│   ├── edges.csv           # Directed graph topology with design allocation ratios
│   └── sub.csv          # SWAT daily sub-basin outputs
└── outputs/                 # Generated automatically at runtime
```

---


## Input Data Format

### `node.csv` — Node attributes

Must contain the following columns:

| Column | Description |
|---|---|
| `node_id` | Unique node identifier (e.g., S1, R6, T1, O1) |
| `node_type` | SRTO category: `S` (Source), `R` (Retention), `T` (Treatment), `O` (Outfall) |
| `SUB` | SWAT sub-basin ID associated with this node |
| `SUB_area_m2` | Sub-basin area (m²) |
| `area_m2` | Engineering unit area (m²) |
| `yihongkoubiaogao_m` | Spillway crest elevation (m) |
| `shuimianmianji_m2` | Pond/reservoir water surface area (m²) |
| `youxiaokurong_m3` | Licensed working storage capacity (m³) |
| `zuidabengpainengli_m3/h` | Rated pump discharge capacity (m³/h) |
| `jiangyuhuishuimianji_m2` | Contributing catchment area (m²) |
| `jingliuxishu` | Empirical runoff coefficient |
| `S`, `R`, `T`, `O` | Binary SRTO type indicator columns (one-hot encoded) |

### `edges.csv` — Graph topology

| Column | Description |
|---|---|
| `from_id` | Source node of the directed transfer edge |
| `to_id` | Destination node of the directed transfer edge |
| `weight_alpha` | Design allocation ratio for this transfer pathway (0–1); must satisfy mass conservation at each non-outfall node |

### `daily_csv` — Dynamic node features

A long-format daily table with columns:

| Column | Description |
|---|---|
| `YYYYDDD` | Date in 7-digit Julian format |
| `node_id` | Node identifier matching `node.csv` |
| `real` | Observed daily discharge volume at outfall nodes (10⁴ m³/d); used as training target |
| `PRECIPmm` | Daily precipitation (mm) from SWAT sub-basin output |
| `SNOWMELTmm` | Snowmelt (mm) |
| `ETmm` | Evapotranspiration (mm) |
| `SWmm` | Soil water content (mm) |
| `PERCmm` | Percolation (mm) |
| `SURQmm` | Surface runoff (mm) |
| `GW_Qmm` | Groundwater contribution (mm) |
| `LAT_Qmm` | Lateral flow (mm) |
| `sim_outflow` | Node-level simulated wastewater volume (10⁴ m³/d) from `node_water_balance.py` |

The `sim_outflow` column is produced by running `node_water_balance.py` before model training (see Workflow below).

### `sub.csv` — SWAT sub-basin outputs

Daily SWAT output table with columns `SUB`, `YYYYDDD` (or `date` / `YEAR`+`DAY`), `PRECIPmm`, `WYLDmm`, `ETmm`. Required by `node_water_balance.py`.

---

## Workflow

### Step 1: Run the node-level water balance

This script simulates daily wastewater volumes at all 22 nodes using process-based physical operators. The output `sim_outflow` series is used as an input feature to the graph model.

```bash
python node_water_balance.py
```

Output is written to `outputs/node_outflow_physcal.csv`. Merge this with SWAT sub-basin outputs and observed outfall discharge records to produce the `daily_csv` required by the training script.

### Step 2: Train and evaluate HyConGAT

```bash
python train_hycongat.py \
    --node_csv ./data/node.csv \
    --edge_csv ./data/edges.csv \
    --daily_csv ./outputs/merge_node_sub_day_real_swat_outflow_phys.csv \
    --out_dir ./hycongat_outputs
```

### Outputs

After training, the following files are saved under `--out_dir`:

```
hycongat_outputs/
├── logs/
│   ├── best_hycongat.pt          # Best model checkpoint (lowest validation loss)
│   ├── training_artifacts.npz    # Reproducibility artifacts (indices, adjacency, scaler stats)
│   └── metadata.json             # Full configuration and file paths
└── tables/
    ├── training_history.csv      # Epoch-level train/validation loss and learning rate
    ├── metrics.csv               # MAE, RMSE, R², MAPE per outfall and globally
    └── test_predictions.csv      # Observed and predicted discharge for the test period
```

