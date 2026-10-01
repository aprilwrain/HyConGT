# HyConGT

HyConGT reconstructs daily flow in engineered mine-water networks by combining hydrological inputs, directed graph attention, recurrent learning and a differentiable water-balance module.

## Model

SWAT supplies daily hydrological forcings, states and fluxes for the subbasin associated with each engineering node. These inputs are combined with operational information, static engineering attributes, antecedent states and previous-day reconstructed discharge.

Three directed graph-attention layers aggregate information along the engineering connections and node self-loops. A single-layer GRU represents temporal dependencies. Node and edge heads transform the hidden states into bounded water-balance parameters and adjustments to existing water-allocation weights. The differentiable Source–Retention–Treatment–Outfall (SRTO) module calculates daily discharge and updates Retention storage. Model parameters are optimized jointly through this calculation.

The SRTO module represents surface runoff, groundwater contributions, mine inflows, external intake, operational water inputs, storage, evaporation, release, spill and hydraulic losses during treatment. Graph-attention weights control information aggregation; physical allocation weights control water routing. The treatment return fraction is a hydraulic quantity.

## Files

All source and documentation files are in this folder:

```text
HyConGT/
    model.py
    physics.py
    data.py
    ops_data.py
    metrics.py
    train_seq.py
    README.md
    DATA_SCHEMA.md
    requirements.txt
    .gitignore
```

| File | Function |
| --- | --- |
| `model.py` | Directed GAT, GRU, parameter heads and initial storage |
| `physics.py` | Differentiable SRTO routing and storage updates |
| `data.py` | Input alignment, node attributes and temporal features |
| `ops_data.py` | Operational water and pond-storage inputs |
| `metrics.py` | R², NSE, KGE, RMSE and MAE |
| `train_seq.py` | Training, validation selection, model replay and output |
| `README.md` | Model description, installation, commands and output definitions |
| `DATA_SCHEMA.md` | Input file layout, field names, units and output columns |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Exclusions for local data, checkpoints, results and caches |

`data.py` calls `ops_data.py` to assemble operational inputs. Both files contain processing code. The training entry point is `train_seq.py`; it advances the GAT–GRU and SRTO states together using the same daily calculation during training and model replay.

## Installation

Use Python 3.10 or 3.11. Run the following commands from this folder:

```bash
python -m pip install -r requirements.txt
python train_seq.py --help
```

The checked CPU environment used Python 3.10.19, PyTorch 2.0.1, NumPy 1.24.4, pandas 2.0.3 and openpyxl 3.1.5.

## Data

Input file structures, variable names and units are described in [DATA_SCHEMA.md](DATA_SCHEMA.md). The study datasets are not publicly distributed. Data remain in a local directory supplied through `--data_root`.

## Training

```bash
python train_seq.py --site dexing --data_root /path/to/data --out_dir /path/to/results
python train_seq.py --site jiama --data_root /path/to/data --out_dir /path/to/results
```

Replace the paths with local directories. Enclose paths containing spaces in quotation marks. `--site all` trains the two sites separately.

Both sites use daily discharge supervision. The composite loss combines Huber errors in log-transformed flow with Huber errors scaled by discharge magnitude. Input standardization uses training-period statistics. Daily predictions and observations enter validation and test metrics directly.

The implementation covers the flow-reconstruction method: the directed GAT–GRU parameterization and SRTO water balance described in supplementary Text S2, together with the chronological training, composite Huber loss and evaluation metrics in Text S4. The daily-supervision setting is specified here as the configuration of this code release.

| Site | Training period | Validation period | Test period |
| --- | --- | --- | --- |
| Dexing (DCM) | January 2021–December 2023 | January–December 2024 | January–July 2025 |
| Jiama (JCPM) | January 2020–August 2024 | September–December 2024 | January–August 2025 |

GRU hidden states and Retention storage are propagated chronologically. Training uses truncated backpropagation across computational segments. The checkpoint with the highest validation mean node NSE is selected and reloaded for final reconstruction. Test observations are excluded from parameter updates and checkpoint selection.

| Setting | Default |
| --- | --- |
| GAT layers / attention heads | 3 / 4 |
| GAT / GRU hidden dimension | 64 / 64 |
| Dropout | 0.10 |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.001 / 0.0001 |
| Maximum epochs / early-stopping patience | 40 / 10 |
| Computational segment | 40 days |
| SRTO routing iterations | 3, followed by final routing and storage update |
| Random seed | 42 |

`--gauge_weights` sets training weights by node; the retained default for Dexing O3 is 2.5, with 1 for other nodes. `--select_weights` sets validation weights, which are equal by default. Full options are available through `--help`, and the selected run configuration is saved with the model.

### Antecedent states

At every node, the preceding day's discharge calculated by the SRTO module enters the next day's neural input after log1p transformation and division by a scale fitted from training-period discharge. The initial discharge state is zero. This state is updated from model predictions throughout training, validation and testing, including at unmonitored nodes.

Observed discharge supplies training targets, the training-period normalization scale, validation selection and evaluation references. It does not enter the time-varying prediction inputs, replace the preceding model discharge, or blend with the SRTO output. The model uses no observed-flow lag, rolling-flow or monthly-flow-summary features. Dexing retains temporal inertia in its operational parameters. Operational water plans and preceding pond-storage records remain separate inputs as specified in DATA_SCHEMA.md.

## Evaluation and output

Evaluation reports each monitored node separately and pools the valid node/date pairs for overall performance. R² is squared Pearson correlation. NSE compares squared prediction error with the variance of the observations. KGE combines correlation, variability and mean-flow ratios. RMSE and MAE are expressed in m³/d.

Each site's output directory contains:

| Output | Contents |
| --- | --- |
| `best_hycongt.pt` | Selected model, normalization, configuration and graph |
| `training_history.csv` | Training loss and validation daily NSE by epoch |
| `all_nodes_daily_flow.csv` | Date, node, observed flow, reconstructed flow and split |
| `daily_metrics.csv` | Node and pooled metrics, with paired sample counts |
| `run_metadata.json` | Training settings, chronological splits, dynamic feature names and reconstructed-state input protocol |

`ALL_OBSERVED` identifies the pooled metrics. Nodes without flow observations receive reconstructed flows but do not contribute to accuracy metrics.

Pooled metrics are calculated from all paired node/date values. Validation selection averages node NSE values, using at least five observations with nonzero variance at each eligible node. Undefined metrics are written as empty CSV cells.

## Load and replay a model

```bash
python train_seq.py --site dexing --data_root /path/to/data --checkpoint /path/to/results/dexing/best_hycongt.pt --out_dir /path/to/replay_results
```

Provide the same input history and node ordering. The command loads the saved model configuration and normalization, replays the daily sequence and writes predictions and metrics.

Checkpoints use format version 3 and record the reconstructed-discharge input protocol. Checkpoints from earlier releases are rejected because their feature sets or model parameters may differ; train a new model with this release before replaying it. Published numerical results and SHAP figures must be checked against the corresponding trained model rather than assumed to carry over from an earlier configuration.

## Code and data scope

This folder contains the main flow-reconstruction model and its input-processing, training and evaluation code. The public files contain no research observations, model checkpoints or generated results. All input/output operations are local. Data and results directories should be kept outside this source folder.
