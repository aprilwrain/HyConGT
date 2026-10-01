# Input data structure

The directory specified by `--data_root` has the following structure. Only the selected site's files and its operational inputs are needed.

```text
data/
    dexing/
        dx_node.csv
        dx_edge.csv
        dx_sub.csv
        daily_observed.csv
    jiama/
        jiama_node.csv
        jiama_edge.csv
        jiama_sub.csv
        daily_observed.csv
    aux/
        dx_recycle_2021_2023.xlsx
        jm_T2_monthly.csv
        jm_T5_daily.csv
```

CSV files support UTF-8 and GB18030. Node identifiers must match across the node, edge and observation tables. Dates must align with the daily SWAT series. If a site's `dx_edge_recalib.csv` or `jiama_edge_recalib.csv` is present, it is used in place of the corresponding edge file.

## Node attributes

| Field | Type / unit | Description |
| --- | --- | --- |
| `node_id` | Text | Unique node identifier |
| `node_type` | Text | `S`, `R`, `T` or `O` |
| `SUB` | Integer | Associated SWAT subbasin |
| `name` | Text | Facility name used for Source classification |
| `jiangyuhuishuimianji_m2` | m² | Surface contributing area at DCM |
| `area_km2` | km² | Surface contributing area |
| `jingliuxishu` | Dimensionless | Baseline runoff coefficient |
| `youxiaokurong_m3` | m³ | Effective storage capacity |
| `shuimianmianji_m2` | m² | Water surface area; secondary area field at DCM |
| `yihongkoubiaogao_m` | m | Spillway elevation |
| `kuangkengyongshui_m3/d` | m³/d | Mine inflow |
| `kuangkengyongshui_m3/h` | m³/h | Hourly mine inflow, converted to daily units when used |
| `chulinengli_m3/d` | m³/d | Treatment capacity |
| `chlinengli_t/d` | t/d | Tonnage-capacity field, using the original table spelling |
| `zuidabengpainengli_m3/h` | m³/h | Pumping capacity |
| `kaicainengli_t/d` | t/d | Mining-capacity field |
| `huiyonglv_%` | % | Water reuse percentage |
| `<field>_ismissing` | 0 or 1 | Attribute availability indicator |

`node_id`, `node_type` and `SUB` are required. Other fields apply according to facility type. Node types represent Source, Retention, Treatment and Outfall. Attribute conversions, capacity bounds and missing-attribute defaults are implemented in `data.py`. `ALL_OBSERVED` is reserved for pooled evaluation output.

## Engineering connections

| Field | Type | Description |
| --- | --- | --- |
| `from_id` | Text | Source node of the connection |
| `to_id` | Text | Destination node of the connection |
| `weight_alpha` | Positive number | Baseline allocation weight |

One row represents one directed engineering connection. Directed pairs must be unique, and both nodes must exist. The SRTO module normalizes allocation weights by source. Self-loops for graph attention are generated internally.

## SWAT daily inputs

| Field | Unit / format | Description |
| --- | --- | --- |
| `YYYYDDD` | Seven digits | Year followed by three-digit day of year |
| `SUB` | Integer | Subbasin identifier |
| `PRECIPmm` | mm | Precipitation |
| `SNOWMELTmm` | mm | Snowmelt |
| `PETmm` | mm | Potential evapotranspiration |
| `ETmm` | mm | Actual evapotranspiration |
| `SWmm` | mm | Soil water |
| `PERCmm` | mm | Percolation |
| `SURQmm` | mm | Surface runoff |
| `GW_Qmm` | mm | Groundwater contribution |
| `WYLDmm` | mm | Water yield |
| `LAT_Qmm` | mm | Lateral flow |

Each mapped subbasin has one row per date over a continuous daily period. The SRTO module converts the applicable depth terms to water volumes using node area.

## Daily discharge

`daily_observed.csv` uses long format:

| Field | Unit / format | Description |
| --- | --- | --- |
| `date` | `YYYY-MM-DD` | Observation date |
| `node_id` | Text | Monitored node identifier |
| `flow_m3_d` | m³/d | Daily discharge |

Each date/node pair is unique. Flow is nonnegative. Nodes without flow supervision do not require observation rows. The daily table supplies training targets and validation/test references. Antecedent observation features are constructed internally in chronological order.

## Operational inputs

### DCM annual operational water table

`dx_recycle_2021_2023.xlsx` uses the first worksheet and its first row as column headers. The adapter reads the following zero-based positions after the header:

| Data-row index | Quantity | Unit |
| --- | --- | --- |
| 0 | Total water | m³/year |
| 1 | Fresh water | m³/year |
| 2 | Recycle water | m³/year |
| 12 | Recycle percentage | % |

Zero-based columns 3, 5 and 7 correspond to 2021, 2022 and 2023. The operational adapter converts annual volumes to daily drivers, applies the seasonal operational factor and carries the latest annual record forward where required.

### JCPM monthly operational water table

`jm_T2_monthly.csv` follows this column ordering:

| Position | Content | Unit / format |
| --- | --- | --- |
| First column | Month | Text `YYYY.MM` |
| Middle columns | Water volumes | m³/month |
| Last column | Recycle percentage | 0–100, optionally followed by `%` |

Middle-column names containing `失` identify water losses; the remaining middle columns contribute to recycle volume. Volumes are divided by the number of days in the calendar month. Retain two digits for the month. Month entries are unique. The existing adapter interprets zero operational entries as missing for forward filling.

### JCPM pond storage

`jm_T5_daily.csv` follows this column ordering:

| Position | Content | Unit / format |
| --- | --- | --- |
| First column | Date | Text `YYYY.MM.DD` |
| Second column | Pond storage | m³ |

Dates are unique. The adapter forward-fills storage records and applies a one-day lag.

## Output structure

| File | Fields |
| --- | --- |
| `all_nodes_daily_flow.csv` | `date`, `node_id`, `observed_m3_d`, `pred_m3_d`, `split` |
| `daily_metrics.csv` | `split`, `node_id`, `R2`, `NSE`, `KGE`, `RMSE`, `MAE`, `n` |
| `training_history.csv` | `epoch`, `train_loss`, `valid_daily_NSE` |

`split` identifies training, validation, test or dates outside those intervals. `n` is the paired evaluation sample count. Model weights and run settings are written to the selected output directory.
