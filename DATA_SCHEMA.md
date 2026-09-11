# Prepared input contract

The repository intentionally does not contain raw mine data, engineering records, operational records, monitoring observations, SWAT project files, or site-specific raw-data parsers.

The public model reads one NumPy NPZ file produced inside the secure environment. The NPZ file is excluded by `.gitignore` and should not be committed.

## Required arrays

Let `T` be the number of days, `N` the number of SRTO nodes, `Fd` the number of dynamic graph-learning features, and `Fs` the number of static graph-learning features.

| Key | Shape | Meaning |
| --- | --- | --- |
| `site` | scalar string | `dexing` or `jiama` |
| `node_ids` | `(N,)` strings | Public or internal node identifiers used consistently within the prepared file |
| `dates` | `(T,)` strings | Daily dates in ISO format |
| `x_dynamic` | `(T,N,Fd)` | Graph-learning dynamic features described in Supplementary Table S5, excluding previous-day reconstructed flow because that channel is generated online by the model |
| `x_static` | `(N,Fs)` | Static engineering attributes and one-hot node representation described in Supplementary Table S5 |
| `dynamic_feature_names` | `(Fd,)` strings | Names for `x_dynamic` columns |
| `static_feature_names` | `(Fs,)` strings | Names for `x_static` columns |
| `precip_mm` | `(T,N)` | Precipitation in physical units for SRTO |
| `snowmelt_mm` | `(T,N)` | Snowmelt in physical units for SRTO |
| `pet_mm` | `(T,N)` | Potential evapotranspiration in physical units for SRTO |
| `gw_mm` | `(T,N)` | SWAT groundwater contribution in physical units for SRTO |
| `ops_scale` | `(T,N)` | Operational intensity indicator used in Eq. S26 |
| `pool_drive` | `(T,N)` | Previous-day process return-water pond storage contribution used in Eq. S28, zero where not applicable |
| `q_plan_recycle` | `(T,N)` | Planned recycled-water volume assigned to source nodes, m3 d-1 |
| `q_plan_treat` | `(T,N)` | Planned treatment throughput proxy assigned to source nodes, m3 d-1 |
| `area_m2` | `(N,)` | Contributing area for surface Source nodes |
| `c0` | `(N,)` | Baseline runoff coefficient |
| `vmax` | `(N,)` | Effective storage capacity |
| `asurf` | `(N,)` | Effective water-surface area |
| `qmine_base` | `(N,)` | Baseline mine inflow |
| `qintake_base` | `(N,)` | Baseline external intake, zero if unavailable |
| `q_recycle_base` | `(N,)` | Baseline recycled-water flow |
| `qcap` | `(N,)` | Treatment capacity, use a sufficiently large nonbinding value when unavailable |
| `is_s`, `is_r`, `is_t`, `is_o` | `(N,)` | One-hot SRTO functional classes |
| `is_surface_s`, `is_mine_s`, `is_intake_s` | `(N,)` | One-hot Source subtypes for Source nodes, zero for non-Source nodes |
| `edge_src`, `edge_dst` | `(E,)` integers | Directed engineering edges |
| `edge_alpha` | `(E,)` | Positive baseline pathway weights |
| `obs_daily` | `(T,N)` | Daily observed flow with `NaN` where unavailable |
| `obs_monthly_total` | `(T,N)` | Monthly cumulative observations placed on the last day of each observed month, `NaN` elsewhere |
| `supervision_mode` | scalar string | `monthly` for DCM and `daily` for JCPM |

## Feature construction boundary

`x_dynamic` should contain the variables stated in Supplementary Table S5. These include the ten SWAT hydrological state and flux variables, month and day-of-year cyclic encodings, operational recycling rate, planned treatment throughput and planned recycled-water volume with the stated raw, log1p, and relative-to-median representations, and plan-month cyclic encoding.

`x_static` should contain the static engineering attributes and node encodings stated in Supplementary Table S5. Dynamic feature standardization is performed by the public training code using training-period statistics only. Static feature standardization is performed across nodes within the site.

The previous-day reconstructed-flow feature is intentionally absent from the prepared NPZ. It is generated during the free-running model rollout from the model prediction at `t-1`, so observed gauge flow cannot enter this channel.
