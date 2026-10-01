from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableSRTOBalance(nn.Module):
    """Unified SRTO daily water balance.

    S: surface, groundwater, mine inflow, intake, recycle and operational inputs
       Q_recycle = k_rec * (Q_recycle_base + Q_plan_recycle)
       Q_plan_treat drives bounded process makeup on S (monthly plan proxy)
       Q_surface = A * C * (PRECIP + k_snow * SNOWMELT) / 1000
       C = clip(C0 * k_c, 0, 1)
       Q_mine = k_mine * Q_mine_base
       Q_intake = k_intake * Q_intake_base (or free non-negative intake if base=0)
    R: continuous storage, evaporation, release, spill (no local generation)
    T: treatment fraction p, hydraulic return fraction eta; Qin = Q_T + Q_loss
    O: Q = Qin
    Edge: score = alpha0 * exp(delta_alpha), |delta_alpha| <= delta_alpha_max, normalize by source
    """

    def __init__(self, n_iters: int = 3, delta_alpha_max: float = 0.25):
        super().__init__()
        self.n_iters = n_iters
        self.delta_alpha_max = float(delta_alpha_max)

    def corrected_alpha(
        self,
        edge_alpha: torch.Tensor,
        delta_alpha: torch.Tensor,
        edge_src: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        dmax = self.delta_alpha_max
        d = delta_alpha.clamp(-dmax, dmax)
        score = edge_alpha.clamp_min(0.0) * torch.exp(d)
        out = torch.zeros_like(score)
        for i in range(n_nodes):
            mask = edge_src == i
            if bool(mask.any()):
                s = score[:, mask]
                out[:, mask] = s / s.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return out

    def forward(
        self,
        precip_mm: torch.Tensor,
        snowmelt_mm: torch.Tensor,
        pet_mm: torch.Tensor,
        V: torch.Tensor,
        k_c: torch.Tensor,
        k_snow: torch.Tensor,
        k_mine: torch.Tensor,
        k_intake: torch.Tensor,
        k_e: torch.Tensor,
        r: torch.Tensor,
        p: torch.Tensor,
        eta: torch.Tensor,
        delta_alpha: torch.Tensor,
        q_intake_free: torch.Tensor,
        k_rec: torch.Tensor,
        k_gw: torch.Tensor,
        q_ext: torch.Tensor,
        gw_mm: torch.Tensor,
        ops_scale: torch.Tensor, # (B,N) exogenous intensity >=0
        pool_drive: torch.Tensor, # (B,N) recycle-pond storage lag (jiama)
        q_plan_recycle: torch.Tensor, # (B,N) monthly planned recycle m3/d
        q_plan_treat: torch.Tensor, # (B,N) monthly planned treat m3/d
        area_m2: torch.Tensor,
        c0: torch.Tensor,
        vmax: torch.Tensor,
        asurf: torch.Tensor,
        qmine_base: torch.Tensor,
        qintake_base: torch.Tensor,
        q_recycle_base: torch.Tensor,
        qcap: torch.Tensor,
        is_s: torch.Tensor,
        is_r: torch.Tensor,
        is_t: torch.Tensor,
        is_o: torch.Tensor,
        is_surface_s: torch.Tensor,
        is_mine_s: torch.Tensor,
        is_intake_s: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = precip_mm.shape
        device = precip_mm.device

        k_c = k_c.clamp(0.5, 1.5)
        k_snow = k_snow.clamp(0.5, 1.5)
        k_mine = k_mine.clamp(0.5, 1.5)
        k_intake = k_intake.clamp(0.0, 2.0)
        k_e = k_e.clamp(0.5, 1.5)
        k_rec = k_rec.clamp(0.0, 3.0)
        k_gw = k_gw.clamp(0.0, 3.0)
        r = r.clamp(0.0, 1.0)
        p = p.clamp(0.0, 1.0)
        eta = eta.clamp(0.0, 1.0)

        C = (c0 * k_c).clamp(0.0, 1.0)
        p_eff = precip_mm + k_snow * snowmelt_mm
        q_surface = area_m2 * C * p_eff / 1000.0 * is_surface_s
        q_gw = k_gw * gw_mm.clamp_min(0.0) * area_m2 / 1000.0 * is_surface_s
        q_mine = k_mine * qmine_base * is_mine_s
        has_intake_base = (qintake_base > 0).float()
        q_intake = (
            k_intake * qintake_base * has_intake_base
            + F.softplus(q_intake_free).clamp_max(8e4) * (1.0 - has_intake_base)
        ) * is_intake_s
        q_recycle = (k_rec * (q_recycle_base + q_plan_recycle.clamp_min(0.0))) * is_s
        # Bounded latent operational-water contribution retained from the main model
        q_ext_c = (8.0e4 * torch.sigmoid(q_ext) * (0.35 + ops_scale.clamp(0.0, 4.0))) * is_s
        q_plan_t = (1.0 * k_rec * q_plan_treat.clamp_min(0.0)).clamp_max(6.0e4) * is_s
        q_pool = (0.25 * pool_drive.clamp_min(0.0)).clamp_max(1.2e4) * is_s
        q_s = (q_surface + q_gw + q_mine + q_intake + q_recycle + q_ext_c + q_plan_t + q_pool) * is_s

        alpha = self.corrected_alpha(edge_alpha, delta_alpha, edge_src, N)
        Q = q_s.clone()
        for _ in range(self.n_iters):
            edge_flow = alpha * Q[:, edge_src]
            Qin = torch.zeros(B, N, device=device, dtype=Q.dtype)
            Qin.index_add_(1, edge_dst, edge_flow)

            W = V + Qin
            q_evap = torch.minimum(k_e * pet_mm * asurf / 1000.0, W.clamp_min(0.0))
            w_avail = (W - q_evap).clamp_min(0.0)
            q_release = r * w_avail
            v_temp = w_avail - q_release
            q_spill = (v_temp - vmax.unsqueeze(0)).clamp_min(0.0)
            q_r = (q_release + q_spill) * is_r

            q_treat = torch.minimum(p * Qin, qcap.unsqueeze(0).expand_as(Qin))
            q_t = (Qin - (1.0 - eta) * q_treat) * is_t
            q_o = Qin * is_o
            Q = q_s + q_r + q_t + q_o

        edge_flow = alpha * Q[:, edge_src]
        Qin = torch.zeros(B, N, device=device, dtype=Q.dtype)
        Qin.index_add_(1, edge_dst, edge_flow)

        W = V + Qin
        q_evap = torch.minimum(k_e * pet_mm * asurf / 1000.0, W.clamp_min(0.0))
        w_avail = (W - q_evap).clamp_min(0.0)
        q_release = r * w_avail
        v_temp = w_avail - q_release
        q_spill = (v_temp - vmax.unsqueeze(0)).clamp_min(0.0)
        q_r = (q_release + q_spill) * is_r
        q_treat = torch.minimum(p * Qin, qcap.unsqueeze(0).expand_as(Qin))
        q_t = (Qin - (1.0 - eta) * q_treat) * is_t
        q_o = Qin * is_o
        Q = q_s + q_r + q_t + q_o

        v_next_r = torch.minimum(v_temp, vmax.unsqueeze(0).expand_as(V)).clamp_min(0.0)
        V_next = torch.where(is_r > 0.5, v_next_r, V)
        return Q.clamp_min(0.0), V_next
