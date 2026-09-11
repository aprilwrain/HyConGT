from __future__ import annotations

import torch
import torch.nn as nn


class DifferentiableSRTOBalance(nn.Module):
    """Differentiable daily Source Retention Treatment Outfall water balance.

    The implementation follows Supplementary Text S3. Engineering-edge weights are
    dynamically corrected, normalized by source node, and used only on edges already
    present in the supplied network. Three intraday routing iterations are used by
    default, followed by one final routing pass that determines the reported discharge
    and the storage carried to the next day.
    """

    def __init__(self, n_iters: int = 3, delta_alpha_max: float = 0.25):
        super().__init__()
        if n_iters < 1:
            raise ValueError("n_iters must be at least 1.")
        self.n_iters = int(n_iters)
        self.delta_alpha_max = float(delta_alpha_max)

    def corrected_alpha(
        self,
        edge_alpha: torch.Tensor,
        delta_alpha: torch.Tensor,
        edge_src: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """Return dynamically adjusted pathway fractions that sum to one per source."""

        delta = delta_alpha.clamp(-self.delta_alpha_max, self.delta_alpha_max)
        score = edge_alpha.clamp_min(0.0).unsqueeze(0) * torch.exp(delta)
        out = torch.zeros_like(score)
        for source in range(n_nodes):
            mask = edge_src == source
            if bool(mask.any()):
                source_score = score[:, mask]
                denom = source_score.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                out[:, mask] = source_score / denom
        return out

    @staticmethod
    def _route(
        q: torch.Tensor,
        alpha: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        edge_flow = alpha * q[:, edge_src]
        qin = q.new_zeros((q.shape[0], n_nodes))
        qin.index_add_(1, edge_dst, edge_flow)
        return qin

    @staticmethod
    def _retention(
        qin: torch.Tensor,
        storage: torch.Tensor,
        pet_mm: torch.Tensor,
        k_e: torch.Tensor,
        r: torch.Tensor,
        asurf: torch.Tensor,
        vmax: torch.Tensor,
        is_r: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        water = storage + qin
        evap_potential = k_e * pet_mm * asurf.unsqueeze(0) / 1000.0
        evap = torch.minimum(evap_potential, water.clamp_min(0.0))
        available = (water - evap).clamp_min(0.0)
        release = r * available
        temp_storage = available - release
        vmax_b = vmax.unsqueeze(0).expand_as(temp_storage)
        spill = (temp_storage - vmax_b).clamp_min(0.0)
        q_r = (release + spill) * is_r.unsqueeze(0)
        next_storage = torch.minimum(temp_storage, vmax_b).clamp_min(0.0)
        return q_r, next_storage

    @staticmethod
    def _treatment(
        qin: torch.Tensor,
        p: torch.Tensor,
        eta: torch.Tensor,
        qcap: torch.Tensor,
        is_t: torch.Tensor,
    ) -> torch.Tensor:
        qcap_b = qcap.unsqueeze(0).expand_as(qin)
        treated = torch.minimum(p * qin, qcap_b)
        return (qin - (1.0 - eta) * treated) * is_t.unsqueeze(0)

    def forward(
        self,
        *,
        precip_mm: torch.Tensor,
        snowmelt_mm: torch.Tensor,
        pet_mm: torch.Tensor,
        gw_mm: torch.Tensor,
        storage: torch.Tensor,
        k_c: torch.Tensor,
        k_snow: torch.Tensor,
        k_gw: torch.Tensor,
        k_mine: torch.Tensor,
        k_intake: torch.Tensor,
        k_rec: torch.Tensor,
        k_e: torch.Tensor,
        r: torch.Tensor,
        p: torch.Tensor,
        eta: torch.Tensor,
        q_intake_latent: torch.Tensor,
        q_op_latent: torch.Tensor,
        delta_alpha: torch.Tensor,
        ops_scale: torch.Tensor,
        pool_drive: torch.Tensor,
        q_plan_recycle: torch.Tensor,
        q_plan_treat: torch.Tensor,
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
        batch, n_nodes = precip_mm.shape

        k_c = k_c.clamp(0.5, 1.5)
        k_snow = k_snow.clamp(0.5, 1.5)
        k_gw = k_gw.clamp(0.0, 3.0)
        k_mine = k_mine.clamp(0.5, 1.5)
        k_intake = k_intake.clamp(0.0, 2.0)
        k_rec = k_rec.clamp(0.0, 3.0)
        k_e = k_e.clamp(0.5, 1.5)
        r = r.clamp(0.0, 1.0)
        p = p.clamp(0.0, 1.0)
        eta = eta.clamp(0.0, 1.0)
        q_intake_latent = q_intake_latent.clamp(0.0, 8.0e4)
        q_op_latent = q_op_latent.clamp(0.0, 8.0e4)

        area = area_m2.unsqueeze(0)
        source = is_s.unsqueeze(0)
        surface_source = is_surface_s.unsqueeze(0)
        mine_source = is_mine_s.unsqueeze(0)
        intake_source = is_intake_s.unsqueeze(0)

        runoff_coefficient = (c0.unsqueeze(0) * k_c).clamp(0.0, 1.0)
        effective_precip = precip_mm + k_snow * snowmelt_mm
        q_leach = area * runoff_coefficient * effective_precip / 1000.0 * surface_source
        q_groundwater = k_gw * gw_mm.clamp_min(0.0) * area / 1000.0 * surface_source
        q_mine = k_mine * qmine_base.unsqueeze(0) * mine_source

        base_intake = qintake_base.unsqueeze(0)
        has_intake_base = (base_intake > 0.0).to(precip_mm.dtype)
        q_intake = (
            k_intake * base_intake * has_intake_base
            + q_intake_latent * (1.0 - has_intake_base)
        ) * intake_source

        q_recycle = k_rec * (
            q_recycle_base.unsqueeze(0) + q_plan_recycle.clamp_min(0.0)
        ) * source
        q_residual = q_op_latent * (0.35 + ops_scale.clamp(0.0, 4.0)) * source
        q_process = torch.minimum(
            k_rec * q_plan_treat.clamp_min(0.0),
            q_plan_treat.new_full(q_plan_treat.shape, 6.0e4),
        ) * source
        q_pool = torch.minimum(
            0.25 * pool_drive.clamp_min(0.0),
            pool_drive.new_full(pool_drive.shape, 1.2e4),
        ) * source

        q_source = (
            q_leach
            + q_groundwater
            + q_mine
            + q_intake
            + q_recycle
            + q_residual
            + q_process
            + q_pool
        ) * source

        alpha = self.corrected_alpha(edge_alpha, delta_alpha, edge_src, n_nodes)
        q = q_source.clone()

        for _ in range(self.n_iters):
            qin = self._route(q, alpha, edge_src, edge_dst, n_nodes)
            q_r, _ = self._retention(qin, storage, pet_mm, k_e, r, asurf, vmax, is_r)
            q_t = self._treatment(qin, p, eta, qcap, is_t)
            q_o = qin * is_o.unsqueeze(0)
            q = q_source + q_r + q_t + q_o

        final_qin = self._route(q, alpha, edge_src, edge_dst, n_nodes)
        q_r, next_storage_r = self._retention(
            final_qin, storage, pet_mm, k_e, r, asurf, vmax, is_r
        )
        q_t = self._treatment(final_qin, p, eta, qcap, is_t)
        q_o = final_qin * is_o.unsqueeze(0)
        q = q_source + q_r + q_t + q_o

        next_storage = torch.where(
            is_r.unsqueeze(0) > 0.5,
            next_storage_r,
            storage,
        )
        return q.clamp_min(0.0), next_storage
