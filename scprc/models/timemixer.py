"""TimeMixer adapter used by the UrbanEV scprc runner.

The vendored model is kept unchanged under ``scprc/vendor/TimeMixer``.
This module only supplies data handling, deterministic training and the small
configuration namespace expected by the upstream implementation.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


_ROOT = Path(__file__).resolve().parents[1]
_VENDOR = _ROOT / "vendor" / "TimeMixer"


def _load_upstream_model():
    """Load the vendored model without importing the repository's model package."""
    vendor = str(_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    import layers
    layer_path = str(_VENDOR / "layers")
    if layer_path not in layers.__path__:
        layers.__path__.append(layer_path)
    path = _VENDOR / "models" / "TimeMixer.py"
    spec = importlib.util.spec_from_file_location("urbanev_upstream_timemixer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored TimeMixer from {path}")
    module = importlib.util.module_from_spec(spec)
    # The upstream file imports ``layers.*``; the vendor root above resolves it.
    spec.loader.exec_module(module)
    # Expose the candidate decomposition for CPU contract tests without
    # importing the vendored ``models`` package a second time.
    globals()["CausalEMADecomp"] = module.CausalEMADecomp
    globals()["CausalDailyDecomp"] = module.CausalDailyDecomp
    globals()["CausalFIRDecomp"] = module.CausalFIRDecomp
    globals()["CausalLevelDriftTransport"] = module.CausalLevelDriftTransport
    globals()["FixedEndpointLevelDriftBridge"] = module.FixedEndpointLevelDriftBridge
    return module.Model


_UpstreamModel = _load_upstream_model()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels make registry retries comparable across workers.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_samples(x: Any, name: str) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [samples, time, channels], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return value


def _config_namespace(config: dict[str, Any], channels: int, pred_len: int) -> SimpleNamespace:
    """Map the runner's concise config to the names used by upstream Model."""
    raw_seq_len = int(config["seq_len"])
    patch_size = int(config.get("patch_size", 1))
    if patch_size < 1 or raw_seq_len % patch_size != 0:
        raise ValueError("patch_size must divide the raw sequence length")
    seq_len = raw_seq_len // patch_size
    window = int(config.get("down_sampling_window", 2))
    # Only the multiscale count belongs in down_sampling_layers.  e_layers is
    # the number of PDM blocks and is intentionally separate.
    down_layers = int(config.get("down_sampling_layers", 3 if config.get("protocol") in {"research_volume", "research_dev"} else 2))
    if window < 2 or down_layers < 1 or seq_len // (window ** down_layers) < 1:
        raise ValueError("TimeMixer requires at least one valid downsampling layer with window >= 2")

    values = dict(
        task_name="long_term_forecast",
        seq_len=seq_len,
        raw_seq_len=raw_seq_len,
        patch_size=patch_size,
        label_len=min(int(config.get("label_len", seq_len)), seq_len),
        pred_len=pred_len,
        enc_in=channels,
        dec_in=channels,
        c_out=channels,
        d_model=int(config.get("d_model", 64)),
        d_ff=int(config.get("d_ff", 128)),
        e_layers=int(config.get("e_layers", 2)),
        moving_avg=int(config.get("moving_avg", 25)),
        causal_ema_alpha=float(config.get("causal_ema_alpha", 0.25)),
        causal_daily_period=int(config.get("causal_daily_period", 24)),
        causal_daily_cycles=int(config.get("causal_daily_cycles", 7)),
        causal_fir_kernel=int(config.get("causal_fir_kernel", 25)),
        use_phase_cycle=int(config.get("use_phase_cycle", 0)),
        phase_cycle_routers=int(config.get("phase_cycle_routers", 2)),
        phase_cycle_heads=int(config.get("phase_cycle_heads", 4)),
        phase_cycle_dropout=float(config.get("phase_cycle_dropout", config.get("dropout", 0.1))),
        phase_cycle_pos_embed=int(config.get("phase_cycle_pos_embed", 0)),
        use_phase_latent_decomp=int(config.get("use_phase_latent_decomp", 0)),
        phase_latent_cycles=int(config.get("phase_latent_cycles", 7)),
        phase_latent_phases=int(config.get("phase_latent_phases", 24)),
        use_joint_phase_region=int(config.get("use_joint_phase_region", 0)),
        joint_phase_routers=int(config.get("joint_phase_routers", 2)),
        joint_region_routers=int(config.get("joint_region_routers", 4)),
        joint_phase_heads=int(config.get("joint_phase_heads", 4)),
        joint_region_heads=int(config.get("joint_region_heads", 4)),
        joint_phase_region_dropout=float(config.get("joint_phase_region_dropout", config.get("dropout", 0.1))),
        use_tariff_operator=int(config.get("use_tariff_operator", 0)),
        tariff_feature_dim=int(config.get("tariff_feature_dim", 2)),
        tariff_rank=int(config.get("tariff_rank", 8)),
        tariff_heads=int(config.get("tariff_heads", 4)),
        tariff_state_rank=int(config.get("tariff_state_rank", 8)),
        tariff_scale_rank=int(config.get("tariff_scale_rank", 4)),
        paired_input_dim=int(config.get("paired_input_dim", 0)),
        dropout=float(config.get("dropout", 0.1)),
        channel_independence=int(
            config.get(
                "channel_independence",
                0 if str(config.get("model", "timemixer_ci")) == "timemixer_cd" else 1,
            )
        ),
        decomp_method=str(config.get("decomp_method", "moving_avg")),
        top_k=int(config.get("top_k", 5)),
        spectral_temperature=float(config.get("spectral_temperature", 1.0)),
        prediction_method=str(config.get("prediction_method", "linear")),
        basis_dim=int(config.get("basis_dim", 16)),
        latent_mixer=str(config.get("latent_mixer", "pointwise")),
        use_mrti=int(config.get("use_mrti", 0)),
        mrti_views=str(config.get("mrti_views", "both")),
        use_phase_mrti=int(config.get("use_phase_mrti", 0)),
        use_event_mrti=int(config.get("use_event_mrti", 0)),
        use_apid=int(config.get("use_apid", 0)),
        use_innovation_exchange=int(config.get("use_innovation_exchange", 0)),
        use_crps_decoder=int(config.get("use_crps_decoder", 0)),
        equilibrium_iterations=int(config.get("equilibrium_iterations", 4)),
        equilibrium_rho=float(config.get("equilibrium_rho", 0.75)),
        transport_scale=float(config.get("transport_scale", 0.5)),
        phase_state_rank=int(config.get("phase_state_rank", 8)),
        phase_state_forecast_step=str(config.get("phase_state_forecast_step", "predictive")),
        phase_router_count=int(config.get("phase_router_count", 4)),
        phase_router_heads=int(config.get("phase_router_heads", 4)),
        dcs_state_rank=int(config.get("dcs_state_rank", 24)),
        dcs_context_basis=int(config.get("dcs_context_basis", 4)),
        dcs_dynamic=int(config.get("dcs_dynamic", 1)),
        dcs_integrator=str(config.get("dcs_integrator", "cayley")),
        dcs_max_decay=float(config.get("dcs_max_decay", 0.25)),
        ntf_frame_rank=int(config.get("ntf_frame_rank", 24)),
        ntf_frame_dropout=float(config.get("ntf_frame_dropout", 0.0)),
        dct_rank=int(config.get("dct_rank", 4)),
        dct_context_hidden=int(config.get("dct_context_hidden", 16)),
        # MDMixer-lite: node/channel-specific adaptive weighting of the
        # official FMM multi-scale forecasts, plus independent trend/season
        # branches inside PDM.  Disabled for every existing route.
        use_mdmixer_lite=int(config.get("use_mdmixer_lite", 0)),
        mdmixer_scale_gate_hidden=int(config.get("mdmixer_scale_gate_hidden", 16)),
        mdmixer_scale_temperature=float(config.get("mdmixer_scale_temperature", 1.0)),
        mdmixer_dual_branch=int(config.get("mdmixer_dual_branch", 1)),
        # Horizon-conditioned representation (HCR): shared low-rank FiLM
        # profiles for the three forecast bins, applied between PDM and FMM.
        use_hcr=int(config.get("use_hcr", 0)),
        hcr_rank=int(config.get("hcr_rank", 8)),
        hcr_bins=str(config.get("hcr_bins", "1-6,7-12,13-24")),
        hcr_strength=float(config.get("hcr_strength", 0.1)),
        use_scm=int(config.get("use_scm", 0)),
        scm_rank=int(config.get("scm_rank", 8)),
        scm_strength=float(config.get("scm_strength", 0.1)),
        # CPR: trend-only low-rank cross-site routing with a CI residual path
        # and separate trend/residual future decoders.
        use_cpr=int(config.get("use_cpr", 0)),
        cpr_rank=int(config.get("cpr_rank", 4)),
        cpr_strength=float(config.get("cpr_strength", 0.1)),
        cpr_ema_alpha=float(config.get("cpr_ema_alpha", 0.25)),
        use_city_adapter=int(config.get("use_city_adapter", 0)),
        city_adapter_count=int(config.get("city_adapter_count", 7)),
        city_adapter_rank=int(config.get("city_adapter_rank", 8)),
        city_adapter_index=int(config.get("city_adapter_index", 0)),
        use_timecap=int(config.get("use_timecap", 0)),
        timecap_group_size=int(config.get("timecap_group_size", 32)),
        timecap_group_stride=int(config.get("timecap_group_stride", 16)),
        timecap_router_rank=int(config.get("timecap_router_rank", 8)),
        timecap_router_strength=float(config.get("timecap_router_strength", 0.1)),
        timecap_ar_steps=int(config.get("timecap_ar_steps", 6)),
        timecap_gate_init=float(config.get("timecap_gate_init", -6.0)),
        use_dynamic_cross_node=int(config.get("use_dynamic_cross_node", 0)),
        dynamic_cross_node_rank=int(config.get("dynamic_cross_node_rank", 8)),
        dynamic_cross_node_strength=float(config.get("dynamic_cross_node_strength", 0.05)),
        use_history_risk_gate=int(config.get("use_history_risk_gate", 0)),
        history_risk_gate_hidden=int(config.get("history_risk_gate_hidden", 16)),
        history_risk_gate_min=float(config.get("history_risk_gate_min", 0.05)),
        history_risk_gate_max=float(config.get("history_risk_gate_max", 0.95)),
        history_risk_gate_init=float(config.get("history_risk_gate_init", 0.25)),
        use_post_pdm_dynamic_cross_node=int(config.get("use_post_pdm_dynamic_cross_node", 0)),
        post_pdm_dynamic_cross_node_rank=int(config.get("post_pdm_dynamic_cross_node_rank", 8)),
        post_pdm_dynamic_cross_node_strength=float(config.get("post_pdm_dynamic_cross_node_strength", 0.05)),
        use_dynamic_phase_cross_node=int(config.get("use_dynamic_phase_cross_node", 0)),
        dynamic_phase_cross_node_rank=int(config.get("dynamic_phase_cross_node_rank", 8)),
        dynamic_phase_cross_node_strength=float(config.get("dynamic_phase_cross_node_strength", 0.05)),
        dynamic_phase_cross_node_phase_strength=float(config.get("dynamic_phase_cross_node_phase_strength", 0.01)),
        dynamic_phase_cross_node_period=int(config.get("dynamic_phase_cross_node_period", 24)),
        use_regime_state=int(config.get("use_regime_state", 0)),
        use_post_regime_state=int(config.get("use_post_regime_state", 0)),
        use_causal_innovation=int(config.get("use_causal_innovation", 0)),
        innovation_lag=int(config.get("innovation_lag", 24)),
        use_harmonic_phase_decoder=int(config.get("use_harmonic_phase_decoder", 0)),
        harmonic_phase_periods=str(config.get("harmonic_phase_periods", "24,12")),
        harmonic_phase_strength=float(config.get("harmonic_phase_strength", 0.02)),
        harmonic_phase_gate_init=float(config.get("harmonic_phase_gate_init", -4.0)),
        use_ungated_harmonic=int(config.get("use_ungated_harmonic", 0)),
        use_cross_phase_attention_decoder=int(config.get("use_cross_phase_attention_decoder", 0)),
        cross_phase_period=int(config.get("cross_phase_period", 24)),
        cross_phase_cycles=int(config.get("cross_phase_cycles", 7)),
        cross_phase_heads=int(config.get("cross_phase_heads", 4)),
        cross_phase_strength=float(config.get("cross_phase_strength", 0.02)),
        cross_phase_gate_init=float(config.get("cross_phase_gate_init", -4.0)),
        use_confidence_residual_gate=int(config.get("use_confidence_residual_gate", 0)),
        confidence_gate_hidden=int(config.get("confidence_gate_hidden", 16)),
        confidence_gate_min=float(config.get("confidence_gate_min", 0.05)),
        confidence_gate_max=float(config.get("confidence_gate_max", 0.95)),
        use_trend_dynamic_cross_node=int(config.get("use_trend_dynamic_cross_node", 0)),
        trend_dynamic_cross_node_rank=int(config.get("trend_dynamic_cross_node_rank", 8)),
        trend_dynamic_cross_node_strength=float(config.get("trend_dynamic_cross_node_strength", 0.05)),
        trend_dynamic_cross_node_kernel=int(config.get("trend_dynamic_cross_node_kernel", 5)),
        use_patch_token_cross_node=int(config.get("use_patch_token_cross_node", 0)),
        patch_token_cross_node_rank=int(config.get("patch_token_cross_node_rank", 8)),
        patch_token_cross_node_strength=float(config.get("patch_token_cross_node_strength", 0.02)),
        patch_token_cross_node_hours=int(config.get("patch_token_cross_node_hours", 6)),
        # PORI: six-hour patch-order recursion, coarse low-rank CM and fine CI
        # detail with a four-by-six-hour segment decoder.
        use_pori=int(config.get("use_pori", 0)),
        pori_rank=int(config.get("pori_rank", 4)),
        pori_patch_len=int(config.get("pori_patch_len", 6)),
        pori_strength=float(config.get("pori_strength", 0.1)),
        pori_segments=int(config.get("pori_segments", 4)),
        pori_segment_horizon=int(config.get("pori_segment_horizon", 6)),
        pori_decoder_strength=float(config.get("pori_decoder_strength", 0.1)),
        pori_router_rank=int(config.get("pori_router_rank", 4)),
        pori_router_strength=float(config.get("pori_router_strength", 0.1)),
        # Minimal PORI: aligned equal-token recursion plus ungated
        # segment-wise FMM; deliberately independent from the legacy PORI
        # channel router/order branch.
        use_pori_minimal=int(config.get("use_pori_minimal", 0)),
        pori_token_count=int(config.get("pori_token_count", 21)),
        tail_basis_dim=int(config.get("tail_basis_dim", 8)),
        tail_context_hidden=int(config.get("tail_context_hidden", 32)),
        horizon_conv_kernel=int(config.get("horizon_conv_kernel", 3)),
        conditional_spatial_topk=int(config.get("conditional_spatial_topk", 8)),
        hybrid_phase_dim=int(config.get("hybrid_phase_dim", 8)),
        hybrid_phase_routers=int(config.get("hybrid_phase_routers", 1)),
        hybrid_phase_heads=int(config.get("hybrid_phase_heads", 2)),
        kd_weight=float(config.get("kd_weight", 0.25)),
        dynamic_norm_hidden=int(config.get("dynamic_norm_hidden", 32)),
        future_stats_aux_weight=float(config.get("future_stats_aux_weight", 0.1)),
        cross_scale_state_rank=int(config.get("cross_scale_state_rank", 8)),
        cross_scale_dropout=float(config.get("cross_scale_dropout", config.get("dropout", 0.1))),
        cross_scale_identity_fusion=int(config.get("cross_scale_identity_fusion", 0)),
        cmamba_state_rank=int(config.get("cmamba_state_rank", 8)),
        cmamba_patch_size=int(config.get("cmamba_patch_size", 4)),
        cmamba_patch_sizes=str(config.get("cmamba_patch_sizes", "")),
        cmamba_gdd_rank=int(config.get("cmamba_gdd_rank", 8)),
        cmamba_dropout=float(config.get("cmamba_dropout", config.get("dropout", 0.1))),
        cmamba_use_tmb=int(config.get("cmamba_use_tmb", 1)),
        cmamba_use_gdd=int(config.get("cmamba_use_gdd", 1)),
        cmamba_node_gdd=int(config.get("cmamba_node_gdd", 0)),
        phase_teacher_checkpoint=str(config.get("phase_teacher_checkpoint", "")),
        phase_teacher_latent_dim=int(config.get("phase_teacher_latent_dim", 32)),
        phase_teacher_layers=int(config.get("phase_teacher_layers", 2)),
        phase_transport_rank=int(config.get("phase_transport_rank", 4)),
        use_scale_lifting=int(config.get("use_scale_lifting", 0)),
        # CLDST: causal level/drift coordinate transport inside every PDM
        # block.  The window is fixed to one daily cycle by default; it is
        # exposed only for reproducibility metadata, not for a search grid.
        use_level_drift_transport=int(config.get("use_level_drift_transport", 0)),
        level_drift_window=int(config.get("level_drift_window", 24)),
        use_endpoint_bridge=int(config.get("use_endpoint_bridge", 0)),
        endpoint_bridge_rank=int(config.get("endpoint_bridge_rank", 4)),
        endpoint_bridge_width=int(config.get("endpoint_bridge_width", 24)),
        innovation_topk=int(config.get("innovation_topk", 4)),
        innovation_rank=int(config.get("innovation_rank", 8)),
        innovation_lags=tuple(int(x) for x in str(config.get("innovation_lags", "0,1,2")).split(",")),
        innovation_stage=str(config.get("innovation_stage", "pre")),
        apid_periods=str(config.get("apid_periods", "24,168,12,84")),
        apid_top_k=int(config.get("apid_top_k", 3)),
        apid_temperature=float(config.get("apid_temperature", 0.7)),
        apid_axis_kernel=int(config.get("apid_axis_kernel", 3)),
        apid_peak_branch=int(config.get("apid_peak_branch", 0)),
        apid_peak_lambda=float(config.get("apid_peak_lambda", 0.15)),
        # Cross-city hierarchical probabilistic route.  These fields are
        # inert for the official TimeMixer and all existing baseline models.
        use_spectral_transport=int(config.get("use_spectral_transport", 0)),
        spectral_dictionary_size=int(config.get("spectral_dictionary_size", 8)),
        spectral_bins=int(config.get("spectral_bins", 8)),
        spectral_sinkhorn_iterations=int(config.get("spectral_sinkhorn_iterations", 8)),
        spectral_transport_temperature=float(config.get("spectral_transport_temperature", 0.2)),
        use_sample_probabilistic=int(config.get("use_sample_probabilistic", 0)),
        probabilistic_sample_count=int(config.get("probabilistic_sample_count", 0)),
        use_norm=int(config.get("use_norm", 1)),
        subtract_last=int(config.get("subtract_last", 0)),
        cpr_regime_level_width=int(config.get("cpr_regime_level_width", 24)),
        cpr_regime_scale_width=int(config.get("cpr_regime_scale_width", 72)),
        cpr_regime_moment_weight=float(config.get("cpr_regime_moment_weight", 0.05)),
        cclt_graph_fusion=int(config.get("cclt_graph_fusion", 0)),
        cclt_graph_order=int(config.get("cclt_graph_order", 1)),
        cclt_graph_volume=int(config.get("cclt_graph_volume", 0)),
        cclt_graph_contrast=int(config.get("cclt_graph_contrast", 0)),
        cclt_trajectory=int(config.get("cclt_trajectory", 0)),
        # Fixed train-prefix precision fusion of the ungated FMM scale
        # forecasts.  The arrays are computed once by the calibration script
        # and are not learned, selected per sample, or searched.
        precision_fusion=bool(config.get("precision_fusion", False)),
        precision_fusion_weights=config.get("precision_fusion_weights"),
        precision_fusion_bias=config.get("precision_fusion_bias"),
        # Fixed cross-resolution reconciliation of the native FMM scale
        # outputs.  This is a deterministic linear projection, not a gate.
        reconciled_fmm=bool(config.get("reconciled_fmm", False)),
        # CSPDM: preserve a causal 24 h history-state decomposition inside
        # the first PDM block.  This is a structural coordinate transform;
        # it is disabled for all legacy routes.
        use_history_state_pdm=int(config.get("use_history_state_pdm", 0)),
        history_state_pdm_window=int(config.get("history_state_pdm_window", 24)),
        down_sampling_layers=down_layers,
        down_sampling_window=window,
        down_sampling_method=str(config.get("down_sampling_method", "avg")),
        use_future_temporal_feature=0,
        embed="fixed",
        freq="h",
    )
    if values["moving_avg"] < 1 or values["moving_avg"] % 2 != 1:
        raise ValueError("moving_avg must be a positive odd integer")
    # ``factor`` is not used by TimeMixer but is retained in metadata by the
    # runner for compatibility with the experiment matrix.
    return SimpleNamespace(**values)


def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Future marks are deliberately disabled for the UrbanEV baseline.  With
    # downsampling enabled the upstream implementation expects a None decoder.
    return model(x, None, None, None)


def _build_tariff_aux(config: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Pack known-future electricity/service tariffs per zone for CI batches.

    The packed layout is ``[electricity_0, service_0, ..., electricity_N,
    service_N]`` at each forecast hour.  Statistics are fitted only on the
    fold's training boundary; future demand/occupancy/duration is never read.
    """
    import pandas as pd
    from scprc.data import DATA

    e = pd.read_csv(DATA / "e_price.csv")
    s = pd.read_csv(DATA / "s_price.csv")
    columns = [str(v) for v in data["metadata"]["TAZ_order"]]
    if list(e.columns[1:].astype(str)) != columns or list(s.columns[1:].astype(str)) != columns:
        raise ValueError("tariff columns do not match the audited TAZ order")
    e_values = e.iloc[:, 1:].to_numpy(dtype=np.float32)
    s_values = s.iloc[:, 1:].to_numpy(dtype=np.float32)
    train_bound = int(data["metadata"]["splits"]["train"]["bounds"][1])
    e_mean = e_values[:train_bound].mean(axis=0)
    s_mean = s_values[:train_bound].mean(axis=0)
    e_std = np.maximum(e_values[:train_bound].std(axis=0), 1e-6)
    s_std = np.maximum(s_values[:train_bound].std(axis=0), 1e-6)
    seq_len = int(config["seq_len"])
    pred_len = int(config.get("pred_len", data["train_y"].shape[1]))
    lead = int(config.get("lead_time", 1))
    zones = len(columns)
    packed_history_route = str(config.get("model", "")).lower() in {
        "tariff_coordinate_timemixer", "tariff_context_timemixer",
        "tariff_bilinear_timemixer", "tariff_bdc_timemixer",
    }
    output: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        key = f"{split}_x"
        if key not in data:
            continue
        starts = np.asarray(data["window_starts"][split], dtype=np.int64)
        packed_len = seq_len + pred_len if packed_history_route else pred_len
        packed = np.empty((len(starts), packed_len, zones * 2), dtype=np.float32)
        for row, start in enumerate(starts):
            if packed_history_route:
                history_start = int(start)
                history_end = history_start + seq_len
                if history_start < 0 or history_end > len(e_values):
                    raise ValueError("tariff history falls outside the published tariff table")
                e_history = (e_values[history_start:history_end] - e_mean[None, :]) / e_std[None, :]
                s_history = (s_values[history_start:history_end] - s_mean[None, :]) / s_std[None, :]
                packed[row, :seq_len, 0::2] = e_history
                packed[row, :seq_len, 1::2] = s_history
            label_start = int(start) + seq_len + lead - 1
            label_end = label_start + pred_len
            if label_start < 0 or label_end > len(e_values):
                raise ValueError("tariff horizon falls outside the published tariff table")
            e_future = (e_values[label_start:label_end] - e_mean[None, :]) / e_std[None, :]
            s_future = (s_values[label_start:label_end] - s_mean[None, :]) / s_std[None, :]
            target_slice = packed[row, seq_len:] if packed_history_route else packed[row]
            target_slice[:, 0::2] = e_future
            target_slice[:, 1::2] = s_future
        output[split] = packed
    output["e_mean"] = e_mean.astype(np.float32)
    output["e_std"] = e_std.astype(np.float32)
    output["s_mean"] = s_mean.astype(np.float32)
    output["s_std"] = s_std.astype(np.float32)
    return output


def _build_history_multimodal_aux(config: dict[str, Any], data: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Build strict-history context summaries for the multimodal CPR route.

    Only rows ``start:start+seq_len`` are materialized for each sample.  All
    normalization statistics are fitted on the training prefix, and future
    occupancy/duration/tariff/weather rows are never read for a sample.  To
    keep the tensor compact, station-wise companion variables are summarized
    by mean and standard deviation over stations at each history timestamp;
    static station descriptors provide the node-specific spatial context.
    """
    import pandas as pd
    from scprc.data import DATA

    columns = [str(v) for v in data["metadata"]["TAZ_order"]]
    seq_len = int(config["seq_len"])
    tr_end = int(data["metadata"]["splits"]["train"]["bounds"][1])
    # Each tuple is (feature name, [time, nodes]) and is derived solely from
    # published historical tables.  Prices are historical inputs here; no
    # future tariff values are used by this route.
    station_sources = []
    requested = config.get("history_features")
    if requested is None:
        requested = ["occupancy", "duration", "electricity_price", "service_price"]
    requested = {str(v).strip().lower() for v in requested}
    # Optional causal residualization removes the portion of each companion
    # variable that is linearly explained by the observed volume coordinate.
    # Coefficients and residual scales are fitted on the train prefix only;
    # applying the fixed projection to a validation/test history uses no
    # future demand.  This is a coordinate change before the latent encoder,
    # not an input-dependent gate or a tuned post-hoc correction.
    residualize_volume = bool(config.get("history_residualize_volume", False))
    volume_timeline = None
    if residualize_volume:
        volume_frame = pd.read_csv(DATA / "volume.csv")
        volume_columns = [str(v) for v in volume_frame.columns[1:]]
        if volume_columns != columns:
            raise ValueError("volume timeline columns do not match audited TAZ order")
        volume_timeline_raw = volume_frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        volume_train_raw = np.asarray(data["raw_splits"]["train"], dtype=np.float64)
        if volume_train_raw.ndim != 2 or volume_train_raw.shape[1] != len(columns):
            raise ValueError("raw volume split shape is incompatible with history residualization")
        volume_mean = volume_train_raw.mean(axis=0)
        volume_std = np.maximum(volume_train_raw.std(axis=0), 1e-6)
        volume_train = (volume_train_raw - volume_mean[None, :]) / volume_std[None, :]
        volume_timeline = (volume_timeline_raw - volume_mean[None, :]) / volume_std[None, :]
    for filename, label in (("occupancy.csv", "occupancy"),
                            ("duration.csv", "duration"),
                            ("e_price.csv", "electricity_price"),
                            ("s_price.csv", "service_price")):
        if label not in requested:
            continue
        frame = pd.read_csv(DATA / filename)
        got = [str(v) for v in frame.columns[1:]]
        if got != columns:
            raise ValueError(f"{filename} columns do not match audited TAZ order")
        station_sources.append((label, frame.iloc[:, 1:].to_numpy(dtype=np.float32)))
    weather_sources = []
    weather_requested = bool(requested.intersection({"weather", "weather_central", "weather_airport"}))
    for filename, label in (("weather_central.csv", "weather_central"),
                            ("weather_airport.csv", "weather_airport")):
        if not weather_requested or (label not in requested and "weather" not in requested):
            continue
        frame = pd.read_csv(DATA / filename)
        timeline_length = len(station_sources[0][1]) if station_sources else len(pd.read_csv(DATA / "volume.csv"))
        if len(frame) != timeline_length:
            raise ValueError(f"{filename} length differs from UrbanEV timeline")
        weather_sources.append((label, frame.iloc[:, 1:].to_numpy(dtype=np.float32)))

    features = []
    node_features = []
    feature_names = []
    for label, values in station_sources:
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains nonfinite values")
        train = values[:tr_end]
        # Preserve TAZ identity: each historical companion channel is
        # standardized per zone using the train prefix, then encoded as a
        # node-aligned feature rather than collapsed to a city mean/std.
        mean = train.mean(axis=0)
        std = np.maximum(train.std(axis=0), 1e-6)
        scaled = (values - mean[None, :]) / std[None, :]
        if residualize_volume:
            # All arrays are standardized with train-prefix statistics before
            # the residual projection, so the fitted slope is dimensionless.
            x = volume_train
            denom = np.sum(x * x, axis=0) + 1e-6
            beta = np.sum(x * np.asarray(scaled[:len(x)], dtype=np.float64), axis=0) / denom
            residual = np.asarray(scaled, dtype=np.float64) - volume_timeline * beta[None, :]
            rtrain = residual[:len(x)]
            rmean = rtrain.mean(axis=0)
            rstd = np.maximum(rtrain.std(axis=0), 1e-3)
            scaled = ((residual - rmean[None, :]) / rstd[None, :]).astype(np.float32)
        lag_bank = config.get("history_lags", [0])
        lag_bank = [int(v) for v in lag_bank] if bool(config.get("use_history_lag_bank", False)) else [0]
        for lag in lag_bank:
            if lag <= 0:
                lagged = scaled
            else:
                lagged = np.concatenate((np.repeat(scaled[:1], lag, axis=0), scaled[:-lag]), axis=0)
            node_features.append(lagged[..., None])
            feature_names.append(f"{label}_lag{lag}")
        if bool(config.get("history_phase_features", False)):
            # Causal same-clock template from preceding daily cycles.  The
            # first rows are padded with the earliest observed value; every
            # sample window therefore uses only rows at or before its
            # prediction origin.  The template and its innovation are kept
            # as separate coordinates so the encoder can distinguish stable
            # phase level from a current-cycle deviation.
            phase_lags = [int(v) for v in config.get(
                "history_phase_lags", [24, 48, 72, 96, 120, 144, 168])]
            phase_lags = [v for v in phase_lags if v > 0]
            if phase_lags:
                phase_views = []
                for lag in phase_lags:
                    phase_views.append(np.concatenate(
                        (np.repeat(scaled[:1], lag, axis=0), scaled[:-lag]), axis=0))
                phase_stack = np.stack(phase_views, axis=-1)
                phase_mean = phase_stack.mean(axis=-1)
                phase_delta = scaled - phase_mean
                node_features.extend((phase_mean[..., None], phase_delta[..., None]))
                feature_names.extend((f"{label}_phase_mean", f"{label}_phase_delta"))
    for label, values in weather_sources:
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains nonfinite values")
        train = values[:tr_end]
        mean = train.mean(axis=0)
        std = np.maximum(train.std(axis=0), 1e-6)
        scaled = (values - mean[None, :]) / std[None, :]
        features.extend(scaled[:, j] for j in range(scaled.shape[1]))
        feature_names.extend(f"{label}_{j}" for j in range(scaled.shape[1]))
    global_timeline = np.stack(features, axis=1).astype(np.float32) if features else None
    if global_timeline is not None and not np.isfinite(global_timeline).all():
        raise ValueError("multimodal history context contains nonfinite values")
    if node_features:
        timeline = np.concatenate(node_features, axis=2).astype(np.float32)
        if global_timeline is not None:
            timeline = np.concatenate((timeline, np.repeat(global_timeline[:, None, :], len(columns), axis=1)), axis=2)
        feature_dim = int(timeline.shape[2])
    elif global_timeline is not None:
        timeline = global_timeline
        feature_dim = int(timeline.shape[1])
    else:
        raise ValueError("history_features selected no usable UrbanEV variables")
    if not np.isfinite(timeline).all():
        raise ValueError("multimodal history context contains nonfinite values")

    # Use audited static zone descriptors plus compact graph statistics as the
    # spatial part of the node representation.  These are independent of the
    # forecast horizon and are therefore safe at deployment time.
    static = np.asarray(data.get("static_features"), dtype=np.float32)
    adjacency = np.asarray(data.get("adj"), dtype=np.float32)
    distance = np.asarray(data.get("distance"), dtype=np.float32)
    graph_static = np.column_stack((
        adjacency.mean(axis=1), adjacency.std(axis=1),
        distance.mean(axis=1), distance.std(axis=1),
    )).astype(np.float32)
    graph_static = (graph_static - graph_static.mean(axis=0)) / np.maximum(graph_static.std(axis=0), 1e-6)
    static_context = np.concatenate((static, graph_static), axis=1).astype(np.float32)
    # Optional fixed graph spectral coordinates.  These are computed once from
    # the audited UrbanEV adjacency and are independent of every sample's
    # history/target.  The symmetric normalized Laplacian gives an orthogonal
    # coordinate system on the station graph; the constant eigenvector is
    # skipped, leaving the first non-trivial spatial modes.  Signs are fixed
    # deterministically so repeated runs/checkpoints have identical metadata.
    spectral_k = int(config.get("hybrid_spectral_k", 0))
    if bool(config.get("hybrid_spectral_static", False)) and spectral_k > 0:
        a = np.maximum(adjacency, 0.0).astype(np.float64)
        a = 0.5 * (a + a.T)
        np.fill_diagonal(a, 0.0)
        degree = a.sum(axis=1)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
        normalized = inv_sqrt[:, None] * a * inv_sqrt[None, :]
        laplacian = np.eye(a.shape[0], dtype=np.float64) - normalized
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        # Skip the graph-constant mode.  For disconnected/near-disconnected
        # graphs this still provides a deterministic non-constant basis.
        order = np.argsort(eigenvalues, kind="mergesort")
        candidates = order[1:1 + min(spectral_k, max(0, a.shape[0] - 1))]
        spectral = eigenvectors[:, candidates]
        for col in range(spectral.shape[1]):
            pivot = int(np.argmax(np.abs(spectral[:, col])))
            if spectral[pivot, col] < 0:
                spectral[:, col] *= -1.0
        spectral = spectral / np.maximum(spectral.std(axis=0, keepdims=True), 1e-6)
        static_context = np.concatenate((static_context, spectral.astype(np.float32)), axis=1)

    aux: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        if split not in data["window_starts"]:
            continue
        starts = np.asarray(data["window_starts"][split], dtype=np.int64)
        values = np.stack([timeline[int(s):int(s) + seq_len] for s in starts]).astype(np.float32)
        expected_tail = (seq_len, feature_dim) if timeline.ndim == 2 else (seq_len, len(columns), feature_dim)
        if values.ndim != len(expected_tail) + 1 or values.shape[1:] != expected_tail:
            raise ValueError(f"invalid {split} multimodal history shape {values.shape}")
        aux[split] = values
    metadata = {
        "feature_names": feature_names,
        "feature_dim": feature_dim,
        "node_aligned": bool(timeline.ndim == 3),
        "requested_features": sorted(requested),
        "history_only": True,
        "normalization_fit": "train_prefix_only",
        "station_reduction": ("per-zone train-prefix standardization" if bool(timeline.ndim == 3) else "per-timestamp mean/std"),
        "static_spatial_features": int(static_context.shape[1]),
        "spectral_static": bool(config.get("hybrid_spectral_static", False)) and spectral_k > 0,
        "spectral_k": int(min(spectral_k, max(0, adjacency.shape[0] - 1))) if bool(config.get("hybrid_spectral_static", False)) else 0,
        "history_residualize_volume": residualize_volume,
    }
    return aux, static_context, metadata


class _ComponentScaleRouter(nn.Module):
    """Sparse node-conditional routing before the official TimeMixer.

    The route is residual and nearly closed at initialization. It tests the
    UrbanEV-specific hypothesis that spatial propagation should depend on the
    seasonal/trend component, temporal scale and receiving region.
    """
    def __init__(self, adjacency: np.ndarray, nodes: int, scales: int = 4,
                 topk: int = 8, enabled: bool = True):
        super().__init__()
        a = np.maximum(np.asarray(adjacency, dtype=np.float32).copy(), 0.0)
        np.fill_diagonal(a, 0.0)
        k = min(int(topk), max(1, nodes - 1))
        idx = np.argsort(-a, axis=1)[:, :k]
        weights = np.take_along_axis(a, idx, axis=1)
        empty = weights.sum(axis=1) <= 1e-8
        if np.any(empty):
            idx[empty, 0] = np.flatnonzero(empty)
            weights[empty, 0] = 1.0
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        self.register_buffer("neighbor_index", torch.as_tensor(idx, dtype=torch.long))
        self.register_buffer("neighbor_weight", torch.as_tensor(weights, dtype=torch.float32))
        self.scales = int(scales)
        self.enabled = bool(enabled)
        hidden = max(8, 2 * self.scales)
        self.context = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, 2 * self.scales))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.constant_(self.context[-1].bias, -4.0)
        self.scale_gain = nn.Parameter(torch.zeros(2, self.scales))

    def _graph_mix(self, z: torch.Tensor) -> torch.Tensor:
        gathered = z[..., self.neighbor_index]
        return (gathered * self.neighbor_weight.view(1, 1, *self.neighbor_weight.shape)).sum(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        b, t, _ = x.shape
        trend = F.avg_pool1d(x.permute(0, 2, 1), 25, 1, 12, count_include_pad=False).permute(0, 2, 1)
        seasonal = x - trend
        context = torch.stack((x.mean(1), x[:, -1]), dim=-1)
        gates = torch.sigmoid(self.context(context).view(b, x.shape[2], 2, self.scales) +
                              self.scale_gain.view(1, 1, 2, self.scales))
        routed_components = []
        for component, signal in enumerate((seasonal, trend)):
            routed = signal
            for m in range(self.scales):
                factor = 2 ** m
                if factor > 1:
                    z = F.avg_pool1d(signal.permute(0, 2, 1), factor, factor).permute(0, 2, 1)
                else:
                    z = signal
                mixed = self._graph_mix(z)
                g = gates[:, :, component, m].unsqueeze(1)
                routed_m = z + g * (mixed - z)
                if factor > 1:
                    routed_m = F.interpolate(routed_m.permute(0, 2, 1), size=t,
                                              mode="linear", align_corners=False).permute(0, 2, 1)
                routed = routed + (routed_m - signal) / self.scales
            routed_components.append(routed)
        return routed_components[0] + routed_components[1]


class _RoutedTimeMixer(nn.Module):
    def __init__(self, args: SimpleNamespace, adjacency: np.ndarray, enabled: bool = True):
        super().__init__()
        self.router = _ComponentScaleRouter(adjacency, int(args.enc_in),
                                            int(getattr(args, "router_scales", 4)),
                                            int(getattr(args, "router_topk", 8)), enabled)
        self.backbone = _UpstreamModel(args)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.backbone(self.router(x_enc), x_mark_enc, x_dec, x_mark_dec, mask)


class _FANTimeMixer(nn.Module):
    """Frequency Adaptive Normalization wrapper for the CPR backbone.

    FAN (Ye et al., NeurIPS 2024) removes the dominant, instance-wise
    Fourier components before the forecasting backbone and predicts those
    components for the output horizon with a small shared MLP.  The residual
    forecast and the frequency forecast are added in the same standardized
    coordinate.  All Fourier statistics are computed from ``x_enc`` only;
    no target rows, future normalization statistics, gate, router, or expert
    branch are used.
    """

    def __init__(self, args: SimpleNamespace, freq_topk: int = 4):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.freq_topk = max(1, min(int(freq_topk), self.seq_len // 2))
        self.backbone = _UpstreamModel(args)
        hidden = max(32, min(128, self.seq_len // 2))
        self.freq_encoder = nn.Sequential(
            nn.Linear(self.seq_len, hidden), nn.ReLU()
        )
        self.freq_predictor = nn.Sequential(
            nn.Linear(hidden + self.seq_len, max(64, hidden * 2)),
            nn.ReLU(),
            nn.Linear(max(64, hidden * 2), self.pred_len),
        )

    def _frequency_residual(self, x: torch.Tensor):
        if x.ndim != 3 or x.size(1) != self.seq_len:
            raise ValueError(
                f"FAN expects [B,{self.seq_len},N] history, got {tuple(x.shape)}"
            )
        # UrbanEV uses standardized float32 tensors; forcing FFT to float32
        # also avoids half-precision cuFFT restrictions for length 168.
        xf = torch.fft.rfft(x.float(), dim=1)
        magnitude = xf.abs()
        magnitude[:, 0, :] = 0.0  # do not treat the DC level as seasonality
        k = min(self.freq_topk, max(1, magnitude.size(1) - 1))
        indices = torch.topk(magnitude, k, dim=1).indices
        mask = torch.zeros_like(magnitude)
        mask.scatter_(1, indices, 1.0)
        dominant = torch.fft.irfft(xf * mask, n=self.seq_len, dim=1)
        return (x.float() - dominant).to(dtype=x.dtype), dominant.to(dtype=x.dtype)

    @staticmethod
    def _dominant_component(x: torch.Tensor, topk: int) -> torch.Tensor:
        """Return top-k non-DC Fourier component for an arbitrary horizon."""
        xf = torch.fft.rfft(x.float(), dim=1)
        magnitude = xf.abs()
        if magnitude.size(1) > 1:
            magnitude[:, 0, :] = 0.0
        k = min(max(1, int(topk)), max(1, magnitude.size(1) - 1))
        indices = torch.topk(magnitude, k, dim=1).indices
        mask = torch.zeros_like(magnitude)
        mask.scatter_(1, indices, 1.0)
        return torch.fft.irfft(xf * mask, n=x.size(1), dim=1).to(dtype=x.dtype)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None):
        residual, dominant = self._frequency_residual(x_enc)
        residual_forecast = self.backbone(
            residual, x_mark_enc, x_dec, x_mark_dec, mask
        )
        # Shared across stations, exactly as FAN's channel-independent MLP.
        bsz, _, nodes = dominant.shape
        flat_dom = dominant.permute(0, 2, 1).reshape(bsz * nodes, self.seq_len)
        flat_raw = x_enc.permute(0, 2, 1).reshape(bsz * nodes, self.seq_len)
        pred_dom = self.freq_predictor(
            torch.cat((self.freq_encoder(flat_dom), flat_raw), dim=-1)
        )
        pred_dom = pred_dom.reshape(bsz, nodes, self.pred_len).permute(0, 2, 1)
        self.last_pred_dom = pred_dom
        return residual_forecast + pred_dom.to(dtype=residual_forecast.dtype)


class _FANIdentityTimeMixer(nn.Module):
    """Identity-preserving FAN ablation.

    The CPR backbone receives the original observed volume, while the
    instance-wise dominant-frequency coordinate is learned as a parallel
    future representation.  The final frequency projection starts at zero,
    so the model is exactly the original CPR predictor before it learns a
    useful frequency component.  No gate or expert selects between paths.
    """

    def __init__(self, args: SimpleNamespace, freq_topk: int = 4):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.freq_topk = max(1, min(int(freq_topk), self.seq_len // 2))
        self.backbone = _UpstreamModel(args)
        hidden = max(32, min(128, self.seq_len // 2))
        self.freq_encoder = nn.Sequential(nn.Linear(self.seq_len, hidden), nn.GELU())
        self.freq_predictor = nn.Sequential(
            nn.Linear(hidden + self.seq_len, max(64, hidden * 2)), nn.GELU(),
            nn.Linear(max(64, hidden * 2), self.pred_len),
        )
        nn.init.zeros_(self.freq_predictor[-1].weight)
        nn.init.zeros_(self.freq_predictor[-1].bias)

    def _dominant(self, x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft(x.float(), dim=1)
        magnitude = xf.abs()
        magnitude[:, 0, :] = 0.0
        k = min(self.freq_topk, max(1, magnitude.size(1) - 1))
        idx = torch.topk(magnitude, k, dim=1).indices
        mask = torch.zeros_like(magnitude)
        mask.scatter_(1, idx, 1.0)
        return torch.fft.irfft(xf * mask, n=self.seq_len, dim=1).to(dtype=x.dtype)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None):
        base = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        dominant = self._dominant(x_enc)
        bsz, _, nodes = dominant.shape
        dom = dominant.permute(0, 2, 1).reshape(bsz * nodes, self.seq_len)
        raw = x_enc.permute(0, 2, 1).reshape(bsz * nodes, self.seq_len)
        pred = self.freq_predictor(torch.cat((self.freq_encoder(dom), raw), dim=-1))
        pred = pred.reshape(bsz, nodes, self.pred_len).permute(0, 2, 1)
        return base + pred.to(dtype=base.dtype)


class _HistoricalMultimodalTimeMixer(nn.Module):
    """CPR-TimeMixer with a causal, compact history context encoder.

    UrbanEV contains several companion measurements (occupancy, duration,
    tariffs and weather) that are available at the forecast origin.  Passing
    every station-feature channel through the CI backbone would multiply the
    node count and break the spatial operators.  This module therefore first
    encodes *global history summaries* of those variables and broadcasts the
    resulting context to each node using static station descriptors.  The
    backbone still predicts one channel per station.  There is no selector or
    gate: the context is an additive representation before PDM, initialized
    at zero so the volume-only CPR model is an exact starting point.
    """
    def __init__(self, args: SimpleNamespace, feature_dim: int,
                 static_features: np.ndarray | None = None,
                 hidden: int = 32):
        super().__init__()
        self.nodes = int(args.enc_in)
        self.feature_dim = int(feature_dim)
        static = np.zeros((self.nodes, 1), dtype=np.float32) if static_features is None else np.asarray(static_features, dtype=np.float32)
        if static.ndim != 2 or static.shape[0] != self.nodes:
            raise ValueError(f"static_features must have shape [{self.nodes}, S], got {static.shape}")
        self.register_buffer("static_features", torch.as_tensor(static, dtype=torch.float32))
        hidden = max(8, int(hidden))
        self.history_encoder = nn.Sequential(nn.Linear(self.feature_dim, hidden), nn.GELU())
        self.static_encoder = nn.Sequential(nn.Linear(static.shape[1], hidden), nn.GELU())
        self.context_norm = nn.LayerNorm(hidden)
        self.context_readout = nn.Linear(hidden, 1)
        # Exact CPR-DX initialization; training can learn a nonzero context
        # from the causal history without a hand-tuned gate parameter.
        nn.init.zeros_(self.context_readout.weight)
        nn.init.zeros_(self.context_readout.bias)
        self.backbone = _UpstreamModel(args)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, history_aux=None):
        if history_aux is None:
            raise ValueError("multimodal CPR requires history_aux")
        if history_aux.ndim not in (3, 4) or history_aux.shape[:2] != x_enc.shape[:2]:
            raise ValueError(
                f"history_aux must have shape [B,T,F] or [B,T,N,F], got {tuple(history_aux.shape)}"
            )
        if history_aux.shape[-1] != self.feature_dim:
            raise ValueError("history_aux feature dimension differs from model")
        temporal = self.history_encoder(history_aux)
        station = self.static_encoder(self.static_features)
        # Keep the auxiliary representation bounded in the standardized input
        # coordinates.  This is a structural stability constraint, not a
        # learned gate or post-hoc residual: the context can only alter the
        # causal volume representation through a smooth, normalized additive
        # coordinate.
        if temporal.ndim == 3:
            temporal = temporal.unsqueeze(2)
        context_latent = self.context_norm(
            temporal + station.view(1, 1, self.nodes, -1)
        )
        context = 0.25 * torch.tanh(self.context_readout(context_latent))
        return self.backbone(x_enc + context.squeeze(-1), x_mark_enc, x_dec, x_mark_dec, mask)


class _NodeMultimodalTokenTimeMixer(nn.Module):
    """Direct node tokenization of volume and causal historical covariates.

    The previous multimodal routes encoded occupancy/duration/price/weather
    after the volume backbone and injected a residual forecast.  This wrapper
    instead builds one token per station and hour, with the observed volume,
    station-aligned history variables and static spatial descriptors entering
    a shared token projection *before* the official TimeMixer PDM/FMM.  A
    the resulting scalar station coordinate is passed through the normal
    ``[B,T,N]`` interface, preserving the CPR cross-site representation.  No
    expert, router, gate, or output residual is introduced.

    The scalar coordinate is initialized to the raw volume coordinate.  This
    makes the new representation an exact volume-only start while allowing
    the learned projection to use the historical covariates and static
    descriptors before PDM/FMM.
    """

    def __init__(self, args: SimpleNamespace, aux_dim: int,
                 static_features: np.ndarray, token_dim: int = 16,
                 freeze_backbone: bool = False):
        super().__init__()
        self.nodes = int(args.enc_in)
        self.aux_dim = int(aux_dim)
        static = np.asarray(static_features, dtype=np.float32)
        if static.ndim != 2 or static.shape[0] != self.nodes:
            raise ValueError(f"static_features must have shape [{self.nodes}, S], got {static.shape}")
        self.static_dim = int(static.shape[1])
        # Bilinear coordinates retain the physical relation between demand
        # level and each covariate while staying inside the input token before
        # normalization.  They are fixed feature construction, not a gate.
        self.raw_dim = 1 + 2 * self.aux_dim + 2 * self.static_dim
        self.register_buffer("static_features", torch.as_tensor(static, dtype=torch.float32))
        # A scalar per-node coordinate keeps the official CPR cross-site
        # operator valid.  The causal temporal encoder lets covariates affect
        # the representation through their history rather than as a noisy
        # instantaneous residual.
        hidden = max(4, int(token_dim))
        self.token_dim = 1
        self.node_token_weight = nn.Parameter(torch.zeros(self.nodes, hidden, self.raw_dim))
        self.node_token_bias = nn.Parameter(torch.zeros(self.nodes, hidden))
        self.causal_token_conv = nn.Conv1d(hidden, hidden, kernel_size=5, padding=0)
        self.token_readout = nn.Linear(hidden, 1)
        with torch.no_grad():
            # The first raw coordinate is standardized volume.
            self.node_token_weight[:, 0, 0] = 1.0
            self.causal_token_conv.weight.zero_()
            self.causal_token_conv.bias.zero_()
            self.causal_token_conv.weight[0, 0, -1] = 1.0
            self.token_readout.weight.zero_()
            self.token_readout.bias.zero_()
            self.token_readout.weight[0, 0] = 1.0
        self.backbone = _UpstreamModel(args)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def _pack_tokens(self, x_enc: torch.Tensor, history_aux: torch.Tensor) -> torch.Tensor:
        if x_enc.ndim != 3:
            raise ValueError("node multimodal volume must have shape [B,T,N]")
        b, t, n = x_enc.shape
        if n != self.nodes:
            raise ValueError("node multimodal volume channel count differs from model")
        if history_aux.ndim == 3:
            history_aux = history_aux.unsqueeze(2).expand(-1, -1, n, -1)
        if history_aux.ndim != 4 or history_aux.shape[:3] != (b, t, n):
            raise ValueError("node multimodal history must have shape [B,T,N,F]")
        if history_aux.shape[-1] != self.aux_dim:
            raise ValueError("node multimodal history feature dimension differs from model")
        static = self.static_features.to(device=x_enc.device, dtype=x_enc.dtype)
        static = static.view(1, 1, n, self.static_dim).expand(b, t, n, -1)
        volume = x_enc.unsqueeze(-1)
        raw = torch.cat((volume, history_aux, volume * history_aux,
                         static, volume * static), dim=-1)
        # Apply the shared station token map independently at every hour and
        # station, then a left-padded causal temporal mixer before restoring
        # the official [B,T,N] CPR layout.
        packed = raw.permute(0, 2, 1, 3).reshape(b * n, t, self.raw_dim)
        # Node-specific linear coordinates model the fact that duration and
        # occupancy have different volume elasticity at different stations.
        # Keep the observed volume coordinate exact and bound only the learned
        # auxiliary deformation.  The bound is a dimension-derived coordinate
        # normalization, not a sample-dependent gate or blend coefficient.
        aux = raw[..., 1:]
        aux_weight = self.node_token_weight[:, 0, 1:]
        aux_bias = self.node_token_bias[:, 0]
        aux_scale = float(max(self.raw_dim - 1, 1) ** -0.5)
        delta = aux_scale * torch.tanh(
            torch.einsum("btna,na->btn", aux, aux_weight) + aux_bias.view(1, 1, n)
        )
        scalar = raw[..., 0] + delta
        latent = torch.zeros(b, t, n, self.causal_token_conv.in_channels,
                             device=raw.device, dtype=raw.dtype)
        latent[..., 0] = scalar
        latent = latent.permute(0, 2, 1, 3).reshape(b * n, t, -1).transpose(1, 2)
        latent = self.causal_token_conv(F.pad(latent, (4, 0))).transpose(1, 2)
        return self.token_readout(latent).reshape(b, n, t).permute(0, 2, 1).contiguous()

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, history_aux=None):
        if history_aux is None:
            raise ValueError("node multimodal TimeMixer requires history_aux")
        b, _, n = x_enc.shape
        tokens = self._pack_tokens(x_enc, history_aux)
        pred = self.backbone(tokens, None, None, None, mask)
        if pred.ndim != 3 or pred.shape[0] != b or pred.shape[-1] != n:
            raise RuntimeError("unexpected packed TimeMixer prediction shape")
        return pred


class _CASTTimeMixer(nn.Module):
    """Causal affine-state canonicalized CPR-TimeMixer.

    The observed volume history is mapped to a local, causal shape coordinate
    before entering one CPR-TimeMixer backbone.  The forecast is reconstructed
    with the same endpoint level and scale, so the state is a coordinate
    transform rather than a second residual predictor.  A four-coordinate
    affine state is transported as a zero-initialized latent before PDM.  No
    selector, gate, router, or expert branch is used.
    """

    def __init__(self, args: SimpleNamespace, checkpoint_path: str | None = None,
                 state_transport: bool = True, preserve_input: bool = False,
                 long_scale: bool = False):
        super().__init__()
        # In the canonicalized variant CAST owns normalization and
        # reconstruction.  The identity-preserving variant keeps the
        # checkpoint's original RevIN/input coordinate and transports only
        # the causal state.  This is a structural ablation: at initialization
        # it is exactly the checkpoint forecast, so state transport cannot
        # silently change the learned input distribution.
        self.preserve_input = bool(preserve_input)
        self.long_scale = bool(long_scale)
        if not self.preserve_input:
            args.use_norm = 0
        self.backbone = _UpstreamModel(args)
        self.pred_len = int(args.pred_len)
        self.state_transport = bool(state_transport)
        self.d_model = int(args.d_model)
        self.register_buffer("eps", torch.tensor(1e-4, dtype=torch.float32))
        if self.state_transport:
            self.state_projection = nn.Linear(4, self.d_model)
            nn.init.zeros_(self.state_projection.weight)
            nn.init.zeros_(self.state_projection.bias)
        if checkpoint_path:
            payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
            state = payload.get("model", payload)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            unexpected = [v for v in unexpected if not v.startswith("normalize_layers")]
            missing = [v for v in missing if not v.startswith("normalize_layers")]
            if missing or unexpected:
                raise ValueError(f"CAST CPR checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    def _state(self, x: torch.Tensor):
        tail = x[:, -24:]
        previous = x[:, -48:-24]
        long = x[:, -168:]
        m24 = tail.mean(dim=1, keepdim=True)
        mprev = previous.mean(dim=1, keepdim=True)
        m168 = long.mean(dim=1, keepdim=True)
        # A fixed train-prefix-scale floor prevents low-activity stations
        # from exploding after local canonicalization.  The UrbanEV arrays
        # are already train-prefix standardized, so 0.25 is a dimensionless
        # quarter-standard-deviation floor, not a validation-tuned value.
        s24 = torch.sqrt(tail.var(dim=1, keepdim=True, unbiased=False) + 1e-4).clamp_min(0.25)
        s168 = torch.sqrt(long.var(dim=1, keepdim=True, unbiased=False) + 1e-4).clamp_min(0.25)
        shape = (x - m24) / s24
        state = torch.cat(((m24 - m168) / s168,
                           torch.log(s24 / s168),
                           (m24 - mprev) / s168,
                           m24 / s168), dim=1)
        if self.long_scale:
            # A 7-day causal scale is less sensitive to a single quiet or
            # spiky day.  The 24-hour level remains the endpoint coordinate;
            # the scale is transported as a separate state coordinate.
            shape = (x - m24) / s168
            return shape, state, m24, s168
        return shape, state, m24, s24

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        shape, state, level, scale = self._state(x_enc)
        if self.state_transport:
            # [B,T,N] -> [B*N,T_s,D] for each official downsampling scale.
            b, _, n = x_enc.shape
            base = self.state_projection(state.transpose(1, 2))  # [B,N,D]
            base = base.reshape(b * n, 1, self.d_model)
            scales = []
            length = int(x_enc.size(1))
            for _ in range(int(self.backbone.configs.down_sampling_layers) + 1):
                scales.append(base.expand(-1, length, -1))
                length //= int(self.backbone.configs.down_sampling_window)
            self.backbone._external_history_latents = scales
        model_input = x_enc if self.preserve_input else shape
        forecast = self.backbone(model_input, x_mark_enc, x_dec, x_mark_dec, mask)
        return forecast if self.preserve_input else forecast * scale + level


class _DailyResidualTimeMixer(nn.Module):
    """One CPR-TimeMixer that predicts corrections to the previous day.

    UrbanEV's deterministic 24-hour persistence is a strong physical
    coordinate.  The backbone therefore sees the causal daily innovation
    ``x_t-x_{t-24}`` and predicts only the future innovation; the forecast is
    reconstructed as ``x_{t-24}+innovation``.  This is a fixed change of
    target/input coordinates, with no gate, router, expert, or forecast
    mixture, and it is identity anchored at zero residual.
    """

    def __init__(self, args: SimpleNamespace, raw_input: bool = False):
        super().__init__()
        self.backbone = _UpstreamModel(args)
        self.pred_len = int(args.pred_len)
        self.period = 24
        self.raw_input = bool(raw_input)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None):
        if x_enc.size(1) < self.period + self.pred_len:
            raise ValueError("daily residual requires at least one daily lag")
        # Causal innovation history.  The first day has no lag and is kept at
        # zero so no future or validation value enters the representation.
        if self.raw_input:
            backbone_input = x_enc
        else:
            innovation = torch.zeros_like(x_enc)
            innovation[:, self.period:] = x_enc[:, self.period:] - x_enc[:, :-self.period]
            backbone_input = innovation
        residual = self.backbone(backbone_input, x_mark_enc, x_dec, x_mark_dec, mask)
        anchor = x_enc[:, -self.period:]
        return anchor[:, :self.pred_len] + residual


class _CedarInvariantTimeMixer(nn.Module):
    """Single CPR backbone with shared-latent domain discrepancy regularization.

    A fixed spatial partition of stations is used only to compare pooled
    pre-PDM representations.  The forecast path remains one CPR-TimeMixer;
    there are no experts, routers, gates, or post-hoc residuals.
    """

    def __init__(self, args: SimpleNamespace, static_features: np.ndarray,
                 partition: np.ndarray | None = None):
        super().__init__()
        args.capture_shared_latent = 1
        static = np.asarray(static_features, dtype=np.float32)
        if static.ndim != 2 or static.shape[0] != int(args.enc_in):
            raise ValueError("Cedar static features must be [nodes, features]")
        self.nodes = int(args.enc_in)
        self.backbone = _UpstreamModel(args)
        if partition is None:
            # Longitude gives two fixed geographic environments using only
            # pre-prediction station descriptors.
            key = static[:, 2] if static.shape[1] > 2 else static[:, 0]
            order = np.argsort(key, kind="mergesort")
            part = np.zeros(self.nodes, dtype=np.int64)
            part[order[self.nodes // 2:]] = 1
        else:
            part = np.asarray(partition, dtype=np.int64)
        if part.shape != (self.nodes,) or len(np.unique(part)) < 2:
            raise ValueError("Cedar partition must contain two nonempty groups")
        self.register_buffer("partition", torch.as_tensor(part, dtype=torch.long))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

    def invariant_penalty(self) -> torch.Tensor:
        latent = getattr(self.backbone, "last_shared_latent", None)
        if latent is None:
            raise RuntimeError("Cedar backbone did not expose shared latent")
        b = latent.shape[0] // self.nodes
        pooled = latent.mean(dim=1).reshape(b, self.nodes, -1)
        groups = []
        for value in torch.unique(self.partition):
            current = pooled[:, self.partition == value].reshape(-1, pooled.shape[-1])
            if current.shape[0] > 1:
                groups.append(current)
        if len(groups) < 2:
            return pooled.new_zeros(())
        # Polynomial-kernel MMD (mean and covariance discrepancy), avoiding a
        # bandwidth hyperparameter and keeping the penalty well-conditioned.
        mmd = pooled.new_zeros(())
        for index in range(1, len(groups)):
            a, c = groups[0], groups[index]
            mean_gap = (a.mean(0) - c.mean(0)).square().mean()
            ac = a - a.mean(0, keepdim=True)
            cc = c - c.mean(0, keepdim=True)
            cov_a = ac.T @ ac / max(a.shape[0] - 1, 1)
            cov_c = cc.T @ cc / max(c.shape[0] - 1, 1)
            mmd = mmd + mean_gap + (cov_a - cov_c).square().mean()
        return mmd / float(len(groups) - 1)


class _LatentHistoryTimeMixer(nn.Module):
    """Inject causal station history directly into PDM latent streams.

    The volume path remains the pretrained CPR-TimeMixer input.  A
    station-specific causal encoder maps occupancy/duration/other history and
    static descriptors to a latent coordinate at every observed hour; the
    coordinate is added before PDM at every official downsampling scale.  The
    injection is bounded and identity initialized, with no output residual,
    gate, router, or expert branch.
    """
    def __init__(self, args: SimpleNamespace, aux_dim: int,
                 static_features: np.ndarray, hidden: int = 32,
                 freeze_backbone: bool = False,
                 scale_specific: bool = False,
                 adjacency: np.ndarray | None = None,
                 graph_fusion: bool = False,
                 phase_coupling: bool = False):
        super().__init__()
        self.nodes = int(args.enc_in)
        self.aux_dim = int(aux_dim)
        static = np.asarray(static_features, dtype=np.float32)
        if static.ndim != 2 or static.shape[0] != self.nodes:
            raise ValueError(f"static_features must have shape [{self.nodes}, S], got {static.shape}")
        self.static_dim = int(static.shape[1])
        self.include_volume_state = bool(
            getattr(args, "history_include_volume_state", False)
        )
        self.raw_dim = self.aux_dim + self.static_dim + self.aux_dim * self.static_dim
        if self.include_volume_state:
            self.raw_dim += 1 + self.static_dim
        self.hidden = int(getattr(args, "d_model", hidden))
        self.history_latent_activation = str(
            getattr(args, "history_latent_activation", "tanh")
        ).lower()
        if self.history_latent_activation not in {"tanh", "none", "layernorm"}:
            raise ValueError("history_latent_activation must be tanh, none, or layernorm")
        self.history_latent_norm = nn.LayerNorm(self.hidden) if self.history_latent_activation == "layernorm" else None
        self.scale_specific = bool(scale_specific)
        self.graph_fusion = bool(graph_fusion)
        self.phase_coupling = bool(phase_coupling)
        self.num_scales = int(getattr(args, "down_sampling_layers", 0)) + 1
        self.register_buffer("static_features", torch.as_tensor(static, dtype=torch.float32))
        if self.graph_fusion:
            if adjacency is None:
                raise ValueError("graph_fusion requires audited adjacency")
            graph = np.asarray(adjacency, dtype=np.float32)
            if graph.shape != (self.nodes, self.nodes):
                raise ValueError(f"adjacency must have shape [{self.nodes},{self.nodes}]")
            graph = graph + np.eye(self.nodes, dtype=np.float32)
            graph = graph / np.maximum(graph.sum(axis=1, keepdims=True), 1e-6)
            self.register_buffer("graph_operator", torch.as_tensor(graph, dtype=torch.float32))
        if self.scale_specific:
            self.scale_projection = nn.Parameter(
                torch.zeros(self.num_scales, self.nodes, self.hidden, self.raw_dim))
            self.scale_bias = nn.Parameter(
                torch.zeros(self.num_scales, self.nodes, self.hidden))
        else:
            self.node_projection = nn.Parameter(torch.zeros(self.nodes, self.hidden, self.raw_dim))
            self.node_bias = nn.Parameter(torch.zeros(self.nodes, self.hidden))
        self.temporal = nn.Conv1d(self.hidden, self.hidden, kernel_size=5,
                                  padding=0, groups=self.hidden, bias=False)
        with torch.no_grad():
            self.temporal.weight.zero_()
            self.temporal.weight[:, 0, -1] = 1.0
        if self.phase_coupling:
            # Shared causal phase-state projection.  The coordinate combines
            # the observed cross-day template, within-day innovation and the
            # cycle-wide departure; it is added before PDM at every scale.
            self.phase_projection = nn.Linear(3 * self.aux_dim, self.hidden)
            nn.init.zeros_(self.phase_projection.weight)
            nn.init.zeros_(self.phase_projection.bias)
        self.backbone = _UpstreamModel(args)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def _history_latent(self, history_aux: torch.Tensor, x_enc: torch.Tensor) -> list[torch.Tensor]:
        b, t, n = x_enc.shape
        if history_aux.ndim == 3:
            history_aux = history_aux.unsqueeze(2).expand(-1, -1, n, -1)
        if history_aux.ndim != 4 or history_aux.shape[:3] != (b, t, n):
            raise ValueError("latent history must have shape [B,T,N,F]")
        if history_aux.shape[-1] != self.aux_dim:
            raise ValueError("latent history feature dimension differs from model")
        static = self.static_features.to(device=x_enc.device, dtype=x_enc.dtype)
        static = static.view(1, 1, n, self.static_dim).expand(b, t, n, -1)
        interaction = history_aux.unsqueeze(-1) * static.unsqueeze(-2)
        raw_parts = [history_aux]
        if self.include_volume_state:
            volume = x_enc.unsqueeze(-1)
            raw_parts.extend((volume, static, interaction.reshape(b, t, n, -1),
                              volume * static))
        else:
            raw_parts.extend((static, interaction.reshape(b, t, n, -1)))
        raw = torch.cat(raw_parts, dim=-1)
        phase_state = None
        if self.phase_coupling:
            period = 24
            if t < period or t % period:
                raise ValueError("phase-coupled latent history requires complete 24-hour cycles")
            # Use only the auxiliary channels for the phase state so static
            # descriptors are not repeatedly copied into the decomposition.
            aux = history_aux[:, -((t // period) * period):].reshape(
                b, t // period, period, n, self.aux_dim
            )
            template = aux.mean(dim=1, keepdim=True)
            innovation = aux - template
            cycle_departure = innovation.mean(dim=2, keepdim=True)
            phase_features = torch.cat(
                (template.expand_as(aux), innovation - cycle_departure,
                 cycle_departure.expand_as(aux)), dim=-1
            ).reshape(b, t, n, 3 * self.aux_dim)
            phase_state = self.phase_projection(phase_features)
        if self.scale_specific:
            # Project each pooled history stream independently.  This keeps
            # hourly, daily and coarse-scale covariate effects distinct
            # before their corresponding PDM/FMM stream, rather than using a
            # shared latent that is repeatedly averaged.
            raw_work = raw.permute(0, 2, 3, 1).reshape(b * n, self.raw_dim, t)
            scales = []
            for scale in range(self.num_scales):
                if scale:
                    raw_work = F.avg_pool1d(
                        raw_work, int(self.backbone.configs.down_sampling_window))
                ts = int(raw_work.shape[-1])
                raw_scale = raw_work.transpose(1, 2).reshape(b, n, ts, self.raw_dim)
                latent = torch.einsum(
                    "bntr,nhr->bnth", raw_scale, self.scale_projection[scale])
                latent = latent + self.scale_bias[scale].view(1, n, 1, self.hidden)
                latent = latent.reshape(b * n, ts, self.hidden).transpose(1, 2)
                latent = self.temporal(F.pad(latent, (4, 0))).transpose(1, 2)
                if phase_state is not None:
                    phase_work = phase_state.permute(0, 2, 3, 1).reshape(b * n, self.hidden, t)
                    if scale:
                        phase_work = F.avg_pool1d(
                            phase_work, int(self.backbone.configs.down_sampling_window) ** scale
                        )
                    phase_scale = phase_work.transpose(1, 2)
                    latent = latent + phase_scale
                scales.append(self._activate(latent))
            return scales
        latent = torch.einsum("btnr,nhr->btnh", raw, self.node_projection)
        latent = latent + self.node_bias.view(1, 1, n, -1)
        latent = latent.permute(0, 2, 1, 3).reshape(b * n, t, self.hidden).transpose(1, 2)
        latent = self.temporal(F.pad(latent, (4, 0))).transpose(1, 2)
        if phase_state is not None:
            phase_latent = phase_state.permute(0, 2, 1, 3).reshape(b * n, t, self.hidden)
            latent = latent + phase_latent
        if self.graph_fusion:
            latent = latent.reshape(b, n, t, self.hidden)
            neighbor = torch.einsum("nm,bmth->bnth", self.graph_operator.to(latent), latent)
            latent = (latent + neighbor).reshape(b * n, t, self.hidden)
        latent = self._activate(latent)
        scales = [latent]
        work = latent.transpose(1, 2)
        for _ in range(int(getattr(self.backbone.configs, "down_sampling_layers", 0))):
            work = F.avg_pool1d(work, int(self.backbone.configs.down_sampling_window))
            scales.append(work.transpose(1, 2))
        return scales

    def _activate(self, latent: torch.Tensor) -> torch.Tensor:
        if self.history_latent_activation == "none":
            return latent
        if self.history_latent_activation == "layernorm":
            return self.history_latent_norm(latent)
        return torch.tanh(latent)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, history_aux=None):
        if history_aux is None:
            raise ValueError("latent history TimeMixer requires history_aux")
        self.backbone._external_history_latents = self._history_latent(history_aux, x_enc)
        return self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)


class _EndToEndMultimodalTimeMixer(nn.Module):
    """End-to-end shared multimodal representation before TimeMixer PDM.

    The earlier latent-history route used a station-specific projection and
    often froze CPR-DX.  This variant learns one shared covariate coordinate
    from all historical variables (occupancy, duration, price and weather),
    adds a static station coordinate, and trains the complete CPR backbone
    end-to-end.  It is a fixed feature encoder: there is no sample gate,
    router, expert bank, output residual, or post-hoc selector.  The same
    representation is average-pooled into every official TimeMixer scale.
    """

    def __init__(self, args: SimpleNamespace, aux_dim: int,
                 static_features: np.ndarray, hidden: int = 32):
        super().__init__()
        self.nodes = int(args.enc_in)
        self.aux_dim = int(aux_dim)
        static = np.asarray(static_features, dtype=np.float32)
        if static.ndim != 2 or static.shape[0] != self.nodes:
            raise ValueError(f"static_features must have shape [{self.nodes}, S], got {static.shape}")
        self.static_dim = int(static.shape[1])
        self.hidden = int(getattr(args, "d_model", hidden))
        self.num_scales = int(getattr(args, "down_sampling_layers", 0)) + 1
        self.register_buffer("static_features", torch.as_tensor(static, dtype=torch.float32))

        # Shared feature-group maps avoid a large station-specific parameter
        # bank and make the route transferable across TAZs/cities.  The first
        # map learns temporal covariate interactions; the second injects the
        # fixed station geometry into the same latent coordinate.
        self.aux_encoder = nn.Sequential(
            nn.Linear(self.aux_dim, self.hidden), nn.GELU(),
            nn.LayerNorm(self.hidden),
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(self.static_dim, self.hidden), nn.GELU(),
            nn.LayerNorm(self.hidden),
        )
        self.fusion = nn.Linear(2 * self.hidden, self.hidden)
        self.temporal = nn.Conv1d(self.hidden, self.hidden, kernel_size=5,
                                  padding=0, groups=self.hidden, bias=False)
        with torch.no_grad():
            self.temporal.weight.zero_()
            self.temporal.weight[:, 0, -1] = 1.0
        # A separate linear map per resolution is an explicit scale
        # coordinate, not a learned scale gate.  Identity initialization
        # keeps the latent norm comparable to the official embedding.
        self.scale_projection = nn.Parameter(
            torch.eye(self.hidden).view(1, self.hidden, self.hidden)
            .repeat(self.num_scales, 1, 1)
        )
        self.scale_bias = nn.Parameter(torch.zeros(self.num_scales, self.hidden))
        self.backbone = _UpstreamModel(args)

    def _history_latent(self, history_aux: torch.Tensor,
                        x_enc: torch.Tensor) -> list[torch.Tensor]:
        b, t, n = x_enc.shape
        if history_aux.ndim == 3:
            history_aux = history_aux.unsqueeze(2).expand(-1, -1, n, -1)
        if history_aux.ndim != 4 or history_aux.shape[:3] != (b, t, n):
            raise ValueError("end-to-end multimodal history must be [B,T,N,F]")
        if history_aux.shape[-1] != self.aux_dim:
            raise ValueError("end-to-end multimodal feature dimension differs from model")
        aux = self.aux_encoder(history_aux)
        static = self.static_encoder(self.static_features.to(
            device=x_enc.device, dtype=x_enc.dtype))
        static = static.view(1, 1, n, self.hidden).expand(b, t, n, -1)
        latent = self.fusion(torch.cat((aux, static), dim=-1))
        latent = latent.permute(0, 2, 1, 3).reshape(b * n, t, self.hidden)
        latent = self.temporal(F.pad(latent.transpose(1, 2), (4, 0))).transpose(1, 2)
        scales = []
        work = latent
        for scale in range(self.num_scales):
            if scale:
                work = F.avg_pool1d(
                    work.transpose(1, 2),
                    int(self.backbone.configs.down_sampling_window),
                ).transpose(1, 2)
            projected = torch.einsum(
                "bth,hk->btk", work, self.scale_projection[scale]
            ) + self.scale_bias[scale].view(1, 1, -1)
            scales.append(projected)
        return scales

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, history_aux=None):
        if history_aux is None:
            raise ValueError("end-to-end multimodal TimeMixer requires history_aux")
        self.backbone._external_history_latents = self._history_latent(history_aux, x_enc)
        return self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)


class _GroupedMultimodalTimeMixer(_EndToEndMultimodalTimeMixer):
    """Physical-group low-rank multimodal representation before CPR PDM.

    Occupancy/duration, prices and weather are encoded in separate shared
    feature groups.  Their low-rank coordinates are concatenated and mapped
    once into the CPR latent space; there is no sample-dependent weighting.
    """

    def __init__(self, args: SimpleNamespace, aux_dim: int,
                 static_features: np.ndarray, group_indices: list[list[int]],
                 hidden: int = 32, group_rank: int = 8):
        # Build the common temporal/scale/backbone members without the
        # unused dense auxiliary encoder, then replace the feature path.
        super().__init__(args, aux_dim, static_features, hidden=hidden)
        self.group_indices = [tuple(int(i) for i in group if 0 <= int(i) < aux_dim)
                              for group in group_indices if group]
        if not self.group_indices:
            self.group_indices = [tuple(range(aux_dim))]
        rank = max(2, int(group_rank))
        self.group_encoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(len(group), rank), nn.GELU(),
                           nn.LayerNorm(rank))
             for group in self.group_indices]
        )
        self.group_fusion = nn.Sequential(
            nn.Linear(rank * len(self.group_indices), self.hidden),
            nn.GELU(), nn.LayerNorm(self.hidden)
        )
        # Keep static geometry as a separate coordinate but combine it through
        # a fixed additive representation map rather than a gate.
        self.static_to_hidden = nn.Linear(self.hidden, self.hidden)
        self.aux_encoder = None
        self.fusion = None

    def _history_latent(self, history_aux: torch.Tensor,
                        x_enc: torch.Tensor) -> list[torch.Tensor]:
        b, t, n = x_enc.shape
        if history_aux.ndim == 3:
            history_aux = history_aux.unsqueeze(2).expand(-1, -1, n, -1)
        if history_aux.ndim != 4 or history_aux.shape[:3] != (b, t, n):
            raise ValueError("grouped multimodal history must be [B,T,N,F]")
        if history_aux.shape[-1] != self.aux_dim:
            raise ValueError("grouped multimodal feature dimension differs from model")
        pieces = [encoder(history_aux[..., list(group)])
                  for group, encoder in zip(self.group_indices, self.group_encoders)]
        aux = self.group_fusion(torch.cat(pieces, dim=-1))
        static = self.static_encoder(self.static_features.to(
            device=x_enc.device, dtype=x_enc.dtype))
        static = self.static_to_hidden(static)
        latent = aux + static.view(1, 1, n, self.hidden)
        latent = latent.permute(0, 2, 1, 3).reshape(b * n, t, self.hidden)
        latent = self.temporal(F.pad(latent.transpose(1, 2), (4, 0))).transpose(1, 2)
        scales = []
        work = latent
        for scale in range(self.num_scales):
            if scale:
                work = F.avg_pool1d(
                    work.transpose(1, 2),
                    int(self.backbone.configs.down_sampling_window),
                ).transpose(1, 2)
            projected = torch.einsum(
                "bth,hk->btk", work, self.scale_projection[scale]
            ) + self.scale_bias[scale].view(1, 1, -1)
            scales.append(projected)
        return scales


class _HybridLatentCCLTTimeMixer(nn.Module):
    """Single CPR backbone with pre-PDM history transport and CCLT readout.

    The two history paths share one frozen CPR-DX backbone: the latent path
    changes the representation before PDM, while CCLT supplies its fixed
    additive, graph-aware history correction.  There is no selector or
    sample-dependent blend between the paths; the CCLT readout is zero
    initialized and the backbone consumes the transported latent once.
    """

    def __init__(self, args: SimpleNamespace, feature_dim: int,
                 static_features: np.ndarray, adjacency: np.ndarray,
                 hidden: int = 32, graph_order: int = 1,
                 graph_contrast: bool = False,
                 latent_scale_specific: bool = False,
                 phase_coupling: bool = False,
                 innovation_projection: bool = False,
                 basis_residual: bool = False,
                 extended_basis: bool = False,
                 cumulative_residual: bool = False, multi_kernel: bool = False,
                 base_feature_indices: np.ndarray | None = None,
                 graph_input_diffusion: bool = False,
                 graph_input_order: int = 1,
                 graph_input_mode: str = "midpoint",
                 state_transport: bool = False,
                 state_stats: tuple[np.ndarray, np.ndarray] | None = None,
                 latent_node_mix: bool = False,
                 latent_node_mix_rank: int = 8):
        super().__init__()
        args.urban_adj = np.asarray(adjacency, dtype=np.float32)
        args.cclt_graph_fusion = 1
        args.cclt_graph_order = int(graph_order)
        args.cclt_graph_contrast = int(bool(graph_contrast))
        self.cclt = _CCLTTimeMixer(
            args, feature_dim, static_features, hidden=hidden,
            freeze_backbone=True, basis_residual=basis_residual,
            innovation_projection=innovation_projection,
            extended_basis=extended_basis,
            endogenous_history=False, trajectory_readout=False,
            cumulative_residual=bool(cumulative_residual),
            multi_kernel=bool(multi_kernel))
        # Fixed causal graph diffusion of the observed volume history before
        # the single CPR backbone.  This is deliberately a deterministic
        # coordinate transform (not a learned gate or route): at each observed
        # time t, node i receives the row-normalized spatial average of its
        # neighbours and itself.  The original volume coordinate is retained
        # through a fixed midpoint residual, so the route cannot erase local
        # station information at initialization.
        self.graph_input_diffusion = bool(graph_input_diffusion)
        self.graph_input_order = max(1, int(graph_input_order))
        self.graph_input_mode = str(graph_input_mode).strip().lower()
        if self.graph_input_mode not in {"midpoint", "contrast"}:
            raise ValueError("graph_input_mode must be midpoint or contrast")
        if self.graph_input_diffusion:
            graph = np.maximum(np.asarray(adjacency, dtype=np.float32), 0.0)
            graph = graph + np.eye(graph.shape[0], dtype=np.float32)
            graph = graph / np.maximum(graph.sum(axis=1, keepdims=True), 1e-6)
            self.register_buffer("graph_input_operator", torch.as_tensor(graph))
        self.latent = _LatentHistoryTimeMixer(
            args, feature_dim, static_features, hidden=hidden,
            freeze_backbone=True, scale_specific=latent_scale_specific,
            adjacency=adjacency, graph_fusion=False,
            phase_coupling=phase_coupling)
        # Both wrappers point to exactly one CPR backbone.  The latent wrapper
        # only contributes its history encoder after this reassignment.
        self.latent.backbone = self.cclt.backbone
        # Optional fixed-rank station-coordinate transport on the historical
        # latent only.  This is deliberately separate from the backbone's
        # dynamic cross-node branch: a shared node projection/reconstruction
        # is applied to every latent time token and scale, with an identity
        # residual path at construction.  There is no sample gate, router or
        # expert choice; the same learned low-rank operator is used for every
        # batch and every horizon.
        self.latent_node_mix = bool(latent_node_mix)
        if self.latent_node_mix:
            rank = max(1, int(latent_node_mix_rank))
            nodes = int(self.cclt.nodes)
            self.latent_node_down = nn.Linear(nodes, rank, bias=False)
            self.latent_node_up = nn.Linear(rank, nodes, bias=False)
            nn.init.normal_(self.latent_node_down.weight, mean=0.0,
                            std=nodes ** -0.5)
            # Exact identity at initialization; the low-rank correction is
            # learned from the training prefix without a searched strength.
            nn.init.zeros_(self.latent_node_up.weight)
            self.latent_node_mix_rank = rank
        else:
            self.latent_node_down = None
            self.latent_node_up = None
            self.latent_node_mix_rank = 0
        if base_feature_indices is None:
            base_feature_indices = np.arange(feature_dim, dtype=np.int64)
        base_feature_indices = np.asarray(base_feature_indices, dtype=np.int64)
        if base_feature_indices.ndim != 1 or base_feature_indices.size > feature_dim:
            raise ValueError("base history feature indices exceed feature_dim")
        self.register_buffer("base_feature_indices", torch.as_tensor(base_feature_indices))

        # Causal state transport is a fixed coordinate system built only from
        # the training prefix.  It carries robust level anomaly, 24 h drift,
        # endpoint innovation and 24 h activity scale alongside the duration
        # stream.  Unlike CAST, this never rewrites the volume input or
        # analytically reconstructs the target; the coordinates are consumed
        # by the same latent/CCLT encoder as the historical companion.
        self.state_transport = bool(state_transport)
        if self.state_transport:
            if state_stats is None or len(state_stats) != 2:
                raise ValueError("state transport requires train-prefix robust stats")
            state_center = np.asarray(state_stats[0], dtype=np.float32)
            state_scale = np.asarray(state_stats[1], dtype=np.float32)
            if state_center.ndim != 1 or state_scale.ndim != 1 or state_center.size != self.cclt.nodes:
                raise ValueError("state transport stats must be [nodes]")
            self.register_buffer("state_center", torch.as_tensor(state_center))
            self.register_buffer("state_scale", torch.as_tensor(np.maximum(state_scale, 1e-3)))
        else:
            self.register_buffer("state_center", torch.zeros(self.cclt.nodes))
            self.register_buffer("state_scale", torch.ones(self.cclt.nodes))

        # Optional fixed cross-variable state fitted on the train prefix.  A
        # zero initialized projection writes this state into PDM scale zero
        # without introducing a gate, router, expert, or forecast selector.
        self.whiten_state = None

    def configure_whitened_state(self, mean: np.ndarray,
                                 components: np.ndarray,
                                 feature_indices: np.ndarray) -> None:
        mean = np.asarray(mean, dtype=np.float32)
        components = np.asarray(components, dtype=np.float32)
        feature_indices = np.asarray(feature_indices, dtype=np.int64)
        if mean.ndim != 1 or components.ndim != 2 or components.shape[0] != mean.size:
            raise ValueError("invalid fixed whitening coordinates")
        if feature_indices.ndim != 1 or feature_indices.size != mean.size:
            raise ValueError("whitening feature index/mean mismatch")
        self.register_buffer("whitening_mean", torch.as_tensor(mean))
        self.register_buffer("whitening_components", torch.as_tensor(components))
        self.register_buffer("whitening_feature_indices", torch.as_tensor(feature_indices))
        self.whiten_state = nn.Linear(int(components.shape[1]), self.latent.hidden)
        nn.init.zeros_(self.whiten_state.weight)
        nn.init.zeros_(self.whiten_state.bias)

    def _whitened_state_token(self, history_aux: torch.Tensor) -> torch.Tensor | None:
        if self.whiten_state is None:
            return None
        if history_aux.ndim == 3:
            history_aux = history_aux.unsqueeze(2).expand(-1, -1, self.cclt.nodes, -1)
        # The state is city-level across the aligned stations, so the fixed
        # whitening coordinates do not create a separate high-dimensional
        # station parameter bank.  The resulting token is broadcast back to
        # each station before PDM scale zero.
        selected = history_aux.mean(dim=2).index_select(
            -1, self.whitening_feature_indices.to(history_aux.device))
        centered = selected - self.whitening_mean.to(
            device=history_aux.device, dtype=history_aux.dtype)
        state = torch.einsum(
            "btf,fr->btr", centered,
            self.whitening_components.to(device=history_aux.device, dtype=history_aux.dtype)
        )
        endpoint = state[:, -1]  # causal state at the forecast origin
        encoded = self.whiten_state(endpoint)
        b, hidden = encoded.shape
        n = self.cclt.nodes
        return encoded[:, None, :].expand(b, n, hidden).reshape(b * n, 1, hidden)

    @property
    def backbone(self):
        return self.cclt.backbone

    @staticmethod
    def _causal_mean(x: torch.Tensor, width: int) -> torch.Tensor:
        """Left-causal trailing mean with no future or unavailable padding."""
        z = x.permute(0, 2, 1)
        z = F.pad(z, (width - 1, 0))
        return F.avg_pool1d(z, width, stride=1, count_include_pad=False).permute(0, 2, 1)

    def _state_coordinates(self, x_enc: torch.Tensor) -> torch.Tensor:
        if not self.state_transport:
            return x_enc.new_zeros((*x_enc.shape, 0))
        # x_enc is already in the train-prefix standardized volume coordinate.
        # Every statistic is evaluated at each observed t using only x[:t].
        m24 = self._causal_mean(x_enc, min(24, x_enc.shape[1]))
        m168 = self._causal_mean(x_enc, min(168, x_enc.shape[1]))
        scale24 = torch.sqrt(torch.clamp(
            self._causal_mean((x_enc - m24).square(), min(24, x_enc.shape[1])),
            min=1e-6))
        center = self.state_center.to(device=x_enc.device, dtype=x_enc.dtype).view(1, 1, -1)
        scale = self.state_scale.to(device=x_enc.device, dtype=x_enc.dtype).view(1, 1, -1)
        level_anomaly = (m24 - center) / scale
        drift = (m24 - m168) / scale
        innovation = (x_enc - m24) / scale
        activity = scale24 / scale
        return torch.stack((level_anomaly, drift, innovation, activity), dim=-1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, history_aux=None):
        if history_aux is None:
            raise ValueError("hybrid latent CCLT requires history_aux")
        base_aux = history_aux.index_select(-1, self.base_feature_indices.to(history_aux.device))
        transport = self._state_coordinates(x_enc)
        aux = torch.cat((base_aux, transport), dim=-1) if self.state_transport else base_aux
        scales = self.latent._history_latent(aux, x_enc)
        if self.latent_node_mix:
            # [B*N,T,D] -> [B,N,T,D].  The projection acts only on the
            # station axis; time and latent channels are untouched.  Keeping
            # the original tensor in the residual path preserves local
            # station amplitudes and gives an exact identity at step zero.
            mixed_scales = []
            batch_size = int(x_enc.shape[0])
            nodes = int(self.cclt.nodes)
            for latent in scales:
                if latent.shape[0] != batch_size * nodes:
                    raise ValueError("latent node mixer received invalid batch shape")
                _, length, hidden = latent.shape
                view = latent.reshape(batch_size, nodes, length, hidden)
                compressed = torch.einsum(
                    "rn,bntd->brtd", self.latent_node_down.weight, view)
                correction = torch.einsum(
                    "nr,brtd->bntd", self.latent_node_up.weight, compressed)
                mixed_scales.append((view + correction).reshape_as(latent))
            scales = mixed_scales
        state = self._whitened_state_token(history_aux)
        if state is not None:
            # Constant state token is visible only to the first PDM/FMM scale.
            scales[0] = scales[0] + state.expand(-1, scales[0].shape[1], -1).to(
                dtype=scales[0].dtype)
        self.backbone._external_history_latents = scales
        # Apply only to the observed history.  No future values are available
        # on this path, and the same fixed operator is used for every sample.
        # Midpoint residual (x + P^k x)/2 is a graph low-pass coordinate,
        # rather than a sample-dependent selector.  The contrast option is
        # the fixed high-pass complement 2x-P^k x and preserves station-local
        # differences that low-pass diffusion can erase.
        backbone_x = x_enc
        if self.graph_input_diffusion:
            propagated = x_enc
            for _ in range(self.graph_input_order):
                propagated = torch.einsum(
                    "ij,btj->bti", self.graph_input_operator.to(
                        device=x_enc.device, dtype=x_enc.dtype), propagated)
            backbone_x = (2.0 * x_enc - propagated
                          if self.graph_input_mode == "contrast"
                          else 0.5 * (x_enc + propagated))
        return self.cclt(backbone_x, x_mark_enc, x_dec, x_mark_dec, mask, aux)


class _PhaseRegionStateTensorTimeMixer(nn.Module):
    """Keep a complete causal 7x24 phase state before the CPR PDM.

    The existing history routes compress a companion stream to one endpoint
    state and add it to the frozen backbone.  This route keeps every observed
    phase token, applies shared linear transitions within a day and across
    days, and then exposes the resulting state at all official TimeMixer
    scales.  The transitions are identity-initialized and the final state
    projection is zero-initialized, so the checkpoint starts at the exact CPR
    forecast while training can learn a new representation.  There is no
    sample-dependent gate, router, expert, or output fusion.
    """

    def __init__(self, args: SimpleNamespace, feature_dim: int,
                 freeze_backbone: bool = True, cycles: int = 7,
                 phases: int = 24):
        super().__init__()
        self.nodes = int(args.enc_in)
        self.feature_dim = int(feature_dim)
        self.hidden = int(getattr(args, "d_model", 32))
        self.cycles = int(cycles)
        self.phases = int(phases)
        self.num_scales = int(getattr(args, "down_sampling_layers", 0)) + 1
        self.window = self.cycles * self.phases
        if int(args.seq_len) != self.window:
            raise ValueError(
                f"phase-region state requires seq_len={self.window}, got {args.seq_len}"
            )
        # Volume is the primary observation; duration/occupancy and any other
        # selected historical channels are independent companion observations.
        self.observation = nn.Linear(self.feature_dim + 1, self.hidden)
        self.temporal = nn.Conv1d(
            self.hidden, self.hidden, kernel_size=5, padding=0,
            groups=self.hidden, bias=False)
        with torch.no_grad():
            self.temporal.weight.zero_()
            self.temporal.weight[:, 0, -1] = 1.0
        self.phase_transition = nn.Linear(2 * self.hidden, self.hidden)
        self.cycle_transition = nn.Linear(2 * self.hidden, self.hidden)
        self._identity_transition(self.phase_transition)
        self._identity_transition(self.cycle_transition)
        # A zero map makes the loaded CPR checkpoint the exact initial model;
        # gradients then learn the state coordinate only when it helps the
        # forecast loss.
        self.state_projection = nn.Linear(self.hidden, self.hidden)
        nn.init.zeros_(self.state_projection.weight)
        nn.init.zeros_(self.state_projection.bias)
        self.backbone = _UpstreamModel(args)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _identity_transition(module: nn.Linear) -> None:
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)
        with torch.no_grad():
            width = module.out_features
            module.weight[:, :width].copy_(torch.eye(width))

    def _state(self, x_enc: torch.Tensor,
               history_aux: torch.Tensor) -> list[torch.Tensor]:
        if history_aux.ndim == 3:
            history_aux = history_aux.unsqueeze(2).expand(
                -1, -1, self.nodes, -1)
        if history_aux.ndim != 4 or history_aux.shape[:3] != x_enc.shape:
            raise ValueError("phase-region history must be [B,T,N,F]")
        b, t, n = x_enc.shape
        observed = torch.cat((x_enc.unsqueeze(-1), history_aux), dim=-1)
        # The upstream CI path flattens each zone's full time series.  A raw
        # reshape of [B,T,N,H] would interleave time and zone memory and send
        # the last observation of only some zones through the causal encoder.
        latent = self.observation(observed).permute(0, 2, 1, 3).reshape(
            b * n, t, self.hidden
        )
        latent = self.temporal(
            F.pad(latent.transpose(1, 2), (4, 0))
        ).transpose(1, 2)
        latent = F.gelu(latent)
        tokens = latent.reshape(b, n, self.cycles, self.phases, self.hidden)
        # Phase transition retains the current phase and adds a shared
        # within-day predecessor coordinate.  At initialization this is an
        # exact identity, while the full 7x24 tensor remains observable.
        previous_phase = torch.cat((tokens[..., :1, :], tokens[..., :-1, :]), dim=3)
        phase_state = self.phase_transition(
            torch.cat((tokens, previous_phase), dim=-1)
        )
        # A second shared transition carries the same phase across daily
        # cycles.  It is causal over the observed seven cycles and preserves
        # the cycle-specific innovation instead of averaging it away.
        previous_cycle = torch.cat((phase_state[:, :, :1], phase_state[:, :, :-1]), dim=2)
        state = self.cycle_transition(
            torch.cat((phase_state, previous_cycle), dim=-1)
        )
        state = self.state_projection(state)
        state = state.reshape(b * n, t, self.hidden)
        scales = [state]
        work = state.transpose(1, 2)
        window = int(getattr(self.backbone.configs, "down_sampling_window", 2))
        for _ in range(self.num_scales - 1):
            work = F.avg_pool1d(work, window)
            scales.append(work.transpose(1, 2))
        return scales

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, history_aux=None):
        if history_aux is None:
            raise ValueError("phase-region state tensor requires history_aux")
        self.backbone._external_history_latents = self._state(x_enc, history_aux)
        return self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)


class _CCLTTimeMixer(nn.Module):
    """Causal Covariate Latent Transport (CCLT) on the CPR backbone.

    The historical companion variables are encoded by a separate causal
    temporal convolution for each zone.  Only the final encoded history state
    is projected to the forecast horizon and added to the volume backbone
    prediction with a fixed coefficient of one.  The residual head is zero
    initialized, so optimization starts from the CPR-TimeMixer solution while
    learning a distinct forecast representation.  There is no gate, router,
    expert selector, or parameter-dependent blending coefficient.
    """
    def __init__(self, args: SimpleNamespace, feature_dim: int,
                 static_features: np.ndarray | None = None,
                 hidden: int = 32, innovation_projection: bool = False,
                 freeze_backbone: bool = False, basis_residual: bool = False,
                 extended_basis: bool = False, endogenous_history: bool = False,
                 recurrent_residual: bool = False, trajectory_readout: bool = False,
                 cumulative_residual: bool = False, multi_kernel: bool = False):
        super().__init__()
        self.nodes = int(args.enc_in)
        self.feature_dim = int(feature_dim)
        self.pred_len = int(args.pred_len)
        self.innovation_projection = bool(innovation_projection)
        self.freeze_backbone = bool(freeze_backbone)
        self.basis_residual = bool(basis_residual)
        self.extended_basis = bool(extended_basis)
        self.endogenous_history = bool(endogenous_history)
        self.recurrent_residual = bool(recurrent_residual)
        self.cumulative_residual = bool(cumulative_residual)
        self.multi_kernel = bool(multi_kernel)
        self.graph_fusion = bool(getattr(args, "cclt_graph_fusion", False))
        self.graph_order = max(1, int(getattr(args, "cclt_graph_order", 1)))
        self.graph_volume = bool(getattr(args, "cclt_graph_volume", False))
        self.graph_contrast = bool(getattr(args, "cclt_graph_contrast", False))
        self.trajectory_readout = bool(trajectory_readout or getattr(args, "cclt_trajectory", False))
        static = np.zeros((self.nodes, 1), dtype=np.float32) if static_features is None else np.asarray(static_features, dtype=np.float32)
        if static.ndim != 2 or static.shape[0] != self.nodes:
            raise ValueError(f"static_features must have shape [{self.nodes}, S], got {static.shape}")
        hidden = max(8, int(hidden))
        self.register_buffer("static_features", torch.as_tensor(static, dtype=torch.float32))
        self.covariate_in = nn.Linear(self.feature_dim, hidden)
        # Left padding makes this convolution causal: output at t only sees
        # covariates at t and earlier.  It is shared across all stations.
        self.causal_conv = nn.Conv1d(hidden, hidden, kernel_size=5, padding=0)
        self.covariate_norm = nn.LayerNorm(hidden)
        if self.multi_kernel:
            self.long_causal_conv = nn.Conv1d(hidden, hidden, kernel_size=25, padding=0)
            self.long_covariate_norm = nn.LayerNorm(hidden)
            self.multi_projection = nn.Linear(2 * hidden, hidden)
            with torch.no_grad():
                self.multi_projection.weight.zero_()
                self.multi_projection.bias.zero_()
                self.multi_projection.weight[:, :hidden] = torch.eye(hidden)
        if self.endogenous_history or self.graph_volume:
            self.volume_in = nn.Linear(1, hidden)
            self.volume_conv = nn.Conv1d(hidden, hidden, kernel_size=5, padding=0)
            self.volume_norm = nn.LayerNorm(hidden)
        self.static_encoder = nn.Sequential(nn.Linear(static.shape[1], hidden), nn.GELU())
        if self.graph_fusion:
            adjacency = np.asarray(getattr(args, "urban_adj", None), dtype=np.float32)
            if adjacency.shape != (self.nodes, self.nodes):
                raise ValueError("cclt_graph_fusion requires [nodes,nodes] urban_adj")
            adjacency = np.maximum(adjacency, 0.0) + np.eye(self.nodes, dtype=np.float32)
            adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-6)
            self.register_buffer("cclt_adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
            graph_terms = 1 + self.graph_order + (self.graph_order if self.graph_contrast else 0)
            self.graph_projection = nn.Linear(graph_terms * hidden, hidden)
            with torch.no_grad():
                self.graph_projection.weight.zero_()
                self.graph_projection.bias.zero_()
                self.graph_projection.weight[:, :hidden] = torch.eye(hidden)
        self.basis_dim = (9 if self.extended_basis else 4) if self.basis_residual else self.pred_len
        self.residual_readout = nn.Linear(hidden, self.basis_dim)
        nn.init.zeros_(self.residual_readout.weight)
        nn.init.zeros_(self.residual_readout.bias)
        if self.trajectory_readout:
            self.trajectory_readout_layer = nn.Linear(5 * hidden, self.basis_dim)
            nn.init.zeros_(self.trajectory_readout_layer.weight)
            nn.init.zeros_(self.trajectory_readout_layer.bias)
        if self.recurrent_residual:
            self.recurrent_cell = nn.GRUCell(4, hidden)
            self.recurrent_readout = nn.Linear(hidden, 1)
            nn.init.zeros_(self.recurrent_readout.weight)
            nn.init.zeros_(self.recurrent_readout.bias)
            hours = torch.arange(self.pred_len, dtype=torch.float32)
            phase24 = 2.0 * torch.pi * hours / 24.0
            phase12 = 2.0 * torch.pi * hours / 12.0
            self.register_buffer("horizon_basis", torch.stack((torch.sin(phase24), torch.cos(phase24), torch.sin(phase12), torch.cos(phase12)), dim=1))
        if self.basis_residual:
            hours = torch.arange(self.pred_len, dtype=torch.float32)
            cols = []
            if self.extended_basis:
                tau = (hours - (self.pred_len - 1) / 2.0) / max(self.pred_len - 1, 1)
                cols.extend((torch.ones_like(hours), tau, tau.square()))
            for period in ((24.0, 12.0, 8.0) if self.extended_basis else (24.0, 12.0)):
                phase = 2.0 * torch.pi * hours / period
                cols.extend((torch.sin(phase), torch.cos(phase)))
            self.register_buffer("innovation_basis", torch.stack(cols, dim=1))
        self.backbone = _UpstreamModel(args)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, history_aux=None):
        if history_aux is None:
            raise ValueError("CCLT-TimeMixer requires history_aux")
        if history_aux.ndim == 3:
            # Global history features are broadcast to every station.  The
            # normal UrbanEV route uses node-aligned [B,T,N,F] features.
            history_aux = history_aux.unsqueeze(2).expand(-1, -1, self.nodes, -1)
        if history_aux.ndim != 4 or history_aux.shape[:3] != (x_enc.shape[0], x_enc.shape[1], self.nodes):
            raise ValueError(
                f"history_aux must have shape [B,T,N,F], got {tuple(history_aux.shape)}"
            )
        if history_aux.shape[-1] != self.feature_dim:
            raise ValueError("history_aux feature dimension differs from CCLT model")
        b, t, n, _ = history_aux.shape
        z = self.covariate_in(history_aux).reshape(b * n, t, -1)
        z = z.transpose(1, 2)
        z = F.pad(z, (4, 0))
        z = self.causal_conv(z).transpose(1, 2)
        z = F.gelu(z)
        if self.multi_kernel:
            long_z = self.covariate_in(history_aux).reshape(b * n, t, -1).transpose(1, 2)
            long_z = F.pad(long_z, (24, 0))
            long_z = self.long_causal_conv(long_z).transpose(1, 2)
            long_z = F.gelu(long_z)
        # Last history state is the only state transported to the forecast;
        # no future or target rows can enter this path.
        if self.trajectory_readout:
            recent = z[:, -24:]
            recent = self.covariate_norm(recent)
            trajectory = recent.reshape(b, n, 24, -1)
            trajectory = torch.cat(
                [trajectory[:, :, start:start + 6].mean(dim=2)
                 for start in (0, 6, 12, 18)], dim=-1
            )
            # The final hidden state is retained separately from the four
            # six-hour summaries; concatenating it preserves the endpoint
            # state without a sample-dependent weighting mechanism.
            endpoint = recent[:, -1].reshape(b, n, -1)
            trajectory_features = torch.cat((trajectory, endpoint), dim=-1)
        z = self.covariate_norm(z[:, -1]).reshape(b, n, -1)
        if self.multi_kernel:
            long_state = self.long_covariate_norm(long_z[:, -1]).reshape(b, n, -1)
            z = self.multi_projection(torch.cat((z, long_state), dim=-1))
        if self.graph_fusion:
            graph_views = [z]
            propagated = z
            for _ in range(self.graph_order):
                propagated = torch.einsum("ij,bjh->bih", self.cclt_adjacency.to(dtype=z.dtype), propagated)
                graph_views.append(propagated)
            if self.graph_contrast:
                propagated = z
                for _ in range(self.graph_order):
                    propagated = torch.einsum("ij,bjh->bih", self.cclt_adjacency.to(dtype=z.dtype), propagated)
                    graph_views.append(z - propagated)
            z = self.graph_projection(torch.cat(graph_views, dim=-1))
        if self.endogenous_history or self.graph_volume:
            v = x_enc.reshape(b * n, t, 1)
            v = self.volume_in(v).transpose(1, 2)
            v = F.pad(v, (4, 0))
            v = self.volume_conv(v).transpose(1, 2)
            v = F.gelu(v)
            v = self.volume_norm(v[:, -1]).reshape(b, n, -1)
            if self.graph_volume:
                if not self.graph_fusion:
                    raise ValueError("cclt_graph_volume requires cclt_graph_fusion")
                # Fixed adjacency diffusion of the observed demand state.  It
                # uses only the history window and has no learned selector.
                v = torch.einsum("ij,bjh->bih", self.cclt_adjacency.to(dtype=v.dtype), v)
            z = z + v
        z = z + self.static_encoder(self.static_features).unsqueeze(0)
        if self.recurrent_residual:
            state = z.reshape(b * n, -1)
            outputs = []
            for h in range(self.pred_len):
                state = self.recurrent_cell(self.horizon_basis[h].to(dtype=state.dtype).expand(b * n, -1), state)
                outputs.append(self.recurrent_readout(state))
            residual = torch.stack(outputs, dim=1).reshape(b, n, self.pred_len).permute(0, 2, 1)
        else:
            coeff = (self.trajectory_readout_layer(trajectory_features)
                     if self.trajectory_readout else self.residual_readout(z))
            if self.basis_residual:
                residual = torch.einsum("bnk,hk->bhn", coeff, self.innovation_basis)
            else:
                residual = coeff.permute(0, 2, 1)
            if self.cumulative_residual:
                # Predict causal level increments and integrate them over the
                # horizon.  The zero-initialized head therefore starts at the
                # exact frozen-backbone forecast and learns a smooth state
                # correction, without a gate or a second forecast selector.
                residual = torch.cumsum(residual, dim=1)
        if self.innovation_projection:
            # Fixed level-preserving projection: the covariate branch can only
            # correct within-horizon shape/innovation and cannot move the
            # backbone's forecast level.  This is an operator, not a gate or
            # a learned blending parameter.
            residual = residual - residual.mean(dim=1, keepdim=True)
        if self.freeze_backbone:
            # Keep the pretrained CPR representation deterministic while the
            # covariate residual head learns train-prefix errors.
            self.backbone.eval()
        forecast = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return forecast + residual


def _smooth(x: torch.Tensor, kernel: int = 25) -> torch.Tensor:
    """Causal-free local trend operator used for both inputs and labels."""
    return F.avg_pool1d(x.permute(0, 2, 1), kernel, 1, kernel // 2,
                        count_include_pad=False).permute(0, 2, 1)


def _phase_align(x: torch.Tensor, period: int = 24) -> torch.Tensor:
    """Replace each phase position by its mean across complete periods.

    UrbanEV has a 24-hour operating cycle, while the 168-hour context contains
    seven complete cycles.  This creates a period-aligned view before mixing;
    it is a change of representation and forecast target, not an output gate
    or a residual graph correction.
    """
    b, t, c = x.shape
    n = max(1, t // period)
    usable = n * period
    tail = x[:, -usable:].reshape(b, n, period, c)
    template = tail.mean(dim=1, keepdim=True).repeat(1, n, 1, 1)
    aligned = template.reshape(b, usable, c)
    if usable < t:
        aligned = torch.cat((aligned[:, :1].repeat(1, t - usable, 1), aligned), dim=1)
    return aligned[:, -t:]


class _PeriodAlignedTimeMixer(nn.Module):
    """Two-target component forecaster with a calendar-aligned periodic view.

    The official TimeMixer is retained as the backbone for each component,
    but the learning problem is changed from one unconstrained forecast to
    two separately supervised forecasts:

    * a smooth demand component from the local trend view;
    * a daily phase-aligned component from the detrended view.

    The final forecast is their reconstruction sum.  Because the two heads
    are trained against their own decomposed targets, this is not a learned
    gate or a post-hoc residual correction.
    """
    def __init__(self, args: SimpleNamespace, period_alignment: bool = True):
        super().__init__()
        self.trend_backbone = _UpstreamModel(args)
        self.period_backbone = _UpstreamModel(args)
        self.period_alignment = bool(period_alignment)

    @staticmethod
    def decompose(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = _smooth(x)
        return trend, x - trend

    def component_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend, seasonal = self.decompose(x)
        # Average the raw detrended view with its 24-hour phase-aligned view.
        # The weekly context remains present in the raw term.
        if self.period_alignment:
            daily = _phase_align(seasonal, period=24)
            period_view = 0.5 * (seasonal + daily)
        else:
            period_view = seasonal
        return trend, period_view

    def forward_components(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                           mask=None):
        trend_x, period_x = self.component_inputs(x_enc)
        trend_pred = self.trend_backbone(trend_x, x_mark_enc, x_dec, x_mark_dec, mask)
        period_pred = self.period_backbone(period_x, x_mark_enc, x_dec, x_mark_dec, mask)
        return trend_pred, period_pred

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        trend_pred, period_pred = self.forward_components(
            x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return trend_pred + period_pred


def _haar_components(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One-level orthonormal Haar decomposition, expanded to the input length."""
    if x.shape[1] % 2:
        raise ValueError("Haar decomposition requires an even time length")
    even, odd = x[:, 0::2], x[:, 1::2]
    scale = float(2 ** -0.5)
    approximation = (even + odd) * scale
    detail = (even - odd) * scale
    # Keep the official TimeMixer interface unchanged while exposing the
    # multi-resolution components as separate forecasting targets.
    return (approximation.repeat_interleave(2, dim=1),
            detail.repeat_interleave(2, dim=1))


class _WaveletTimeMixer(nn.Module):
    """Wavelet component forecaster built from two official TimeMixer mixers."""
    def __init__(self, args: SimpleNamespace):
        super().__init__()
        self.approximation_backbone = _UpstreamModel(args)
        self.detail_backbone = _UpstreamModel(args)

    @staticmethod
    def decompose(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _haar_components(x)

    def forward_components(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                           mask=None):
        approx_x, detail_x = self.decompose(x_enc)
        approx_pred = self.approximation_backbone(
            approx_x, x_mark_enc, x_dec, x_mark_dec, mask)
        detail_pred = self.detail_backbone(
            detail_x, x_mark_enc, x_dec, x_mark_dec, mask)
        return approx_pred, detail_pred

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        approx_pred, detail_pred = self.forward_components(
            x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return approx_pred + detail_pred


def _wavelet_loss(model: _WaveletTimeMixer, x: torch.Tensor,
                  target: torch.Tensor, reconstruction_weight: float = 0.2) -> torch.Tensor:
    target_approx, target_detail = model.decompose(target)
    pred_approx, pred_detail = model.forward_components(x)
    component = F.mse_loss(pred_approx, target_approx) + F.mse_loss(pred_detail, target_detail)
    reconstruction = F.mse_loss(pred_approx + pred_detail, target)
    return component + float(reconstruction_weight) * reconstruction


def _fourier_phase_view(x: torch.Tensor, period: int, harmonics: int) -> torch.Tensor:
    """Project each zone onto fixed sinusoidal bases for a known period."""
    b, t, _ = x.shape
    grid = torch.arange(t, device=x.device, dtype=x.dtype)
    bases = []
    for harmonic in range(1, harmonics + 1):
        phase = 2.0 * torch.pi * harmonic * grid / float(period)
        bases.extend((torch.sin(phase), torch.cos(phase)))
    basis = torch.stack(bases, dim=1)  # [time, basis]
    centered = x - x.mean(dim=1, keepdim=True)
    coeff = torch.einsum("btc,tk->bkc", centered, basis) / max(1, t) * 2.0
    return torch.einsum("tk,bkc->btc", basis, coeff)


class _SpectralTimeMixer(nn.Module):
    """Single-head TimeMixer with fixed daily/weekly spectral input views."""
    def __init__(self, args: SimpleNamespace):
        super().__init__()
        channels = int(args.enc_in)
        self.feature_projection = nn.Linear(3 * channels, channels)
        with torch.no_grad():
            self.feature_projection.weight.zero_()
            self.feature_projection.weight[:, :channels].copy_(torch.eye(channels))
            self.feature_projection.bias.zero_()
        self.backbone = _UpstreamModel(args)

    def spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        daily = _fourier_phase_view(x, period=24, harmonics=2)
        weekly = _fourier_phase_view(x, period=168, harmonics=1)
        return self.feature_projection(torch.cat((x, daily, weekly), dim=-1))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        features = self.spectral_features(x_enc)
        return self.backbone(features, x_mark_enc, x_dec, x_mark_dec, mask)


def _spectral_loss(model: _SpectralTimeMixer, x: torch.Tensor,
                   target: torch.Tensor, spectral_weight: float = 0.1) -> torch.Tensor:
    pred = _forward(model, x)
    time_loss = F.mse_loss(pred, target)
    pred_amp = torch.log1p(torch.abs(torch.fft.rfft(pred, dim=1)))
    target_amp = torch.log1p(torch.abs(torch.fft.rfft(target, dim=1)))
    frequency_loss = F.mse_loss(pred_amp, target_amp)
    return time_loss + float(spectral_weight) * frequency_loss


def _cedar_invariant_loss(model: _CedarInvariantTimeMixer, x: torch.Tensor,
                          target: torch.Tensor, discrepancy_weight: float = 0.05,
                          mmd_weight: float = 0.05) -> torch.Tensor:
    """Forecast loss plus representation MMD and group loss discrepancy."""
    pred = _forward(model, x)
    forecast = F.mse_loss(pred, target)
    partition = model.partition.to(device=pred.device)
    group_losses = []
    for value in torch.unique(partition):
        mask = partition == value
        if bool(mask.any()):
            group_losses.append((pred[..., mask] - target[..., mask]).square().mean())
    if len(group_losses) < 2:
        return forecast
    discrepancy = torch.stack(group_losses).std(unbiased=False)
    mmd = model.invariant_penalty()
    return forecast + float(mmd_weight) * mmd + float(discrepancy_weight) * discrepancy


class _LocalizedTimeFrequencyTimeMixer(nn.Module):
    """TimeMixer with a localized 24-hour time-frequency latent mixer.

    The FFT is computed independently inside each 24-hour patch. Frequency
    coefficients are mixed back into a patch-length latent sequence, then the
    seven daily patches are mixed across the week before a single raw-volume
    TimeMixer head. This is a representation block, not an output gate or a
    residual correction.
    """
    def __init__(self, args: SimpleNamespace, patch_len: int = 24,
                 spectral_hidden: int = 16):
        super().__init__()
        self.backbone = _UpstreamModel(args)
        self.patch_len = int(patch_len)
        self.n_patches = int(args.seq_len) // self.patch_len
        if self.n_patches * self.patch_len != int(args.seq_len):
            raise ValueError("localized time-frequency mixer requires divisible seq_len")
        channels = int(args.enc_in)
        freq_bins = self.patch_len // 2 + 1
        self.frequency_mixer = nn.Sequential(
            nn.Linear(2 * freq_bins, int(spectral_hidden)),
            nn.GELU(),
            nn.Linear(int(spectral_hidden), self.patch_len),
        )
        self.patch_mixer = nn.Sequential(
            nn.Linear(self.n_patches, self.n_patches),
            nn.GELU(),
            nn.Linear(self.n_patches, self.n_patches),
        )
        self.feature_fusion = nn.Linear(2 * channels, channels)
        with torch.no_grad():
            # Keep the raw-input path intact at initialization while allowing
            # the localized time-frequency representation to learn in.
            self.feature_fusion.weight.zero_()
            self.feature_fusion.weight[:, :channels].copy_(torch.eye(channels))
            # Normalize across the 275 local spectral channels; an unscaled
            # constant here would shift the standardized input mean by O(C).
            self.feature_fusion.weight[:, channels:].fill_(0.05 / channels)
            self.feature_fusion.bias.zero_()

    def localized_view(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        patches = x.reshape(b, self.n_patches, self.patch_len, c)
        spectrum = torch.fft.rfft(patches, dim=2, norm="ortho")
        spectral_tokens = torch.cat((spectrum.real, spectrum.imag), dim=2)
        spectral_tokens = spectral_tokens.permute(0, 1, 3, 2)
        patch_latent = self.frequency_mixer(spectral_tokens)
        patch_latent = patch_latent.permute(0, 1, 3, 2)
        patch_latent = patch_latent.permute(0, 2, 3, 1)
        patch_latent = self.patch_mixer(patch_latent)
        patch_latent = patch_latent.permute(0, 3, 1, 2)
        return patch_latent.reshape(b, t, c)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        local_tf = self.localized_view(x_enc)
        fused = self.feature_fusion(torch.cat((x_enc, local_tf), dim=-1))
        return self.backbone(fused, x_mark_enc, x_dec, x_mark_dec, mask)


def _localized_tf_loss(model: _LocalizedTimeFrequencyTimeMixer, x: torch.Tensor,
                       target: torch.Tensor, spectral_weight: float = 0.01) -> torch.Tensor:
    pred = model(x)
    time_loss = F.mse_loss(pred, target)
    pred_amp = torch.log1p(torch.abs(torch.fft.rfft(pred, dim=1, norm="ortho")))
    target_amp = torch.log1p(torch.abs(torch.fft.rfft(target, dim=1, norm="ortho")))
    return time_loss + float(spectral_weight) * F.mse_loss(pred_amp, target_amp)


def _apid_peak_loss(model: nn.Module, x: torch.Tensor, target: torch.Tensor,
                    peak_lambda: float = 0.15) -> torch.Tensor:
    """Preserve rare high-demand events during APID representation learning."""
    pred = _forward(model, x)
    flat = target.detach().reshape(target.size(0), -1)
    threshold = torch.quantile(flat, 0.85, dim=1, keepdim=True).view(-1, 1, 1)
    peak_mask = (target.detach() >= threshold).to(pred.dtype)
    weights = 1.0 + float(peak_lambda) * peak_mask
    return (weights * (pred - target).square()).mean()


def _scale_weighted_loss(model: nn.Module, x: torch.Tensor, target: torch.Tensor,
                         channel_weights: torch.Tensor) -> torch.Tensor:
    """Match training emphasis to UrbanEV's heteroscedastic raw-volume error."""
    pred = _forward(model, x)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    return (weights * (pred - target).square()).mean()


def _multiresolution_scale_loss(model: nn.Module, x: torch.Tensor, target: torch.Tensor,
                                channel_weights: torch.Tensor, consistency_weight: float,
                                factors: tuple[int, ...]) -> torch.Tensor:
    """Jointly supervise point forecasts and their temporal aggregates.

    UrbanEV volumes are evaluated at the original hourly resolution, but the
    same demand process has meaningful 2/3/4/6/12-hour aggregates.  The
    auxiliary term makes the representation preserve those cross-resolution
    constraints instead of only fitting each hourly endpoint independently.
    It is a training objective coupled to the multi-resolution mixer, not an
    output gate or forecast residual.
    """
    pred = _forward(model, x)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    point = (weights * (pred - target).square()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor < 2 or pred.size(1) % factor:
            continue
        pred_agg = pred.reshape(pred.size(0), pred.size(1) // factor, factor, pred.size(2)).mean(dim=2)
        target_agg = target.reshape(target.size(0), target.size(1) // factor, factor, target.size(2)).mean(dim=2)
        terms.append((weights * (pred_agg - target_agg).square()).mean())
    if not terms or float(consistency_weight) <= 0:
        return point
    return point + float(consistency_weight) * torch.stack(terms).mean()


def _robust_multiresolution_scale_loss(model: nn.Module, x: torch.Tensor, target: torch.Tensor,
                                       channel_weights: torch.Tensor,
                                       consistency_weight: float,
                                       factors: tuple[int, ...],
                                       robust_mae_weight: float = 0.1,
                                       residual_strength_reg: float = 0.0) -> torch.Tensor:
    """MRC objective with a bounded pointwise L1 component for MAE calibration."""
    pred = _forward(model, x)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    residual = pred - target
    mix = float(max(0.0, min(1.0, robust_mae_weight)))
    point_mse = (weights * residual.square()).mean()
    point_mae = (weights * residual.abs()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor < 2 or pred.size(1) % factor:
            continue
        pred_agg = pred.reshape(pred.size(0), pred.size(1) // factor, factor, pred.size(2)).mean(dim=2)
        target_agg = target.reshape(target.size(0), target.size(1) // factor, factor, target.size(2)).mean(dim=2)
        terms.append((weights * (pred_agg - target_agg).square()).mean())
    consistency = torch.stack(terms).mean() if terms else point_mse.new_zeros(())
    total = (1.0 - mix) * point_mse + mix * point_mae + float(consistency_weight) * consistency
    if float(residual_strength_reg) > 0:
        strengths = []
        for module_name in ("dynamic_cross_node", "harmonic_phase_decoder"):
            module = getattr(model, module_name, None)
            value = getattr(module, "strength", None)
            if value is not None:
                strengths.append(torch.clamp(value, min=0.0).square())
        if strengths:
            total = total + float(residual_strength_reg) * torch.stack(strengths).sum()
    return total


def _transition_weighted_multiresolution_loss(model: nn.Module, x: torch.Tensor,
                                              target: torch.Tensor,
                                              channel_weights: torch.Tensor,
                                              consistency_weight: float,
                                              factors: tuple[int, ...],
                                              robust_mae_weight: float = 0.1) -> torch.Tensor:
    """Train-only weighting for causal level/drift transitions.

    The sample weight is a deterministic z-scored magnitude of the observed
    24 h versus 72 h level drift.  It is computed from the input prefix only;
    validation selection remains the unweighted MRC objective.  This changes
    representation learning emphasis without introducing a forecast gate.
    """
    pred = _forward(model, x)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    history = x.mean(dim=2)
    t = history.size(1)
    short = history[:, -min(24, t):].mean(dim=1)
    long = history[:, -min(72, t):].mean(dim=1)
    drift = (short - long).abs()
    drift = drift / drift.detach().std(unbiased=False).clamp_min(1e-4)
    sample_weight = (1.0 + drift).detach().view(-1, 1, 1)
    residual = pred - target
    mix = float(max(0.0, min(1.0, robust_mae_weight)))
    point_mse = (sample_weight * weights * residual.square()).mean()
    point_mae = (sample_weight * weights * residual.abs()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor >= 2 and pred.size(1) % factor == 0:
            p = pred.reshape(pred.size(0), pred.size(1) // factor, factor, pred.size(2)).mean(dim=2)
            y = target.reshape(target.size(0), target.size(1) // factor, factor, target.size(2)).mean(dim=2)
            terms.append((sample_weight * weights * (p - y).square()).mean())
    consistency = torch.stack(terms).mean() if terms else point_mse.new_zeros(())
    return (1.0 - mix) * point_mse + mix * point_mae + float(consistency_weight) * consistency


def _cpr_regime_moment_loss(model: nn.Module, x: torch.Tensor, target: torch.Tensor,
                            channel_weights: torch.Tensor,
                            consistency_weight: float,
                            factors: tuple[int, ...],
                            robust_mae_weight: float = 0.1,
                            moment_weight: float = 0.05) -> torch.Tensor:
    """CPR objective with explicit future level/scale moment supervision.

    The affine state is represented causally by the model; this objective
    trains its forecast innovation to preserve the first two moments of the
    observed target window.  Unlike a future-statistics *predictor*, the
    auxiliary term is attached to the forecast itself and therefore cannot
    introduce a second inference branch or a learned selector.
    """
    pred = _forward(model, x)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    residual = pred - target
    mix = float(max(0.0, min(1.0, robust_mae_weight)))
    point_mse = (weights * residual.square()).mean()
    point_mae = (weights * residual.abs()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor < 2 or pred.size(1) % factor:
            continue
        pred_agg = pred.reshape(pred.size(0), pred.size(1) // factor, factor, pred.size(2)).mean(dim=2)
        target_agg = target.reshape(target.size(0), target.size(1) // factor, factor, target.size(2)).mean(dim=2)
        terms.append((weights * (pred_agg - target_agg).square()).mean())
    consistency = torch.stack(terms).mean() if terms else point_mse.new_zeros(())
    forecast = (1.0 - mix) * point_mse + mix * point_mae + float(consistency_weight) * consistency
    pred_mean = pred.mean(dim=1)
    target_mean = target.mean(dim=1)
    pred_scale = pred.std(dim=1, unbiased=False).clamp_min(1e-4)
    target_scale = target.std(dim=1, unbiased=False).clamp_min(1e-4)
    moment = (weights[:, 0] * (pred_mean - target_mean).square()).mean()
    moment = moment + (weights[:, 0] * (pred_scale - target_scale).square()).mean()
    return forecast + float(moment_weight) * moment


def _history_aux_multiresolution_loss(model: nn.Module, x: torch.Tensor,
                                      target: torch.Tensor, aux: torch.Tensor,
                                      channel_weights: torch.Tensor,
                                      consistency_weight: float,
                                      factors: tuple[int, ...],
                                      robust_mae_weight: float = 0.1) -> torch.Tensor:
    """Multiresolution CPR loss whose forward path consumes history auxiliaries."""
    pred = model(x, None, None, None, None, aux)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    residual = pred - target
    mix = float(max(0.0, min(1.0, robust_mae_weight)))
    point = (1.0 - mix) * (weights * residual.square()).mean() + mix * (weights * residual.abs()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor >= 2 and pred.size(1) % factor == 0:
            p = pred.reshape(pred.size(0), pred.size(1) // factor, factor, pred.size(2)).mean(dim=2)
            y = target.reshape(target.size(0), target.size(1) // factor, factor, target.size(2)).mean(dim=2)
            terms.append((weights * (p - y).square()).mean())
    return point + float(consistency_weight) * (torch.stack(terms).mean() if terms else point.new_zeros(()))


def _history_raw_mae_multiresolution_loss(model: nn.Module, x: torch.Tensor,
                                          target: torch.Tensor, aux: torch.Tensor,
                                          scaler_std: np.ndarray,
                                          consistency_weight: float,
                                          factors: tuple[int, ...]) -> torch.Tensor:
    """Metric-aligned raw-volume MAE objective for history-aware models.

    ``target`` is standardized by the train-prefix scaler, while UrbanEV's
    primary MAE is reported in raw volume units.  Multiplying the absolute
    standardized residual by the train-prefix standard deviation therefore
    gives the exact raw-unit MAE surrogate, without choosing a channel
    exponent or a validation-derived weight.  The same reconstruction is
    imposed on fixed temporal aggregates so the history transport cannot
    improve point MAE by producing an incoherent hourly trajectory.
    """
    pred = model(x, None, None, None, None, aux)
    scale = torch.as_tensor(scaler_std, device=pred.device,
                            dtype=pred.dtype).view(1, 1, -1)
    point = (scale * (pred - target).abs()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor >= 2 and pred.size(1) % factor == 0:
            p = pred.reshape(pred.size(0), pred.size(1) // factor,
                             factor, pred.size(2)).mean(dim=2)
            y = target.reshape(target.size(0), target.size(1) // factor,
                               factor, target.size(2)).mean(dim=2)
            # An aggregate in standardized coordinates has the same raw
            # scale as its constituent channel; retain exact unit alignment.
            terms.append((scale * (p - y).abs()).mean())
    if not terms or float(consistency_weight) <= 0:
        return point
    return point + float(consistency_weight) * torch.stack(terms).mean()


def _history_aux_l1_multiresolution_loss(model: nn.Module, x: torch.Tensor,
                                         target: torch.Tensor, aux: torch.Tensor,
                                         channel_weights: torch.Tensor,
                                         consistency_weight: float,
                                         factors: tuple[int, ...]) -> torch.Tensor:
    """Causal history route objective aligned with the reported MAE.

    The latent and CCLT paths are unchanged.  This is a fixed Laplace-style
    objective: point forecasts and their causal temporal aggregates are both
    supervised with absolute error.  It is intentionally a separate
    representation objective rather than a tunable MAE/MSE interpolation,
    so the experiment tests whether the hybrid representation can calibrate
    its conditional median without a gate or post-hoc correction.
    """
    pred = model(x, None, None, None, None, aux)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    point = (weights * (pred - target).abs()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor >= 2 and pred.size(1) % factor == 0:
            p = pred.reshape(pred.size(0), pred.size(1) // factor, factor, pred.size(2)).mean(dim=2)
            y = target.reshape(target.size(0), target.size(1) // factor, factor, target.size(2)).mean(dim=2)
            terms.append((weights * (p - y).abs()).mean())
    consistency = torch.stack(terms).mean() if terms else point.new_zeros(())
    return point + float(consistency_weight) * consistency


def _history_raw_scale_mse_loss(model: nn.Module, x: torch.Tensor,
                                target: torch.Tensor, aux: torch.Tensor,
                                scaler_std: np.ndarray) -> torch.Tensor:
    """Validation-only raw-volume MSE for history-aware routes."""
    pred = model(x, None, None, None, None, aux)
    scale = torch.as_tensor(scaler_std, device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    return ((pred - target) * scale).square().mean()


def _foil_invariant_multiresolution_loss(model: nn.Module, x: torch.Tensor,
                                         target: torch.Tensor,
                                         channel_weights: torch.Tensor,
                                         consistency_weight: float,
                                         factors: tuple[int, ...],
                                         robust_mae_weight: float = 0.1,
                                         invariant_weight: float = 0.1) -> torch.Tensor:
    """Train-only invariant-risk objective using causal drift environments.

    Each mini-batch is partitioned into three environments using only the
    observed-window 24-hour level drift.  The forecast loss remains the usual
    robust multi-resolution objective; a fixed variance penalty discourages
    the representation from fitting one drift regime at the expense of the
    others.  Environment labels are detached and never use future targets.
    """
    pred = _forward(model, x)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    residual = pred - target
    mix = float(max(0.0, min(1.0, robust_mae_weight)))
    point = (1.0 - mix) * (weights * residual.square()).mean()
    point = point + mix * (weights * residual.abs()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor >= 2 and pred.size(1) % factor == 0:
            p = pred.reshape(pred.size(0), pred.size(1) // factor, factor, pred.size(2)).mean(dim=2)
            y = target.reshape(target.size(0), target.size(1) // factor, factor, target.size(2)).mean(dim=2)
            terms.append((weights * (p - y).square()).mean())
    base = point + float(consistency_weight) * (torch.stack(terms).mean() if terms else point.new_zeros(()))
    if pred.size(0) < 3 or invariant_weight <= 0.0 or x.size(1) < 48:
        return base
    drift = (x[:, -24:].mean(dim=1) - x[:, -48:-24].mean(dim=1)).mean(dim=-1)
    q = torch.quantile(drift.detach(), torch.tensor((1.0 / 3.0, 2.0 / 3.0), device=drift.device))
    group_losses = []
    for mask in (drift <= q[0], (drift > q[0]) & (drift <= q[1]), drift > q[1]):
        if bool(mask.any()):
            group_losses.append((weights * residual[mask].square()).mean())
    if len(group_losses) < 2:
        return base
    stacked = torch.stack(group_losses)
    penalty = (stacked - stacked.detach().mean()).square().mean()
    return base + float(invariant_weight) * penalty


def _history_foil_invariant_multiresolution_loss(
        model: nn.Module, x: torch.Tensor, target: torch.Tensor,
        aux: torch.Tensor, channel_weights: torch.Tensor,
        consistency_weight: float, factors: tuple[int, ...],
        robust_mae_weight: float = 0.1, invariant_weight: float = 0.1,
        drift_feature_index: int = 0) -> torch.Tensor:
    """History-aware invariant-risk loss with strictly causal environments.

    ``aux`` contains only the observed companion variables for each training
    window.  The environment label is the tercile of the latest 24-hour
    change relative to the preceding 24 hours, computed from one selected
    history feature (duration by default) and averaged over aligned stations.
    Thus no target/future value is used to assign environments.  The forecast
    objective remains the ordinary scale-weighted robust MRC loss; the
    additional term equalizes the squared risk across the causal drift
    environments.  It is a training representation penalty, not a gate,
    router, expert, or prediction-time selector.
    """
    if aux is None or aux.ndim not in (3, 4):
        raise ValueError("history invariant loss requires [B,T,F] or [B,T,N,F] aux")
    pred = model(x, None, None, None, None, aux)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    residual = pred - target
    mix = float(max(0.0, min(1.0, robust_mae_weight)))
    point = (1.0 - mix) * (weights * residual.square()).mean()
    point = point + mix * (weights * residual.abs()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor >= 2 and pred.size(1) % factor == 0:
            p = pred.reshape(pred.size(0), pred.size(1) // factor,
                             factor, pred.size(2)).mean(dim=2)
            y = target.reshape(target.size(0), target.size(1) // factor,
                               factor, target.size(2)).mean(dim=2)
            terms.append((weights * (p - y).square()).mean())
    base = point + float(consistency_weight) * (
        torch.stack(terms).mean() if terms else point.new_zeros(())
    )
    # The history route uses seq_len=168 in the audited protocol.  If a
    # shorter prefix is supplied, silently skip the penalty rather than
    # introducing padded/non-causal pseudo-environments.
    if pred.size(0) < 3 or invariant_weight <= 0.0 or aux.size(1) < 48:
        return base
    index = int(drift_feature_index)
    if index < 0 or index >= aux.shape[-1]:
        raise ValueError("history invariant drift feature index out of bounds")
    selected = aux[..., index]
    if selected.ndim == 3:
        # [B,T,N] -> city-level causal drift.
        recent = selected[:, -24:].mean(dim=(1, 2))
        previous = selected[:, -48:-24].mean(dim=(1, 2))
    else:
        # [B,T] global companion variable.
        recent = selected[:, -24:].mean(dim=1)
        previous = selected[:, -48:-24].mean(dim=1)
    drift = recent - previous
    # Quantiles are computed on detached observed-history coordinates.  The
    # group assignment cannot backpropagate or depend on future labels.
    q = torch.quantile(
        drift.detach(), torch.tensor((1.0 / 3.0, 2.0 / 3.0),
                                     device=drift.device, dtype=drift.dtype))
    group_losses = []
    group_counts = []
    for mask in (drift <= q[0], (drift > q[0]) & (drift <= q[1]), drift > q[1]):
        if bool(mask.any()):
            group_losses.append((weights * residual[mask].square()).mean())
            group_counts.append(int(mask.sum()))
    if len(group_losses) < 2:
        return base
    stacked = torch.stack(group_losses)
    # Equalize environments while retaining the base forecast objective.  The
    # detached mean makes this a pure risk-dispersion penalty: it cannot lower
    # all groups by simply scaling the main loss.
    penalty = (stacked - stacked.detach().mean()).square().mean()
    return base + float(invariant_weight) * penalty


def _history_cmi_pdm_loss(
        model: nn.Module, x: torch.Tensor, target: torch.Tensor,
        aux: torch.Tensor, channel_weights: torch.Tensor,
        consistency_weight: float, factors: tuple[int, ...],
        robust_mae_weight: float = 0.1, invariant_weight: float = 0.05,
        drift_feature_index: int = 0) -> torch.Tensor:
    """Causal multi-resolution invariant PDM (CMI-PDM) objective.

    The model exposes the *ungated* forecast emitted by every PDM/FMM scale.
    We supervise the usual summed forecast and, using only observed history,
    equalize the squared risk of each scale across causal drift environments.
    This makes the PDM scales learn a common transportable representation
    instead of allowing one scale to absorb city/regime-specific residuals.
    No target value enters the environment assignment and no scale gate is
    introduced at inference.
    """
    if aux is None or aux.ndim not in (3, 4):
        raise ValueError("CMI-PDM loss requires [B,T,F] or [B,T,N,F] aux")
    pred = model(x, None, None, None, None, aux)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    residual = pred - target
    mix = float(max(0.0, min(1.0, robust_mae_weight)))
    point = (1.0 - mix) * (weights * residual.square()).mean()
    point = point + mix * (weights * residual.abs()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor >= 2 and pred.size(1) % factor == 0:
            p = pred.reshape(pred.size(0), pred.size(1) // factor,
                             factor, pred.size(2)).mean(dim=2)
            y = target.reshape(target.size(0), target.size(1) // factor,
                               factor, target.size(2)).mean(dim=2)
            terms.append((weights * (p - y).square()).mean())
    base = point + float(consistency_weight) * (
        torch.stack(terms).mean() if terms else point.new_zeros(())
    )
    scale_preds = getattr(getattr(model, "backbone", model),
                          "last_scale_forecasts", None)
    if scale_preds is None or len(scale_preds) < 2:
        return base
    if pred.size(0) < 3 or invariant_weight <= 0.0 or aux.size(1) < 48:
        return base
    index = int(drift_feature_index)
    if index < 0 or index >= aux.shape[-1]:
        raise ValueError("CMI-PDM drift feature index out of bounds")
    selected = aux[..., index]
    if selected.ndim == 3:
        recent = selected[:, -24:].mean(dim=(1, 2))
        previous = selected[:, -48:-24].mean(dim=(1, 2))
    else:
        recent = selected[:, -24:].mean(dim=1)
        previous = selected[:, -48:-24].mean(dim=1)
    drift = recent - previous
    q = torch.quantile(
        drift.detach(), torch.tensor((1.0 / 3.0, 2.0 / 3.0),
                                     device=drift.device, dtype=drift.dtype))
    masks = (drift <= q[0], (drift > q[0]) & (drift <= q[1]), drift > q[1])
    risks = []
    # Each FMM scale forecast is already in the same [B,H,N] coordinate as
    # the aggregate, so risks can be compared without an arbitrary rescale.
    for scale_pred in scale_preds:
        if scale_pred.shape != target.shape:
            continue
        scale_residual = scale_pred - target
        env = []
        for mask in masks:
            if bool(mask.any()):
                env.append((weights * scale_residual[mask].square()).mean())
        if len(env) >= 2:
            risks.append(torch.stack(env))
    # Include the summed forecast as the final member.  The penalty therefore
    # protects both the PDM streams and the deployed forecast.
    env = []
    for mask in masks:
        if bool(mask.any()):
            env.append((weights * residual[mask].square()).mean())
    if len(env) >= 2:
        risks.append(torch.stack(env))
    if not risks:
        return base
    risk_matrix = torch.stack(risks)  # [scale_or_sum, environment]
    # Center each scale separately so a global error reduction remains the
    # responsibility of the forecast objective rather than this penalty.
    penalty = (risk_matrix - risk_matrix.detach().mean(dim=1, keepdim=True)).square().mean()
    return base + float(invariant_weight) * penalty


def _future_stats_supervised_loss(model: nn.Module, x: torch.Tensor,
                                  target: torch.Tensor,
                                  channel_weights: torch.Tensor,
                                  consistency_weight: float,
                                  factors: tuple[int, ...],
                                  aux_weight: float) -> torch.Tensor:
    """Scale/MRC forecast loss plus causal train-window future-statistics loss.

    The dynamic-normalization predictor emits a per-horizon mean delta and
    log-standard-deviation delta relative to the current history window.  For
    each training window, the future target's per-zone mean and standard
    deviation are converted to the same deltas.  The auxiliary target is
    repeated over the horizon because it describes the complete 24-step future
    window, and is never used by the validation prediction path.
    """
    if not hasattr(model, "future_statistics"):
        raise TypeError("future-statistics supervision requires DynamicNormalizationTimeMixer")
    gt = _multiresolution_scale_loss(model, x, target, channel_weights,
                                     consistency_weight, factors)
    current_mean = x.mean(dim=1)
    current_std = x.std(dim=1, unbiased=False).clamp_min(1e-4)
    future_mean = target.mean(dim=1)
    future_std = target.std(dim=1, unbiased=False).clamp_min(1e-4)
    pred_mean_delta, pred_logstd_delta = model.future_statistics(x)
    target_mean_delta = (future_mean - current_mean).unsqueeze(1).expand_as(pred_mean_delta)
    target_logstd_delta = (
        torch.log(future_std) - torch.log(current_std)
    ).unsqueeze(1).expand_as(pred_logstd_delta)
    weights = channel_weights.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
    mean_loss = (weights * (pred_mean_delta - target_mean_delta).square()).mean()
    logstd_loss = (weights * (pred_logstd_delta - target_logstd_delta).square()).mean()
    return gt + float(aux_weight) * (mean_loss + logstd_loss)


def _tail_balanced_multiresolution_loss(model: nn.Module, x: torch.Tensor,
                                        target: torch.Tensor,
                                        channel_weights: torch.Tensor,
                                        consistency_weight: float,
                                        factors: tuple[int, ...]) -> torch.Tensor:
    """Train-only zone-balanced MSE/MRC objective.

    ``channel_weights`` is fitted from the training-prefix zone demand
    quantiles.  It is deliberately bounded and normalized, so large zones do
    not dominate every gradient while the forecast/scorer remain in raw kWh.
    """
    pred = _forward(model, x)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    point = (weights * (pred - target).square()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor < 2 or pred.size(1) % factor:
            continue
        pred_agg = pred.reshape(pred.size(0), pred.size(1) // factor,
                                factor, pred.size(2)).mean(dim=2)
        target_agg = target.reshape(target.size(0), target.size(1) // factor,
                                   factor, target.size(2)).mean(dim=2)
        terms.append((weights * (pred_agg - target_agg).square()).mean())
    if not terms or float(consistency_weight) <= 0:
        return point
    return point + float(consistency_weight) * torch.stack(terms).mean()


def _tariff_multiresolution_loss(model: nn.Module, x: torch.Tensor,
                                 target: torch.Tensor, tariff: torch.Tensor,
                                 channel_weights: torch.Tensor,
                                 consistency_weight: float,
                                 factors: tuple[int, ...]) -> torch.Tensor:
    """Scale/MRC loss for the known-future tariff operator."""
    pred = model(x, None, None, tariff)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    point = (weights * (pred - target).square()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor < 2 or pred.size(1) % factor:
            continue
        pred_agg = pred.reshape(pred.size(0), pred.size(1) // factor, factor, pred.size(2)).mean(dim=2)
        target_agg = target.reshape(target.size(0), target.size(1) // factor, factor, target.size(2)).mean(dim=2)
        terms.append((weights * (pred_agg - target_agg).square()).mean())
    if not terms or float(consistency_weight) <= 0:
        return point
    return point + float(consistency_weight) * torch.stack(terms).mean()


def _distilled_multiresolution_loss(model: nn.Module, x: torch.Tensor,
                                    target: torch.Tensor, teacher: torch.Tensor,
                                    channel_weights: torch.Tensor,
                                    consistency_weight: float,
                                    factors: tuple[int, ...],
                                    kd_weight: float) -> torch.Tensor:
    """Ground-truth scale/MRC objective plus frozen-teacher response loss."""
    pred = _forward(model, x)
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    point = (weights * (pred - target).square()).mean()
    terms = []
    for factor in factors:
        factor = int(factor)
        if factor < 2 or pred.size(1) % factor:
            continue
        pred_agg = pred.reshape(pred.size(0), pred.size(1) // factor, factor,
                                pred.size(2)).mean(dim=2)
        target_agg = target.reshape(target.size(0), target.size(1) // factor,
                                    factor, target.size(2)).mean(dim=2)
        terms.append((weights * (pred_agg - target_agg).square()).mean())
    gt = point + float(consistency_weight) * torch.stack(terms).mean() if terms and consistency_weight > 0 else point
    teacher = teacher.to(device=pred.device, dtype=pred.dtype)
    kd = torch.mean((pred - teacher.detach()) ** 2)
    return gt + float(kd_weight) * kd


def _component_loss(model: _PeriodAlignedTimeMixer, x: torch.Tensor,
                    target: torch.Tensor, reconstruction_weight: float = 0.2) -> torch.Tensor:
    target_trend, target_period = model.decompose(target)
    pred_trend, pred_period = model.forward_components(x)
    # Component supervision is primary.  The smaller reconstruction term
    # prevents incompatible component scales without turning this into a
    # residual-only objective.
    component = F.mse_loss(pred_trend, target_trend) + F.mse_loss(pred_period, target_period)
    reconstruction = F.mse_loss(pred_trend + pred_period, target)
    return component + float(reconstruction_weight) * reconstruction


def _predict(model: nn.Module, x: np.ndarray, device: torch.device,
             batch_size: int, forward: Callable = _forward,
             aux: np.ndarray | None = None) -> np.ndarray:
    if aux is not None:
        if len(aux) != len(x):
            raise ValueError("auxiliary prediction samples do not match x")
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(aux)),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
    else:
        loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if aux is None:
                output.append(forward(model, batch.to(device)).detach().cpu().numpy())
            else:
                bx, ba = batch
                output.append(forward(model, bx.to(device), ba.to(device)).detach().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32, copy=False)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_history(history: list[dict], path: Path) -> None:
    # The last-state checkpoint is the authoritative epoch commit.  Atomic
    # replacement lets an interrupted write be reconstructed from that state.
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            for row in history:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _resume_signature(config: dict, data: dict) -> str:
    manifest = {"config": config, "data_metadata": data.get("metadata", {}),
                "shapes": {k: list(np.shape(data[k])) for k in
                           ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")
                           if k in data}}
    manifest["aux_shapes"] = {k: list(np.shape(data[k])) for k in data
                               if k.endswith("_aux")}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()


def _fit_model(config: dict[str, Any], data: dict[str, Any], run_dir: Path,
               model_factory: Callable[[], nn.Module], forward: Callable,
               model_metadata: dict[str, Any],
               loss_fn: Callable | None = None,
               validation_loss_fn: Callable | None = None,
               aux_data: dict[str, np.ndarray] | None = None,
               aux_loss_fn: Callable | None = None,
               aux_validation_loss_fn: Callable | None = None) -> dict[str, Any]:
    """Common research training loop with resumable epoch commits.

    Model code is kept faithful to its source.  This training loop is the
    frozen UrbanEV research adapter (AdamW, constant LR, clipping), rather than
    an assertion that the source repository's entire training script matches.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    torch.set_num_threads(int(config.get("torch_threads", 2)))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    validation_only = bool(config.get("validation_only", False)) or config.get("protocol") == "research_dev"
    sample_keys = ("train_x", "train_y", "val_x", "val_y")
    if not validation_only:
        sample_keys += ("test_x", "test_y")
    arrays = {key: _as_samples(data[key], key) for key in sample_keys}
    train_x, train_y = arrays["train_x"], arrays["train_y"]
    val_x, val_y = arrays["val_x"], arrays["val_y"]
    if aux_data is not None:
        for split in ("train", "val") if validation_only else ("train", "val", "test"):
            if split not in aux_data:
                raise ValueError(f"missing auxiliary {split} samples")
            aux_data[split] = np.asarray(aux_data[split], dtype=np.float32)
            if len(aux_data[split]) != len(arrays[f"{split}_x"]):
                raise ValueError(f"auxiliary {split} sample count differs from x")
    seq_len, input_channels = train_x.shape[1:]
    target_channels = train_y.shape[2]
    pred_len = int(config.get("pred_len", train_y.shape[1]))
    if seq_len != int(config["seq_len"]):
        raise ValueError(f"train_x length {seq_len} != seq_len {config['seq_len']}")
    for split in ("train", "val") if validation_only else ("train", "val", "test"):
        x, y = arrays[f"{split}_x"], arrays[f"{split}_y"]
        if not len(x) or len(x) != len(y):
            raise ValueError(f"{split} has empty or unmatched samples")
        if x.shape[1:] != (seq_len, input_channels) or y.shape[1:] != (pred_len, target_channels):
            raise ValueError(f"{split} input/target shape differs from model dimensions")

    device_name = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = model_factory().float().to(device)
    batch_size = max(1, int(config.get("batch_size", 16)))
    generator = torch.Generator().manual_seed(seed)
    if aux_data is None:
        train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    else:
        train_dataset = TensorDataset(
            torch.from_numpy(train_x), torch.from_numpy(train_y),
            torch.from_numpy(aux_data["train"]),
        )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0,
        pin_memory=(device.type == "cuda"), generator=generator,
    )
    learning_rate = float(config.get("lr", config.get("learning_rate", 1e-3)))
    weight_decay = float(config.get("weight_decay", 1e-4))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    amp_enabled = bool(config.get("amp", device.type == "cuda")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    max_epochs = max(1, int(config.get("epochs", config.get("train_epochs", 50))))
    patience = max(1, int(config.get("patience", 8)))
    checkpoint = run_dir / "checkpoint.pth"
    last_checkpoint = run_dir / "last_state.pth"
    history_path = run_dir / "history.jsonl"
    signature = _resume_signature(config, data)
    best_val, best_epoch, stale = float("inf"), 0, 0
    start_epoch, prior_seconds = 1, 0.0
    best_state: dict | None = None
    history: list[dict[str, Any]] = []
    resumed = False
    training_complete = False

    if last_checkpoint.exists():
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if state.get("resume_signature") != signature:
            raise ValueError("last_state.pth belongs to a different config/data; use a new run directory")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["loader_rng"].cpu())
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        history = state["history"]
        best_val, best_epoch, stale = state["best_val"], state["best_epoch"], state["stale"]
        best_state = state["best_model"]
        start_epoch = state["epoch"] + 1
        prior_seconds = float(state["training_seconds"])
        training_complete = bool(state.get("training_complete", False))
        resumed = True
        _atomic_write_history(history, history_path)
        if best_state is not None:
            _atomic_torch_save({"model": best_state, "epoch": best_epoch,
                                "resume_signature": signature, "model_metadata": model_metadata}, checkpoint)
    elif checkpoint.exists() or history_path.exists():
        raise ValueError("old checkpoint/history lacks resumable last state; use a new run directory")

    started = time.perf_counter()
    if not training_complete:
        for epoch in range(start_epoch, max_epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            loss_sum, n_values = 0.0, 0
            for batch in train_loader:
                if aux_data is None:
                    batch_x, batch_y = batch
                    batch_aux = None
                else:
                    batch_x, batch_y, batch_aux = batch
                optimizer.zero_grad(set_to_none=True)
                target = batch_y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    batch_x = batch_x.to(device, non_blocking=True)
                    if batch_aux is not None:
                        batch_aux = batch_aux.to(device, non_blocking=True)
                    # Optional train-only sample/channel mixup.  The same
                    # convex permutation is applied to inputs and targets;
                    # validation and test paths never enter this branch.
                    mix_sigma = float(config.get("channel_mixup_sigma", 0.0))
                    if mix_sigma > 0.0:
                        if not 0.0 < mix_sigma <= 1.0:
                            raise ValueError("channel_mixup_sigma must be in (0,1]")
                        permutation = torch.randperm(batch_x.size(0), device=device)
                        lam = torch.rand(batch_x.size(0), 1, 1, device=device) * mix_sigma
                        batch_x = (1.0 - lam) * batch_x + lam * batch_x[permutation]
                        target = (1.0 - lam) * target + lam * target[permutation]
                    if aux_data is not None and aux_loss_fn is not None:
                        loss = aux_loss_fn(model, batch_x, target, batch_aux)
                    elif loss_fn is None:
                        pred = forward(model, batch_x) if batch_aux is None else forward(model, batch_x, batch_aux)
                        loss = criterion(pred, target)
                    else:
                        loss = loss_fn(model, batch_x, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
                scaler.scale(loss).backward()
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * target.numel()
                n_values += target.numel()
            val_pred = _predict(model, val_x, device, batch_size, forward,
                                None if aux_data is None else aux_data["val"])
            if aux_data is not None and aux_validation_loss_fn is not None:
                weighted_loss, n_batches = 0.0, 0
                with torch.no_grad():
                    for start in range(0, len(val_x), batch_size):
                        stop = min(start + batch_size, len(val_x))
                        xb = torch.from_numpy(val_x[start:stop]).to(device)
                        yb = torch.from_numpy(val_y[start:stop]).to(device)
                        ab = torch.from_numpy(aux_data["val"][start:stop]).to(device)
                        batch_loss = aux_validation_loss_fn(model, xb, yb, ab)
                        count = stop - start
                        weighted_loss += float(batch_loss.detach().cpu()) * count
                        n_batches += count
                val_loss = weighted_loss / max(n_batches, 1)
            elif aux_data is not None and aux_loss_fn is not None:
                weighted_loss, n_batches = 0.0, 0
                with torch.no_grad():
                    for start in range(0, len(val_x), batch_size):
                        stop = min(start + batch_size, len(val_x))
                        xb = torch.from_numpy(val_x[start:stop]).to(device)
                        yb = torch.from_numpy(val_y[start:stop]).to(device)
                        ab = torch.from_numpy(aux_data["val"][start:stop]).to(device)
                        batch_loss = aux_loss_fn(model, xb, yb, ab)
                        count = stop - start
                        weighted_loss += float(batch_loss.detach().cpu()) * count
                        n_batches += count
                val_loss = weighted_loss / max(n_batches, 1)
            elif validation_loss_fn is None:
                val_loss = float(np.mean((val_pred - val_y) ** 2))
            else:
                # Custom criteria may retain large intermediate tensors (for
                # example a coordinate inverse map).  Keep validation
                # memory-bounded just like the prediction path instead of
                # materializing the entire validation fold at once.
                weighted_loss, n_batches = 0.0, 0
                with torch.no_grad():
                    for start in range(0, len(val_x), batch_size):
                        stop = min(start + batch_size, len(val_x))
                        xb = torch.from_numpy(val_x[start:stop]).to(device)
                        yb = torch.from_numpy(val_y[start:stop]).to(device)
                        batch_loss = validation_loss_fn(model, xb, yb)
                        count = stop - start
                        weighted_loss += float(batch_loss.detach().cpu()) * count
                        n_batches += count
                val_loss = weighted_loss / max(n_batches, 1)
            if not np.isfinite(val_loss):
                raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
            if val_loss < best_val:
                best_val, best_epoch, stale = val_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                _atomic_torch_save({"model": best_state, "epoch": epoch,
                                    "resume_signature": signature, "model_metadata": model_metadata}, checkpoint)
            else:
                stale += 1
            row = {"epoch": epoch, "train_loss": loss_sum / n_values,
                   "val_loss": val_loss, "epoch_seconds": time.perf_counter() - epoch_started,
                   "lr": float(optimizer.param_groups[0]["lr"]), "best_epoch": best_epoch,
                   "stale_epochs": stale}
            history.append(row)
            complete = stale >= patience or epoch >= max_epochs
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_model": best_state, "best_val": best_val,
                     "best_epoch": best_epoch, "stale": stale, "history": history,
                     "training_seconds": prior_seconds + time.perf_counter() - started,
                     "training_complete": complete, "resume_signature": signature,
                     "loader_rng": generator.get_state(), "python_rng": random.getstate(),
                     "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
            _atomic_torch_save(state, last_checkpoint)
            _atomic_write_history(history, history_path)
            print(json.dumps({"model": config.get("model"), **row}, sort_keys=True), flush=True)
            if complete:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a best model")
    model.load_state_dict(best_state)
    # The test split is first accessed by the model after all model selection.
    val_pred = _predict(model, val_x, device, batch_size, forward,
                        None if aux_data is None else aux_data["val"])
    # Optional train-prefix predictions are used only by the history-only
    # expert selector.  They never enter the validation loss or checkpoint
    # selection, and are saved explicitly so that a deployment-time risk
    # calibrator can be audited without touching future labels.
    train_pred = None
    if bool(config.get("save_train_predictions", False)):
        train_pred = _predict(model, train_x, device, batch_size, forward,
                              None if aux_data is None else aux_data["train"])
    test_pred = None if validation_only else _predict(
        model, arrays["test_x"], device, batch_size, forward,
        None if aux_data is None else aux_data["test"]
    )
    elapsed = prior_seconds + time.perf_counter() - started
    metadata = {
        **model_metadata, "best_epoch": best_epoch, "best_val_mse": best_val,
        "device": str(device), "n_parameters": sum(p.numel() for p in model.parameters()),
        "epochs_completed": len(history), "max_epochs": max_epochs, "patience": patience,
        "torch_threads": torch.get_num_threads(), "resumed": resumed,
        "trainer": "UrbanEV research adapter; not source repository training-script scprc",
        "optimizer": "AdamW", "learning_rate": learning_rate, "lr_schedule": "constant",
        "weight_decay": weight_decay, "max_grad_norm": max_grad_norm,
        "loss": model_metadata.get(
            "loss_name", "custom" if loss_fn is not None else "MSE"),
        "test_evaluated_during_training": False,
        "validation_only": validation_only,
        "drop_last": False, "amp": amp_enabled,
    }
    (run_dir / "training.json").write_text(
        json.dumps({"seconds": elapsed, **metadata}, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"val_pred": val_pred, "test_pred": test_pred,
            "train_pred": train_pred, "train_true": train_y if train_pred is not None else None,
            "history": history,
            "training_seconds": elapsed, "checkpoint": str(checkpoint),
            "last_checkpoint": str(last_checkpoint), "metadata": metadata}


def _fit_capacity_normalized(config: dict[str, Any], data: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Train CPR on volume/capacity and decode by fixed station capacity.

    Capacity is static and available before prediction; this is a structural
    target coordinate, not a learned gate or a post-hoc residual.
    """
    mean = np.asarray(data["scaler_mean"], dtype=np.float32)
    std = np.asarray(data["scaler_std"], dtype=np.float32)
    cap = np.maximum(np.asarray(data["capacity"], dtype=np.float32), 1e-6)
    train_raw = np.asarray(data["raw_splits"]["train"], dtype=np.float32) * std[None, :] + mean[None, :]
    q_train = train_raw / cap[None, :]
    qmean = q_train.mean(0); qstd = np.maximum(q_train.std(0), 1e-6)
    # A fixed convex coordinate blend preserves both utilization dynamics and
    # the raw-load scale.  This is a target reparameterization, not a learned
    # gate or an ensemble: the inverse map is analytic at decode time.
    alpha = float(config.get("capacity_coordinate_alpha", 1.0))
    alpha = min(max(alpha, 0.0), 1.0)
    denom = alpha * std / (cap * qstd) + (1.0 - alpha)
    denom = np.maximum(denom, 1e-6)
    # Express the blended coordinate as z = denom * y_std + intercept.
    # q_norm contains the per-node raw mean term as well as qmean.
    intercept = alpha * ((mean / cap) - qmean) / qstd
    transformed = dict(data)
    for split in ("train", "val", "test"):
        for suffix in ("x", "y"):
            key = f"{split}_{suffix}"
            if key not in data: continue
            raw = np.asarray(data[key], dtype=np.float32) * std[None, None, :] + mean[None, None, :]
            ystd = (raw - mean[None, None, :]) / std[None, None, :]
            qnorm = (raw / cap[None, None, :] - qmean[None, None, :]) / qstd[None, None, :]
            transformed[key] = (alpha * qnorm + (1.0 - alpha) * ystd).astype(np.float32)
    # The inner adapter expects a scaler pair only for metadata; all arrays
    # above are already in the blended coordinate. Keep a neutral scaler.
    transformed["scaler_mean"] = np.zeros_like(mean, dtype=np.float32)
    transformed["scaler_std"] = np.ones_like(std, dtype=np.float32)
    transformed["raw_splits"] = {}
    for k, v in data["raw_splits"].items():
        raw = np.asarray(v, dtype=np.float32)
        ystd = (raw - mean[None, :]) / std[None, :]
        qnorm = (raw / cap[None, :] - qmean[None, :]) / qstd[None, :]
        transformed["raw_splits"][k] = (alpha * qnorm + (1.0 - alpha) * ystd).astype(np.float32)
    inner = dict(config); inner["model"] = "timemixer_ci"; inner["implementation"] = str(config.get("implementation", "capacity")) + "-inner"
    result = fit_predict(inner, transformed, run_dir / "capacity_inner")
    for key in ("val_pred", "test_pred", "train_pred"):
        if key not in result or result[key] is None: continue
        z = np.asarray(result[key], dtype=np.float32)
        # z = alpha*q_norm + (1-alpha)*y_std; solve for y_std and restore raw volume.
        ystd = (z - intercept[None, None, :]) / denom[None, None, :]
        volume = ystd * std[None, None, :] + mean[None, None, :]
        result[key] = ((volume - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    result.setdefault("metadata", {})["capacity_normalized_target"] = True
    result.setdefault("metadata", {})["capacity_coordinate_alpha"] = alpha
    return result


def _fit_log_capacity_normalized(config: dict[str, Any], data: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Capacity-normalized log utilization target with analytic volume decode."""
    mean = np.asarray(data["scaler_mean"], dtype=np.float32)
    std = np.asarray(data["scaler_std"], dtype=np.float32)
    cap = np.maximum(np.asarray(data["capacity"], dtype=np.float32), 1e-6)
    raw_train = np.asarray(data["raw_splits"]["train"], dtype=np.float32) * std[None, :] + mean[None, :]
    z_train = np.log1p(np.maximum(raw_train, 0.0) / cap[None, :])
    zmean = z_train.mean(0)
    zstd = np.maximum(z_train.std(0), 1e-6)
    transformed = dict(data)
    for split in ("train", "val", "test"):
        for suffix in ("x", "y"):
            key = f"{split}_{suffix}"
            if key not in data:
                continue
            raw = np.asarray(data[key], dtype=np.float32) * std[None, None, :] + mean[None, None, :]
            transformed[key] = ((np.log1p(np.maximum(raw, 0.0) / cap[None, None, :]) - zmean[None, None, :]) /
                                zstd[None, None, :]).astype(np.float32)
    transformed["scaler_mean"] = np.zeros_like(mean, dtype=np.float32)
    transformed["scaler_std"] = np.ones_like(std, dtype=np.float32)
    transformed["raw_splits"] = {}
    for k, v in data["raw_splits"].items():
        raw = np.asarray(v, dtype=np.float32)
        transformed["raw_splits"][k] = ((np.log1p(np.maximum(raw, 0.0) / cap[None, :]) - zmean[None, :]) /
                                         zstd[None, :]).astype(np.float32)
    inner = dict(config)
    inner["model"] = "timemixer_ci"
    inner["implementation"] = str(config.get("implementation", "log-capacity")) + "-inner"
    result = fit_predict(inner, transformed, run_dir / "log_capacity_inner")
    for key in ("val_pred", "test_pred", "train_pred"):
        if key not in result or result[key] is None:
            continue
        z = np.asarray(result[key], dtype=np.float32) * zstd[None, None, :] + zmean[None, None, :]
        volume = np.expm1(np.clip(z, -20.0, 20.0)) * cap[None, None, :]
        result[key] = ((volume - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    result.setdefault("metadata", {})["log_capacity_normalized_target"] = True
    return result


def fit_predict(config: dict[str, Any], data: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Fit the unchanged official model using the UrbanEV research adapter.

    Targets already encode endpoint lead time or a continuous output horizon.
    Lead time never causes an extra shift in this model interface.  Applying
    TimeMixer on UrbanEV is an UrbanEV baseline scprc; it does not claim
    scprc of the paper's ETT benchmark tables.
    """
    channels = np.shape(data["train_x"])[2]
    pred_len = int(config.get("pred_len", np.shape(data["train_y"])[1]))
    args = _config_namespace(config, channels, pred_len)
    model_name = str(config.get("model", "")).lower()
    if model_name in {"capacity_norm_timemixer", "capacity_normalized_timemixer"}:
        return _fit_capacity_normalized(config, data, run_dir)
    if model_name in {"log_capacity_timemixer", "log_capacity_normalized_timemixer"}:
        return _fit_log_capacity_normalized(config, data, run_dir)
    cmpd = model_name in {"cmpd_timemixer", "causal_moment_timemixer", "moment_decimation_timemixer"}
    causal_ema = model_name == "causal_ema_timemixer"
    causal_daily = model_name in {"causal_daily_timemixer", "phase_causal_timemixer"}
    causal_fir = model_name in {"causal_fir_timemixer", "learned_causal_decomp_timemixer"}
    robust_huber = model_name in {"timemixer_robust_huber", "robust_huber_timemixer"}
    phase_cycle = model_name in {"phase_cycle_timemixer", "phase_cycle_pos_timemixer"}
    phase_latent_decomp = model_name in {"phase_latent_decomp_timemixer", "cpr_phase_latent_decomp"}
    joint_phase_region = model_name == "joint_phase_region_timemixer"
    tariff = model_name in {"tariff_timemixer", "tariff_fusion_timemixer", "tariff_cross_timemixer", "tariff_state_timemixer", "tariff_flow_timemixer", "tariff_coordinate_timemixer", "tariff_scale_timemixer", "tariff_context_timemixer", "tariff_bilinear_timemixer", "tariff_bdc_timemixer"}
    tariff_fusion = model_name == "tariff_fusion_timemixer"
    tariff_cross = model_name == "tariff_cross_timemixer"
    tariff_state = model_name == "tariff_state_timemixer"
    tariff_flow = model_name == "tariff_flow_timemixer"
    tariff_coordinate = model_name == "tariff_coordinate_timemixer"
    tariff_scale = model_name == "tariff_scale_timemixer"
    tariff_context = model_name in {"tariff_context_timemixer", "tariff_bdc_timemixer"}
    tariff_bilinear = model_name == "tariff_bilinear_timemixer"
    tariff_bdc = model_name == "tariff_bdc_timemixer"
    regime_state = model_name in {"regime_state_timemixer", "cpr_regime_state_timemixer", "causal_regime_cpr_timemixer"}
    post_regime_state = model_name in {"post_regime_state_timemixer", "cpr_post_regime_state_timemixer"}
    if regime_state:
        args.use_regime_state = 1
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone + shared causal level/drift/volatility regime-state coordinates before PDM"
    elif post_regime_state:
        args.use_post_regime_state = 1
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone + shared causal level/drift/volatility regime-state coordinates between PDM and FMM"
    elif cmpd:
        from scprc.models.cmpd import fit_cmpd_bank
        args.down_sampling_method = "cmpd"
        args.cmpd_order = int(config.get("cmpd_order", 1))
        args.cmpd_kernel_size = int(config.get("cmpd_kernel_size", 5))
        args.cmpd_regularization = float(config.get("cmpd_regularization", 0.05))
        args.cmpd_kernels = [kernel.tolist() for kernel in fit_cmpd_bank(
            np.asarray(data["raw_splits"]["train"], dtype=np.float32),
            int(args.down_sampling_layers), factor=int(args.down_sampling_window),
            kernel_size=args.cmpd_kernel_size, order=args.cmpd_order,
            regularization=args.cmpd_regularization,
        )]
    routed = model_name == "routed_timemixer"
    period_aligned = model_name == "period_timemixer"
    wavelet = model_name == "wavelet_timemixer"
    spectral = model_name == "spectral_timemixer"
    localized_tf = model_name == "localized_tf_timemixer"
    dft = model_name == "dft_timemixer"
    temporal_dft = model_name == "temporal_dft_timemixer"
    adaptive_latent_spectral = model_name == "adaptive_latent_spectral_timemixer"
    horizon_basis = model_name == "horizon_basis_timemixer"
    patch_token = model_name == "patch_timemixer"
    latent_temporal = model_name == "latent_temporal_timemixer"
    mrti = model_name == "mrti_timemixer"
    phase_mrti = model_name == "phase_mrti_timemixer"
    event_mrti = model_name == "event_mrti_timemixer"
    apid = model_name == "apid_timemixer"
    innovation = model_name == "innovation_timemixer"
    crps_decoder = model_name in {"crpsd_timemixer", "crps_decoder_timemixer"}
    shared_query = model_name in {"sfqd_timemixer", "shared_query_timemixer"}
    reconciled_future = model_name in {"rcfd_timemixer", "reconciled_future_timemixer"}
    equilibrium = model_name in {"cred_timemixer", "equilibrium_timemixer"}
    transport = model_name in {"ortd_timemixer", "transport_timemixer"}
    phase_state = model_name in {"phase_state_timemixer", "ps001_timemixer"}
    phase_grid = model_name in {"phase_grid_timemixer", "cross_resolution_phase_grid_timemixer", "crpg_timemixer", "phase_grid_fusion_timemixer", "crpg_fusion_timemixer"}
    phase_grid_fusion = model_name in {"phase_grid_fusion_timemixer", "crpg_fusion_timemixer"}
    phase_token_fmm = model_name in {"phase_token_fmm_timemixer", "phase_token_timemixer", "cpr_phase_token_timemixer"}
    nlinear_fmm = model_name in {"nlinear_fmm_timemixer", "nlinear_fine_fmm_timemixer", "cpr_nlinear_fmm_timemixer"}
    regime_trajectory_fmm = model_name in {"regime_trajectory_fmm_timemixer", "cpr_regime_trajectory_timemixer"}
    phase_resolution = model_name in {"mpr_timemixer", "phase_resolution_timemixer"}
    bilateral_coupling = model_name in {"bdc_timemixer", "bilateral_coupled_timemixer", "tariff_bdc_timemixer"}
    scale_bilateral = model_name in {"sbdc_timemixer", "scale_bilateral_timemixer"}
    dcs_fmm = model_name in {"dcs_fmm_timemixer", "dcs_timemixer", "dual_conservative_semigroup_timemixer"}
    ntf_fmm = model_name in {"ntf_fmm_timemixer", "ntf_timemixer", "nested_temporal_frame_timemixer"}
    dct_fmm = model_name in {"dct_fmm_timemixer", "dct_timemixer", "drift_conditioned_fmm_timemixer"}
    cclt_innovation = model_name in {"cclt_innovation_timemixer", "cpr_cclt_innovation"}
    cclt_frozen = model_name in {"cclt_frozen_residual_timemixer", "cpr_cclt_frozen_residual", "cclt_contrast_frozen_timemixer", "cclt_trajectory_frozen_timemixer"}
    cclt_basis = model_name in {"cclt_basis_residual_timemixer", "cpr_cclt_basis_residual"}
    cclt_extended_basis = model_name in {"cclt_extended_basis_timemixer", "cpr_cclt_extended_basis"}
    cclt_joint = model_name in {"cclt_joint_residual_timemixer", "cpr_cclt_joint_residual"}
    cclt_recurrent = model_name in {"cclt_recurrent_residual_timemixer", "cpr_cclt_recurrent_residual"}
    cclt_contrast = model_name in {"cclt_contrast_frozen_timemixer"}
    cclt_trajectory = model_name in {"cclt_trajectory_frozen_timemixer"}
    cldst = model_name in {"cldst_timemixer", "level_drift_transport_timemixer", "timemixer_cldst"}
    cclt_history = model_name in {"cclt_timemixer", "cpr_cclt_timemixer", "cpr_cclt"} or cclt_innovation or cclt_frozen or cclt_basis or cclt_extended_basis or cclt_joint or cclt_recurrent
    latent_multimodal = model_name in {"latent_history_timemixer", "cpr_latent_multimodal_timemixer"}
    e2e_multimodal = model_name in {"e2e_multimodal_timemixer", "cpr_e2e_multimodal_timemixer", "multimodal_token_backbone_timemixer"}
    grouped_multimodal = model_name in {"grouped_e2e_multimodal_timemixer", "cpr_grouped_multimodal_timemixer"}
    e2e_multimodal = e2e_multimodal or grouped_multimodal
    hybrid_latent_cclt = model_name in {"hybrid_latent_cclt_timemixer", "cpr_hybrid_latent_cclt", "timemixer_hybrid_robust_huber"}
    phase_region_state = model_name in {
        "phase_region_state_timemixer",
        "cpr_phase_region_state_timemixer",
    }
    node_multimodal = model_name in {"node_multimodal_timemixer", "cpr_node_multimodal_timemixer", "direct_token_cpr_timemixer"}
    multimodal_history = (model_name in {"multimodal_timemixer", "cpr_multimodal_timemixer", "cpr_dx_multimodal_timemixer"} or cclt_history or hybrid_latent_cclt or e2e_multimodal or phase_region_state) and not node_multimodal
    dynamic_scale = model_name in {"timemixer_tail_conditional", "tail_conditional_timemixer", "dscfd_timemixer"}
    horizon_convolution = model_name in {"horizon_conv_timemixer", "hscd_timemixer"}
    conditional_spatial = model_name in {"conditional_spatial_timemixer", "edge_contrast_timemixer", "r3_spatial_timemixer"}
    phase_conditioned = model_name in {"phase_conditioned_timemixer", "timemixer_phase_hybrid", "phase_hybrid_timemixer"}
    phase_kd = model_name in {"phase_kd_timemixer", "timemixer_phase_kd"}
    dynamic_normalization = model_name in {"dynamic_norm_timemixer", "evomsn_timemixer", "future_stats_norm_timemixer", "future_stats_supervised_timemixer"}
    future_stats_supervised = model_name == "future_stats_supervised_timemixer"
    cpr_regime = model_name in {"cpr_regime_timemixer", "causal_affine_regime_timemixer"}
    cpr_periodic_state = model_name in {"cpr_periodic_state_timemixer", "periodic_state_timemixer", "causal_periodic_state_timemixer"}
    fan_frequency = model_name in {"fan_timemixer", "fan_cpr_timemixer", "frequency_adaptive_timemixer"}
    fan_identity = model_name in {"fan_identity_timemixer", "fan_preserve_timemixer", "cpr_fan_identity_timemixer"}
    pori_minimal = model_name in {"pori_timemixer", "pori_minimal_timemixer", "pori_equal_token_timemixer"}
    cast = model_name in {"cast_timemixer", "cast_cpr_timemixer"}
    cedar = model_name in {"cedar_timemixer", "cpr_cedar_timemixer"}
    daily_residual = model_name in {"daily_residual_timemixer", "cpr_daily_residual_timemixer"}
    phase_residual_transport = model_name in {"phase_residual_transport_timemixer", "phase_transport_timemixer"}
    phase_lowrank_transport = model_name in {"phase_lowrank_transport_timemixer", "phase_rank4_transport_timemixer"}
    phase_error_aware_transport = model_name in {"phase_error_aware_transport_timemixer", "error_aware_phase_transport_timemixer"}
    scale_lifting = model_name in {"scale_lifting_timemixer", "lifting_timemixer", "coarse_detail_lifting_timemixer"}
    static_chart = model_name in {"static_chart_timemixer", "zone_chart_timemixer", "node_chart_timemixer"}
    cross_scale_ssm = model_name in {"cross_scale_ssm_timemixer", "cross_scale_ssm_v2_timemixer", "ssm_timemixer", "multiscale_ssm_timemixer"}
    pdm_cmamba = model_name in {"pdm_cmamba_timemixer", "cmamba_timemixer"}
    shared_basis_transport = model_name == "shared_basis_transport_timemixer"
    endpoint_bridge = model_name in {"endpoint_bridge_timemixer", "cpr_endpoint_bridge_timemixer", "level_drift_endpoint_timemixer"}
    cointegration = model_name in {"cointegration_timemixer", "common_spread_timemixer", "graph_cointegration_timemixer"}
    if static_chart:
        args.static_features = np.asarray(data["static_features"], dtype=np.float32)
    if conditional_spatial or cointegration:
        args.urban_adj = np.asarray(data["adj"], dtype=np.float32)
    if innovation:
        args.use_innovation_exchange = 1
        args.urban_adj = np.asarray(data["adj"], dtype=np.float32)
    period_alignment = bool(config.get("period_alignment", 1))
    component_supervision = bool(config.get("component_supervision", 1))
    reconstruction_weight = float(config.get("reconstruction_weight", 0.2))
    tariff_aux = None
    cast_metadata = None
    if pori_minimal:
        args.use_pori_minimal = 1
        args.pori_token_count = int(config.get("pori_token_count", 21))
        factory = lambda: _UpstreamModel(args)
        source = ("official TimeMixer PDM with equal-token coarse-to-fine patch recursion "
                  "and ungated four-segment FMM readout; no order auxiliary, router, "
                  "expert, or post-hoc forecast selector")
    elif fan_identity:
        factory = lambda: _FANIdentityTimeMixer(
            args, freq_topk=int(config.get("fan_freq_topk", 4)))
        source = ("identity-preserving FAN frequency coordinate in parallel with a single CPR-TimeMixer "
                  "backbone; zero-initialized shared Fourier forecast, no gate/router/expert")
    elif fan_frequency:
        factory = lambda: _FANTimeMixer(
            args, freq_topk=int(config.get("fan_freq_topk", 4)))
        source = ("FAN frequency-residual normalization before a single CPR-TimeMixer "
                  "backbone; dominant causal Fourier components are forecast by a "
                  "shared MLP and added without gate/router/expert")
    elif daily_residual:
        factory = lambda: _DailyResidualTimeMixer(
            args, raw_input=bool(config.get("daily_residual_raw_input", False)))
        source = ("single CPR-TimeMixer on causal 24-hour innovation coordinates "
                  "with fixed previous-day reconstruction; no gate/router/expert/post-hoc residual")
    elif cedar:
        args.capture_shared_latent = 1
        static_context = np.asarray(data["static_features"], dtype=np.float32)
        factory = lambda: _CedarInvariantTimeMixer(args, static_context)
        source = ("single CPR-TimeMixer with shared pre-PDM latent and fixed "
                  "spatial polynomial-MMD/loss-discrepancy regularization; "
                  "no gate/router/expert/post-hoc residual")
    elif cast:
        checkpoint_path = config.get("cast_backbone_checkpoint") or config.get("latent_backbone_checkpoint")
        if not checkpoint_path:
            raise ValueError("CAST requires cast_backbone_checkpoint")
        args.use_norm = 0
        def _make_cast():
            return _CASTTimeMixer(
                args, checkpoint_path=checkpoint_path,
                state_transport=bool(config.get("cast_state_transport", True)),
                preserve_input=bool(config.get("cast_preserve_input", False)),
                long_scale=bool(config.get("cast_long_scale", False)))
        factory = _make_cast
        source = ("single CPR-TimeMixer with causal affine-state canonicalization "
                  "and analytic level/scale reconstruction; no gate/router/expert")
        cast_metadata = {
            "statistics": "causal trailing 24h/168h mean and standard deviation",
            "state": "level-shift, log-scale-ratio, one-day drift, normalized level",
            "reconstruction": "forecast_shape * s24 + m24",
            "state_transport": bool(config.get("cast_state_transport", True)),
            "preserve_input": bool(config.get("cast_preserve_input", False)),
            "long_scale": bool(config.get("cast_long_scale", False)),
            "learned_gate": False, "router": False, "experts": False,
            "posthoc_residual": False,
        }
    elif e2e_multimodal:
        history_aux, static_context, history_meta = _build_history_multimodal_aux(config, data)
        if not history_meta["node_aligned"]:
            raise ValueError("end-to-end multimodal TimeMixer requires node-aligned history variables")
        args.enc_in = channels
        def _make_e2e_multimodal():
            if grouped_multimodal:
                names = history_meta["feature_names"]
                groups = [
                    [i for i, name in enumerate(names)
                     if str(name).split("_lag", 1)[0] in {"occupancy", "duration"}],
                    [i for i, name in enumerate(names)
                     if str(name).split("_lag", 1)[0] in {"electricity_price", "service_price"}],
                    [i for i, name in enumerate(names)
                     if str(name).startswith("weather_")],
                ]
                model = _GroupedMultimodalTimeMixer(
                    args, history_meta["feature_dim"], static_context, groups,
                    hidden=int(config.get("multimodal_hidden", 32)),
                    group_rank=int(config.get("multimodal_group_rank", 8)))
            else:
                model = _EndToEndMultimodalTimeMixer(
                    args, history_meta["feature_dim"], static_context,
                    hidden=int(config.get("multimodal_hidden", 32)))
            checkpoint_path = config.get("e2e_backbone_checkpoint") or config.get("latent_backbone_checkpoint")
            if checkpoint_path:
                payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
                state = payload.get("model", payload)
                target_state = model.backbone.state_dict()
                compatible = {key: value for key, value in state.items()
                              if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)}
                model.backbone.load_state_dict(compatible, strict=False)
                if len(compatible) < int(0.9 * max(len(target_state), 1)):
                    raise ValueError("e2e multimodal checkpoint has too few compatible CPR parameters")
            return model
        factory = _make_e2e_multimodal
        source = (
            "single end-to-end CPR-TimeMixer with physical-group shared low-rank "
            "multimodal history/geometry representation before PDM at every scale; "
            "full backbone trainable; no gate/router/expert/posthoc selector"
            if grouped_multimodal else
            "single end-to-end CPR-TimeMixer with shared multimodal history/geometry "
            "representation before PDM at every scale; full backbone trainable; "
            "no gate/router/expert/posthoc selector"
        )
    elif phase_region_state:
        history_aux, static_context, history_meta = _build_history_multimodal_aux(config, data)
        if not history_meta["node_aligned"]:
            raise ValueError("phase-region state requires node-aligned history variables")
        args.enc_in = channels
        checkpoint_path = config.get("phase_region_backbone_checkpoint") or config.get("latent_backbone_checkpoint")
        if not checkpoint_path:
            raise ValueError("phase-region state requires phase_region_backbone_checkpoint")

        def _make_phase_region_state():
            model = _PhaseRegionStateTensorTimeMixer(
                args, history_meta["feature_dim"],
                freeze_backbone=bool(config.get("phase_region_freeze_backbone", True)),
                cycles=int(config.get("phase_region_cycles", 7)),
                phases=int(config.get("phase_region_phases", 24)),
            )
            payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
            state = payload.get("model", payload)
            target_state = model.backbone.state_dict()
            compatible = {
                key: value for key, value in state.items()
                if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
            }
            model.backbone.load_state_dict(compatible, strict=False)
            required_keys = [key for key in target_state
                             if not (bool(getattr(args, "use_mrti", False)) and
                                     key.startswith("mrti."))]
            if len([key for key in compatible if key in required_keys]) < int(0.9 * max(len(required_keys), 1)):
                raise ValueError("phase-region backbone checkpoint has too few compatible CPR parameters")
            return model

        factory = _make_phase_region_state
        source = (
            "single CPR-DX backbone with complete causal phase-region state tensor, "
            "shared within-day/cross-day transitions and fixed multiscale transport "
            "before PDM; no gate/router/expert/posthoc selector"
        )
    elif hybrid_latent_cclt:
        history_aux, static_context, history_meta = _build_history_multimodal_aux(config, data)
        if not history_meta["node_aligned"]:
            raise ValueError("hybrid latent CCLT requires node-aligned history variables")
        args.enc_in = channels
        if model_name == "timemixer_hybrid_robust_huber":
            args.decomp_method = "robust_huber"
        args.urban_adj = np.asarray(data["adj"], dtype=np.float32)
        args.history_latent_activation = str(config.get("history_latent_activation", "tanh"))
        args.history_include_volume_state = bool(config.get("history_include_volume_state", False))
        def _make_hybrid_latent_cclt():
            history_names = [str(v) for v in history_meta["feature_names"]]
            base_prefixes = tuple(str(v).strip().lower() for v in config.get(
                "hybrid_base_history_prefixes", ("duration",)))
            selected_indices = np.asarray([
                i for i, name in enumerate(history_names)
                if name.lower().startswith(base_prefixes)
            ], dtype=np.int64)
            if selected_indices.size == 0:
                raise ValueError("hybrid base history selected no features")
            use_state_transport = bool(config.get("hybrid_state_transport", False))
            state_dim = 4 if use_state_transport else 0
            # The latent encoder receives the selected companion variables
            # followed by four deterministic, train-prefix-calibrated volume
            # state coordinates.  Indices are local to this compact tensor.
            base_indices = selected_indices
            state_stats = None
            if use_state_transport:
                train_series = np.asarray(data["raw_splits"]["train"], dtype=np.float64)
                center = np.median(train_series, axis=0)
                mad = np.median(np.abs(train_series - center[None, :]), axis=0) * 1.4826
                state_stats = (center.astype(np.float32), np.maximum(mad, 0.05).astype(np.float32))
            model = _HybridLatentCCLTTimeMixer(
                args, int(base_indices.size + state_dim), static_context,
                adjacency=np.asarray(data["adj"], dtype=np.float32),
                hidden=int(config.get("multimodal_hidden", 32)),
                graph_order=int(config.get("cclt_graph_order", 1)),
                graph_contrast=bool(config.get("cclt_graph_contrast", False)),
                latent_scale_specific=bool(config.get("latent_scale_specific", False)),
                phase_coupling=bool(config.get("phase_cclt_coupling", False)),
                innovation_projection=bool(config.get("hybrid_innovation_projection", False)),
                basis_residual=bool(config.get("hybrid_basis_residual", False)),
                extended_basis=bool(config.get("hybrid_extended_basis", False)),
                cumulative_residual=bool(config.get("hybrid_cumulative_residual", False)),
                multi_kernel=bool(config.get("hybrid_multi_kernel", False)),
                base_feature_indices=base_indices,
                graph_input_diffusion=bool(config.get("hybrid_graph_input_diffusion", False)),
                graph_input_order=int(config.get("hybrid_graph_input_order", 1)),
                graph_input_mode=str(config.get("hybrid_graph_input_mode", "midpoint")),
                state_transport=use_state_transport, state_stats=state_stats,
                latent_node_mix=bool(config.get("hybrid_latent_node_mix", False)),
                latent_node_mix_rank=int(config.get("hybrid_latent_node_mix_rank", 8)))
            if bool(config.get("hybrid_whitened_state", False)):
                # Fit a fixed PCA-whitening map on the training prefix only.
                # The covariance is accumulated in chunks to avoid materializing
                # the full [windows,time,nodes,features] matrix in memory.
                names = [str(v) for v in history_meta["feature_names"]]
                prefixes = tuple(str(v).strip().lower() for v in config.get(
                    "whiten_history_prefixes",
                    ("occupancy", "electricity_price", "service_price", "weather")))
                indices = np.asarray([
                    i for i, name in enumerate(names)
                    if name.lower().startswith(prefixes)
                ], dtype=np.int64)
                if indices.size == 0:
                    raise ValueError("hybrid_whitened_state selected no history features")
                train_aux = np.asarray(history_aux["train"], dtype=np.float64)
                count = 0
                mean = np.zeros(indices.size, dtype=np.float64)
                for start in range(0, train_aux.shape[0], 32):
                    chunk = train_aux[start:start + 32].mean(axis=2)[..., indices].reshape(-1, indices.size)
                    mean += chunk.sum(axis=0)
                    count += chunk.shape[0]
                mean /= max(count, 1)
                covariance = np.zeros((indices.size, indices.size), dtype=np.float64)
                for start in range(0, train_aux.shape[0], 32):
                    chunk = train_aux[start:start + 32].mean(axis=2)[..., indices].reshape(-1, indices.size)
                    centered = chunk - mean
                    covariance += centered.T @ centered
                covariance /= max(count - 1, 1)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                order = np.argsort(eigenvalues)[::-1]
                rank = min(int(config.get("hybrid_whiten_rank", 8)), indices.size)
                keep = order[:rank]
                # Whitening is fixed and train-prefix-only; the small floor
                # avoids amplifying numerical null directions.
                scale = np.sqrt(np.maximum(eigenvalues[keep], 1e-5))
                components = eigenvectors[:, keep] / scale[None, :]
                model.configure_whitened_state(mean, components, indices)
            if bool(config.get("latent_ols_init", False)):
                aux_train = np.asarray(history_aux["train"], dtype=np.float64)[..., selected_indices]
                volume_train = np.asarray(data["train_x"], dtype=np.float64)
                if use_state_transport:
                    with torch.no_grad():
                        state_train = model._state_coordinates(
                            torch.as_tensor(volume_train, dtype=torch.float32)).cpu().numpy()
                    aux_train = np.concatenate((aux_train, state_train.astype(np.float64)), axis=-1)
                feature_dim = int(aux_train.shape[-1])
                static_np = np.asarray(static_context, dtype=np.float64)
                raw_dim = int(model.latent.raw_dim)
                coefficients = np.zeros((channels, raw_dim), dtype=np.float64)
                eye = np.eye(raw_dim, dtype=np.float64)
                for node in range(channels):
                    hist = aux_train[:, :, node, :].reshape(-1, feature_dim)
                    st = np.repeat(static_np[node:node + 1], hist.shape[0], axis=0)
                    inter = hist[:, :, None] * static_np[node][None, None, :]
                    parts = [hist]
                    if model.latent.include_volume_state:
                        vol = volume_train[:, :, node].reshape(-1, 1)
                        parts.extend((vol, st, inter.reshape(hist.shape[0], -1),
                                      vol * static_np[node][None, :]))
                    else:
                        parts.extend((st, inter.reshape(hist.shape[0], -1)))
                    raw = np.concatenate(parts, axis=1)
                    target = volume_train[:, :, node].reshape(-1)
                    coefficients[node] = np.linalg.solve(raw.T @ raw + eye, raw.T @ target)
                coefficients *= 0.25
                with torch.no_grad():
                    if model.latent.scale_specific:
                        model.latent.scale_projection[:, :, 0, :].copy_(
                            torch.as_tensor(coefficients, dtype=model.latent.scale_projection.dtype).unsqueeze(0).expand(model.latent.num_scales, -1, -1))
                    else:
                        model.latent.node_projection[:, 0, :].copy_(
                            torch.as_tensor(coefficients, dtype=model.latent.node_projection.dtype))
            checkpoint_path = config.get("latent_backbone_checkpoint") or config.get("cclt_backbone_checkpoint")
            if not checkpoint_path:
                raise ValueError("hybrid latent CCLT requires latent_backbone_checkpoint")
            payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
            state = payload.get("model", payload)
            missing, unexpected = model.backbone.load_state_dict(state, strict=False)
            missing = [name for name in missing if not (
                name.startswith("precision_fusion_") or
                name.startswith("reconciled_aggregation_") or
                name == "reconciled_normal_inverse"
            )]
            if missing or unexpected:
                raise ValueError(f"hybrid CPR backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}")
            # Optional precision-fusion experiments start from a previously
            # trained single-backbone CPR checkpoint.  This is only an
            # initialization for the fixed FMM projection; the projection
            # itself is computed analytically from train-prefix residuals.
            fusion_init = config.get("precision_fusion_calibration_checkpoint")
            if bool(config.get("precision_fusion", False)) and fusion_init:
                fusion_payload = torch.load(Path(fusion_init), map_location="cpu", weights_only=False)
                fusion_state = fusion_payload.get("model", fusion_payload)
                compatible = {key: value for key, value in fusion_state.items()
                              if key in model.state_dict() and tuple(value.shape) == tuple(model.state_dict()[key].shape)}
                model.load_state_dict(compatible, strict=False)
                if len(compatible) < int(0.9 * max(len(model.state_dict()), 1)):
                    raise ValueError("precision fusion checkpoint has too few compatible hybrid parameters")
            if bool(config.get("hybrid_unfreeze_all", False)):
                # Full end-to-end adaptation audit: every official CPR-DX
                # embedding, PDM/FMM and fixed cross-node/phase module is
                # trainable.  This is a fixed module set, not a gate or
                # parameter search; the CCLT/latent coordinates remain in
                # the same representation path.
                for parameter in model.backbone.parameters():
                    parameter.requires_grad_(True)
            elif bool(config.get("hybrid_unfreeze_pdm", False)):
                # Structural adaptation audit: retain the pretrained CPR
                # node/phase operators, but allow the PDM/FMM representation
                # to re-coordinate the transported history.  This is a fixed
                # module selection, not a learning-rate or gate search.
                for parameter in model.backbone.parameters():
                    parameter.requires_grad_(False)
                for name, parameter in model.backbone.named_parameters():
                    if (name.startswith("pdm_blocks") or
                            name.startswith("predict_layers") or
                            name.startswith("projection_layer") or
                            name.startswith("enc_embedding")):
                        parameter.requires_grad_(True)
            if bool(config.get("hybrid_residual_median_init", False)):
                # Calibrate the zero-initialized CCLT horizon bias from the
                # training prefix only.  This is a robust location estimate,
                # not a validation-selected coefficient or a sample gate.
                model.eval()
                train_x = np.asarray(data["train_x"], dtype=np.float32)
                train_y = np.asarray(data["train_y"], dtype=np.float32)
                residuals = []
                batch_size = max(1, int(config.get("batch_size", 32)))
                with torch.no_grad():
                    for start in range(0, len(train_x), batch_size):
                        xb = torch.from_numpy(train_x[start:start + batch_size])
                        ab = torch.from_numpy(np.asarray(history_aux["train"][start:start + batch_size], dtype=np.float32))
                        model.backbone._external_history_latents = model.latent._history_latent(ab, xb)
                        pred = model.backbone(xb, None, None, None, None)
                        residuals.append(train_y[start:start + batch_size] - pred.detach().cpu().numpy())
                residual = np.concatenate(residuals, axis=0)
                median_bias = np.median(residual, axis=(0, 2)).astype(np.float32)
                if model.cclt.basis_residual:
                    raise ValueError("residual median initialization requires direct horizon residual")
                with torch.no_grad():
                    model.cclt.residual_readout.bias.copy_(torch.as_tensor(median_bias))
            return model
        factory = _make_hybrid_latent_cclt
        source = ("single frozen CPR-DX backbone with causal multimodal latent transport before PDM "
                  "and fixed graph-aware CCLT history readout; no gate/router/expert/residual selector")
    elif latent_multimodal:
        history_aux, static_context, history_meta = _build_history_multimodal_aux(config, data)
        if not history_meta["node_aligned"]:
            raise ValueError("latent multimodal representation requires node-aligned history variables")
        args.enc_in = channels
        def _make_latent_multimodal():
            model = _LatentHistoryTimeMixer(
                args, history_meta["feature_dim"], static_context,
                hidden=int(config.get("multimodal_hidden", 32)),
                freeze_backbone=bool(config.get("latent_freeze_backbone", False)),
                scale_specific=bool(config.get("latent_scale_specific", False)),
                adjacency=np.asarray(data["adj"], dtype=np.float32),
                graph_fusion=bool(config.get("latent_graph_fusion", False)))
            if bool(config.get("latent_ols_init", False)):
                # Train-prefix-only station-wise regression initializes the
                # first latent coordinate in the observed demand direction.
                # The latent route is additive before PDM, so this is a small
                # data-derived context coordinate rather than a searched gate.
                aux_train = np.asarray(history_aux["train"], dtype=np.float64)
                volume_train = np.asarray(data["train_x"], dtype=np.float64)
                feature_dim = int(history_meta["feature_dim"])
                static_np = np.asarray(static_context, dtype=np.float64)
                raw_dim = int(model.raw_dim)
                coefficients = np.zeros((channels, raw_dim), dtype=np.float64)
                eye = np.eye(raw_dim, dtype=np.float64)
                for node in range(channels):
                    hist = aux_train[:, :, node, :].reshape(-1, feature_dim)
                    st = np.repeat(static_np[node:node + 1], hist.shape[0], axis=0)
                    inter = hist[:, :, None] * static_np[node][None, None, :]
                    raw = np.concatenate((hist, st, inter.reshape(hist.shape[0], -1)), axis=1)
                    target = volume_train[:, :, node].reshape(-1)
                    gram = raw.T @ raw + eye
                    rhs = raw.T @ target
                    coefficients[node] = np.linalg.solve(gram, rhs)
                # Calibrate the additive coordinate to a quarter of the
                # direct OLS signal; the factor is a fixed representation
                # initialization, not a validation-selected hyperparameter.
                coefficients *= 0.25
                with torch.no_grad():
                    if model.scale_specific:
                        model.scale_projection[:, :, 0, :].copy_(
                            torch.as_tensor(coefficients, dtype=model.scale_projection.dtype).unsqueeze(0).expand(model.num_scales, -1, -1))
                    else:
                        model.node_projection[:, 0, :].copy_(
                            torch.as_tensor(coefficients, dtype=model.node_projection.dtype))
            checkpoint_path = config.get("latent_backbone_checkpoint")
            if checkpoint_path:
                payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
                state = payload.get("model", payload)
                missing, unexpected = model.backbone.load_state_dict(state, strict=False)
                if missing or unexpected:
                    raise ValueError(f"latent multimodal checkpoint mismatch: missing={missing}, unexpected={unexpected}")
            return model
        factory = _make_latent_multimodal
        source = ("single CPR-TimeMixer PDM/FMM with bounded station-specific causal "
                  "multimodal latent transport before every PDM scale; no gate/router/expert/residual")
    elif node_multimodal:
        history_aux, static_context, history_meta = _build_history_multimodal_aux(config, data)
        if not history_meta["node_aligned"]:
            raise ValueError("node multimodal tokenization requires node-aligned history variables")
        args.enc_in = channels
        def _make_node_multimodal():
            model = _NodeMultimodalTokenTimeMixer(
                args, history_meta["feature_dim"], static_context,
                token_dim=int(config.get("multimodal_token_dim", 16)),
                freeze_backbone=bool(config.get("node_freeze_backbone", False)))
            if bool(config.get("node_ols_init", False)):
                # Train-prefix-only station-wise regression initializes the
                # auxiliary coordinate in the physical direction of demand.
                # It uses no validation/test rows and is a deterministic
                # representation initialization, not a searched parameter.
                aux_train = np.asarray(history_aux["train"], dtype=np.float64)
                volume_train = np.asarray(data["train_x"], dtype=np.float64)
                feature_dim = int(history_meta["feature_dim"])
                coefficients = np.zeros((channels, feature_dim), dtype=np.float64)
                eye = np.eye(feature_dim, dtype=np.float64)
                for node in range(channels):
                    features = aux_train[:, :, node, :].reshape(-1, feature_dim)
                    target = volume_train[:, :, node].reshape(-1)
                    gram = features.T @ features + eye
                    rhs = features.T @ target
                    coefficients[node] = np.linalg.solve(gram, rhs)
                scale = float(max(model.raw_dim - 1, 1) ** -0.5)
                normalized = np.clip(coefficients / max(scale, 1e-6), -0.8, 0.8)
                with torch.no_grad():
                    model.node_token_weight[:, 0, 1:1 + feature_dim].copy_(
                        torch.as_tensor(normalized, dtype=model.node_token_weight.dtype))
            checkpoint_path = config.get("node_backbone_checkpoint")
            if checkpoint_path:
                payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
                state = payload.get("model", payload)
                missing, unexpected = model.backbone.load_state_dict(state, strict=False)
                if missing or unexpected:
                    raise ValueError(f"node multimodal backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}")
            return model
        factory = _make_node_multimodal
        source = ("single TimeMixer PDM/FMM with direct per-node multimodal tokens: "
                  "volume + causal occupancy/duration/price/weather history + static spatial descriptors; "
                  "no experts, router, gate, or post-hoc residual")
    elif multimodal_history:
        history_aux, static_context, history_meta = _build_history_multimodal_aux(config, data)
        args.enc_in = channels
        if bool(config.get("cclt_graph_fusion", 0)):
            args.urban_adj = np.asarray(data["adj"], dtype=np.float32)
        if cclt_contrast:
            args.cclt_graph_contrast = 1
        if cclt_trajectory:
            args.cclt_trajectory = 1
        if cclt_history:
            def _make_cclt():
                model = _CCLTTimeMixer(
                    args, history_meta["feature_dim"], static_context,
                    hidden=int(config.get("multimodal_hidden", 32)),
                    innovation_projection=cclt_innovation,
                    freeze_backbone=(cclt_frozen or cclt_basis or cclt_extended_basis or cclt_joint or cclt_recurrent),
                    basis_residual=(cclt_basis or cclt_extended_basis),
                    extended_basis=cclt_extended_basis,
                    endogenous_history=(cclt_joint or cclt_recurrent),
                    recurrent_residual=cclt_recurrent,
                    trajectory_readout=cclt_trajectory)
                if cclt_frozen or cclt_basis or cclt_extended_basis or cclt_joint or cclt_recurrent:
                    checkpoint_path = config.get("cclt_backbone_checkpoint")
                    if not checkpoint_path:
                        raise ValueError("frozen CCLT requires cclt_backbone_checkpoint")
                    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
                    state = payload.get("model", payload)
                    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
                    if missing or unexpected:
                        raise ValueError(f"CPR backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}")
                return model
            factory = _make_cclt
            source = ("official CPR-TimeMixer backbone + CCLT causal covariate encoder + fixed additive forecast residual"
                      + (" + level-preserving innovation projection" if cclt_innovation else "")
                      + (" + frozen CPR-DX representation" if (cclt_frozen or cclt_basis or cclt_extended_basis or cclt_joint or cclt_recurrent) else "")
                      + (" + fixed daily/half-daily innovation basis" if cclt_basis else "")
                      + (" + fixed level/drift/Fourier innovation basis" if cclt_extended_basis else "")
                      + (" + joint causal volume-history encoder" if (cclt_joint or cclt_recurrent) else "")
                      + (" + causal recurrent residual decoder" if cclt_recurrent else ""))
        else:
            factory = lambda: _HistoricalMultimodalTimeMixer(
                args, history_meta["feature_dim"], static_context,
                hidden=int(config.get("multimodal_hidden", 32)))
            source = "official CPR-TimeMixer backbone + causal historical multimodal context encoder (occupancy/duration/tariff/weather + static spatial descriptors)"
    elif tariff:
        args.use_tariff_operator = 1
        args.prediction_method = ("tariff_context" if tariff_bdc else
                                  "tariff_bilinear" if tariff_bilinear else
                                  "tariff_context" if tariff_context else
                                  "tariff_scale" if tariff_scale else
                                  "tariff_coordinate" if tariff_coordinate else
                                  "tariff_flow" if tariff_flow else
                                  "tariff_state" if tariff_state else "tariff_cross" if tariff_cross
                                  else "tariff_fusion" if tariff_fusion else "tariff_tensor")
        tariff_aux = _build_tariff_aux(config, data)
        if tariff_bdc:
            from scprc.models.decomposition_coupling import BilateralCoupledTimeMixer
            factory = lambda: BilateralCoupledTimeMixer(args)
        else:
            factory = lambda: _UpstreamModel(args)
        source = ("official TimeMixer backbone + bilateral season-trend coupling + historical/future tariff context"
                  if tariff_bdc else
                  "official TimeMixer backbone + bilinear volume-tariff context embedding with tariff FMM fusion"
                  if tariff_bilinear else
                  "official TimeMixer backbone + historical/future tariff context embedding with tariff FMM fusion"
                  if tariff_context else
                  "official TimeMixer backbone + tariff-conditioned energy-preserving cross-scale latent transport"
                  if tariff_scale else
                  "official TimeMixer backbone + invertible tariff-neutral coordinate and learned elasticity"
                  if tariff_coordinate else
                  "official TimeMixer backbone + known-future tariff stable controlled latent flow"
                  if tariff_flow else
                  "official TimeMixer backbone + known-future tariff controlled latent state-transition horizon operator"
                  if tariff_state else
                  "official TimeMixer backbone + known-future tariff cross-attention horizon operator"
                  if tariff_cross else
                  "official TimeMixer backbone + known-future tariff latent fusion horizon operator"
                  if tariff_fusion else
                  "official TimeMixer backbone + known-future tariff tensor-product horizon operator")
    elif joint_phase_region:
        args.use_joint_phase_region = 1
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone + joint phase-region routing before PDM"
    elif phase_latent_decomp:
        args.use_phase_latent_decomp = 1
        factory = lambda: _UpstreamModel(args)
        source = ("official TimeMixer backbone + causal seven-cycle phase-template/"
                  "innovation/level latent decomposition before PDM; fixed linear"
                  " re-coordinate, no gate/router/expert/post-hoc residual")
    elif phase_cycle:
        # The alias is explicit so provenance cannot silently omit the
        # representation operator when a config is copied from a baseline.
        args.use_phase_cycle = 1
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone + phase-cycle tokenization and cross-phase router before PDM"
    elif causal_ema:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone + fixed causal EMA PDM decomposition"
    elif causal_daily:
        args.decomp_method = "causal_daily"
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone + causal same-phase seven-cycle PDM decomposition"
    elif causal_fir:
        args.decomp_method = "causal_fir"
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone + learnable positive causal FIR PDM decomposition"
    elif robust_huber:
        args.decomp_method = "robust_huber"
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone + fixed Hampel/Huber robust PDM decomposition"
    elif routed:
        factory = lambda: _RoutedTimeMixer(args, data["adj"], enabled=bool(config.get("router_enabled", 1)))
        source = "official TimeMixer backbone + component-scale sparse router"
    elif period_aligned:
        factory = lambda: _PeriodAlignedTimeMixer(args, period_alignment=period_alignment)
        source = "official TimeMixer backbone + trend/24-hour phase-aligned component forecasting"
    elif wavelet:
        factory = lambda: _WaveletTimeMixer(args)
        source = "official TimeMixer backbone + Haar wavelet approximation/detail forecasting"
    elif spectral:
        factory = lambda: _SpectralTimeMixer(args)
        source = "official TimeMixer backbone + fixed 24h/168h Fourier input views"
    elif localized_tf:
        factory = lambda: _LocalizedTimeFrequencyTimeMixer(args)
        source = "official TimeMixer backbone + localized 24h patch time-frequency latent mixer"
    elif dft:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone with DFT seasonal/trend decomposition"
    elif temporal_dft:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone with temporal-axis dynamic DFT decomposition"
    elif adaptive_latent_spectral:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone with adaptive latent spectral decomposition"
    elif horizon_basis:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone with shared daily-weekly horizon basis projection"
    elif patch_token:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone with learnable 4-hour patch tokenization"
    elif latent_temporal:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone with scale-consistent latent temporal mixing"
    elif mrti:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone with UrbanEV multi-resolution time imaging"
    elif phase_mrti:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone with phase-aligned UrbanEV multi-resolution time imaging"
    elif event_mrti:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone with event-aware UrbanEV multi-resolution time imaging"
    elif apid:
        factory = lambda: _UpstreamModel(args)
        source = "official TimeMixer backbone with UrbanEV adaptive periodic image decomposition"
    elif innovation:
        factory = lambda: _UpstreamModel(args)
        source = "TimeMixer PDM seasonal innovation with sparse phase-lag regional exchange"
    elif crps_decoder:
        from scprc.models.crps_decoder import CrossResolutionStateTimeMixer
        factory = lambda: CrossResolutionStateTimeMixer(args)
        source = "TimeMixer PDM + cross-resolution coarse-to-fine future-state decoder"
    elif shared_query:
        from scprc.models.crps_decoder import SharedFutureQueryTimeMixer
        factory = lambda: SharedFutureQueryTimeMixer(args)
        source = "TimeMixer PDM + shared future-query cross-resolution decoder"
    elif reconciled_future:
        from scprc.models.crps_decoder import ReconciledFutureTimeMixer
        factory = lambda: ReconciledFutureTimeMixer(args)
        source = "TimeMixer PDM + deterministic cross-resolution forecast reconciliation"
    elif equilibrium:
        from scprc.models.equilibrium_decoder import ContractiveEquilibriumTimeMixer
        factory = lambda: ContractiveEquilibriumTimeMixer(args)
        source = "TimeMixer PDM + contractive bidirectional cross-resolution equilibrium decoder"
    elif transport:
        from scprc.models.transport_decoder import OrthogonalTransportTimeMixer
        factory = lambda: OrthogonalTransportTimeMixer(args)
        source = "TimeMixer PDM + phase-conditioned orthogonal cross-resolution transport decoder"
    elif phase_state:
        from scprc.models.phase_state import PhaseStateTimeMixer
        factory = lambda: PhaseStateTimeMixer(args)
        source = "TimeMixer PDM + shared daily phase-state transition decoder"
    elif phase_grid:
        if phase_grid_fusion:
            from scprc.models.phase_grid import PhaseGridFusionTimeMixer
            factory = lambda: PhaseGridFusionTimeMixer(args)
            source = "TimeMixer PDM + identity-preserving finest latent fusion with cross-resolution 24-phase grid state"
        else:
            from scprc.models.phase_grid import PhaseGridTimeMixer
            factory = lambda: PhaseGridTimeMixer(args)
            source = "TimeMixer PDM + cross-resolution 24-phase grid with shared daily state transition"
    elif phase_token_fmm:
        from scprc.models.phase_token_fmm import PhaseTokenFMMTimeMixer
        factory = lambda: PhaseTokenFMMTimeMixer(args)
        source = "TimeMixer PDM + phase-token latent finest FMM with identity-preserving representation fusion"
    elif nlinear_fmm:
        from scprc.models.nlinear_fmm import NLinearFMMTimeMixer
        factory = lambda: NLinearFMMTimeMixer(args)
        source = "TimeMixer PDM + NLinear level-anchored finest decoder with unchanged coarse FMM scales"
    elif regime_trajectory_fmm:
        from scprc.models.regime_trajectory_fmm import RegimeTrajectoryFMMTimeMixer
        factory = lambda: RegimeTrajectoryFMMTimeMixer(args)
        source = "TimeMixer PDM + causal level/drift regime trajectory fused into finest latent FMM"
    elif phase_resolution:
        from scprc.models.phase_resolution import PhaseResolutionTimeMixer
        factory = lambda: PhaseResolutionTimeMixer(args)
        source = "TimeMixer PDM + shared multi-resolution phase routing"
    elif bilateral_coupling:
        from scprc.models.decomposition_coupling import BilateralCoupledTimeMixer
        factory = lambda: BilateralCoupledTimeMixer(args)
        source = "TimeMixer PDM + bilateral season-trend coupling before scale mixing"
    elif scale_bilateral:
        from scprc.models.decomposition_coupling import ScaleSpecificBilateralTimeMixer
        factory = lambda: ScaleSpecificBilateralTimeMixer(args)
        source = "TimeMixer PDM + scale-specific bilateral season-trend coupling"
    elif dcs_fmm:
        from scprc.models.dcs_fmm import DualConservativeSemigroupTimeMixer
        factory = lambda: DualConservativeSemigroupTimeMixer(args)
        source = "TimeMixer PDM + dynamic dual-conservative semigroup FMM"
    elif ntf_fmm:
        from scprc.models.temporal_frame_fmm import NestedTemporalFrameTimeMixer
        factory = lambda: NestedTemporalFrameTimeMixer(args)
        source = "TimeMixer PDM + nested temporal-frame factorized FMM"
    elif dct_fmm:
        from scprc.models.drift_conditioned_fmm import DriftConditionedFMMTimeMixer
        factory = lambda: DriftConditionedFMMTimeMixer(args)
        source = "TimeMixer PDM + causal drift-conditioned temporal FMM operators"
    elif dynamic_scale:
        from scprc.models.tail_conditioned_fmm import DynamicScaleConditionedTimeMixer
        factory = lambda: DynamicScaleConditionedTimeMixer(args)
        source = "TimeMixer PDM + dynamic scale/tail-conditioned horizon-basis decoder"
    elif horizon_convolution:
        from scprc.models.horizon_conv_decoder import HorizonConvolutionTimeMixer
        factory = lambda: HorizonConvolutionTimeMixer(args)
        source = "TimeMixer PDM + identity-initialized joint horizon convolution decoder"
    elif conditional_spatial:
        from scprc.models.conditional_spatial import ConditionalSpatialTimeMixer
        factory = lambda: ConditionalSpatialTimeMixer(args)
        source = "TimeMixer PDM + history-conditioned fixed-edge latent edge-contrast projection"
    elif phase_conditioned:
        from scprc.models.phase_conditioned_hybrid import PhaseConditionedTimeMixer
        factory = lambda: PhaseConditionedTimeMixer(args)
        source = "TimeMixer PDM/FMM + PhaseFormer phase encoder latent conditioning"
    elif phase_kd:
        factory = lambda: _UpstreamModel(args)
        source = "TimeMixer-CI PDM/FMM trained with frozen PhaseFormer response distillation"
    elif phase_residual_transport:
        from scprc.models.phase_residual_transport import PhaseResidualTransportTimeMixer
        factory = lambda: PhaseResidualTransportTimeMixer(args)
        source = "TimeMixer CI forecast + frozen PhaseFormer phase-token residual transport"
    elif phase_lowrank_transport:
        from scprc.models.phase_residual_transport import PhaseResidualLowRankTransportTimeMixer
        factory = lambda: PhaseResidualLowRankTransportTimeMixer(args)
        source = "TimeMixer CI forecast + frozen PhaseFormer low-rank horizon-phase residual transport"
    elif phase_error_aware_transport:
        from scprc.models.phase_residual_transport import ErrorAwarePhaseResidualTransportTimeMixer
        factory = lambda: ErrorAwarePhaseResidualTransportTimeMixer(args)
        source = "TimeMixer CI forecast + frozen PhaseFormer error-aware phase residual transport"
    elif dynamic_normalization:
        from scprc.models.dynamic_normalization import DynamicNormalizationTimeMixer
        factory = lambda: DynamicNormalizationTimeMixer(args)
        source = ("TimeMixer PDM/FMM + causal future-statistics normalization with train-only mean/std supervision"
                  if future_stats_supervised else
                  "TimeMixer PDM/FMM + causal multi-scale future-statistics normalization")
    elif cpr_regime:
        from scprc.models.cpr_regime import CausalAffineRegimeTimeMixer
        factory = lambda: CausalAffineRegimeTimeMixer(
            args,
            level_width=int(config.get("cpr_regime_level_width", 24)),
            scale_width=int(config.get("cpr_regime_scale_width", 72)),
        )
        source = ("single CPR-TimeMixer backbone in a causal affine regime coordinate: "
                  "recent level + drift continuation + robust scale restoration; no gate/router")
    elif cpr_periodic_state:
        from scprc.models.cpr_regime import CausalPeriodicStateTimeMixer
        factory = lambda: CausalPeriodicStateTimeMixer(
            args,
            period=24,
            cycles=7,
        )
        source = ("single CPR-TimeMixer backbone in a causal daily-template state coordinate: "
                  "six-cycle phase template + seventh-cycle level transport + robust innovation scale; "
                  "no gate/router/expert")
    elif scale_lifting:
        from scprc.models.scale_lifting_timemixer import ScaleLiftingTimeMixer
        factory = lambda: ScaleLiftingTimeMixer(args)
        source = "TimeMixer embedding + reversible coarse/detail scale-lifting before PDM"
    elif static_chart:
        from scprc.models.static_chart_timemixer import StaticChartTimeMixer
        factory = lambda: StaticChartTimeMixer(args)
        source = "TimeMixer embedding + static-zone latent chart before PDM"
    elif cross_scale_ssm:
        from scprc.models.cross_scale_ssm import CrossScaleSSMTimeMixer
        factory = lambda: CrossScaleSSMTimeMixer(args)
        source = "TimeMixer PDM + shared diagonal selective state-space scan with coarse-to-fine cross-scale terminal-state communication"
    elif pdm_cmamba:
        from scprc.models.pdm_cmamba import PDMCMambaTimeMixer
        factory = lambda: PDMCMambaTimeMixer(args)
        source = "TimeMixer PDM/FMM + shared patch temporal selective scan (TMB) + low-rank statistic-conditioned GDD-MLP"
    elif shared_basis_transport:
        from scprc.models.shared_basis_transport import SharedBasisTransportTimeMixer
        factory = lambda: SharedBasisTransportTimeMixer(args)
        source = ("TimeMixer PDM/FMM + shared fixed rank-four history-to-horizon "
                  "temporal basis transport inside FMM (identity initialized)")
    elif cldst:
        args.use_level_drift_transport = 1
        factory = lambda: _UpstreamModel(args)
        source = ("TimeMixer PDM/FMM + causal level-drift state transport "
                  "inside every PDM block (identity initialized, no gate/router)")
    elif endpoint_bridge:
        args.use_endpoint_bridge = 1
        factory = lambda: _UpstreamModel(args)
        source = ("TimeMixer PDM/FMM + shared rank-four endpoint level-drift "
                  "bridge in latent coordinates before PDM (identity initialized, no gate/router)")
    elif cointegration:
        from scprc.models.cointegration_timemixer import CointegrationCommonSpreadTimeMixer
        factory = lambda: CointegrationCommonSpreadTimeMixer(args)
        source = ("TimeMixer PDM/FMM + fixed UrbanEV graph common-trend/Laplacian-spread "
                  "coordinate transform before PDM (identity initialized, no gate/router)")
    elif cmpd:
        from scprc.models.cmpd_timemixer import CMPDTimeMixer
        factory = lambda: CMPDTimeMixer(args)
        source = "official TimeMixer backbone + frozen causal moment-preserving decimation"
    else:
        factory = lambda: _UpstreamModel(args)
        source = "unchanged official models/TimeMixer.py"
    metadata = {"official_timemixer_commit": "e24610583b36fdd8c76cc17a8df4e65759a5f460",
                "model_source": source,
                "model_config": {k: v for k, v in vars(args).items()
                                 if k not in {"urban_adj", "static_features"}},
                "cast_canonicalization": cast_metadata,
                "cedar_invariant": {
                    "representation": "pooled pre-PDM shared latent",
                    "environments": "fixed median-longitude station partition",
                    "discrepancy": "polynomial-kernel MMD (mean+covariance) plus group-MSE std",
                    "mmd_weight": float(config.get("cedar_mmd_weight", 0.05)),
                    "loss_discrepancy_weight": float(config.get("cedar_discrepancy_weight", 0.05)),
                    "learned_gate": False, "router": False, "experts": False,
                    "posthoc_residual": False,
                } if cedar else None,
                "future_calendar_features": False,
                "innovation_exchange": {
                    "location": "PDM seasonal component before multiscale mixing",
                    "topk": args.innovation_topk,
                    "rank": args.innovation_rank,
                    "lags": args.innovation_lags,
                    "stage": args.innovation_stage,
                    "city_mode": "regional innovation only; equal latent send/receive",
                } if innovation else None,
                "benchmark_scope": "UrbanEV; not scprc of original ETT result tables",
                "router": routed, "router_enabled": bool(config.get("router_enabled", 1)),
                "cross_resolution_decoder": crps_decoder,
                "cross_resolution_decoder_spec": {
                    "direction": "coarse_to_fine",
                    "latent_state": True,
                    "output": "finest_scale_only",
                    "fusion": "concat_linear_gelu_linear",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if crps_decoder else None,
                "shared_future_query_decoder": shared_query,
                "shared_future_query_spec": {
                    "query_bank": "learned_horizon_queries_conditioned_on_all_scale_summaries",
                    "scale_readout": "shared_query_cross_attention_per_scale",
                    "fusion": "concatenate_all_scale_states_then_mlp",
                    "independent_scalar_heads": False,
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if shared_query else None,
                "reconciled_future_decoder": reconciled_future,
                "reconciled_future_spec": {
                    "operator": "closed_form_multiresolution_least_squares_projection",
                    "constraints": "hourly trajectory matched to scale block means",
                    "learned_gate": False,
                    "posthoc_residual": False,
                } if reconciled_future else None,
                "contractive_equilibrium_decoder": equilibrium,
                "contractive_equilibrium_spec": {
                    "direction": "bidirectional_synchronous",
                    "state": "future_latent_equilibrium",
                    "iterations": int(getattr(args, "equilibrium_iterations", 4)),
                    "rho": float(getattr(args, "equilibrium_rho", 0.75)),
                    "spectral_norm_bound": 1.0,
                    "output": "finest_equilibrium_state_only",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                    "implicit_differentiation": False,
                } if equilibrium else None,
                "orthogonal_transport_decoder": transport,
                "orthogonal_transport_spec": {
                    "operator": "phase-conditioned matrix exponential of bounded skew-symmetric adjacent-scale generator",
                    "state": "future_latent_resolution_field",
                    "output": "all-scale transported latent readout",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                    "fixed_point": False,
                } if transport else None,
                "phase_state_decoder": phase_state,
                "phase_state_spec": {
                    "operator": "shared_low_rank_day_axis_gru_with_explicit_next_day_state",
                    "rank": int(getattr(args, "phase_state_rank", 8)),
                    "forecast_step": str(getattr(args, "phase_state_forecast_step", "predictive")),
                    "replaces": "finest_scale_fmm_only",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if phase_state else None,
                "multi_resolution_phase_router": phase_resolution,
                "multi_resolution_phase_router_spec": {
                    "phase_lengths": [24, 12, 6, 3],
                    "routers": int(getattr(args, "phase_router_count", 4)),
                    "heads": int(getattr(args, "phase_router_heads", 4)),
                    "location": "embedding_before_PDM",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if phase_resolution else None,
                "bilateral_decomposition_coupling": bilateral_coupling,
                "bilateral_decomposition_coupling_spec": {
                    "location": "PDM_decomposition_before_season_trend_scale_mixing",
                    "operator": "pointwise_2d_change_of_basis_initialized_identity",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if bilateral_coupling else None,
                "scale_specific_bilateral_coupling": scale_bilateral,
                "scale_specific_bilateral_coupling_spec": {
                    "location": "PDM_decomposition_before_scale_mixing",
                    "operator": "independent_identity_initialized_2d_change_of_basis_per_scale",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if scale_bilateral else None,
                "dcs_fmm": dcs_fmm,
                "dcs_fmm_spec": {
                    "location": "replace_all_FMM_temporal_heads_after_PDM",
                    "operator": "context_conditioned_J_minus_diagonal_decay_discrete_semigroup",
                    "state_rank": int(getattr(args, "dcs_state_rank", 24)),
                    "context_basis": int(getattr(args, "dcs_context_basis", 4)),
                    "dynamic": bool(getattr(args, "dcs_dynamic", 1)),
                    "integrator": str(getattr(args, "dcs_integrator", "cayley")),
                    "max_decay": float(getattr(args, "dcs_max_decay", 0.25)),
                    "duality": "adjoint_semigroup_exposed_for_CPU_contract",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if dcs_fmm else None,
                "nested_temporal_frame_fmm": ntf_fmm,
                "nested_temporal_frame_spec": {
                    "location": "replace_all_FMM_temporal_heads_after_PDM",
                    "operator": "shared_horizon_frame_with_nested_scale_restriction",
                    "frame_rank": int(getattr(args, "ntf_frame_rank", 24)),
                    "dropout": float(getattr(args, "ntf_frame_dropout", 0.0)),
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if ntf_fmm else None,
                "drift_conditioned_fmm": dct_fmm,
                "drift_conditioned_fmm_spec": {
                    "location": "replace_all_FMM_temporal_heads_after_PDM",
                    "operator": "history-drift-conditioned low-rank temporal linear operator",
                    "rank": int(getattr(args, "dct_rank", 4)),
                    "context_hidden": int(getattr(args, "dct_context_hidden", 16)),
                    "drift_features": "m24-m72,m72-m168 from causal normalized history",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if dct_fmm else None,
                "dynamic_scale_conditioned_decoder": dynamic_scale,
                "dynamic_scale_conditioned_spec": {
                    "location": "replace_all_FMM_temporal_heads_after_PDM",
                    "operator": "causal level/volatility/upper-tail/drift conditioned low-rank horizon basis",
                    "context_features": "window mean, log std, q90, q95-q75/std, recent24-minus-mean, q95",
                    "basis_dim": int(getattr(args, "tail_basis_dim", 8)),
                    "context_hidden": int(getattr(args, "tail_context_hidden", 32)),
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if dynamic_scale else None,
                "horizon_convolution_decoder": horizon_convolution,
                "horizon_convolution_spec": {
                    "location": "replace_all_FMM_temporal_heads_after_PDM",
                    "operator": "scale-concatenated depthwise horizon convolution plus pointwise scale mixer",
                    "kernel": int(getattr(args, "horizon_conv_kernel", 3)),
                    "initialization": "identity temporal convolution and sum-equivalent scale mixer",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if horizon_convolution else None,
                "conditional_spatial_decoder": conditional_spatial,
                "conditional_spatial_spec": {
                    "location": "after_each_PDM_scale_before_FMM",
                    "operator": "fixed_topk_adjacency self/neighbor/contrast projection conditioned on history agreement",
                    "topk": int(getattr(args, "conditional_spatial_topk", 8)),
                    "history_feature": "mean exp(-|recent24-neighbor_recent24|/train_window_std)",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if conditional_spatial else None,
                "phase_conditioned_hybrid": phase_conditioned,
                "phase_conditioned_hybrid_spec": {
                    "location": "after_each_PDM_scale_before_FMM",
                    "operator": "PhaseFormer 7x24 embedding and single-router cross-phase latent summary",
                    "phase_dim": int(getattr(args, "hybrid_phase_dim", 8)),
                    "routers": int(getattr(args, "hybrid_phase_routers", 1)),
                    "heads": int(getattr(args, "hybrid_phase_heads", 2)),
                    "predictor": "TimeMixer FMM only",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if phase_conditioned else None,
                "phase_distillation": phase_kd,
                "phase_distillation_spec": {
                    "teacher": "frozen Dec PhaseFormer checkpoint",
                    "student": "TimeMixer-CI unchanged PDM/FMM",
                    "kd_weight": float(getattr(args, "kd_weight", 0.25)),
                    "teacher_future_labels": False,
                    "inference_teacher_required": False,
                } if phase_kd else None,
                "dynamic_normalization": dynamic_normalization,
                "dynamic_normalization_spec": {
                    "statistics_scales": [24, 72, 168],
                    "statistics": "mean/logstd/recent-drift from observed history",
                    "hidden": int(getattr(args, "dynamic_norm_hidden", 32)),
                    "future_labels_in_statistics": False,
                    "identity_initialized": True,
                } if dynamic_normalization else None,
                "causal_affine_regime": cpr_regime,
                "causal_affine_regime_spec": {
                    "location": "input coordinate before the single TimeMixer PDM/FMM backbone",
                    "state": "causal recent-24 level, recent-vs-previous-day drift, 72-hour MAD scale",
                    "decoder": "fixed linear drift continuation plus affine innovation restoration",
                    "training_objective": "robust scale/MRC forecast loss + future level/scale moment alignment",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                    "router": False,
                } if cpr_regime else None,
                "causal_periodic_state": cpr_periodic_state,
                "causal_periodic_state_spec": {
                    "location": "input coordinate before the single TimeMixer PDM/FMM backbone",
                    "state": "six-cycle 24-hour phase template plus seventh-cycle causal level shift",
                    "scale": "fixed 72-hour-equivalent robust absolute innovation scale",
                    "decoder": "deterministic repeated phase template plus affine innovation restoration",
                    "future_labels_in_state": False,
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                    "router": False,
                    "experts": False,
                } if cpr_periodic_state else None,
                "phase_residual_transport": phase_residual_transport,
                "phase_residual_transport_spec": {
                    "teacher": "frozen PhaseFormer checkpoint",
                    "latent": "24 phase tokens per zone",
                    "adapter": "zero-initialized D->1 residual head scaled by history std",
                    "checkpoint": str(config.get("phase_teacher_checkpoint", "")),
                } if phase_residual_transport else None,
                "phase_lowrank_transport": phase_lowrank_transport,
                "phase_lowrank_transport_spec": {
                    "teacher": "frozen PhaseFormer checkpoint",
                    "latent": "24 phase tokens per zone",
                    "rank": int(config.get("phase_transport_rank", 4)),
                    "adapter": "zero-initialized D->rank x rank->phase basis scaled by history std",
                    "checkpoint": str(config.get("phase_teacher_checkpoint", "")),
                } if phase_lowrank_transport else None,
                "phase_error_aware_transport": phase_error_aware_transport,
                "phase_error_aware_transport_spec": {
                    "teacher": "frozen PhaseFormer checkpoint",
                    "conditioning": "per-horizon CI forecast value concatenated to phase latent",
                    "adapter": "zero-initialized D+1->1 residual head scaled by history std",
                    "checkpoint": str(config.get("phase_teacher_checkpoint", "")),
                } if phase_error_aware_transport else None,
                "scale_lifting": scale_lifting,
                "scale_lifting_spec": {
                    "location": "embedding_before_first_PDM",
                    "operator": "coarse/detail reversible lifting with learned predict/update coordinates",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if scale_lifting else None,
                "static_chart": static_chart,
                "static_chart_spec": {
                    "location": "embedding_before_first_PDM",
                    "operator": "static-zone descriptor concatenation and shared latent coordinate projection",
                    "features": "log capacity/stations/area/perimeter plus weighted centroid longitude/latitude",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if static_chart else None,
                "cmpd": cmpd,
                "cmpd_spec": {
                    "location": "replace official AvgPool1d before every PDM scale",
                    "operator": "training-spectrum-fitted causal FIR with polynomial moment constraints",
                    "order": int(getattr(args, "cmpd_order", 1)),
                    "kernel_size": int(getattr(args, "cmpd_kernel_size", 5)),
                    "regularization": float(getattr(args, "cmpd_regularization", 0.05)),
                    "fit_split": "train_only",
                    "learned_gate": False,
                    "posthoc_residual": False,
                } if cmpd else None,
                "causal_ema": causal_ema,
                "causal_ema_spec": {
                    "location": "replace official centered moving-average decomposition inside every PDM block",
                    "operator": "one-sided exponential level state with season=x-level",
                    "alpha": float(getattr(args, "causal_ema_alpha", 0.25)),
                    "learned_gate": False,
                    "posthoc_residual": False,
                } if causal_ema else None,
                "causal_daily": causal_daily,
                "causal_daily_spec": {
                    "location": "replace official centered moving-average decomposition inside every PDM block",
                    "operator": "causal same-phase mean over preceding daily cycles with exact x=innovation+season reconstruction",
                    "period": int(getattr(args, "causal_daily_period", 24)),
                    "cycles": int(getattr(args, "causal_daily_cycles", 7)),
                    "learned_gate": False,
                    "posthoc_residual": False,
                } if causal_daily else None,
                "causal_fir": causal_fir,
                "causal_fir_spec": {
                    "location": "replace official centered moving-average decomposition inside every PDM block",
                    "operator": "per-latent positive causal FIR initialized as 25-step moving average with exact innovation reconstruction",
                    "kernel": int(getattr(args, "causal_fir_kernel", 25)),
                    "learned_gate": False,
                    "posthoc_residual": False,
                } if causal_fir else None,
                "phase_cycle_tokenization": phase_cycle,
                "phase_cycle_spec": {
                    "location": "embedding_before_first_PDM",
                    "layout": "7 observed daily cycles x 24 clock phases",
                    "operator": "phase-wise seven-cycle encoding + shared router cross-phase attention + cycle decode",
                    "routers": int(getattr(args, "phase_cycle_routers", 2)),
                    "heads": int(getattr(args, "phase_cycle_heads", 4)),
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if phase_cycle else None,
                "phase_latent_decomposition": phase_latent_decomp,
                "phase_latent_decomposition_spec": {
                    "location": "embedding_before_first_PDM",
                    "layout": "7 observed daily cycles x 24 clock phases",
                    "coordinates": "raw latent + cross-cycle phase template + within-cycle innovation + cycle-level departure",
                    "operator": "shared linear re-coordinate initialized as identity on raw latent",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if phase_latent_decomp else None,
                "joint_phase_region": joint_phase_region,
                "joint_phase_region_spec": {
                    "location": "embedding_before_first_PDM",
                    "layout": "zone x 7 observed daily cycles x 24 clock phases",
                    "operator": "full phase routing followed by per-phase region routing, then cycle decode",
                    "phase_routers": int(getattr(args, "joint_phase_routers", 2)),
                    "region_routers": int(getattr(args, "joint_region_routers", 4)),
                    "phase_heads": int(getattr(args, "joint_phase_heads", 4)),
                    "region_heads": int(getattr(args, "joint_region_heads", 4)),
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if joint_phase_region else None,
                "tariff_tensor_operator": tariff and not tariff_fusion and not tariff_cross and not tariff_state and not tariff_flow and not tariff_coordinate and not tariff_scale and not tariff_context and not tariff_bilinear,
                "tariff_tensor_spec": {
                    "location": "replace_FMM_horizon_projection",
                    "known_future_inputs": "zone-wise electricity and service tariffs for the 24 forecast hours",
                    "operator": "history latent coefficients tensor-product with future tariff coefficients",
                    "rank": int(getattr(args, "tariff_rank", 8)),
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if tariff and not tariff_fusion and not tariff_cross and not tariff_state and not tariff_flow and not tariff_coordinate and not tariff_scale and not tariff_context and not tariff_bilinear else None,
                "tariff_fusion_operator": tariff_fusion,
                "tariff_fusion_spec": {
                    "location": "replace_FMM_horizon_projection",
                    "known_future_inputs": "zone-wise electricity and service tariffs for the 24 forecast hours",
                    "operator": "direct history horizon projection concatenated with tariff tokens before latent fusion",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if tariff_fusion else None,
                "tariff_cross_operator": tariff_cross,
                "tariff_cross_spec": {
                    "location": "replace_FMM_horizon_projection",
                    "known_future_inputs": "zone-wise electricity and service tariffs for the 24 forecast hours",
                    "operator": "future tariff queries cross-attend to historical TimeMixer latent tokens",
                    "heads": int(getattr(args, "tariff_heads", 4)),
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if tariff_cross else None,
                "tariff_state_transition_operator": tariff_state,
                "tariff_state_transition_spec": {
                    "location": "replace_FMM_horizon_projection",
                    "known_future_inputs": "zone-wise electricity and service tariffs for the 24 forecast hours",
                    "operator": "future tariff token bilinearly controls a low-rank latent state transition; history projection is the forcing sequence",
                    "rank": int(getattr(args, "tariff_state_rank", 8)),
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if tariff_state else None,
                "tariff_controlled_flow_operator": tariff_flow,
                "tariff_controlled_flow_spec": {
                    "location": "replace_FMM_horizon_projection",
                    "known_future_inputs": "zone-wise electricity and service tariffs for the 24 forecast hours",
                    "operator": "bounded continuous-time latent flow with non-positive decay and low-rank tariff-driven vector field",
                    "rank": int(getattr(args, "tariff_state_rank", 8)),
                    "stable_step": "1/pred_len explicit Euler step",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if tariff_flow else None,
                "tariff_coordinate_operator": tariff_coordinate,
                "tariff_coordinate_spec": {
                    "location": "invertible input/output coordinate around TimeMixer normalization and FMM",
                    "known_future_inputs": "zone-wise electricity and service tariffs for observed and forecast hours",
                    "operator": "learned bounded elasticity removes tariff effect from history and applies the same inverse coordinate at future horizon",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if tariff_coordinate else None,
                "tariff_scale_transport_operator": tariff_scale,
                "tariff_scale_transport_spec": {
                    "location": "between per-scale FMM latent forecasts and channel projection",
                    "known_future_inputs": "zone-wise electricity and service tariffs for the 24 forecast hours",
                    "operator": "tariff-conditioned skew-symmetric generator with matrix-exponential cross-scale transport",
                    "rank": int(getattr(args, "tariff_scale_rank", 4)),
                    "energy_preserving": True,
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if tariff_scale else None,
                "tariff_context_operator": tariff_context,
                "tariff_context_spec": {
                    "location": "historical embedding before PDM plus known-future tariff-conditioned FMM",
                    "known_future_inputs": "zone-wise electricity and service tariffs for observed and forecast hours",
                    "operator": "causal multiscale tariff context channels followed by direct latent fusion at FMM",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if tariff_context else None,
                "tariff_bilinear_context_operator": tariff_bilinear,
                "tariff_bilinear_context_spec": {
                    "location": "historical embedding before PDM plus known-future tariff-conditioned FMM",
                    "known_future_inputs": "zone-wise electricity and service tariffs for observed and forecast hours",
                    "operator": "volume, tariff, and explicit volume-times-tariff bilinear coordinates before multiscale PDM",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if tariff_bilinear else None,
                "tariff_bdc_operator": tariff_bdc,
                "tariff_bdc_spec": {
                    "location": "historical tariff context before PDM plus bilateral season-trend coupling plus future tariff FMM fusion",
                    "operator": "causal tariff context coordinates and pointwise season-trend change of basis",
                    "posthoc_gate": False,
                    "posthoc_residual": False,
                } if tariff_bdc else None,
                "period_aligned": period_aligned,
                "period_alignment": "24-hour phase template + raw detrended context" if period_aligned and period_alignment else "raw detrended context" if period_aligned else None,
                "component_supervision": period_aligned and component_supervision,
                "reconstruction_weight": reconstruction_weight if period_aligned and component_supervision else 0.0,
                "wavelet_decomposition": "haar_level_1" if wavelet else None,
                "wavelet_component_supervision": wavelet,
                "spectral_input_views": "daily_harmonics_1_2+weekly_harmonic_1" if spectral else None,
                "spectral_loss_weight": float(config.get("spectral_lambda", 0.1)) if spectral or localized_tf else 0.0,
                "localized_time_frequency": "patch24_fft+patch_mixer" if localized_tf else None,
                "dft_decomposition": str(config.get("decomp_method", "")) if dft else None,
                "temporal_dft_decomposition": str(config.get("decomp_method", "")) if temporal_dft else None,
                "adaptive_latent_spectral_decomposition": {
                    "method": str(config.get("decomp_method", "")),
                    "top_k": int(config.get("top_k", 5)),
                    "temperature": float(config.get("spectral_temperature", 1.0)),
                } if adaptive_latent_spectral else None,
                "horizon_basis_projection": {
                    "method": str(config.get("prediction_method", "")),
                    "basis_dim": int(config.get("basis_dim", 16)),
                    "phase_priors": "polynomial+daily_harmonics+weekly_harmonics",
                } if horizon_basis else None,
                "patch_tokenization": {
                    "patch_size": int(config.get("patch_size", 1)),
                    "token_length": int(args.seq_len),
                    "initialization": "channel-wise average pooling",
                } if patch_token else None,
                "latent_temporal_mixer": {
                    "method": str(config.get("latent_mixer", "")),
                    "kernel_size": 5,
                    "scale_sharing": True,
                } if latent_temporal else None,
                "mrti": {
                    "periods": [24, 168],
                    "views": str(config.get("mrti_views", "both")),
                    "axes": "daily-phase x cycle-column",
                    "fusion": "raw+daily-image+weekly-image",
                } if mrti else None,
                "phase_mrti": {
                    "periods": [24, 168],
                    "alignment": "per-cycle circular cross-correlation to reference cycle",
                    "axes": "daily-phase x cycle-column",
                    "fusion": "raw+aligned-daily-image+aligned-weekly-image",
                } if phase_mrti else None,
                "event_mrti": {
                    "periods": [24, 168],
                    "components": "cycle-trend + phase-seasonal + positive-event peak",
                    "axes": "daily-phase x cycle-column",
                    "fusion": "raw+event-aware-daily-image+event-aware-weekly-image",
                    "scale_aware_loss": bool(config.get("scale_aware_loss", 0)),
                    "scale_exponent": float(config.get("scale_exponent", 0.5)),
                } if event_mrti else None,
                "apid": {
                    "periods": str(config.get("apid_periods", "24,168,12,84")),
                    "top_k": int(config.get("apid_top_k", 3)),
                    "temperature": float(config.get("apid_temperature", 0.7)),
                    "axis_kernel": int(config.get("apid_axis_kernel", 3)),
                    "decomposition": "adaptive-period-bank + phase/trend image axes + cross-resolution mixer",
                    "selection": "batch-shared spectral top-k; sample-specific soft weights",
                    "peak_branch": bool(config.get("apid_peak_branch", 0)),
                    "peak_lambda": float(config.get("apid_peak_lambda", 0.15)),
                    "fusion": "period-weighted component reconstruction before PDM",
                } if apid else None,
                "loss_name": "cpr_periodic_state_mrc" if cpr_periodic_state else
                             "cpr_regime_moment_mrc" if cpr_regime else
                             "robust_mae_mrc" if float(config.get("robust_mae_weight", 0.0)) > 0 else
                             "future_stats_supervised_scale_mrc" if future_stats_supervised else
                             "spectral_mse+lambda_log_amplitude" if spectral else
                             "localized_time_mse+lambda_log_amplitude" if localized_tf else
                             "peak_weighted_mse" if apid and bool(config.get("apid_peak_branch", 0)) else
                             "tail_balanced_multi_resolution_scale_consistency" if bool(config.get("tail_balanced_loss", 0)) and float(config.get("consistency_weight", 0.0)) > 0 else
                             "tail_balanced_mse" if bool(config.get("tail_balanced_loss", 0)) else
                             "multi_resolution_scale_consistency" if float(config.get("consistency_weight", 0.0)) > 0 else
                             "scale_weighted_mse" if bool(config.get("scale_aware_loss", 0)) else
                             "wavelet_component_mse+reconstruction" if wavelet else
                             "component_mse+reconstruction" if period_aligned and component_supervision else
                             "MSE"}
    if phase_region_state:
        metadata["phase_region_state"] = {
            "historical_multimodal": history_meta,
            "location": "complete causal 7x24 history tensor before CPR PDM",
            "operator": "shared within-day predecessor and cross-day same-phase linear transitions with fixed average-pool scale transport",
            "cycles": int(config.get("phase_region_cycles", 7)),
            "phases": int(config.get("phase_region_phases", 24)),
            "backbone_training": "frozen CPR-DX backbone" if bool(config.get("phase_region_freeze_backbone", True)) else "trainable CPR-DX backbone",
            "identity_initialization": True,
            "learned_gate": False,
            "router": False,
            "experts": False,
            "posthoc_residual": False,
        }
    elif e2e_multimodal:
        metadata["e2e_multimodal_history"] = {
            "historical_multimodal": history_meta,
            "location": "shared covariate/geometry token encoder after volume embedding and before every PDM scale",
            "operator": ("physical group shared low-rank projections (occupancy/duration, price, weather) + "
                         "fixed concatenation + causal depthwise temporal transport + fixed average-pool scale maps"
                         if grouped_multimodal else
                         "shared feature-group MLP + causal depthwise temporal transport + fixed average-pool scale maps"),
            "group_rank": int(config.get("multimodal_group_rank", 8)) if grouped_multimodal else None,
            "backbone_training": "full CPR-DX backbone trainable end-to-end",
            "learned_gate": False,
            "router": False,
            "experts": False,
            "posthoc_residual": False,
        }
    elif hybrid_latent_cclt:
        metadata["hybrid_latent_cclt"] = {
            "historical_multimodal": history_meta,
            "location": "causal latent transport before PDM plus fixed additive CCLT graph readout",
            "operator": "shared frozen CPR-DX backbone with duration/history latent and zero-initialized CCLT residual",
            "prefix_ols_initialization": bool(config.get("latent_ols_init", False)),
            "fixed_graph": True,
            "learned_gate": False,
            "router": False,
            "experts": False,
            "posthoc_residual": True,
            "residual_selector": False,
            "cumulative_residual": bool(config.get("hybrid_cumulative_residual", False)),
            "selective_pdm_adaptation": bool(config.get("hybrid_unfreeze_pdm", False)),
            "full_backbone_adaptation": bool(config.get("hybrid_unfreeze_all", False)),
            "phase_cclt_coupling": bool(config.get("phase_cclt_coupling", False)),
            "graph_input_diffusion": bool(config.get("hybrid_graph_input_diffusion", False)),
            "graph_input_order": int(config.get("hybrid_graph_input_order", 1)),
            "graph_input_mode": str(config.get("hybrid_graph_input_mode", "midpoint")),
            "graph_input_operator": "fixed row-normalized (I+A) diffusion with midpoint residual x'=(x+P^k x)/2" if bool(config.get("hybrid_graph_input_diffusion", False)) and str(config.get("hybrid_graph_input_mode", "midpoint")) == "midpoint" else (
                "fixed graph high-pass residual x'=2x-P^k x" if bool(config.get("hybrid_graph_input_diffusion", False)) else None),
            "latent_node_mixing": bool(config.get("hybrid_latent_node_mix", False)),
            "latent_node_mixing_rank": int(config.get("hybrid_latent_node_mix_rank", 8)) if bool(config.get("hybrid_latent_node_mix", False)) else None,
            "latent_node_mixing_operator": "shared learned node projection/reconstruction with exact identity residual (same map at all latent scales; no sample gate/router)" if bool(config.get("hybrid_latent_node_mix", False)) else None,
            "spectral_static_coordinates": bool(config.get("hybrid_spectral_static", False)),
            "spectral_k": int(config.get("hybrid_spectral_k", 0)),
            "spectral_operator": "first non-trivial eigenvectors of symmetric normalized Laplacian; deterministic sign; static latent encoder only" if bool(config.get("hybrid_spectral_static", False)) else None,
        }
    elif latent_multimodal:
        metadata["latent_multimodal_history"] = {
            "historical_multimodal": history_meta,
            "location": "after official value embedding and before PDM at all downsampling scales",
            "operator": ("station-specific causal latent projection + fixed average-pool scale transport"
                         if not bool(config.get("latent_scale_specific", False)) else
                         "scale-specific station latent projection after fixed average-pool transport"),
            "identity_initialization": not bool(config.get("latent_ols_init", False)),
            "prefix_ols_initialization": bool(config.get("latent_ols_init", False)),
            "scale_specific": bool(config.get("latent_scale_specific", False)),
            "fixed_graph_fusion": bool(config.get("latent_graph_fusion", False)),
            "bounded": "tanh latent coordinates",
            "learned_gate": False,
            "router": False,
            "experts": False,
            "posthoc_residual": False,
        }
    elif node_multimodal:
        metadata["node_multimodal_tokens"] = {
            "historical_multimodal": history_meta,
            "token_layout": "[volume, covariates, volume×covariates, static descriptors, volume×static]",
            "location": "station-specific bounded causal token coordinate before TimeMixer normalization, PDM and FMM",
            "station_batching": "station-specific coefficients with shared causal temporal operator",
            "token_dim": int(config.get("multimodal_token_dim", 16)),
            "learned_gate": False,
            "router": False,
            "experts": False,
            "posthoc_residual": False,
        }
    elif multimodal_history:
        metadata["historical_multimodal"] = history_meta
    loss_fn = None
    validation_loss_fn = None
    aux_loss_fn = None
    if fan_frequency:
        metadata["fan_frequency_adaptive_normalization"] = {
            "top_k": int(config.get("fan_freq_topk", 4)),
            "frequency_axis": "history time axis; DC excluded",
            "normalization": "remove instance-wise dominant Fourier components before CPR backbone",
            "adaptation": "shared MLP predicts output-horizon dominant component",
            "loss": "forecast MSE + target-horizon dominant-component MSE",
            "future_statistics": False,
            "learned_gate": False, "router": False, "experts": False,
        }
        fan_weight = float(config.get("fan_nonstat_weight", 1.0))
        fan_topk = int(config.get("fan_freq_topk", 4))
        def _fan_loss(model, x, target):
            pred = _forward(model, x)
            target_dom = model._dominant_component(target, fan_topk)
            pred_dom = getattr(model, "last_pred_dom", None)
            if pred_dom is None or pred_dom.shape != target_dom.shape:
                raise RuntimeError("FAN dominant forecast was not produced")
            return F.mse_loss(pred, target) + fan_weight * F.mse_loss(pred_dom, target_dom)
        loss_fn = _fan_loss
        validation_loss_fn = lambda model, x, target: F.mse_loss(
            _forward(model, x), target)
        metadata["loss_name"] = "fan_forecast_plus_dominant_frequency_mse"
    elif daily_residual:
        metadata["daily_residual_coordinate"] = {
            "period_hours": 24,
            "input": "x_t-x_{t-24} with zero warm-up",
            "backbone_input": "raw volume" if bool(config.get("daily_residual_raw_input", False)) else "causal daily innovation",
            "reconstruction": "previous_day + predicted_innovation",
            "learned_gate": False, "router": False, "experts": False,
            "posthoc_residual": False,
        }
        metadata["loss_name"] = "direct_forecast_loss_on_daily_residual_coordinate"
    elif cedar:
        loss_fn = lambda model, x, target: _cedar_invariant_loss(
            model, x, target,
            discrepancy_weight=float(config.get("cedar_discrepancy_weight", 0.05)),
            mmd_weight=float(config.get("cedar_mmd_weight", 0.05)))
        validation_loss_fn = lambda model, x, target: F.mse_loss(
            _forward(model, x), target)
        metadata["loss_name"] = "cedar_forecast_plus_shared_latent_mmd_loss_discrepancy"
    elif tariff:
        source_std = np.asarray(data["scaler_std"], dtype=np.float32)
        median_std = max(float(np.median(source_std)), 1e-6)
        exponent = float(config.get("scale_exponent", 0.5))
        channel_weights = np.clip((source_std / median_std) ** exponent, 0.5, 4.0)
        channel_weights /= max(float(channel_weights.mean()), 1e-6)
        channel_weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        factors = tuple(int(v.strip()) for v in str(
            config.get("consistency_factors", "2,3,4,6,12")
        ).split(",") if v.strip())
        aux_loss_fn = lambda model, x, target, aux: _tariff_multiresolution_loss(
            model, x, target, aux, channel_weights=channel_weights,
            consistency_weight=float(config.get("consistency_weight", 0.15)),
            factors=factors)
    elif period_aligned and component_supervision:
        loss_fn = lambda model, x, target: _component_loss(
            model, x, target, reconstruction_weight=reconstruction_weight)
    elif wavelet:
        loss_fn = lambda model, x, target: _wavelet_loss(
            model, x, target, reconstruction_weight=reconstruction_weight)
    elif spectral:
        loss_fn = lambda model, x, target: _spectral_loss(
            model, x, target, spectral_weight=float(config.get("spectral_lambda", 0.1)))
    elif localized_tf:
        loss_fn = lambda model, x, target: _localized_tf_loss(
            model, x, target, spectral_weight=float(config.get("spectral_lambda", 0.01)))
    elif apid and bool(config.get("apid_peak_branch", 0)):
        loss_fn = lambda model, x, target: _apid_peak_loss(
            model, x, target, peak_lambda=float(config.get("apid_peak_lambda", 0.15)))
    elif cpr_regime:
        source_std = np.asarray(data["scaler_std"], dtype=np.float32)
        median_std = max(float(np.median(source_std)), 1e-6)
        exponent = float(config.get("scale_exponent", 0.5))
        channel_weights = np.clip((source_std / median_std) ** exponent, 0.5, 4.0)
        channel_weights /= max(float(channel_weights.mean()), 1e-6)
        channel_weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        raw_factors = str(config.get("consistency_factors", "2,3,4,6,12"))
        factors = tuple(int(v.strip()) for v in raw_factors.split(",") if v.strip())
        robust_weight = float(config.get("robust_mae_weight", 0.1))
        moment_weight = float(config.get("cpr_regime_moment_weight", 0.05))
        loss_fn = lambda model, x, target: _cpr_regime_moment_loss(
            model, x, target, channel_weights=channel_weights,
            consistency_weight=float(config.get("consistency_weight", 0.15)),
            factors=factors, robust_mae_weight=robust_weight,
            moment_weight=moment_weight)
        validation_loss_fn = lambda model, x, target: _multiresolution_scale_loss(
            model, x, target, channel_weights=channel_weights,
            consistency_weight=float(config.get("consistency_weight", 0.15)),
            factors=factors)
        metadata["causal_affine_regime_loss"] = {
            "forecast": "scale-weighted robust MSE/MAE + multi-resolution consistency",
            "moment_terms": "future 24-hour mean and standard deviation",
            "moment_weight": moment_weight,
            "fit_split": "train_windows_only",
            "validation_selection": "forecast scale/MRC loss only",
        }
    elif cpr_periodic_state:
        source_std = np.asarray(data["scaler_std"], dtype=np.float32)
        median_std = max(float(np.median(source_std)), 1e-6)
        exponent = float(config.get("scale_exponent", 0.5))
        channel_weights = np.clip((source_std / median_std) ** exponent, 0.5, 4.0)
        channel_weights /= max(float(channel_weights.mean()), 1e-6)
        channel_weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        raw_factors = str(config.get("consistency_factors", "2,3,4,6,12"))
        factors = tuple(int(v.strip()) for v in raw_factors.split(",") if v.strip())
        robust_weight = float(config.get("robust_mae_weight", 0.1))
        loss_fn = lambda model, x, target: _robust_multiresolution_scale_loss(
            model, x, target, channel_weights=channel_weights,
            consistency_weight=float(config.get("consistency_weight", 0.15)),
            factors=factors, robust_mae_weight=robust_weight)
        validation_loss_fn = lambda model, x, target: _multiresolution_scale_loss(
            model, x, target, channel_weights=channel_weights,
            consistency_weight=float(config.get("consistency_weight", 0.15)),
            factors=factors)
        metadata["causal_periodic_state_loss"] = {
            "forecast": "scale-weighted robust MSE/MAE + multi-resolution consistency",
            "state_fit": "causal observed six-cycle phase template and latest-cycle level shift",
            "fit_split": "train_windows_only",
            "validation_selection": "forecast scale/MRC loss only",
        }
    elif bool(config.get("tail_balanced_loss", 0)):
        # Fit zone weights only from the train-prefix raw demand.  The values
        # in raw_splits are train-standardized; reconstructing them with the
        # train-only scaler keeps validation/test labels out of the rule.
        train_scaled = np.asarray(data["raw_splits"]["train"], dtype=np.float32)
        train_mean = np.asarray(data["scaler_mean"], dtype=np.float32)
        train_std = np.asarray(data["scaler_std"], dtype=np.float32)
        zone_mean = (train_scaled * train_std[None, :] + train_mean[None, :]).mean(axis=0)
        ranks = np.argsort(np.argsort(zone_mean)).astype(np.float32)
        ranks = ranks / max(float(len(zone_mean) - 1), 1.0)
        lo = float(config.get("tail_weight_low", 1.25))
        hi = float(config.get("tail_weight_high", 0.75))
        if lo < hi or lo <= 0 or hi <= 0:
            raise ValueError("tail_weight_low must be >= tail_weight_high > 0")
        channel_weights = hi + (lo - hi) * (1.0 - ranks)
        channel_weights = np.clip(channel_weights, min(hi, lo), max(hi, lo))
        channel_weights /= max(float(channel_weights.mean()), 1e-6)
        channel_weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        raw_factors = str(config.get("consistency_factors", "2,3,4,6,12"))
        factors = tuple(int(v.strip()) for v in raw_factors.split(",") if v.strip())
        loss_fn = lambda model, x, target: _tail_balanced_multiresolution_loss(
            model, x, target, channel_weights=channel_weights,
            consistency_weight=float(config.get("consistency_weight", 0.15)),
            factors=factors)
        metadata["tail_balanced_loss_spec"] = {
            "fit_split": "train_prefix_only",
            "zone_statistic": "raw_train_zone_mean_rank",
            "low_demand_weight": lo,
            "high_demand_weight": hi,
            "weight_min": float(channel_weights.min()),
            "weight_max": float(channel_weights.max()),
            "weight_mean": float(channel_weights.mean()),
            "consistency_factors": list(factors),
        }
    elif future_stats_supervised:
        source_std = np.asarray(data["scaler_std"], dtype=np.float32)
        median_std = max(float(np.median(source_std)), 1e-6)
        exponent = float(config.get("scale_exponent", 0.5))
        channel_weights = np.clip((source_std / median_std) ** exponent, 0.5, 4.0)
        channel_weights /= max(float(channel_weights.mean()), 1e-6)
        channel_weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        raw_factors = str(config.get("consistency_factors", "2,3,4,6,12"))
        factors = tuple(int(v.strip()) for v in raw_factors.split(",") if v.strip())
        aux_weight = float(config.get("future_stats_aux_weight", 0.1))
        loss_fn = lambda model, x, target: _future_stats_supervised_loss(
            model, x, target, channel_weights=channel_weights,
            consistency_weight=float(config.get("consistency_weight", 0.15)),
            factors=factors, aux_weight=aux_weight)
        validation_loss_fn = lambda model, x, target: _multiresolution_scale_loss(
            model, x, target, channel_weights=channel_weights,
            consistency_weight=float(config.get("consistency_weight", 0.15)),
            factors=factors)
        metadata["future_stats_supervision"] = {
            "fit_split": "train_prefix_only",
            "target": "future_window_zone_mean_and_log_std_delta",
            "history_features": "24/72/168h mean/logstd + recent drift",
            "aux_weight": aux_weight,
            "validation_auxiliary_loss": False,
            "consistency_factors": list(factors),
        }
    elif bool(config.get("scale_aware_loss", 0)) or float(config.get("consistency_weight", 0.0)) > 0:
        source_std = np.asarray(data["scaler_std"], dtype=np.float32)
        median_std = max(float(np.median(source_std)), 1e-6)
        exponent = float(config.get("scale_exponent", 0.5))
        channel_weights = np.clip((source_std / median_std) ** exponent, 0.5, 4.0)
        channel_weights /= max(float(channel_weights.mean()), 1e-6)
        channel_weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        if float(config.get("robust_mae_weight", 0.0)) > 0:
            raw_factors = str(config.get("consistency_factors", "2,3,4,6,12"))
            factors = tuple(int(v.strip()) for v in raw_factors.split(",") if v.strip())
            if bool(config.get("foil_invariant_loss", 0)):
                invariant_weight = float(config.get("foil_invariant_weight", 0.1))
                loss_fn = lambda model, x, target: _foil_invariant_multiresolution_loss(
                    model, x, target, channel_weights=channel_weights,
                    consistency_weight=float(config.get("consistency_weight", 0.15)),
                    factors=factors,
                    robust_mae_weight=float(config.get("robust_mae_weight", 0.1)),
                    invariant_weight=invariant_weight)
                validation_loss_fn = lambda model, x, target: _multiresolution_scale_loss(
                    model, x, target, channel_weights=channel_weights,
                    consistency_weight=float(config.get("consistency_weight", 0.15)),
                    factors=factors)
                metadata["foil_invariant_loss"] = {
                    "environments": "three causal 24-hour level-drift terciles",
                    "fit_split": "train_windows_only",
                    "invariant_weight": invariant_weight,
                    "validation_selection": "forecast scale/MRC loss only",
                }
            elif bool(config.get("transition_weighted_loss", 0)):
                loss_fn = lambda model, x, target: _transition_weighted_multiresolution_loss(
                    model, x, target, channel_weights=channel_weights,
                    consistency_weight=float(config.get("consistency_weight", 0.15)),
                    factors=factors,
                    robust_mae_weight=float(config.get("robust_mae_weight", 0.1)))
            else:
                loss_fn = lambda model, x, target: _robust_multiresolution_scale_loss(
                    model, x, target, channel_weights=channel_weights,
                    consistency_weight=float(config.get("consistency_weight", 0.15)),
                    factors=factors,
                    robust_mae_weight=float(config.get("robust_mae_weight", 0.1)),
                    residual_strength_reg=float(config.get("residual_strength_reg", 0.0)))
        elif float(config.get("consistency_weight", 0.0)) > 0:
            raw_factors = str(config.get("consistency_factors", "2,3,4,6,12"))
            factors = tuple(int(v.strip()) for v in raw_factors.split(",") if v.strip())
            loss_fn = lambda model, x, target: _multiresolution_scale_loss(
                model, x, target, channel_weights=channel_weights,
                consistency_weight=float(config.get("consistency_weight", 0.15)),
                factors=factors)
        else:
            loss_fn = lambda model, x, target: _scale_weighted_loss(
                model, x, target, channel_weights=channel_weights)
    if latent_multimodal or e2e_multimodal or node_multimodal or multimodal_history:
        source_std = np.asarray(data["scaler_std"], dtype=np.float32)
        median_std = max(float(np.median(source_std)), 1e-6)
        exponent = float(config.get("scale_exponent", 0.5))
        channel_weights = np.clip((source_std / median_std) ** exponent, 0.5, 4.0)
        channel_weights /= max(float(channel_weights.mean()), 1e-6)
        channel_weights = torch.as_tensor(channel_weights, dtype=torch.float32)
        factors = tuple(int(v.strip()) for v in str(config.get("consistency_factors", "2,3,4,6,12")).split(",") if v.strip())
        loss_key = ("hybrid_latent_cclt" if hybrid_latent_cclt else
                    "e2e_multimodal_history" if e2e_multimodal else
                    "latent_multimodal_history" if latent_multimodal else
                    "node_multimodal_tokens" if node_multimodal else
                    "phase_region_state" if phase_region_state else "historical_multimodal")
        if bool(config.get("history_cmi_pdm_loss", False)):
            # CMI-PDM applies the causal environment penalty to every
            # ungated PDM/FMM scale forecast exposed by the single backbone.
            # Validation still selects the ordinary raw-scale forecast loss.
            aux_loss_fn = lambda model, x, target, aux: _history_cmi_pdm_loss(
                model, x, target, aux, channel_weights=channel_weights,
                consistency_weight=float(config.get("consistency_weight", 0.15)),
                factors=factors,
                robust_mae_weight=float(config.get("robust_mae_weight", 0.1)),
                invariant_weight=float(config.get("history_cmi_pdm_weight", 0.05)),
                drift_feature_index=int(config.get("history_cmi_pdm_drift_feature_index", 0)))
            metadata[loss_key]["loss"] = "causal multi-resolution invariant PDM"
            metadata[loss_key]["history_cmi_pdm"] = {
                "environment_source": "observed auxiliary history only",
                "drift": "latest 24h mean minus preceding 24h mean",
                "partition": "batch-local causal drift terciles",
                "scale_supervision": "individual ungated PDM/FMM forecasts plus summed forecast",
                "invariant_weight": float(config.get("history_cmi_pdm_weight", 0.05)),
                "validation_selection": "raw-scale forecast MSE/MRC only",
                "future_labels_used_for_environment": False,
                "gate_router_expert": False,
            }
            metadata["loss_name"] = "history_cmi_pdm_invariant_multiresolution"
        elif bool(config.get("history_foil_invariant_loss", False)):
            # Environment construction is deliberately attached to the
            # history-aware auxiliary callback.  The callback receives the
            # observed ``aux`` tensor and therefore cannot accidentally fall
            # back to the future target or a volume coordinate.  Validation
            # remains the raw scale/MRC objective below.
            aux_loss_fn = lambda model, x, target, aux: _history_foil_invariant_multiresolution_loss(
                model, x, target, aux, channel_weights=channel_weights,
                consistency_weight=float(config.get("consistency_weight", 0.15)),
                factors=factors,
                robust_mae_weight=float(config.get("robust_mae_weight", 0.1)),
                invariant_weight=float(config.get("history_foil_invariant_weight", 0.1)),
                drift_feature_index=int(config.get("history_foil_drift_feature_index", 0)))
            metadata[loss_key]["loss"] = "history-aware causal drift invariant MRC"
            metadata[loss_key]["history_foil_invariant"] = {
                "environment_source": "observed auxiliary history only",
                "drift": "latest 24h mean minus preceding 24h mean",
                "partition": "batch-local terciles",
                "feature_index": int(config.get("history_foil_drift_feature_index", 0)),
                "invariant_weight": float(config.get("history_foil_invariant_weight", 0.1)),
                "validation_selection": "raw-scale forecast MSE/MRC only",
                "future_labels_used_for_environment": False,
            }
            metadata["loss_name"] = "history_foil_invariant_multiresolution"
        elif bool(config.get("history_raw_mae_objective", False)):
            # The labels are standardized, but the protocol scores raw
            # volumes.  This branch multiplies |e_std| by the train-prefix
            # standard deviation, giving an exact raw-unit MAE objective.
            # It is a fixed metric alignment, not a tuned exponent/gate.
            aux_loss_fn = lambda model, x, target, aux: _history_raw_mae_multiresolution_loss(
                model, x, target, aux, scaler_std=source_std,
                consistency_weight=float(config.get("consistency_weight", 0.15)),
                factors=factors)
        elif bool(config.get("history_l1_objective", False)):
            aux_loss_fn = lambda model, x, target, aux: _history_aux_l1_multiresolution_loss(
                model, x, target, aux, channel_weights=channel_weights,
                consistency_weight=float(config.get("consistency_weight", 0.15)),
                factors=factors)
        else:
            aux_loss_fn = lambda model, x, target, aux: _history_aux_multiresolution_loss(
                model, x, target, aux, channel_weights=channel_weights,
                consistency_weight=float(config.get("consistency_weight", 0.15)),
                factors=factors, robust_mae_weight=float(config.get("robust_mae_weight", 0.1)))
        metadata[loss_key]["loss"] = (
            "history-aware causal drift invariant MRC"
            if bool(config.get("history_foil_invariant_loss", False)) else
            "history-aware raw-unit MAE multiresolution loss"
            if bool(config.get("history_raw_mae_objective", False)) else
            "history-aware absolute-error multiresolution loss"
            if bool(config.get("history_l1_objective", False))
            else "history-aware robust multiresolution scale loss")
        # Keep the top-level provenance field synchronized with the actual
        # auxiliary objective; otherwise a raw-MAE run is mislabeled as the
        # default robust MRC in result.json.
        if bool(config.get("history_raw_mae_objective", False)):
            metadata["loss_name"] = "history_raw_unit_mae_multiresolution"
        elif bool(config.get("history_l1_objective", False)):
            metadata["loss_name"] = "history_l1_multiresolution"
    if phase_kd:
        # Teacher responses are precomputed from source train/validation
        # windows only.  The teacher checkpoint is frozen and never receives
        # validation labels; inference later uses the student alone.
        teacher_path = config.get("teacher_checkpoint")
        if not teacher_path:
            raise ValueError("phase_kd_timemixer requires teacher_checkpoint")
        import sys
        optional = str(_ROOT / "baselines")
        if optional not in sys.path:
            sys.path.insert(0, optional)
        from scprc.baselines.phaseformer_adapter import PhaseFormerUrbanEV
        teacher = PhaseFormerUrbanEV(
            seq_len=int(config["seq_len"]), pred_len=pred_len, channels=channels,
            period_len=24, latent_dim=int(config.get("teacher_latent_dim", 32)),
            phase_layers=int(config.get("teacher_phase_layers", 2)),
            routers=int(config.get("teacher_routers", 1)),
            heads=int(config.get("teacher_heads", 8)),
            dropout=float(config.get("teacher_dropout", 0.1)),
            pos_embed=bool(config.get("teacher_pos_embed", True)),
        ).float()
        payload = torch.load(Path(teacher_path), map_location="cpu", weights_only=False)
        teacher.load_state_dict(payload["model"], strict=True)
        teacher.eval()
        teacher_device = torch.device(str(config.get("device", "cuda")))
        if teacher_device.type == "cuda" and not torch.cuda.is_available():
            teacher_device = torch.device("cpu")
        teacher.to(teacher_device)

        def teacher_predict(values: np.ndarray) -> np.ndarray:
            outputs = []
            with torch.no_grad():
                bs = max(1, int(config.get("batch_size", 32)))
                for start in range(0, len(values), bs):
                    xb = torch.from_numpy(values[start:start + bs]).to(teacher_device)
                    outputs.append(teacher(xb).detach().cpu().numpy())
            return np.concatenate(outputs, axis=0).astype(np.float32)

        teacher_train = teacher_predict(np.asarray(data["train_x"], dtype=np.float32))
        teacher_val = teacher_predict(np.asarray(data["val_x"], dtype=np.float32))
        source_std = np.asarray(data["scaler_std"], dtype=np.float32)
        median_std = max(float(np.median(source_std)), 1e-6)
        exponent = float(config.get("scale_exponent", 0.5))
        channel_weights_kd = np.clip((source_std / median_std) ** exponent, 0.5, 4.0)
        channel_weights_kd /= max(float(channel_weights_kd.mean()), 1e-6)
        channel_weights_kd = torch.as_tensor(channel_weights_kd, dtype=torch.float32)
        raw_factors_kd = str(config.get("consistency_factors", "2,3,4,6,12"))
        factors_kd = tuple(int(v.strip()) for v in raw_factors_kd.split(",") if v.strip())
        kd_weight = float(config.get("kd_weight", 0.25))
        kd_loss = lambda model, x, target, teacher_target: _distilled_multiresolution_loss(
            model, x, target, teacher_target,
            channel_weights=channel_weights_kd,
            consistency_weight=float(config.get("consistency_weight", 0.15)),
            factors=factors_kd, kd_weight=kd_weight)
        metadata["phase_distillation_teacher_checkpoint"] = str(teacher_path)
        metadata["phase_distillation_kd_weight"] = kd_weight
        aux_data = {"train": teacher_train, "val": teacher_val}
        kd_forward = lambda model, x, _teacher: _forward(model, x)
        return _fit_model(config, data, run_dir, factory, kd_forward, metadata,
                          aux_data=aux_data, aux_loss_fn=kd_loss)
    if (bool(config.get("precision_fusion", False)) and hybrid_latent_cclt
            and not config.get("precision_fusion_weights")):
        # Build one copy without the fixed projection, load the previously
        # trained CPR checkpoint, and estimate a single global BLUE projection
        # from train-prefix forecast residuals.  The estimate is then frozen
        # in the actual model factory; no validation labels or weight search
        # enter this calibration.
        calibration_checkpoint = config.get("precision_fusion_calibration_checkpoint")
        if not calibration_checkpoint:
            raise ValueError("precision fusion requires calibration checkpoint")
        args.precision_fusion = False
        calibration_model = factory().float()
        payload = torch.load(Path(calibration_checkpoint), map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        compatible = {key: value for key, value in state.items()
                      if key in calibration_model.state_dict()
                      and tuple(value.shape) == tuple(calibration_model.state_dict()[key].shape)}
        calibration_model.load_state_dict(compatible, strict=False)
        if len(compatible) < int(0.9 * max(len(calibration_model.state_dict()), 1)):
            raise ValueError("precision fusion calibration checkpoint is incompatible")
        device_name = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        calibration_device = torch.device(device_name)
        if calibration_device.type == "cuda" and not torch.cuda.is_available():
            calibration_device = torch.device("cpu")
        calibration_model.to(calibration_device).eval()
        scale_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []
        batch_size = max(1, int(config.get("batch_size", 32)))
        with torch.no_grad():
            for start in range(0, len(data["train_x"]), batch_size):
                stop = min(start + batch_size, len(data["train_x"]))
                xb = torch.from_numpy(np.asarray(data["train_x"][start:stop], dtype=np.float32)).to(calibration_device)
                ab = torch.from_numpy(np.asarray(history_aux["train"][start:stop], dtype=np.float32)).to(calibration_device)
                _ = calibration_model(xb, None, None, None, None, ab)
                scales = getattr(calibration_model.backbone, "last_scale_forecasts", None)
                if scales is None or len(scales) < 2:
                    raise RuntimeError("precision fusion calibration could not capture FMM scales")
                scale_chunks.append(torch.stack(scales, dim=-1).cpu().numpy())
                target_chunks.append(np.asarray(data["train_y"][start:stop], dtype=np.float32))
        scale_values = np.concatenate(scale_chunks, axis=0)  # [B,H,N,S]
        target_values = np.concatenate(target_chunks, axis=0)
        residual = scale_values - target_values[..., None]
        bias = residual.mean(axis=(0, 2))
        centered = residual - bias[None, :, None, :]
        horizon_weights = []
        for horizon in range(centered.shape[1]):
            matrix = centered[:, horizon].reshape(-1, centered.shape[-1]).astype(np.float64)
            covariance = (matrix.T @ matrix) / max(matrix.shape[0] - 1, 1)
            # A deterministic numerical floor keeps the analytic precision
            # projection well-conditioned without a searched shrinkage knob.
            covariance = covariance + np.eye(covariance.shape[0], dtype=np.float64) * 1e-6
            try:
                inverse = np.linalg.pinv(covariance, hermitian=True)
            except TypeError:
                inverse = np.linalg.pinv(covariance)
            ones = np.ones(covariance.shape[0], dtype=np.float64)
            weight = inverse @ ones
            denominator = float(ones @ weight)
            weight = weight / denominator if abs(denominator) > 1e-10 else ones / len(ones)
            # Fixed simplex projection avoids a negative scale contribution;
            # this is a deterministic feasibility rule, not parameter tuning.
            weight = np.maximum(weight, 0.0)
            weight = weight / max(float(weight.sum()), 1e-12)
            horizon_weights.append(weight.astype(np.float32))
        precision_weights = np.stack(horizon_weights, axis=0)
        config["precision_fusion_weights"] = precision_weights.tolist()
        config["precision_fusion_bias"] = bias.astype(np.float32).tolist()
        args.precision_fusion = True
        args.precision_fusion_weights = config["precision_fusion_weights"]
        args.precision_fusion_bias = config["precision_fusion_bias"]
        metadata["precision_fusion"] = {
            "operator": "train-prefix residual covariance BLUE with fixed simplex feasibility projection",
            "calibration_checkpoint": str(calibration_checkpoint),
            "fit_split": "train_windows_only",
            "sample_conditioning": False,
            "future_labels_used": False,
            "gate_router_expert": False,
        }
        metadata["loss_name"] = metadata.get("loss_name", "history-aware robust multiresolution scale loss")

    if latent_multimodal or node_multimodal or multimodal_history:
        data_for_resume = dict(data)
        for split, values in history_aux.items():
            data_for_resume[f"{split}_aux"] = values
        forward = lambda model, x, aux: model(x, None, None, None, None, aux)
        return _fit_model(config, data_for_resume, run_dir, factory, forward, metadata,
                          aux_data=history_aux, loss_fn=loss_fn,
                          validation_loss_fn=validation_loss_fn,
                          aux_loss_fn=aux_loss_fn,
                          aux_validation_loss_fn=(
                              (lambda model, x, target, aux: _history_raw_scale_mse_loss(
                                  model, x, target, aux,
                                  np.asarray(data["scaler_std"], dtype=np.float32)))
                              if (node_multimodal or latent_multimodal or e2e_multimodal or hybrid_latent_cclt) else None))
    if tariff:
        data_for_resume = dict(data)
        data_for_resume["train_aux"] = tariff_aux["train"]
        data_for_resume["val_aux"] = tariff_aux["val"]
        forward = lambda model, x, aux: model(x, None, None, aux)
        return _fit_model(config, data_for_resume, run_dir, factory, forward, metadata,
                          aux_data=tariff_aux, aux_loss_fn=aux_loss_fn)
    return _fit_model(config, data, run_dir, factory, _forward, metadata,
                      loss_fn=loss_fn, validation_loss_fn=validation_loss_fn)


__all__ = ["fit_predict"]
