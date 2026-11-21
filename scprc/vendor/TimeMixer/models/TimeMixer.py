import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp



class CausalEMADecomp(nn.Module):
    """Causal exponential level/trend decomposition for UrbanEV.

    The official moving-average decomposition is centered inside the observed
    window and can dilute the state at the forecast boundary. This operator
    carries a one-sided level state to the right edge, then exposes the causal
    innovation as season. ``alpha`` is fixed by the experiment config.
    """

    def __init__(self, alpha: float = 0.25):
        super().__init__()
        alpha = float(alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("causal_ema_alpha must lie in (0, 1]")
        self.alpha = alpha

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("causal EMA expects [batch,time,channels]")
        level = [x[:, :1]]
        for index in range(1, x.size(1)):
            level.append(self.alpha * x[:, index:index + 1] +
                         (1.0 - self.alpha) * level[-1])
        trend = torch.cat(level, dim=1)
        return x - trend, trend


class RobustHuberDecomp(nn.Module):
    """Fixed Hampel/Huber robust moving-average decomposition.

    The official centered average can turn isolated charging spikes into a
    low-frequency trend.  This replacement computes a local median/MAD and a
    deterministic Huber influence weight in the same 25-step window, then
    keeps the exact residual ``season=x-trend``.  It has no learned scale,
    gate, router, or searched threshold.
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size < 3 or self.kernel_size % 2 != 1:
            raise ValueError("robust Huber window must be odd and at least three")

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("robust Huber decomposition expects [batch,time,channels]")
        half = self.kernel_size // 2
        # The first implementation materialized ``[B,C,T,K]`` windows.  On
        # UrbanEV's long context this makes autograd retain several hundred
        # MB at every TimeMixer scale.  The same fixed Huber influence rule
        # can be written with two depthwise convolutions, so its memory is
        # linear in ``[B,T,C]`` and it remains entirely deterministic.
        channels = int(x.size(-1))
        kernel = x.new_full((channels, 1, self.kernel_size),
                            1.0 / float(self.kernel_size))
        values = F.pad(x.transpose(1, 2), (half, half), mode="replicate")
        center = F.conv1d(values, kernel, groups=channels).transpose(1, 2)
        abs_dev = (x - center).abs()
        mad = F.conv1d(F.pad(abs_dev.transpose(1, 2), (half, half), mode="replicate"),
                       kernel, groups=channels).transpose(1, 2)
        delta = (1.345 * 1.4826 * mad).clamp_min(1e-3)
        influence = torch.minimum(torch.ones_like(abs_dev),
                                  delta / abs_dev.clamp_min(1e-6))
        numerator = F.conv1d(
            F.pad((x * influence).transpose(1, 2), (half, half), mode="replicate"),
            kernel, groups=channels).transpose(1, 2)
        denominator = F.conv1d(
            F.pad(influence.transpose(1, 2), (half, half), mode="replicate"),
            kernel, groups=channels).transpose(1, 2)
        trend = numerator / denominator.clamp_min(1e-6)
        return x - trend, trend


class CausalDailyDecomp(nn.Module):
    """Causal same-phase decomposition with exact reconstruction.

    For each observed position, the seasonal coordinate is the mean of the
    preceding same-clock positions from the available daily cycles.  The
    trend/innovation coordinate is the exact difference ``x-season``.  The
    period is inferred from the current multiscale length (24, 12, 6, ...),
    so no future row and no searched smoothing parameter enters the split.
    """

    def __init__(self, base_period: int = 24, cycles: int = 7):
        super().__init__()
        self.base_period = int(base_period)
        self.cycles = int(cycles)
        if self.base_period < 1 or self.cycles < 1:
            raise ValueError("causal daily decomposition requires positive period/cycles")

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("causal daily decomposition expects [batch,time,channels]")
        length = int(x.size(1))
        # UrbanEV's downsampled scales preserve the 24-hour phase grid.
        period = max(1, int(round(self.base_period * length / 168.0)))
        seasonal = torch.zeros_like(x)
        counts = torch.zeros(length, device=x.device, dtype=x.dtype)
        for lag_index in range(1, self.cycles + 1):
            lag = lag_index * period
            if lag >= length:
                break
            seasonal[:, lag:] += x[:, :-lag]
            counts[lag:] += 1.0
        seasonal = seasonal / counts.clamp_min(1.0).view(1, length, 1)
        return x - seasonal, seasonal


class CausalFIRDecomp(nn.Module):
    """Learnable positive causal FIR level with exact innovation residual.

    The kernel starts as the official moving-average window but is learned per
    latent channel.  Softmax normalization keeps it a stable averaging
    operator, while left-only padding prevents future leakage.  The two
    returned coordinates always reconstruct the input exactly.
    """

    def __init__(self, d_model: int, kernel_size: int = 25):
        super().__init__()
        self.d_model = int(d_model)
        self.kernel_size = int(kernel_size)
        if self.d_model < 1 or self.kernel_size < 1:
            raise ValueError("causal FIR decomposition dimensions must be positive")
        self.logits = nn.Parameter(torch.zeros(self.d_model, self.kernel_size))

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("causal FIR decomposition expects [batch,time,d_model]")
        weights = torch.softmax(self.logits, dim=-1).unsqueeze(1)
        value = x.transpose(1, 2)
        value = F.pad(value, (self.kernel_size - 1, 0))
        trend = F.conv1d(value, weights, groups=self.d_model).transpose(1, 2)
        return x - trend, trend


class CausalRegimeState(nn.Module):
    """Fixed causal level/drift/volatility coordinates for regime shifts.

    The coordinates are computed only from the observed prefix and injected
    into every PDM scale through one shared linear map.  This is a state
    representation, not a sample-dependent gate or an output residual.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(5, int(d_model))
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    @staticmethod
    def _mean_tail(x: torch.Tensor, width: int) -> torch.Tensor:
        # x is [B,N,T]; all widths are available for UrbanEV's 168-hour
        # history.  The final window is causal and contains no target rows.
        return x[..., -width:].mean(dim=-1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.size(1) < 168:
            raise ValueError("regime state expects [B,time,nodes] with >=168 history")
        x = history.permute(0, 2, 1)
        m24 = self._mean_tail(x, 24)
        m72 = self._mean_tail(x, 72)
        m168 = self._mean_tail(x, 168)
        previous24 = x[..., -48:-24].mean(dim=-1)
        volatility = x[..., -24:].std(dim=-1, unbiased=False)
        phase_lag = (x[..., -24:] - x[..., -48:-24]).mean(dim=-1)
        features = torch.stack((m24 - m72, m72 - m168,
                                m24 - previous24, volatility, phase_lag), dim=-1)
        return self.projection(features)


class PhaseCycleTokenOperator(nn.Module):
    """Explicit 24-phase x 7-cycle tokenization before TimeMixer PDM.

    The operator encodes the seven observed values at each clock phase, routes
    information across the 24 phase tokens with shared routers, then decodes
    phase tokens back to seven cycle-specific latent tokens. It changes the
    representation seen by PDM; it is not a forecast gate or output residual.
    """

    def __init__(self, d_model: int, cycles: int = 7, phases: int = 24,
                 routers: int = 2, heads: int = 4, dropout: float = 0.1,
                 use_pos_embed: bool = False):
        super().__init__()
        if d_model < 1 or cycles < 1 or phases < 1 or d_model % heads:
            raise ValueError("invalid phase-cycle operator dimensions")
        self.d_model, self.cycles, self.phases = int(d_model), int(cycles), int(phases)
        self.use_pos_embed = bool(use_pos_embed)
        self.encode = nn.Sequential(nn.Linear(self.cycles * self.d_model, self.d_model),
                                    nn.LayerNorm(self.d_model))
        if self.use_pos_embed:
            self.pos_embedding = nn.Parameter(torch.zeros(self.phases, self.d_model))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.router = nn.Parameter(torch.randn(int(routers), self.d_model) *
                                   (self.d_model ** -0.5))
        self.sender = nn.MultiheadAttention(self.d_model, int(heads),
                                            dropout=float(dropout), batch_first=True)
        self.receiver = nn.MultiheadAttention(self.d_model, int(heads),
                                              dropout=float(dropout), batch_first=True)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, 4 * self.d_model),
                                 nn.GELU(), nn.Linear(4 * self.d_model, self.d_model))
        self.dropout = nn.Dropout(float(dropout))
        self.decode = nn.Linear(self.d_model, self.cycles * self.d_model)

    def forward(self, x):
        if x.ndim != 3 or x.size(1) != self.cycles * self.phases:
            raise ValueError("phase-cycle operator expects [batch, 168, d]")
        batch = x.size(0)
        # [B, day, phase, d] -> [B, phase, day, d]. Each phase token sees all
        # seven historical cycles before cross-phase routing.
        phase_series = x.reshape(batch, self.cycles, self.phases, self.d_model)
        phase_series = phase_series.permute(0, 2, 1, 3)
        encoded = self.encode(phase_series.reshape(batch, self.phases,
                                                    self.cycles * self.d_model))
        if self.use_pos_embed:
            encoded = encoded + self.pos_embedding.unsqueeze(0)
        query = self.router.unsqueeze(0).expand(batch, -1, -1)
        pooled, _ = self.sender(query, encoded, encoded, need_weights=False)
        received, _ = self.receiver(encoded, pooled, pooled, need_weights=False)
        hidden = self.norm1(encoded + self.dropout(received))
        hidden = self.norm2(hidden + self.dropout(self.mlp(hidden)))
        decoded = self.decode(hidden).reshape(batch, self.phases, self.cycles, self.d_model)
        return decoded.permute(0, 2, 1, 3).reshape(batch, self.cycles * self.phases, self.d_model)


class JointPhaseRegionOperator(nn.Module):
    """Separable phase-then-region routing on the full UrbanEV tensor.

    Unlike ``PhaseCycleTokenOperator``, this block retains the zone axis while
    routing: seven daily states are encoded for each of 24 phases, then each
    phase routes across all zones with a second router bank.  The result is
    decoded back to the complete 7x24 time grid before PDM.  It is a
    representation operator, not a forecast gate or output correction.
    """

    def __init__(self, d_model: int, nodes: int, cycles: int = 7,
                 phases: int = 24, phase_routers: int = 2,
                 region_routers: int = 4, phase_heads: int = 4,
                 region_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if (d_model < 1 or nodes < 1 or cycles < 1 or phases < 1 or
                d_model % phase_heads or d_model % region_heads):
            raise ValueError("invalid joint phase-region operator dimensions")
        self.d_model, self.nodes = int(d_model), int(nodes)
        self.cycles, self.phases = int(cycles), int(phases)
        self.phase_encode = nn.Sequential(
            nn.Linear(self.cycles * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.phase_router = nn.Parameter(
            torch.randn(int(phase_routers), self.d_model) * (self.d_model ** -0.5)
        )
        self.phase_sender = nn.MultiheadAttention(
            self.d_model, int(phase_heads), dropout=float(dropout), batch_first=True
        )
        self.phase_receiver = nn.MultiheadAttention(
            self.d_model, int(phase_heads), dropout=float(dropout), batch_first=True
        )
        self.phase_norm1 = nn.LayerNorm(self.d_model)
        self.phase_norm2 = nn.LayerNorm(self.d_model)
        self.phase_mlp = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model), nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model)
        )
        self.region_embedding = nn.Parameter(
            torch.randn(self.nodes, self.d_model) * (self.d_model ** -0.5)
        )
        self.region_router = nn.Parameter(
            torch.randn(int(region_routers), self.d_model) * (self.d_model ** -0.5)
        )
        self.region_sender = nn.MultiheadAttention(
            self.d_model, int(region_heads), dropout=float(dropout), batch_first=True
        )
        self.region_receiver = nn.MultiheadAttention(
            self.d_model, int(region_heads), dropout=float(dropout), batch_first=True
        )
        self.region_norm1 = nn.LayerNorm(self.d_model)
        self.region_norm2 = nn.LayerNorm(self.d_model)
        self.region_mlp = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model), nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model)
        )
        self.dropout = nn.Dropout(float(dropout))
        self.decode = nn.Linear(self.d_model, self.cycles * self.d_model)

    def forward(self, x):
        if x.ndim != 3 or x.size(1) != self.cycles * self.phases:
            raise ValueError("joint phase-region operator expects [batch*zones, 168, d]")
        flat_batch = x.size(0)
        if flat_batch % self.nodes:
            raise ValueError("flattened CI batch is not divisible by zone count")
        batch = flat_batch // self.nodes
        tensor = x.reshape(batch, self.nodes, self.cycles, self.phases, self.d_model)

        # Encode all seven observed cycles for each phase, independently per
        # zone, then route across phases.
        phase_series = tensor.permute(0, 1, 3, 2, 4).reshape(
            batch * self.nodes, self.phases, self.cycles * self.d_model
        )
        phase_tokens = self.phase_encode(phase_series)
        phase_query = self.phase_router.unsqueeze(0).expand(batch * self.nodes, -1, -1)
        pooled, _ = self.phase_sender(
            phase_query, phase_tokens, phase_tokens, need_weights=False
        )
        received, _ = self.phase_receiver(
            phase_tokens, pooled, pooled, need_weights=False
        )
        phase_tokens = self.phase_norm1(phase_tokens + self.dropout(received))
        phase_tokens = self.phase_norm2(
            phase_tokens + self.dropout(self.phase_mlp(phase_tokens))
        )
        phase_tokens = phase_tokens.reshape(batch, self.nodes, self.phases, self.d_model)

        # Retain every zone token and route each phase across zones.  A learned
        # node coordinate prevents permutation collapse without a fixed graph
        # smoothing kernel or output correction.
        region_tokens = phase_tokens.permute(0, 2, 1, 3).reshape(
            batch * self.phases, self.nodes, self.d_model
        )
        region_tokens = region_tokens + self.region_embedding.view(1, self.nodes, self.d_model)
        region_query = self.region_router.unsqueeze(0).expand(batch * self.phases, -1, -1)
        region_pool, _ = self.region_sender(
            region_query, region_tokens, region_tokens, need_weights=False
        )
        region_received, _ = self.region_receiver(
            region_tokens, region_pool, region_pool, need_weights=False
        )
        region_tokens = self.region_norm1(region_tokens + self.dropout(region_received))
        region_tokens = self.region_norm2(
            region_tokens + self.dropout(self.region_mlp(region_tokens))
        )
        phase_tokens = region_tokens.reshape(batch, self.phases, self.nodes, self.d_model).permute(
            0, 2, 1, 3
        )
        decoded = self.decode(phase_tokens).reshape(
            batch, self.nodes, self.phases, self.cycles, self.d_model
        )
        return decoded.permute(0, 1, 3, 2, 4).reshape(
            batch * self.nodes, self.cycles * self.phases, self.d_model
        )


class PhaseLatentDecomposition(nn.Module):
    """Cycle/phase factorization of the latent history before PDM.

    The UrbanEV input contains seven complete daily cycles.  Instead of asking
    the PDM block to infer this factorization from a flat sequence, this
    operator exposes three causal coordinates at every timestamp: the
    cross-cycle phase template, the within-cycle innovation, and the
    cycle-level departure from the template.  A shared linear re-coordinate
    maps the concatenated coordinates back to ``d_model``.  The map is
    initialized to the identity on the original latent, so the ablation has a
    controlled starting point while gradients can learn a different phase
    manifold.  No selector, gate, expert, or forecast residual is used.
    """

    def __init__(self, d_model: int, cycles: int = 7, phases: int = 24):
        super().__init__()
        if d_model < 1 or cycles < 1 or phases < 1:
            raise ValueError("phase latent decomposition dimensions must be positive")
        self.d_model = int(d_model)
        self.cycles = int(cycles)
        self.phases = int(phases)
        self.recoordinate = nn.Linear(4 * self.d_model, self.d_model)
        # Keep the initial representation exactly equal to the original
        # latent.  The decomposition coordinates become useful as training
        # learns the re-coordinate, rather than changing the baseline by a
        # hand-tuned output coefficient.
        linear = self.recoordinate
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)
        with torch.no_grad():
            linear.weight[:, :self.d_model].copy_(torch.eye(self.d_model))

    def forward(self, x):
        if x.ndim != 3 or x.size(1) != self.cycles * self.phases:
            raise ValueError(
                "phase latent decomposition expects [batch, cycles*phases, d]"
            )
        b, _, d = x.shape
        cycles = x.reshape(b, self.cycles, self.phases, d)
        template = cycles.mean(dim=1, keepdim=True)
        innovation = cycles - template
        cycle_level = innovation.mean(dim=2, keepdim=True)
        # Remove the cycle-wide offset from the innovation coordinate.  The
        # four coordinates (raw, template, phase innovation, cycle departure)
        # are jointly re-coordinated before the ordinary PDM decomposition.
        phase_innovation = innovation - cycle_level
        features = torch.cat(
            (cycles, template.expand_as(cycles), phase_innovation,
             cycle_level.expand_as(cycles)), dim=-1
        )
        return self.recoordinate(features).reshape(b, self.cycles * self.phases, d)
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class Temporal_DFT_series_decomp(nn.Module):
    """Frequency decomposition along the temporal axis.

    The original DFT_series_decomp applies rFFT to the last tensor dimension.
    TimeMixer's PDM receives [batch, time, feature], so this UrbanEV adapter
    explicitly decomposes along dim=1 and selects dominant temporal modes per
    sample/channel.
    """
    def __init__(self, top_k=5):
        super(Temporal_DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        length = x.size(1)
        original_dtype = x.dtype
        # cuFFT half precision supports only power-of-two lengths; UrbanEV
        # uses 168-hour windows, so perform the transform in float32.
        xf = torch.fft.rfft(x.float(), dim=1)
        freq = torch.abs(xf)
        freq[:, 0, :] = 0
        k = min(int(self.top_k), max(1, freq.size(1) - 1))
        threshold = torch.topk(freq, k, dim=1).values[:, -1:, :]
        mask = freq >= threshold
        seasonal = torch.fft.irfft(torch.where(mask, xf, torch.zeros_like(xf)),
                                   n=length, dim=1).to(original_dtype)
        trend = x - seasonal
        return seasonal, trend


class Adaptive_Latent_DFT_series_decomp(nn.Module):
    """Soft adaptive spectral decomposition in the TimeMixer latent space.

    Unlike the source DFT path, which applies a hard top-k mask to the last
    tensor dimension, this block receives the embedded ``[batch, time,
    latent]`` tokens from PDM.  It computes a temporal FFT, combines the
    input-conditioned log-energy with a learnable frequency prior for each
    latent channel, and uses a differentiable soft band allocation to split
    seasonal and trend tokens.  The split remains exactly reconstructive:
    ``trend = x - seasonal``.
    """
    def __init__(self, top_k=5, latent_dim=32, temperature=1.0):
        super(Adaptive_Latent_DFT_series_decomp, self).__init__()
        self.top_k = int(top_k)
        self.latent_dim = int(latent_dim)
        self.temperature = float(temperature)
        # A small coordinate MLP produces a frequency prior per latent
        # channel. It is evaluated at the current scale's normalized FFT
        # coordinates, so the same latent factors can specialize across all
        # TimeMixer resolutions (168, 84, 42, 21 steps).
        hidden = max(8, self.latent_dim // 2)
        self.frequency_prior = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, self.latent_dim)
        )
        nn.init.zeros_(self.frequency_prior[-1].weight)
        nn.init.zeros_(self.frequency_prior[-1].bias)
        self.channel_temperature = nn.Parameter(torch.zeros(1, 1, self.latent_dim))

    def forward(self, x):
        length = x.size(1)
        original_dtype = x.dtype
        latent_dim = x.size(-1)
        if latent_dim != self.latent_dim:
            raise ValueError(
                f"adaptive latent DFT expected latent dim {self.latent_dim}, got {latent_dim}"
            )
        # The 168-step input is non-power-of-two. Keep FFT numerically stable
        # under AMP, then restore the model dtype after inverse transform.
        xf = torch.fft.rfft(x.float(), dim=1, norm="ortho")
        energy = torch.log1p(torch.abs(xf))
        energy = torch.cat((torch.full_like(energy[:, :1, :], -torch.inf),
                            energy[:, 1:, :]), dim=1)
        bins = energy.size(1)
        k = min(self.top_k, max(1, bins - 1))
        positions = torch.linspace(0.0, 1.0, bins, device=x.device, dtype=xf.real.dtype)
        coords = torch.stack((positions, torch.sin(torch.pi * positions)), dim=-1)
        prior = self.frequency_prior(coords).unsqueeze(0)
        temp = F.softplus(self.channel_temperature) + self.temperature
        scores = energy + prior
        # Use the k-th energy as a detached centre and soften the top-k
        # decision. This preserves a dominant k-band allocation while keeping
        # gradients for nearby frequencies and for the learned prior.
        threshold = torch.topk(scores, k, dim=1).values[:, -1:, :].detach()
        allocation = torch.sigmoid((scores - threshold) / temp)
        dc_mask = torch.ones_like(allocation)
        dc_mask[:, 0, :] = 0.0
        allocation = allocation * dc_mask
        seasonal = torch.fft.irfft(xf * allocation, n=length, dim=1,
                                   norm="ortho").to(original_dtype)
        trend = x - seasonal
        return seasonal, trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class CausalLevelDriftTransport(nn.Module):
    """Transport causal level/drift coordinates inside PDM.

    TimeMixer's PDM decomposes each scale independently and only exchanges
    the resulting season/trend tensors.  UrbanEV's long horizon errors show a
    different failure mode: the fine scale can carry a recent level shift
    while the coarse scale still carries the previous level.  This block
    exposes two causal coordinates (a fixed 24-step trailing level and its
    first difference), lifts the adjacent coarse coordinates to the fine
    grid, and rewrites the latent token through a shared linear state map.

    The map is initialized as the identity on the original latent ``x``.  It
    therefore changes no prediction at construction and has no sigmoid,
    softmax, expert, router, or output residual.  The coarse state is an
    ordinary additive input to the PDM representation; it is never selected
    sample by sample.  All coordinates are computed from the observed prefix
    at the current scale, and the trailing level uses left replication only
    so no future token is read.
    """

    def __init__(self, d_model: int, scales: int, window: int = 24):
        super().__init__()
        if int(d_model) < 1 or int(scales) < 1:
            raise ValueError("d_model and scales must be positive")
        if int(window) < 2:
            raise ValueError("level drift window must be at least two")
        self.d_model = int(d_model)
        self.scales = int(scales)
        self.window = int(window)
        # Inputs are [x, level, drift, lifted_coarse_level,
        # lifted_coarse_drift].  Every scale has its own fixed coordinate map
        # because the sampling interval changes with resolution; weights are
        # still initialized identically as an identity on x.
        self.state_maps = nn.ModuleList([
            nn.Linear(5 * self.d_model, self.d_model)
            for _ in range(self.scales)
        ])
        for layer in self.state_maps:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            with torch.no_grad():
                layer.weight[:, :self.d_model].copy_(torch.eye(self.d_model))

    def _state(self, x: torch.Tensor):
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("level/drift state expects [batch,time,d_model]")
        # Causal trailing mean.  Replicating the first observed token on the
        # left is equivalent to a shorter prefix window and avoids zero-padding
        # transients at every forecast origin.
        values = x.transpose(1, 2)
        padded = F.pad(values, (self.window - 1, 0), mode="replicate")
        level = F.avg_pool1d(padded, kernel_size=self.window, stride=1)
        level = level.transpose(1, 2)
        drift = torch.cat((torch.zeros_like(level[:, :1]),
                           level[:, 1:] - level[:, :-1]), dim=1)
        return level, drift

    @staticmethod
    def _lift(x: torch.Tensor, length: int) -> torch.Tensor:
        if x.size(1) == int(length):
            return x
        return F.interpolate(x.transpose(1, 2), size=int(length),
                             mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, x_list):
        if not isinstance(x_list, (list, tuple)) or len(x_list) != self.scales:
            raise ValueError("level/drift transport received an unexpected number of scales")
        states = [self._state(value) for value in x_list]
        output = []
        for index, (value, (level, drift)) in enumerate(zip(x_list, states)):
            if index + 1 < self.scales:
                coarse_level, coarse_drift = states[index + 1]
                lifted_level = self._lift(coarse_level, value.size(1))
                lifted_drift = self._lift(coarse_drift, value.size(1))
            else:
                lifted_level = torch.zeros_like(value)
                lifted_drift = torch.zeros_like(value)
            coordinates = torch.cat((value, level, drift,
                                     lifted_level, lifted_drift), dim=-1)
            output.append(self.state_maps[index](coordinates))
        return output


class FixedEndpointLevelDriftBridge(nn.Module):
    """Low-rank endpoint state bridge for the latent PDM representation.

    The bridge summarizes only the observed endpoint of each latent scale
    (last token and its fixed one-day difference), compresses that state to a
    shared rank-four subspace, and writes it back through a deterministic
    trailing ramp.  The ramp is a coordinate transport inside PDM, not a
    forecast correction; its expansion is zero initialized so the official
    representation is exactly preserved at construction.  There is no gate,
    router, expert choice, or searched branch weight.
    """

    def __init__(self, d_model: int, scales: int, rank: int = 4,
                 endpoint_width: int = 24):
        super().__init__()
        if int(d_model) < 1 or int(scales) < 1 or int(rank) < 1:
            raise ValueError("d_model, scales, and rank must be positive")
        if int(endpoint_width) < 2:
            raise ValueError("endpoint width must be at least two")
        self.d_model = int(d_model)
        self.scales = int(scales)
        self.rank = int(rank)
        self.endpoint_width = int(endpoint_width)
        # A shared low-rank map enforces the same endpoint semantics across
        # the multi-resolution streams.  The expansion starts at zero.
        self.compress = nn.Linear(2 * self.d_model, self.rank, bias=False)
        self.expand = nn.Linear(self.rank, self.d_model, bias=False)
        nn.init.normal_(self.compress.weight, mean=0.0,
                        std=(2 * self.d_model) ** -0.5)
        nn.init.zeros_(self.expand.weight)
        # Scale-specific normalization keeps the bridge magnitude comparable
        # when the coarser streams contain fewer than 24 tokens.
        self.scale_norm = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(self.scales)]
        )

    def forward(self, x_list):
        if not isinstance(x_list, (list, tuple)) or len(x_list) != self.scales:
            raise ValueError("endpoint bridge received an unexpected scale list")
        output = []
        for index, value in enumerate(x_list):
            if value.ndim != 3 or value.size(-1) != self.d_model:
                raise ValueError("endpoint bridge expects [batch,time,d_model]")
            width = min(self.endpoint_width, value.size(1))
            # Fixed one-day endpoint difference; for a coarse scale this uses
            # the available tail and therefore remains causal and deterministic.
            tail = value[:, -width:]
            last = tail[:, -1]
            previous = tail[:, 0]
            drift = last - previous
            state = torch.cat((last, drift), dim=-1)
            update = self.expand(self.compress(state))
            # A fixed linear ramp writes the endpoint state into the tail,
            # preserving the ordering of the observed latent trajectory.
            ramp = torch.linspace(0.0, 1.0, width, device=value.device,
                                  dtype=value.dtype).view(1, width, 1)
            bridge = ramp * update.unsqueeze(1)
            # LayerNorm is applied only to the learned state; zero expansion
            # guarantees an exact identity path before training.
            bridge = self.scale_norm[index](bridge)
            output.append(torch.cat((value[:, :-width],
                                     value[:, -width:] + bridge), dim=1))
        return output


class LatentTemporalMixer(nn.Module):
    """Depthwise local temporal mixing followed by latent channel mixing.

    PDM currently applies a point-wise latent MLP after cross-scale mixing.
    This block adds a scale-consistent local temporal operator at that exact
    latent stage, so fine-scale demand spikes and short ramps can interact
    before the future predictors. The depthwise kernel is initialized as an
    identity filter; learning comes from the temporal operator and channel
    projection, not from a forecast gate.
    """
    def __init__(self, configs):
        super(LatentTemporalMixer, self).__init__()
        d_model = int(configs.d_model)
        self.temporal = nn.Conv1d(
            d_model, d_model, kernel_size=5, padding=2, groups=d_model, bias=True
        )
        with torch.no_grad():
            self.temporal.weight.zero_()
            self.temporal.weight[:, 0, 2] = 1.0
            self.temporal.bias.zero_()
        self.channel = nn.Sequential(
            nn.Linear(d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, d_model),
        )

    def forward(self, x):
        mixed = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        return self.channel(mixed)


class DualAxisTimeImageMixer(nn.Module):
    """Mix rows (phase) and columns (repeated cycles) of a time image."""
    def __init__(self, d_model):
        super(DualAxisTimeImageMixer, self).__init__()
        self.row = nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1),
            groups=d_model, bias=True
        )
        self.column = nn.Conv2d(
            d_model, d_model, kernel_size=(3, 1), padding=(1, 0),
            groups=d_model, bias=True
        )
        with torch.no_grad():
            for layer in (self.row, self.column):
                layer.weight.zero_()
                if layer is self.row:
                    layer.weight[:, 0, 0, 1] = 1.0
                else:
                    layer.weight[:, 0, 1, 0] = 1.0
                layer.bias.zero_()
        self.channel = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        with torch.no_grad():
            self.channel.weight.zero_()
            self.channel.weight[:, :, 0, 0].copy_(torch.eye(d_model))
            self.channel.bias.zero_()

    def forward(self, image):
        return self.channel(self.column(self.row(image)))


class UrbanMRTI(nn.Module):
    """UrbanEV multi-resolution time imaging in the embedded scale streams.

    Each scale is reshaped at daily and weekly periods into 2D images. A
    shared dual-axis mixer extracts phase and cycle patterns, then a learned
    linear fusion returns one sequence at the original scale length. The raw
    embedding is the identity path at initialization; the model learns to
    use image resolutions through representation fusion before PDM.
    """
    def __init__(self, configs):
        super(UrbanMRTI, self).__init__()
        self.d_model = int(configs.d_model)
        self.window = int(configs.down_sampling_window)
        self.n_scales = int(configs.down_sampling_layers) + 1
        view_name = str(getattr(configs, "mrti_views", "both")).lower()
        if view_name not in {"daily", "weekly", "both"}:
            raise ValueError("mrti_views must be daily, weekly, or both")
        self.views = ("daily", "weekly") if view_name == "both" else (view_name,)
        self.image_mixers = nn.ModuleList(
            [DualAxisTimeImageMixer(self.d_model) for _ in range(self.n_scales)]
        )
        self.fusions = nn.ModuleList(
            [nn.Linear((1 + len(self.views)) * self.d_model, self.d_model)
             for _ in range(self.n_scales)]
        )
        for fusion in self.fusions:
            with torch.no_grad():
                fusion.weight.zero_()
                fusion.weight[:, :self.d_model].copy_(torch.eye(self.d_model))
                fusion.bias.zero_()

    @staticmethod
    def _image_view(x, period, mixer):
        b, length, d = x.shape
        period = max(2, min(int(period), length))
        padding = (period - length % period) % period
        if padding:
            x = F.pad(x, (0, 0, 0, padding))
        cycles = x.size(1) // period
        image = x.reshape(b, cycles, period, d).permute(0, 3, 1, 2)
        image = mixer(image)
        return image.permute(0, 2, 3, 1).reshape(b, cycles * period, d)[:, :length]

    def forward(self, enc_out_list):
        output = []
        for scale, (x, mixer, fusion) in enumerate(
                zip(enc_out_list, self.image_mixers, self.fusions)):
            daily = max(2, round(24 / (self.window ** scale)))
            weekly = max(daily + 1, round(168 / (self.window ** scale)))
            daily_view = self._image_view(x, daily, mixer)
            weekly_view = self._image_view(x, weekly, mixer)
            image_views = {"daily": daily_view, "weekly": weekly_view}
            output.append(fusion(torch.cat(
                (x, *(image_views[name] for name in self.views)), dim=-1
            )))
        return output


class EventAwareTimeImageMixer(nn.Module):
    """Factor a time image into trend, seasonal phase and positive events."""
    def __init__(self, d_model):
        super(EventAwareTimeImageMixer, self).__init__()
        self.phase = nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1),
            groups=d_model, bias=True
        )
        self.trend = nn.Conv2d(
            d_model, d_model, kernel_size=(3, 1), padding=(1, 0),
            groups=d_model, bias=True
        )
        self.event = nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1),
            groups=d_model, bias=True
        )
        self.components = nn.Conv2d(3 * d_model, d_model, kernel_size=1, bias=True)
        with torch.no_grad():
            for layer, axis in ((self.phase, 1), (self.trend, 0), (self.event, 1)):
                layer.weight.zero_()
                if axis == 0:
                    layer.weight[:, 0, 1, 0] = 1.0
                else:
                    layer.weight[:, 0, 0, 1] = 1.0
                layer.bias.zero_()
            self.components.weight.zero_()
            eye = torch.eye(d_model)
            self.components.weight[:, :d_model, 0, 0].copy_(eye)
            self.components.weight[:, d_model:2 * d_model, 0, 0].copy_(eye)
            self.components.bias.zero_()

    def forward(self, image):
        trend = self.trend(image)
        seasonal = self.phase(image - trend)
        event_input = F.max_pool2d(
            F.relu(image - trend), kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        event = self.event(event_input)
        return self.components(torch.cat((seasonal, trend, event), dim=1))


class EventAwareUrbanMRTI(UrbanMRTI):
    """MRTI v1 with explicit cycle/phase/event image decomposition."""
    def __init__(self, configs):
        super(EventAwareUrbanMRTI, self).__init__(configs)
        self.image_mixers = nn.ModuleList(
            [EventAwareTimeImageMixer(self.d_model) for _ in range(self.n_scales)]
        )


class PhaseAlignedUrbanMRTI(UrbanMRTI):
    """MRTI with deterministic phase alignment across repeated cycles.

    UrbanEV's daily demand peaks can shift between adjacent days.  Before the
    existing 2-D image mixer, each repeated cycle is circularly aligned to the
    most recent cycle using cross-correlation on the latent mean signal.  The
    operation changes the temporal coordinate system seen by the mixer; it is
    not a learned gate and does not alter the forecast after prediction.
    """
    @staticmethod
    def _align_cycles(image):
        batch, channels, cycles, period = image.shape
        if cycles < 2 or period < 2:
            return image
        reference = image[:, :, -1, :].mean(dim=1)
        previous = image[:, :, :-1, :]
        signal = previous.mean(dim=1)
        max_shift = max(1, period // 4)
        offsets = torch.arange(-max_shift, max_shift + 1,
                               device=image.device, dtype=torch.long)
        with torch.no_grad():
            scores = []
            for offset in offsets.tolist():
                shifted = torch.roll(signal, shifts=int(offset), dims=-1)
                scores.append((shifted * reference.unsqueeze(1)).mean(dim=-1))
            scores = torch.stack(scores, dim=1)
            best_shift = offsets[scores.argmax(dim=1)]
            phase = torch.arange(period, device=image.device, dtype=torch.long)
            indices = (phase.view(1, 1, -1) - best_shift.unsqueeze(-1)) % period
            gather_index = indices.unsqueeze(1).expand(-1, channels, -1, -1)
        aligned = torch.gather(previous, dim=3, index=gather_index)
        return torch.cat((aligned, image[:, :, -1:, :]), dim=2)

    @staticmethod
    def _image_view(x, period, mixer):
        batch, length, d_model = x.shape
        period = max(2, min(int(period), length))
        padding = (period - length % period) % period
        if padding:
            x = F.pad(x, (0, 0, 0, padding))
        cycles = x.size(1) // period
        image = x.reshape(batch, cycles, period, d_model).permute(0, 3, 1, 2)
        image = PhaseAlignedUrbanMRTI._align_cycles(image)
        image = mixer(image)
        return image.permute(0, 2, 3, 1).reshape(
            batch, cycles * period, d_model
        )[:, :length]


class AdaptivePeriodicImageDecomposition(nn.Module):
    """Decompose a latent stream through an input-conditioned period bank.

    Each candidate period is represented as a 2-D image whose axes are cycle
    index and within-cycle phase.  A depthwise phase operator acts on the
    seasonal component and a depthwise cycle operator acts on the trend
    component.  The two components are reconstructed with a channel mixing
    projection before the period views are combined.  This is a structural
    representation change before PDM, rather than an output gate or forecast
    residual.
    """
    def __init__(self, d_model, periods, top_k=3, temperature=0.7, axis_kernel=3,
                 use_peak=False):
        super(AdaptivePeriodicImageDecomposition, self).__init__()
        kernel = int(axis_kernel)
        if kernel < 1 or kernel % 2 != 1:
            raise ValueError("apid_axis_kernel must be a positive odd integer")
        self.d_model = int(d_model)
        self.periods = tuple(sorted({max(2, int(p)) for p in periods}))
        if not self.periods:
            raise ValueError("APID requires at least one period")
        self.top_k = max(1, int(top_k))
        self.temperature = max(0.05, float(temperature))
        self.use_peak = bool(use_peak)
        padding = kernel // 2
        self.phase_mixer = nn.Conv2d(
            self.d_model, self.d_model, kernel_size=(1, kernel),
            padding=(0, padding), groups=self.d_model, bias=True
        )
        self.trend_mixer = nn.Conv2d(
            self.d_model, self.d_model, kernel_size=(kernel, 1),
            padding=(padding, 0), groups=self.d_model, bias=True
        )
        if self.use_peak:
            self.peak_mixer = nn.Conv2d(
                self.d_model, self.d_model, kernel_size=(1, kernel),
                padding=(0, padding), groups=self.d_model, bias=True
            )
        self.component_projection = nn.Conv2d(
            (3 if self.use_peak else 2) * self.d_model,
            self.d_model, kernel_size=1, bias=True
        )
        self._initialize_identity_decomposition()

    def _initialize_identity_decomposition(self):
        with torch.no_grad():
            self.phase_mixer.weight.zero_()
            self.phase_mixer.weight[:, 0, 0, self.phase_mixer.kernel_size[1] // 2] = 1.0
            self.phase_mixer.bias.zero_()
            self.trend_mixer.weight.zero_()
            self.trend_mixer.weight[:, 0, self.trend_mixer.kernel_size[0] // 2, 0] = 1.0
            self.trend_mixer.bias.zero_()
            if self.use_peak:
                self.peak_mixer.weight.zero_()
                self.peak_mixer.weight[:, 0, 0, self.peak_mixer.kernel_size[1] // 2] = 1.0
                self.peak_mixer.bias.zero_()
            self.component_projection.weight.zero_()
            eye = torch.eye(self.d_model)
            self.component_projection.weight[:, :self.d_model, 0, 0].copy_(eye)
            self.component_projection.weight[:, self.d_model:2 * self.d_model, 0, 0].copy_(eye)
            # The peak branch starts as a strict feature extractor, not an
            # output correction.  Its contribution is learned from the
            # peak-aware objective below.
            self.component_projection.bias.zero_()

    @staticmethod
    def _image(x, period):
        b, length, d = x.shape
        padding = (int(period) - length % int(period)) % int(period)
        if padding:
            x = F.pad(x, (0, 0, 0, padding))
        cycles = x.size(1) // int(period)
        return x.reshape(b, cycles, int(period), d).permute(0, 3, 1, 2), length

    def _period_view(self, x, period):
        image, length = self._image(x, period)
        trend = self.trend_mixer(image)
        seasonal = self.phase_mixer(image - trend)
        components = [seasonal, trend]
        if self.use_peak:
            peak = F.max_pool2d(
                F.relu(seasonal), kernel_size=(1, 3), stride=1, padding=(0, 1)
            )
            components.append(self.peak_mixer(peak))
        reconstructed = self.component_projection(torch.cat(components, dim=1))
        return reconstructed.permute(0, 2, 3, 1).reshape(
            x.size(0), -1, self.d_model
        )[:, :length]

    def _period_weights(self, x):
        # FFT is kept in float32 because UrbanEV's 168-step sequence is not a
        # power of two and half-precision cuFFT is unsupported for this size.
        # The period score uses the mean latent signal, avoiding a redundant
        # FFT for every latent channel.  Period ranking is batch-shared while
        # the soft weights remain sample-specific.
        signal = x.float().mean(dim=-1, keepdim=True)
        centered = signal - signal.mean(dim=1, keepdim=True)
        with torch.no_grad():
            energy = torch.abs(torch.fft.rfft(centered, dim=1, norm="ortho")).squeeze(-1)
        bins = energy.size(1)
        scores = []
        for period in self.periods:
            index = max(1, min(bins - 1, int(round(x.size(1) / float(period)))))
            score = energy[:, index]
            harmonic = min(bins - 1, 2 * index)
            if harmonic != index:
                score = score + 0.5 * energy[:, harmonic]
            scores.append(torch.log1p(score))
        scores = torch.stack(scores, dim=1)
        # Retain a compact batch-level top-k period bank.  The selected image
        # operators are the expensive part; sample-specific weights still
        # preserve input-conditioned periodicity inside the selected bank.
        k = min(self.top_k, len(self.periods))
        indices = torch.topk(scores.mean(dim=0), k, dim=0).indices
        selected = scores[:, indices]
        return indices, torch.softmax(selected / self.temperature, dim=1)

    def forward(self, x):
        indices, weights = self._period_weights(x)
        selected_periods = [self.periods[int(index)] for index in indices.tolist()]
        views = torch.stack([self._period_view(x, period) for period in selected_periods], dim=2)
        return (views * weights[:, None, :, None].to(views.dtype)).sum(dim=2)


class APIDCrossResolutionMixer(nn.Module):
    """Mix aligned APID streams across resolution and phase axes."""
    def __init__(self, d_model):
        super(APIDCrossResolutionMixer, self).__init__()
        self.resolution = nn.Conv2d(
            d_model, d_model, kernel_size=(3, 1), padding=(1, 0),
            groups=d_model, bias=True
        )
        self.phase = nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1),
            groups=d_model, bias=True
        )
        self.channel = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        with torch.no_grad():
            for layer, axis in ((self.resolution, 0), (self.phase, 1)):
                layer.weight.zero_()
                if axis == 0:
                    layer.weight[:, 0, 1, 0] = 1.0
                else:
                    layer.weight[:, 0, 0, 1] = 1.0
                layer.bias.zero_()
            self.channel.weight.zero_()
            self.channel.weight[:, :, 0, 0].copy_(torch.eye(d_model))
            self.channel.bias.zero_()

    def forward(self, streams):
        base_length = streams[0].size(1)
        aligned = []
        for stream in streams:
            value = stream.transpose(1, 2)
            if value.size(-1) != base_length:
                value = F.interpolate(value, size=base_length, mode="linear", align_corners=False)
            aligned.append(value)
        grid = torch.stack(aligned, dim=2)
        mixed = self.channel(self.phase(self.resolution(grid)))
        output = []
        for index, stream in enumerate(streams):
            value = mixed[:, :, index, :]
            if stream.size(1) != base_length:
                value = F.interpolate(value, size=stream.size(1), mode="linear", align_corners=False)
            output.append(value.transpose(1, 2))
        return output


class UrbanAPID(nn.Module):
    """UrbanEV adaptive periodic image decomposition at all TimeMixer scales."""
    def __init__(self, configs):
        super(UrbanAPID, self).__init__()
        self.window = int(configs.down_sampling_window)
        self.n_scales = int(configs.down_sampling_layers) + 1
        raw_periods = tuple(int(p.strip()) for p in str(
            getattr(configs, "apid_periods", "24,168,12,84")
        ).split(",") if p.strip())
        if not raw_periods:
            raise ValueError("apid_periods cannot be empty")
        self.blocks = nn.ModuleList()
        for scale in range(self.n_scales):
            factor = self.window ** scale
            scaled = tuple(max(2, int(round(period / factor))) for period in raw_periods)
            length = int(configs.seq_len) // factor
            scaled = tuple(sorted({min(length, period) for period in scaled if length >= 2}))
            self.blocks.append(AdaptivePeriodicImageDecomposition(
                int(configs.d_model), scaled,
                top_k=int(getattr(configs, "apid_top_k", 3)),
                temperature=float(getattr(configs, "apid_temperature", 0.7)),
                axis_kernel=int(getattr(configs, "apid_axis_kernel", 3)),
                use_peak=bool(getattr(configs, "apid_peak_branch", False)),
            ))
        self.cross_resolution = APIDCrossResolutionMixer(int(configs.d_model))

    def forward(self, enc_out_list):
        streams = [block(value) for block, value in zip(self.blocks, enc_out_list)]
        return self.cross_resolution(streams)


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence
        # Optional causal history-state decomposition used by the UrbanEV
        # hybrid route.  The CPR backbone still receives one additive history
        # coordinate, but the first PDM block keeps its low-frequency history
        # state on the trend stream and its innovation on the seasonal stream.
        # This avoids letting the centered volume decomposition erase a
        # covariate level shift.  It is a fixed 24-hour causal coordinate,
        # with no gate, router, expert branch, or learned coefficient.
        self.use_history_state_pdm = bool(
            getattr(configs, "use_history_state_pdm", False)
        )
        self.history_state_pdm_window = int(
            getattr(configs, "history_state_pdm_window", 24)
        )
        if self.history_state_pdm_window < 2:
            raise ValueError("history_state_pdm_window must be at least two")
        self.use_level_drift_transport = bool(
            getattr(configs, "use_level_drift_transport", False)
        )
        if self.use_level_drift_transport:
            # This coordinate transform lives inside every PDM block, before
            # seasonal/trend decomposition.  The trailing window is fixed by
            # the UrbanEV 24-hour cycle; it is not a searched hyperparameter.
            self.level_drift_transport = CausalLevelDriftTransport(
                int(configs.d_model),
                int(configs.down_sampling_layers) + 1,
                window=int(getattr(configs, "level_drift_window", 24)),
            )
        self.use_mdmixer_dual_branch = bool(getattr(configs, "use_mdmixer_lite", False) and
                                             getattr(configs, "mdmixer_dual_branch", 1))
        self.paired_input_dim = int(getattr(configs, "paired_input_dim", 0))
        if self.paired_input_dim < 0:
            raise ValueError("paired_input_dim must be nonnegative")
        self.innovation_exchange = None
        self.innovation_stage = str(getattr(configs, "innovation_stage", "pre"))
        if self.innovation_stage not in {"pre", "post"}:
            raise ValueError("innovation_stage must be pre or post")

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "causal_ema":
            self.decompsition = CausalEMADecomp(
                getattr(configs, "causal_ema_alpha", 0.25)
            )
        elif configs.decomp_method == "causal_daily":
            self.decompsition = CausalDailyDecomp(
                base_period=int(getattr(configs, "causal_daily_period", 24)),
                cycles=int(getattr(configs, "causal_daily_cycles", 7)),
            )
        elif configs.decomp_method == "causal_fir":
            self.decompsition = CausalFIRDecomp(
                int(configs.d_model),
                kernel_size=int(getattr(configs, "causal_fir_kernel", 25)),
            )
        elif configs.decomp_method == "robust_huber":
            self.decompsition = RobustHuberDecomp(
                kernel_size=int(getattr(configs, "moving_avg", 25))
            )
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        elif configs.decomp_method == "temporal_dft":
            self.decompsition = Temporal_DFT_series_decomp(configs.top_k)
        elif configs.decomp_method == "adaptive_latent_dft":
            self.decompsition = Adaptive_Latent_DFT_series_decomp(
                configs.top_k, configs.d_model, getattr(configs, "spectral_temperature", 1.0)
            )
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        if getattr(configs, "latent_mixer", "pointwise") == "depthwise_temporal":
            self.out_cross_layer = LatentTemporalMixer(configs)
        else:
            self.out_cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )
        # MDMixer's trend and seasonal branches are independent after
        # cross-scale mixing.  Initialize both maps as identity so enabling
        # the route does not destroy the official TimeMixer representation at
        # step zero; training can then specialize the two components.
        if self.use_mdmixer_dual_branch:
            self.mdmixer_season_branch = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model), nn.GELU(),
                nn.Linear(configs.d_model, configs.d_model)
            )
            self.mdmixer_trend_branch = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model), nn.GELU(),
                nn.Linear(configs.d_model, configs.d_model)
            )
            for branch in (self.mdmixer_season_branch, self.mdmixer_trend_branch):
                nn.init.zeros_(branch[-1].weight)
                nn.init.zeros_(branch[-1].bias)

    def forward(self, x_list, history_list=None):
        if history_list is not None:
            if not self.use_history_state_pdm:
                raise ValueError("history_list requires use_history_state_pdm")
            if not isinstance(history_list, (list, tuple)) or len(history_list) != len(x_list):
                raise ValueError("history-state PDM received an unexpected scale list")
            for value, history in zip(x_list, history_list):
                if history.shape != value.shape:
                    raise ValueError("history-state PDM scale shape mismatch")
        if self.use_level_drift_transport:
            # Rewrite the history coordinates before the ordinary PDM
            # decomposition.  No output correction is introduced; the
            # unchanged season/trend mixer and FMM consume this representation.
            x_list = self.level_drift_transport(x_list)
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for scale, x in enumerate(x_list):
            if history_list is not None:
                # ``x`` already contains the additive external history
                # coordinate.  Decompose the original volume latent with the
                # official operator, then decompose the history coordinate
                # with a fixed trailing 24-hour state.  The two components
                # still reconstruct x exactly before cross-scale mixing.
                history = history_list[scale].to(dtype=x.dtype, device=x.device)
                base = x - history
                season, trend = self.decompsition(base)
                values = history.transpose(1, 2)
                window = min(self.history_state_pdm_window, values.size(-1))
                padded = F.pad(values, (window - 1, 0), mode="replicate")
                history_trend = F.avg_pool1d(
                    padded, kernel_size=window, stride=1
                ).transpose(1, 2)
                history_season = history - history_trend
                season = season + history_season
                trend = trend + history_trend
            else:
                season, trend = self.decompsition(x)
            if self.innovation_exchange is not None and self.innovation_stage == "pre":
                season = self.innovation_exchange(season, scale)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        if self.innovation_exchange is not None and self.innovation_stage == "post":
            out_season_list = [self.innovation_exchange(season, scale)
                               for scale, season in enumerate(out_season_list)]
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            if self.use_mdmixer_dual_branch:
                # The zero-initialized branch maps are residual refinements,
                # preserving the official sum while allowing independent
                # trend/season statistics to be learned.
                out = (out_season + self.mdmixer_season_branch(out_season) +
                       out_trend + self.mdmixer_trend_branch(out_trend))
            else:
                out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class HorizonBasisProjection(nn.Module):
    """Low-rank phase-aware projection from context tokens to forecast steps.

    TimeMixer's default future mixer uses one unconstrained Linear(L, H) per
    scale. UrbanEV has a fixed 24-step horizon with daily and weekly phase
    structure, so this projection first predicts coefficients from the
    context and reconstructs the horizon with a shared, learnable dictionary
    initialized by polynomial, daily and weekly bases. It is a direct
    coefficient decoder: no forecast gate and no residual output correction.
    """
    def __init__(self, input_len, pred_len, basis_dim=16):
        super(HorizonBasisProjection, self).__init__()
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.basis_dim = int(basis_dim)
        grid = torch.arange(self.pred_len, dtype=torch.float32)
        grid = grid / max(1, self.pred_len - 1)
        columns = [torch.ones_like(grid), grid, grid ** 2, grid ** 3]
        for period, harmonics in ((24.0, 4), (168.0, 2)):
            for harmonic in range(1, harmonics + 1):
                phase = 2.0 * torch.pi * harmonic * torch.arange(self.pred_len) / period
                columns.extend((torch.sin(phase), torch.cos(phase)))
        while len(columns) < self.basis_dim:
            columns.append(torch.cos(torch.pi * len(columns) * grid))
        basis = torch.stack(columns[:self.basis_dim], dim=0)
        basis = basis / torch.clamp(torch.linalg.vector_norm(basis, dim=1, keepdim=True), min=1e-6)
        self.basis = nn.Parameter(basis)
        self.coefficient = nn.Linear(self.input_len, self.basis_dim)

    def forward(self, x):
        # x: [batch, latent, context]. Reconstruct [batch, latent, horizon].
        coefficients = self.coefficient(x)
        return torch.einsum("bdk,kp->bdp", coefficients, self.basis)


class TariffTensorProjection(nn.Module):
    """Direct tensor-product horizon projection conditioned on known tariffs.

    Historical latent coefficients and future tariff coefficients interact
    multiplicatively to generate every forecast latent token.  The module
    replaces the FMM linear readout; it does not add a forecast correction or
    sigmoid gate after prediction.
    """
    def __init__(self, input_len: int, pred_len: int, d_model: int,
                 rank: int = 8):
        super().__init__()
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.history_coeff = nn.Linear(self.input_len, self.rank)
        self.tariff_coeff = nn.Linear(self.d_model, self.rank)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_bias = nn.Parameter(torch.zeros(1, self.pred_len, self.d_model))

    def forward(self, history: torch.Tensor, tariff_tokens: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tariff_tokens.ndim != 3:
            raise ValueError("tariff tensor projection expects [B,D,L] and [B,H,D]")
        if history.shape[1] != self.d_model or history.shape[2] != self.input_len:
            raise ValueError("history latent shape does not match tariff projection")
        if tariff_tokens.shape[1:] != (self.pred_len, self.d_model):
            raise ValueError("tariff token shape does not match horizon/model dimensions")
        history_coeff = self.history_coeff(history)  # [B,D,R]
        tariff_coeff = self.tariff_coeff(tariff_tokens)  # [B,H,R]
        output = torch.einsum("bdr,bhr->bhd", history_coeff, tariff_coeff)
        return self.output_norm(output + self.output_bias)


class TariffFusionProjection(nn.Module):
    """Joint history/tariff latent projection without a post-hoc correction.

    A direct historical horizon projection and a future-tariff token stream are
    concatenated before one learned latent fusion map.  The fusion output is
    the sole latent forecast supplied to the channel projection.
    """
    def __init__(self, input_len: int, pred_len: int, d_model: int):
        super().__init__()
        self.input_len, self.pred_len, self.d_model = int(input_len), int(pred_len), int(d_model)
        self.history_projection = nn.Linear(self.input_len, self.pred_len)
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model), nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.horizon_embedding = nn.Parameter(torch.zeros(1, self.pred_len, self.d_model))

    def forward(self, history: torch.Tensor, tariff_tokens: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tariff_tokens.ndim != 3:
            raise ValueError("tariff fusion expects [B,D,L] and [B,H,D]")
        if history.shape[1:] != (self.d_model, self.input_len):
            raise ValueError("history latent shape does not match tariff fusion")
        if tariff_tokens.shape[1:] != (self.pred_len, self.d_model):
            raise ValueError("tariff token shape does not match tariff fusion")
        historical = self.history_projection(history).transpose(1, 2)
        joint = torch.cat((historical, tariff_tokens + self.horizon_embedding), dim=-1)
        return self.fusion(joint)


class TariffCrossAttentionProjection(nn.Module):
    """Use known-future tariff tokens as queries over historical latent tokens."""
    def __init__(self, input_len: int, pred_len: int, d_model: int,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % heads:
            raise ValueError("tariff cross-attention heads must divide d_model")
        self.input_len, self.pred_len, self.d_model = int(input_len), int(pred_len), int(d_model)
        self.attention = nn.MultiheadAttention(
            self.d_model, int(heads), dropout=float(dropout), batch_first=True
        )
        self.horizon_embedding = nn.Parameter(torch.zeros(1, self.pred_len, self.d_model))
        self.norm1 = nn.LayerNorm(self.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model), nn.GELU(),
            nn.Linear(4 * self.d_model, self.d_model),
        )
        self.norm2 = nn.LayerNorm(self.d_model)

    def forward(self, history: torch.Tensor, tariff_tokens: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tariff_tokens.ndim != 3:
            raise ValueError("tariff cross-attention expects [B,D,L] and [B,H,D]")
        if history.shape[1:] != (self.d_model, self.input_len):
            raise ValueError("history latent shape does not match tariff cross-attention")
        if tariff_tokens.shape[1:] != (self.pred_len, self.d_model):
            raise ValueError("tariff token shape does not match tariff cross-attention")
        key_value = history.transpose(1, 2)
        query = tariff_tokens + self.horizon_embedding
        attended, _ = self.attention(query, key_value, key_value, need_weights=False)
        hidden = self.norm1(attended + query)
        return self.norm2(hidden + self.mlp(hidden))


class TariffStateTransitionProjection(nn.Module):
    """Use known-future tariffs as an exogenous latent state transition.

    A horizon-specific historical drive initializes a latent state, then the
    state is rolled through the forecast horizon.  At each step, the
    transition operator is a low-rank bilinear function of that step's tariff
    token.  The resulting states are the sole input to channel projection;
    this is neither token concatenation nor an output gate/residual correction.
    """
    def __init__(self, input_len: int, pred_len: int, d_model: int,
                 rank: int = 8):
        super().__init__()
        self.input_len, self.pred_len, self.d_model = int(input_len), int(pred_len), int(d_model)
        self.rank = max(1, int(rank))
        self.history_projection = nn.Linear(self.input_len, self.pred_len)
        self.drive_projection = nn.Linear(self.d_model, self.d_model)
        self.base_transition = nn.Linear(self.d_model, self.d_model, bias=False)
        self.control_projection = nn.Linear(self.d_model, self.rank)
        self.transition_left = nn.Parameter(torch.randn(self.rank, self.d_model) * 0.02)
        self.transition_right = nn.Parameter(torch.randn(self.rank, self.d_model) * 0.02)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.horizon_bias = nn.Parameter(torch.zeros(1, self.pred_len, self.d_model))
        with torch.no_grad():
            self.base_transition.weight.copy_(0.5 * torch.eye(self.d_model))

    def forward(self, history: torch.Tensor, tariff_tokens: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tariff_tokens.ndim != 3:
            raise ValueError("tariff state transition expects [B,D,L] and [B,H,D]")
        if history.shape[1:] != (self.d_model, self.input_len):
            raise ValueError("history latent shape does not match tariff state transition")
        if tariff_tokens.shape[1:] != (self.pred_len, self.d_model):
            raise ValueError("tariff token shape does not match tariff state transition")
        drive = self.drive_projection(self.history_projection(history).transpose(1, 2))
        controls = self.control_projection(tariff_tokens)
        state = drive[:, 0]
        states = [state]
        for step in range(1, self.pred_len):
            base = self.base_transition(state)
            state_coordinates = torch.einsum("bd,rd->br", state, self.transition_right)
            controlled = torch.einsum(
                "br,rd->bd", state_coordinates * controls[:, step], self.transition_left
            )
            state = torch.tanh(base + drive[:, step] + controlled)
            states.append(state)
        output = torch.stack(states, dim=1)
        return self.output_norm(output + self.horizon_bias)


class TariffControlledFlowProjection(nn.Module):
    """Stable continuous-time latent flow driven by known-future tariffs.

    Unlike the discrete tanh recurrence above, this projection defines a
    bounded Euler discretisation of a controlled latent ODE.  The diagonal
    decay is strictly non-positive and the tariff-dependent interaction is a
    low-rank vector field.  The resulting state sequence is the sole FMM
    output; there is no post-hoc correction or forecast gate.
    """
    def __init__(self, input_len: int, pred_len: int, d_model: int,
                 rank: int = 8):
        super().__init__()
        self.input_len, self.pred_len, self.d_model = int(input_len), int(pred_len), int(d_model)
        self.rank = max(1, int(rank))
        self.history_projection = nn.Linear(self.input_len, self.pred_len)
        self.drive_projection = nn.Linear(self.d_model, self.d_model)
        self.control_projection = nn.Linear(self.d_model, self.rank)
        self.state_projection = nn.Linear(self.d_model, self.rank)
        self.vector_left = nn.Parameter(torch.randn(self.rank, self.d_model) * 0.02)
        self.vector_right = nn.Parameter(torch.randn(self.rank, self.d_model) * 0.02)
        self.decay_logits = nn.Parameter(torch.full((self.d_model,), -2.0))
        self.output_norm = nn.LayerNorm(self.d_model)
        self.horizon_bias = nn.Parameter(torch.zeros(1, self.pred_len, self.d_model))

    def forward(self, history: torch.Tensor, tariff_tokens: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tariff_tokens.ndim != 3:
            raise ValueError("tariff controlled flow expects [B,D,L] and [B,H,D]")
        if history.shape[1:] != (self.d_model, self.input_len):
            raise ValueError("history latent shape does not match tariff controlled flow")
        if tariff_tokens.shape[1:] != (self.pred_len, self.d_model):
            raise ValueError("tariff token shape does not match tariff controlled flow")
        forcing = self.drive_projection(
            self.history_projection(history).transpose(1, 2)
        )
        controls = 0.25 * torch.tanh(self.control_projection(tariff_tokens.float()))
        decay = 0.25 * F.softplus(self.decay_logits).to(dtype=forcing.dtype)
        state = forcing[:, 0]
        states = [state]
        # A bounded step size keeps the explicit flow stable while preserving
        # the ordering of the known tariff path across the forecast horizon.
        step = 1.0 / max(1, self.pred_len)
        for index in range(1, self.pred_len):
            state_coordinates = self.state_projection(state)
            interaction = torch.einsum(
                "br,brd->bd", state_coordinates * controls[:, index],
                self.vector_left.unsqueeze(0).expand(state.size(0), -1, -1)
            )
            interaction = interaction + torch.einsum(
                "br,rd->bd", controls[:, index], self.vector_right
            )
            derivative = -decay * state + interaction + forcing[:, index]
            state = state + step * derivative
            states.append(state)
        output = torch.stack(states, dim=1)
        return self.output_norm(output + self.horizon_bias)


class TariffScaleTransport(nn.Module):
    """Energy-preserving transport between future multiscale latent states.

    A tariff token generates a skew-symmetric cross-scale generator.  Its
    matrix exponential is therefore an exactly invertible, norm-preserving
    transport operator on the stack of scale-specific future states.  This
    changes how PDM/FMM scales exchange future information, rather than gating
    or correcting a finished forecast.
    """
    def __init__(self, scales: int, d_model: int, rank: int = 4):
        super().__init__()
        self.scales, self.d_model, self.rank = int(scales), int(d_model), max(1, int(rank))
        self.control_projection = nn.Linear(self.d_model, self.rank)
        self.generator_basis = nn.Parameter(torch.randn(self.rank, self.scales, self.scales) * 0.02)

    def forward(self, scale_latents: list[torch.Tensor], tariff_tokens: torch.Tensor) -> list[torch.Tensor]:
        if len(scale_latents) != self.scales:
            raise ValueError("tariff scale transport received an unexpected number of scales")
        if tariff_tokens.ndim != 3:
            raise ValueError("tariff scale transport expects [B,H,D] tariff tokens")
        stacked = torch.stack(scale_latents, dim=2)
        if stacked.shape[-1] != self.d_model or stacked.shape[:2] != tariff_tokens.shape[:2]:
            raise ValueError("tariff scale transport latent/tariff shapes do not match")
        # Bound the generator coordinates before the matrix exponential.  The
        # bounded coordinates are a stability constraint on the transport
        # operator, not a forecast gate.
        controls = 0.25 * torch.tanh(self.control_projection(tariff_tokens.float()))
        basis = self.generator_basis.float()
        generator = torch.einsum("bhr,rst->bhst", controls, basis).float()
        generator = generator - generator.transpose(-1, -2)
        transport = torch.matrix_exp(generator.reshape(-1, self.scales, self.scales))
        transport = transport.reshape_as(generator).to(dtype=stacked.dtype)
        transported = torch.einsum("bhst,bhtd->bhsd", transport, stacked)
        return [transported[:, :, index, :] for index in range(self.scales)]


class HorizonConditionedRepresentation(nn.Module):
    """Shared low-rank FiLM for horizon-specific history representations.

    FMM receives a history latent and emits all 24 future steps with a linear
    temporal projection.  This module injects a small bank of horizon-bin
    (1--6, 7--12, 13--24) context vectors into each scale *before* that
    projection.  The bank is projected from the scale's history summary and
    mapped onto its temporal length using fixed smooth profiles, so no future
    target or calendar feature enters the representation.  The same weights
    are shared across all scales.
    """
    def __init__(self, d_model: int, rank: int = 8, strength: float = 0.1):
        super().__init__()
        self.d_model, self.rank, self.bins = int(d_model), int(rank), 3
        if self.rank < 1:
            raise ValueError("hcr_rank must be positive")
        self.context = nn.Linear(self.d_model, 2 * self.bins * self.rank)
        self.low_rank = nn.Parameter(torch.empty(2, self.bins, self.rank, self.d_model))
        nn.init.normal_(self.low_rank, std=self.d_model ** -0.5)
        self.strength = nn.Parameter(torch.tensor(float(strength)).clamp(min=0.0))

    @staticmethod
    def _profiles(length: int, device, dtype):
        # Smooth, non-overlapping-ish temporal profiles associated with short,
        # medium and long forecast bins. They only provide coordinates; all
        # values are learned through the context coefficients.
        t = torch.linspace(0.0, 1.0, int(length), device=device, dtype=dtype)
        centers = torch.tensor((0.15, 0.50, 0.85), device=device, dtype=dtype)
        width = 0.32
        p = torch.exp(-0.5 * ((t[None, :] - centers[:, None]) / width) ** 2)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, latent_list):
        out = []
        for latent in latent_list:
            if latent.ndim != 3 or latent.size(-1) != self.d_model:
                raise ValueError("HCR expects each latent as [batch,time,d_model]")
            b, length, _ = latent.shape
            coeff = self.context(latent.mean(dim=1)).view(
                b, 2, self.bins, self.rank
            )
            profiles = self._profiles(length, latent.device, latent.dtype)
            # [B,component,bin,rank] x [component,bin,rank,D]
            comp = torch.einsum("bckr,ckrd->bckd", coeff, self.low_rank)
            # Split the two component channels into a bounded FiLM scale and
            # additive shift. A small learnable strength keeps the initial
            # route close to official TimeMixer while allowing specialization.
            # ``comp`` is indexed as [batch, component, bin, d_model]; the
            # fixed profiles are [bin, time].  Use the bin index here (the
            # previous implementation accidentally contracted the component
            # axis with the three-bin profile).
            scale_vec = torch.einsum("bkd,kt->btd", comp[:, 0], profiles)
            shift_vec = torch.einsum("bkd,kt->btd", comp[:, 1], profiles)
            alpha = torch.clamp(self.strength, min=0.0)
            out.append(latent * (1.0 + alpha * torch.tanh(scale_vec)) + alpha * shift_vec)
        return out


class ScaleConsistentMixer(nn.Module):
    """Low-rank cross-scale token mixer between PDM and FMM.

    Each PDM scale is aligned to the finest temporal grid, mixed jointly by a
    shared low-rank token map, and returned at its original length.  The last
    projection is zero-initialized so the operator starts as an exact identity;
    it changes history representations rather than adding a forecast gate.
    """
    def __init__(self, d_model: int, scales: int, rank: int = 8,
                 strength: float = 0.1):
        super().__init__()
        self.d_model, self.scales, self.rank = int(d_model), int(scales), int(rank)
        if self.rank < 1:
            raise ValueError("scm_rank must be positive")
        # Compress the aligned scale ensemble before the low-rank map. This
        # keeps CI memory bounded when B*N is large (UrbanEV has 275 zones)
        # while still letting every scale contribute to each scale residual.
        self.norm = nn.LayerNorm(self.d_model)
        self.down = nn.Linear(self.d_model, self.rank)
        self.up = nn.Linear(self.rank, self.scales * self.d_model)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        self.strength = nn.Parameter(torch.tensor(float(strength)).clamp(min=0.0))

    def forward(self, latent_list):
        if len(latent_list) != self.scales:
            raise ValueError("SCM scale count does not match latent list")
        finest = latent_list[0].size(1)
        original_lengths = [value.size(1) for value in latent_list]
        aligned = []
        factors = []
        for value in latent_list:
            if value.size(1) == finest:
                aligned.append(value)
                factors.append(1)
            else:
                # UrbanEV scales are exact powers-of-two.  Nearest repeat is
                # equivalent to zero-order interpolation here and avoids the
                # very expensive CUDA linear interpolation over B*N streams.
                if finest % value.size(1):
                    raise ValueError("SCM requires integer scale ratios")
                factor = finest // value.size(1)
                aligned.append(value.repeat_interleave(factor, dim=1))
                factors.append(factor)
        merged = torch.stack(aligned, dim=2).mean(dim=2)
        residual = self.up(torch.tanh(self.down(self.norm(merged))))
        residual = residual.view(merged.size(0), finest, self.scales, self.d_model)
        alpha = torch.clamp(self.strength, min=0.0)
        output = []
        for index, (value, length, factor) in enumerate(zip(latent_list, original_lengths, factors)):
            refined = aligned[index] + alpha * residual[:, :, index]
            if length != finest:
                refined = refined.reshape(refined.size(0), length, factor, refined.size(-1)).mean(dim=2)
                output.append(refined)
            else:
                output.append(refined)
        return output


class MDMixerScaleGate(nn.Module):
    """Node/channel-specific adaptive multi-scale forecast weighting.

    MDMixer reports that different variables prefer different temporal
    granularities.  TimeMixer's FMM emits one forecast per PDM scale and
    currently sums them with a fixed coefficient.  This gate predicts a
    simplex weight for each flattened CI variable from its scale-specific
    latent history.  It has no access to future labels or target values.
    """
    def __init__(self, d_model: int, scales: int, hidden: int = 16,
                 temperature: float = 1.0):
        super().__init__()
        self.scales = int(scales)
        self.temperature = max(float(temperature), 1e-3)
        hidden = max(4, int(hidden))
        self.score = nn.Sequential(
            nn.Linear(3 * int(d_model), hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, latent_list):
        if len(latent_list) != self.scales:
            raise ValueError("MDMixer scale count does not match latent list")
        # Each latent is [B*N, T_i, D].  Pooling over time retains one
        # descriptor per CI zone/channel, exactly the channel-specific axis
        # used by MDMixer's adaptive multi-granularity gate.
        pooled = []
        for latent in latent_list:
            # Mean, max and volatility summarize complementary temporal
            # evidence for a node's preferred granularity.
            pooled.append(torch.cat((latent.mean(dim=1), latent.amax(dim=1),
                                     latent.std(dim=1, unbiased=False)), dim=-1))
        descriptors = torch.stack(pooled, dim=1)  # [B*N,S,3D]
        logits = self.score(descriptors).squeeze(-1) / self.temperature
        return torch.softmax(logits, dim=1)


class CPRTrendCrossSite(nn.Module):
    """Cross-site trend routing with a channel-independent residual path.

    CPR (Cross-site Patch Residual) is deliberately asymmetric: only the
    causal trend component is pooled across sites through a low-rank latent
    bottleneck.  The innovation/residual component remains per-site, avoiding
    the noise propagation observed with full channel-dependent TimeMixer.
    The zero-initialized decoder makes the operator an identity at step zero.
    """

    def __init__(self, d_model: int, nodes: int, rank: int = 4,
                 strength: float = 0.1, alpha: float = 0.25):
        super().__init__()
        self.d_model = int(d_model)
        self.nodes = int(nodes)
        self.rank = max(1, int(rank))
        self.decomposition = CausalEMADecomp(alpha)
        self.norm = nn.LayerNorm(self.d_model)
        self.down = nn.Linear(self.d_model, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.d_model, bias=False)
        # Static site coefficients preserve site-specific direction while the
        # pooled latent carries information shared by all sites.
        self.site_gate = nn.Parameter(torch.ones(self.nodes, self.rank))
        nn.init.zeros_(self.up.weight)
        self.strength = nn.Parameter(torch.tensor(float(strength)).clamp(min=0.0))

    def forward(self, latent_list, batch_size: int):
        trend_list, residual_list, merged_list = [], [], []
        for latent in latent_list:
            if latent.ndim != 3 or latent.size(-1) != self.d_model:
                raise ValueError("CPR expects each latent as [B*N,time,d_model]")
            if latent.size(0) != int(batch_size) * self.nodes:
                raise ValueError("CPR latent batch does not match configured site count")
            residual, trend = self.decomposition(latent)
            _, length, _ = trend.shape
            trend4 = trend.reshape(batch_size, self.nodes, length, self.d_model)
            residual4 = residual.reshape(batch_size, self.nodes, length, self.d_model)
            # Compress each site's trend, pool the latent across sites, then
            # decode a site-conditioned shared correction.
            low = self.down(self.norm(trend4))
            pooled = low.mean(dim=1, keepdim=True)
            conditioned = pooled * self.site_gate.view(1, self.nodes, 1, self.rank)
            correction = self.up(conditioned)
            refined = trend4 + torch.clamp(self.strength, min=0.0) * correction
            refined = refined.reshape(batch_size * self.nodes, length, self.d_model)
            residual_flat = residual4.reshape(batch_size * self.nodes, length, self.d_model)
            trend_list.append(refined)
            residual_list.append(residual_flat)
            merged_list.append(refined + residual_flat)
        return merged_list, trend_list, residual_list


class PORIPatchOrder(nn.Module):
    """Patch-order recursion with coarse CM and fine CI streams.

    The history is partitioned into six-hour patches.  Coarse patch tokens
    exchange information through a low-rank site latent, while the detail
    residual is left channel independent.  A causal patch-order state scans
    the coarse sequence; it only uses observed patches and therefore adds no
    future information.
    """

    def __init__(self, d_model: int, nodes: int, rank: int = 4,
                 patch_len: int = 6, strength: float = 0.1):
        super().__init__()
        self.d_model, self.nodes = int(d_model), int(nodes)
        self.rank, self.patch_len = max(1, int(rank)), max(1, int(patch_len))
        self.norm = nn.LayerNorm(self.d_model)
        self.down = nn.Linear(self.d_model, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.d_model, bias=False)
        self.site_gate = nn.Parameter(torch.ones(self.nodes, self.rank))
        self.order_in = nn.Linear(self.d_model, self.d_model)
        self.order_state = nn.Linear(self.d_model, self.d_model, bias=False)
        self.order_out = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.order_out.weight)
        self.strength = nn.Parameter(torch.tensor(float(strength)).clamp(min=0.0))

    def forward(self, latent_list, batch_size: int):
        merged_list, coarse_list, detail_list = [], [], []
        for latent in latent_list:
            if latent.ndim != 3 or latent.size(-1) != self.d_model:
                raise ValueError("PORI expects each latent as [B*N,time,d_model]")
            if latent.size(0) != int(batch_size) * self.nodes:
                raise ValueError("PORI latent batch does not match configured site count")
            _, length, _ = latent.shape
            pad = (self.patch_len - length % self.patch_len) % self.patch_len
            work = latent.transpose(1, 2)
            if pad:
                work = F.pad(work, (0, pad), mode="replicate")
            patch_count = work.size(-1) // self.patch_len
            tokens = F.avg_pool1d(work, kernel_size=self.patch_len,
                                  stride=self.patch_len).transpose(1, 2)
            tokens = tokens.reshape(batch_size, self.nodes, patch_count, self.d_model)
            global_tokens = tokens.mean(dim=1)
            low = self.down(self.norm(global_tokens))
            cross = low.unsqueeze(1) * self.site_gate.view(1, self.nodes, 1, self.rank)
            cross = self.up(cross)
            # Causal order recursion over coarse patches.
            state = torch.zeros(batch_size, self.d_model, device=latent.device,
                                dtype=latent.dtype)
            ordered = []
            for index in range(patch_count):
                state = torch.tanh(self.order_in(global_tokens[:, index]) +
                                   self.order_state(state))
                ordered.append(self.order_out(state))
            ordered = torch.stack(ordered, dim=1).unsqueeze(1)
            coarse_tokens = tokens + torch.clamp(self.strength, min=0.0) * (cross + ordered)
            coarse_flat = coarse_tokens.reshape(batch_size * self.nodes,
                                                patch_count, self.d_model)
            coarse_up = coarse_flat.repeat_interleave(self.patch_len, dim=1)[:, :length]
            detail = latent - coarse_up
            merged_list.append(coarse_up + detail)
            coarse_list.append(coarse_up)
            detail_list.append(detail)
        return merged_list, coarse_list, detail_list


class PORISegmentDecoder(nn.Module):
    """Four six-hour segment decoder for the coarse PORI stream."""

    def __init__(self, input_len: int, d_model: int, segments: int = 4,
                 segment_horizon: int = 6):
        super().__init__()
        self.segments = int(segments)
        self.segment_horizon = int(segment_horizon)
        self.segment_projection = nn.Linear(int(input_len), self.segments)
        self.value_projection = nn.Sequential(
            nn.LayerNorm(int(d_model)), nn.Linear(int(d_model), self.segment_horizon)
        )
        # The decoder starts as a zero residual so PORI is exactly paired
        # with the official FMM at initialization; training learns only the
        # segment correction.
        nn.init.zeros_(self.value_projection[-1].weight)
        nn.init.zeros_(self.value_projection[-1].bias)

    def forward(self, latent):
        if latent.ndim != 3:
            raise ValueError("PORI segment decoder expects [batch,time,d_model]")
        tokens = self.segment_projection(latent.transpose(1, 2)).transpose(1, 2)
        values = self.value_projection(tokens)
        return values.reshape(values.size(0), self.segments * self.segment_horizon)


class PORISalientChannelRouter(nn.Module):
    """Dual CI/CM patch router with zero-initialized channel enhancement.

    The CI view preserves each site's local patch token.  The CM view is a
    low-rank summary shared across sites and is injected only through a
    zero-initialized residual projection, so enabling the router cannot change
    the paired TimeMixer at initialization.
    """

    def __init__(self, d_model: int, nodes: int, rank: int = 4,
                 strength: float = 0.1):
        super().__init__()
        self.d_model, self.nodes, self.rank = int(d_model), int(nodes), max(1, int(rank))
        self.local_down = nn.Linear(self.d_model, self.rank, bias=False)
        self.global_down = nn.Linear(self.d_model, self.rank, bias=False)
        self.global_up = nn.Linear(self.rank, self.d_model, bias=False)
        self.saliency = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model), nn.GELU(),
            nn.Linear(self.d_model, self.d_model), nn.Sigmoid()
        )
        self.residual = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.zeros_(self.residual.weight)
        self.strength = nn.Parameter(torch.tensor(float(strength)).clamp(min=0.0))

    def forward(self, latent_list, batch_size: int):
        output = []
        for latent in latent_list:
            if latent.size(0) != int(batch_size) * self.nodes:
                raise ValueError("PORI router latent batch does not match site count")
            _, length, _ = latent.shape
            view = latent.reshape(batch_size, self.nodes, length, self.d_model)
            cm = view.mean(dim=1, keepdim=True)
            cm_low = self.global_down(cm)
            cm_view = self.global_up(cm_low)
            summary = view.mean(dim=2)
            volatility = view.std(dim=2, unbiased=False)
            gate = self.saliency(torch.cat((summary, volatility), dim=-1))
            local_low = self.local_down(view)
            fused = torch.tanh(local_low + self.global_down(cm_view))
            fused_up = self.global_up(fused)
            residual = self.residual(fused_up * gate.unsqueeze(2))
            refined = view + torch.clamp(self.strength, min=0.0) * residual
            output.append(refined.reshape(batch_size * self.nodes, length, self.d_model))
        return output


class PORIEqualTokenRecursion(nn.Module):
    """Equal-token temporal recursion used by the minimal PORI route.

    Every official TimeMixer resolution is converted to the same number of
    non-overlapping temporal tokens (21 for the 168/84/42/21 UrbanEV
    pyramid).  The recursion runs from coarse to fine on the aligned token
    axis and is station independent: no cross-site router, selector or
    sample-dependent gate is involved.  The reconstructed token correction is
    added to the original latent, preserving the official backbone exactly at
    initialization because each cross-scale projection starts at zero.
    """

    def __init__(self, d_model: int, scale_lengths: list[int],
                 token_count: int = 21):
        super().__init__()
        self.d_model = int(d_model)
        self.scale_lengths = tuple(int(x) for x in scale_lengths)
        self.token_count = int(token_count)
        if self.d_model < 1 or self.token_count < 1:
            raise ValueError("PORI equal-token dimensions must be positive")
        if any(length < 1 or length % self.token_count for length in self.scale_lengths):
            raise ValueError(
                "PORI equal-token recursion requires every scale length to "
                "be divisible by token_count"
            )
        self.patch_lengths = tuple(length // self.token_count for length in self.scale_lengths)
        # Index i transports the already refined coarser scale i+1 to scale i.
        self.coarse_to_fine = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model, bias=False)
             for _ in range(max(0, len(self.scale_lengths) - 1))]
        )
        for projection in self.coarse_to_fine:
            nn.init.zeros_(projection.weight)

    def forward(self, latent_list: list[torch.Tensor], batch_size: int):
        if len(latent_list) != len(self.scale_lengths):
            raise ValueError("PORI equal-token scale count mismatch")
        base_tokens = []
        for latent, expected_length in zip(latent_list, self.scale_lengths):
            if latent.ndim != 3 or latent.size(-1) != self.d_model:
                raise ValueError("PORI equal-token latent must be [B*N,time,d]")
            if latent.size(0) % int(batch_size):
                raise ValueError("PORI equal-token latent batch is incompatible with B")
            if latent.size(1) != expected_length:
                raise ValueError(
                    f"PORI expected scale length {expected_length}, got {latent.size(1)}"
                )
            patch = self.patch_lengths[len(base_tokens)]
            tokens = F.avg_pool1d(
                latent.transpose(1, 2), kernel_size=patch, stride=patch
            ).transpose(1, 2).contiguous()
            if tokens.size(1) != self.token_count:
                raise RuntimeError("PORI equal-token patchification produced wrong token count")
            base_tokens.append(tokens)

        refined = [None] * len(base_tokens)
        # Coarse-to-fine recursion is possible because all scales share the
        # same token axis.  A zero initialized projection makes this an exact
        # identity path before training learns cross-scale structure.
        refined[-1] = base_tokens[-1]
        for index in range(len(base_tokens) - 2, -1, -1):
            refined[index] = base_tokens[index] + self.coarse_to_fine[index](refined[index + 1])

        outputs = []
        for latent, base, tokens, patch in zip(
                latent_list, base_tokens, refined, self.patch_lengths):
            base_up = base.transpose(1, 2).repeat_interleave(patch, dim=-1)
            refined_up = tokens.transpose(1, 2).repeat_interleave(patch, dim=-1)
            # Only the learned patch recursion is inserted; original
            # within-patch detail remains untouched.
            outputs.append(latent + (refined_up - base_up).transpose(1, 2))
        return outputs, refined


class PORIEqualTokenSegmentFMM(nn.Module):
    """Segment-wise FMM readout over the aligned PORI token sequence.

    Each scale emits four six-hour segments from the 21 aligned tokens.  The
    official FMM remains the main path; this decoder is a structural,
    horizon-segmented companion summed per scale, with a zero-initialized
    value projection so the paired baseline is unchanged at step zero.  It
    has no gate or post-hoc forecast selector.
    """

    def __init__(self, d_model: int, pred_len: int, token_count: int = 21,
                 segments: int = 4, segment_horizon: int = 6):
        super().__init__()
        self.d_model = int(d_model)
        self.pred_len = int(pred_len)
        self.token_count = int(token_count)
        self.segments = int(segments)
        self.segment_horizon = int(segment_horizon)
        if self.segments * self.segment_horizon != self.pred_len:
            raise ValueError("PORI segment FMM must cover pred_len exactly")
        self.segment_projection = nn.Linear(self.token_count, self.segments)
        self.value_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.segment_horizon),
        )
        # The branch learns an additive structured FMM while preserving the
        # official prediction at initialization.  No scalar strength/gate is
        # optimized to select between the two forecasts.
        nn.init.zeros_(self.value_projection[-1].weight)
        nn.init.zeros_(self.value_projection[-1].bias)

    def forward(self, token_list: list[torch.Tensor], batch_size: int,
                nodes: int):
        if len(token_list) < 1:
            raise ValueError("PORI segment FMM requires at least one scale")
        outputs = []
        for tokens in token_list:
            if tokens.ndim != 3 or tokens.size(1) != self.token_count:
                raise ValueError("PORI segment FMM expects [B*N,21,d] tokens")
            if tokens.size(0) != int(batch_size) * int(nodes):
                raise ValueError("PORI segment FMM latent batch does not match B*N")
            # [B*N,21,d] -> [B*N,segments,d], then emit six values per segment.
            segment_tokens = self.segment_projection(tokens.transpose(1, 2)).transpose(1, 2)
            values = self.value_projection(segment_tokens)
            values = values.reshape(tokens.size(0), self.pred_len)
            outputs.append(values.reshape(batch_size, nodes, self.pred_len).permute(0, 2, 1).contiguous())
        return outputs


class CitySpecificAdapter(nn.Module):
    """Small zero-initialized residual adapter for a source/target city."""

    def __init__(self, d_model: int, rank: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(int(d_model))
        self.down = nn.Linear(int(d_model), max(1, int(rank)), bias=False)
        self.up = nn.Linear(max(1, int(rank)), int(d_model), bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(torch.tanh(self.down(self.norm(x))))


class TimeCAPGroupRouter(nn.Module):
    """Overlapping station-group CI MLP with a low-rank meta-router."""

    def __init__(self, d_model: int, nodes: int, group_size: int = 32,
                 stride: int = 16, rank: int = 8, strength: float = 0.1):
        super().__init__()
        self.d_model, self.nodes = int(d_model), int(nodes)
        self.group_size, self.stride = max(2, int(group_size)), max(1, int(stride))
        self.rank = max(1, int(rank))
        starts = list(range(0, max(1, self.nodes - self.group_size + 1), self.stride))
        if not starts or starts[-1] + self.group_size < self.nodes:
            starts.append(max(0, self.nodes - self.group_size))
        self.register_buffer("group_starts", torch.tensor(starts, dtype=torch.long), persistent=False)
        self.local_mlp = nn.Sequential(
            nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model),
            nn.GELU(), nn.Linear(self.d_model, self.d_model, bias=False)
        )
        self.meta_down = nn.Linear(self.d_model, self.rank, bias=False)
        self.meta_up = nn.Linear(self.rank, self.d_model, bias=False)
        self.group_out = nn.Linear(self.d_model, self.d_model, bias=False)
        # Keep the residual branch small at initialization without making the
        # meta-router a dead path.  Zero-initializing both meta_up and
        # group_out cuts the gradient to each layer (their product is the
        # only route from the forecast loss), so the advertised group router
        # never learns under the short strict budget.
        nn.init.zeros_(self.local_mlp[-1].weight)
        nn.init.normal_(self.meta_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.group_out.weight, mean=0.0, std=0.02)
        self.strength = nn.Parameter(torch.tensor(float(strength)).clamp(min=0.0))

    def forward(self, latent_list, batch_size: int):
        output = []
        for latent in latent_list:
            if latent.size(0) != batch_size * self.nodes:
                raise ValueError("TimeCAP group router expects [B*N,time,d_model]")
            _, length, _ = latent.shape
            view = latent.reshape(batch_size, self.nodes, length, self.d_model)
            local = view + self.local_mlp(view)
            groups = []
            for start in self.group_starts.tolist():
                groups.append(local[:, start:start + self.group_size].mean(dim=1))
            group_tokens = torch.stack(groups, dim=1)  # [B,G,T,D]
            meta = self.meta_down(group_tokens)
            meta_context = self.meta_up(meta.mean(dim=1, keepdim=True))
            updates = torch.zeros_like(local)
            counts = torch.zeros(self.nodes, device=latent.device, dtype=latent.dtype)
            for group_index, start in enumerate(self.group_starts.tolist()):
                end = min(self.nodes, start + self.group_size)
                update = self.group_out(meta_context[:, 0])
                updates[:, start:end] = updates[:, start:end] + update.unsqueeze(1)
                counts[start:end] += 1.0
            refined = local + updates / counts.clamp_min(1.0).view(1, self.nodes, 1, 1)
            output.append((view + torch.clamp(self.strength, min=0.0) *
                           (refined - view)).reshape_as(latent))
        return output


class TimeCAPARHead(nn.Module):
    """Short recursive six-step head from the finest latent state."""

    def __init__(self, d_model: int, steps: int = 6):
        super().__init__()
        self.d_model, self.steps = int(d_model), int(steps)
        self.state = nn.GRUCell(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, 1)
        # A zero output head blocks gradients from the recursive state.  Use a
        # small head so the AR branch starts as a near-zero residual but can
        # train within the fixed 16-epoch protocol.
        nn.init.normal_(self.value.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value.bias)

    def forward(self, latent):
        state = latent[:, -1]
        context = latent.mean(dim=1)
        values = []
        for _ in range(self.steps):
            state = self.state(context, state)
            values.append(self.value(state).squeeze(-1))
        return torch.stack(values, dim=1)


class FactorizedDynamicCrossNode(nn.Module):
    """Station-identity-preserving, context-gated low-rank node mixing.

    The module does not pool stations into a single global token.  Instead it
    projects each station's observed-history summary to a gated rank space,
    aggregates rank statistics with a learned source factor, and projects the
    result back with a target factor tied to station identity.  A small
    residual projection is applied independently at each time step before
    PDM.  This keeps the cross-node operator factorized (O(B*N*D*R)) while
    allowing the mixing weights to change with the observed context.
    """

    def __init__(self, d_model: int, nodes: int, rank: int = 8,
                 strength: float = 0.05, use_history_risk_gate: bool = False,
                 risk_gate_hidden: int = 16, risk_gate_min: float = 0.05,
                 risk_gate_max: float = 0.95, risk_gate_init: float = 0.25):
        super().__init__()
        self.d_model, self.nodes = int(d_model), int(nodes)
        self.rank = max(1, int(rank))
        self.source_factor = nn.Parameter(torch.empty(self.nodes, self.rank))
        self.target_factor = nn.Parameter(torch.empty(self.nodes, self.rank))
        nn.init.normal_(self.source_factor, mean=0.0, std=0.08)
        nn.init.normal_(self.target_factor, mean=0.0, std=0.08)
        self.context_proj = nn.Linear(self.d_model, self.rank, bias=False)
        self.target_gate = nn.Sequential(
            nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.rank)
        )
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        self.use_history_risk_gate = bool(use_history_risk_gate)
        self.risk_gate_min = float(risk_gate_min)
        self.risk_gate_max = float(risk_gate_max)
        if not 0.0 < self.risk_gate_min < self.risk_gate_max < 1.0:
            raise ValueError("history risk gate bounds must satisfy 0 < min < max < 1")
        if self.use_history_risk_gate:
            hidden = max(1, int(risk_gate_hidden))
            self.history_risk_gate = nn.Sequential(
                nn.Linear(3 * self.d_model, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )
            # Choose the final bias so the bounded gate starts at the requested
            # effective reliability (default 0.25), while retaining gradients.
            target = min(max(float(risk_gate_init), self.risk_gate_min + 1e-4),
                         self.risk_gate_max - 1e-4)
            q = (target - self.risk_gate_min) / (self.risk_gate_max - self.risk_gate_min)
            bias = math.log(q / (1.0 - q))
            # Tiny nonzero weights preserve a near-constant initial gate while
            # keeping the hidden history descriptor trainable on the first
            # backward pass.
            nn.init.normal_(self.history_risk_gate[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.history_risk_gate[-1].bias, bias)
        # Start with a weak, positive residual while preserving a learnable
        # gate.  Unlike a zero projection this keeps all factors trainable on
        # the first backward pass.
        self.strength = nn.Parameter(torch.tensor(float(strength)).clamp(min=0.0))

    def forward(self, latent_list, batch_size: int):
        output = []
        for latent in latent_list:
            if latent.size(0) != batch_size * self.nodes:
                raise ValueError("dynamic cross-node expects [B*N,time,d_model]")
            _, length, _ = latent.shape
            view = latent.reshape(batch_size, self.nodes, length, self.d_model)
            summary = view.mean(dim=2)  # [B,N,D], history only
            src_gate = torch.sigmoid(self.context_proj(summary))
            src = self.source_factor.view(1, self.nodes, self.rank) * src_gate
            # Rank statistics retain source identity through the learned source
            # factors instead of replacing all stations with a global mean.
            rank_ctx = torch.einsum("bnr,bnd->brd", src, summary)
            tgt_gate = torch.sigmoid(self.target_gate(summary))
            target = self.target_factor.view(1, self.nodes, self.rank) * tgt_gate
            mixed_summary = torch.einsum("bnr,brd->bnd", target, rank_ctx)
            delta = self.output_proj(mixed_summary).unsqueeze(2)
            # Context gate limits negative transfer for stations whose history
            # is unlike the learned source factors.
            gate = torch.sigmoid(summary.mean(dim=-1, keepdim=True)).unsqueeze(2)
            if self.use_history_risk_gate:
                # The reliability descriptor is causal: last token, temporal
                # mean, and temporal standard deviation from observed history.
                history_stats = torch.cat(
                    (view[:, :, -1], view.mean(dim=2),
                     view.std(dim=2, unbiased=False)), dim=-1
                )
                risk = torch.sigmoid(self.history_risk_gate(history_stats))
                risk = self.risk_gate_min + (self.risk_gate_max - self.risk_gate_min) * risk
                gate = gate * risk.unsqueeze(2)
            refined = view + torch.clamp(self.strength, min=0.0) * gate * delta
            output.append(refined.reshape_as(latent))
        return output


class TrendGatedFactorizedCrossNode(FactorizedDynamicCrossNode):
    """Factorized dynamic node mixing restricted to low-frequency trends."""

    def __init__(self, d_model: int, nodes: int, rank: int = 8,
                 strength: float = 0.05, trend_kernel: int = 5):
        super().__init__(d_model, nodes, rank=rank, strength=strength)
        kernel = max(3, int(trend_kernel))
        if kernel % 2 == 0:
            kernel += 1
        self.trend_kernel = kernel

    def forward(self, latent_list, batch_size: int):
        output = []
        for latent in latent_list:
            if latent.size(0) != batch_size * self.nodes:
                raise ValueError("trend cross-node expects [B*N,time,d_model]")
            _, length, _ = latent.shape
            view = latent.reshape(batch_size, self.nodes, length, self.d_model)
            # Causal-history representation is already encoded in latent.  A
            # short symmetric average here is only a latent low-pass filter;
            # it never reads future labels or decoder inputs.
            work = view.reshape(batch_size * self.nodes, length, self.d_model).transpose(1, 2)
            trend = F.avg_pool1d(
                work, kernel_size=self.trend_kernel,
                stride=1, padding=self.trend_kernel // 2,
                count_include_pad=False,
            ).transpose(1, 2).reshape(batch_size, self.nodes, length, self.d_model)
            summary = trend.mean(dim=2)
            src_gate = torch.sigmoid(self.context_proj(summary))
            src = self.source_factor.view(1, self.nodes, self.rank) * src_gate
            rank_trend = torch.einsum("bnr,bntd->brtd", src, trend)
            tgt_gate = torch.sigmoid(self.target_gate(summary))
            target = self.target_factor.view(1, self.nodes, self.rank) * tgt_gate
            mixed_trend = torch.einsum("bnr,brtd->bntd", target, rank_trend)
            delta = self.output_proj(mixed_trend)
            # Stationwise trend energy gates the residual; high-frequency
            # latent residual is copied exactly, preventing noise exchange.
            energy = torch.sigmoid(summary.abs().mean(dim=-1, keepdim=True)).unsqueeze(2)
            refined = view + torch.clamp(self.strength, min=0.0) * energy * delta
            output.append(refined.reshape_as(latent))
        return output


class PatchTokenFactorizedCrossNode(nn.Module):
    """Multi-scale patch-token node mixer with an identity residual path.

    Patch tokens are formed only inside a residual branch (6-hour equivalent
    patches at the finest scale).  The original latent, including its
    high-frequency detail, is retained and receives a small station-specific
    cross-node correction reconstructed from the mixed patch tokens.
    """

    def __init__(self, d_model: int, nodes: int, rank: int = 8,
                 strength: float = 0.02, patch_hours: int = 6):
        super().__init__()
        self.d_model, self.nodes = int(d_model), int(nodes)
        self.rank = max(1, int(rank))
        self.patch_hours = max(1, int(patch_hours))
        self.source_factor = nn.Parameter(torch.empty(self.nodes, self.rank))
        self.target_factor = nn.Parameter(torch.empty(self.nodes, self.rank))
        nn.init.normal_(self.source_factor, mean=0.0, std=0.08)
        nn.init.normal_(self.target_factor, mean=0.0, std=0.08)
        self.context_proj = nn.Linear(self.d_model, self.rank, bias=False)
        self.patch_gate = nn.Sequential(
            nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.rank)
        )
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        self.strength = nn.Parameter(torch.tensor(float(strength)).clamp(min=0.0))

    def forward(self, latent_list, batch_size: int):
        output = []
        for scale, latent in enumerate(latent_list):
            if latent.size(0) != batch_size * self.nodes:
                raise ValueError("patch cross-node expects [B*N,time,d_model]")
            _, length, _ = latent.shape
            view = latent.reshape(batch_size, self.nodes, length, self.d_model)
            # Convert the fixed 6-hour physical patch to each downsampled
            # scale.  The residual branch is aligned back to every token.
            ratio = 2 ** int(scale)
            patch = max(1, int(round(self.patch_hours / ratio)))
            n_patches = (length + patch - 1) // patch
            padded = F.pad(view, (0, 0, 0, n_patches * patch - length))
            tokens = padded.reshape(batch_size, self.nodes, n_patches, patch, self.d_model).mean(dim=3)
            summary = tokens.mean(dim=2)
            src_gate = torch.sigmoid(self.context_proj(summary))
            src = self.source_factor.view(1, self.nodes, self.rank) * src_gate
            rank_tokens = torch.einsum("bnr,bnpd->brpd", src, tokens)
            tgt_gate = torch.sigmoid(self.patch_gate(summary))
            target = self.target_factor.view(1, self.nodes, self.rank) * tgt_gate
            mixed_tokens = torch.einsum("bnr,brpd->bnpd", target, rank_tokens)
            delta_tokens = self.output_proj(mixed_tokens)
            delta = delta_tokens.unsqueeze(3).expand(
                batch_size, self.nodes, n_patches, patch, self.d_model
            ).reshape(batch_size, self.nodes, n_patches * patch, self.d_model)[:, :, :length]
            # Identity path preserves all un-mixed latent detail; only a small
            # dynamic patch correction is added before PDM.
            gate = torch.sigmoid(summary.abs().mean(dim=-1, keepdim=True)).unsqueeze(2)
            refined = view + torch.clamp(self.strength, min=0.0) * gate * delta
            output.append(refined.reshape_as(latent))
        return output


class DynamicPhaseFactorizedCrossNode(nn.Module):
    """Pre-PDM node factors plus a learned multi-period phase residual."""

    def __init__(self, d_model: int, nodes: int, rank: int = 8,
                 node_strength: float = 0.05, phase_strength: float = 0.01,
                 base_period: int = 24):
        super().__init__()
        self.d_model, self.nodes = int(d_model), int(nodes)
        self.rank = max(1, int(rank))
        self.base_period = max(1, int(base_period))
        self.source_factor = nn.Parameter(torch.empty(self.nodes, self.rank))
        self.target_factor = nn.Parameter(torch.empty(self.nodes, self.rank))
        nn.init.normal_(self.source_factor, mean=0.0, std=0.08)
        nn.init.normal_(self.target_factor, mean=0.0, std=0.08)
        self.context_proj = nn.Linear(self.d_model, self.rank, bias=False)
        self.target_gate = nn.Sequential(
            nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.rank)
        )
        self.node_output = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.normal_(self.node_output.weight, mean=0.0, std=0.02)
        # A shared 24-hour phase dictionary is sliced at coarser scales.
        self.phase_embed = nn.Parameter(torch.zeros(self.base_period, self.d_model))
        nn.init.normal_(self.phase_embed, mean=0.0, std=0.02)
        self.phase_output = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.normal_(self.phase_output.weight, mean=0.0, std=0.005)
        self.phase_gate = nn.Linear(self.d_model, 1)
        nn.init.zeros_(self.phase_gate.weight)
        nn.init.constant_(self.phase_gate.bias, -2.0)
        self.node_strength = nn.Parameter(torch.tensor(float(node_strength)).clamp(min=0.0))
        self.phase_strength = nn.Parameter(torch.tensor(float(phase_strength)).clamp(min=0.0))

    def forward(self, latent_list, batch_size: int):
        output = []
        for scale, latent in enumerate(latent_list):
            if latent.size(0) != batch_size * self.nodes:
                raise ValueError("dynamic phase cross-node expects [B*N,time,d_model]")
            _, length, _ = latent.shape
            view = latent.reshape(batch_size, self.nodes, length, self.d_model)
            summary = view.mean(dim=2)
            src_gate = torch.sigmoid(self.context_proj(summary))
            src = self.source_factor.view(1, self.nodes, self.rank) * src_gate
            rank_ctx = torch.einsum("bnr,bnd->brd", src, summary)
            tgt_gate = torch.sigmoid(self.target_gate(summary))
            target = self.target_factor.view(1, self.nodes, self.rank) * tgt_gate
            mixed_summary = torch.einsum("bnr,brd->bnd", target, rank_ctx)
            node_delta = self.node_output(mixed_summary).unsqueeze(2)
            node_gate = torch.sigmoid(summary.mean(dim=-1, keepdim=True)).unsqueeze(2)

            # Exact period alignment at each downsampled scale: 24, 12, 6,
            # 3 for the standard 168/2^scale lengths.  If a custom length is
            # used, fall back to the largest divisor not exceeding 24.
            period = max(1, self.base_period // (2 ** int(scale)))
            while period > 1 and length % period:
                period -= 1
            cycles = max(1, length // period)
            usable = cycles * period
            phase_view = view[:, :, :usable].reshape(
                batch_size, self.nodes, cycles, period, self.d_model
            ).mean(dim=2)
            phase_token = phase_view + self.phase_embed[:period].view(1, 1, period, self.d_model)
            phase_delta = self.phase_output(phase_token)
            phase_delta = phase_delta.unsqueeze(2).expand(
                batch_size, self.nodes, cycles, period, self.d_model
            ).reshape(batch_size, self.nodes, usable, self.d_model)
            if usable < length:
                phase_delta = F.pad(phase_delta, (0, 0, 0, length - usable))
            phase_gate = torch.sigmoid(self.phase_gate(summary)).unsqueeze(2)
            refined = view + torch.clamp(self.node_strength, min=0.0) * node_gate * node_delta
            refined = refined + torch.clamp(self.phase_strength, min=0.0) * phase_gate * phase_delta
            output.append(refined.reshape_as(latent))
        return output


class HarmonicPhaseResidualDecoder(nn.Module):
    """Bounded 24/12-hour harmonic residual on top of the official FMM."""

    def __init__(self, d_model: int, pred_len: int, periods=(24, 12),
                 strength: float = 0.02, gate_init: float = -4.0):
        super().__init__()
        self.d_model, self.pred_len = int(d_model), int(pred_len)
        self.periods = tuple(int(p) for p in periods if int(p) > 0)
        self.n_harmonics = 2 * len(self.periods)  # sin/cos per period
        self.coeff = nn.Linear(self.d_model, self.n_harmonics)
        nn.init.normal_(self.coeff.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.coeff.bias)
        horizon = torch.arange(self.pred_len, dtype=torch.float32)
        columns = []
        for period in self.periods:
            phase = 2.0 * math.pi * horizon / float(period)
            columns.extend([torch.sin(phase), torch.cos(phase)])
        self.register_buffer("basis", torch.stack(columns, dim=1), persistent=False)
        self.horizon_gate = nn.Parameter(torch.full((self.pred_len,), float(gate_init)))
        self.strength = nn.Parameter(torch.tensor(float(strength)).clamp(min=0.0))

    def forward(self, latent, batch_size: int, nodes: int):
        if latent.size(0) != batch_size * nodes:
            raise ValueError("harmonic decoder expects [B*N,time,d_model]")
        context = latent[:, -1].reshape(batch_size, nodes, self.d_model)
        coeff = self.coeff(context)
        basis = self.basis.to(device=latent.device, dtype=latent.dtype)
        raw = torch.einsum("bnk,hk->bhn", coeff, basis)
        gate = torch.sigmoid(self.horizon_gate).view(1, self.pred_len, 1)
        # A positive bounded strength prevents an accidentally dominant
        # harmonic branch while keeping gradients alive from initialization.
        return torch.clamp(self.strength, min=0.0) * gate * raw


class CrossPhaseAttentionResidualDecoder(nn.Module):
    """History-only cross-phase residual decoder on top of official FMM.

    The finest latent is grouped into seven observed 24-hour cycles.  Cycle
    averaging forms one token per phase, self-attention exchanges information
    across the 24 phase positions, and learned horizon phase queries read the
    attended tokens.  A weak bounded residual preserves the official FMM path
    at initialization and never consumes future labels.
    """

    def __init__(self, d_model: int, pred_len: int, period: int = 24,
                 cycles: int = 7, heads: int = 4, strength: float = 0.02,
                 gate_init: float = -4.0):
        super().__init__()
        self.d_model, self.pred_len = int(d_model), int(pred_len)
        self.period, self.cycles = max(1, int(period)), max(1, int(cycles))
        self.phase_tokens = nn.Parameter(torch.zeros(self.period, self.d_model))
        self.horizon_queries = nn.Parameter(torch.zeros(self.pred_len, self.d_model))
        self.input_norm = nn.LayerNorm(self.d_model)
        self.phase_attention = nn.MultiheadAttention(
            self.d_model, num_heads=max(1, int(heads)), batch_first=True,
        )
        self.query_attention = nn.MultiheadAttention(
            self.d_model, num_heads=max(1, int(heads)), batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, 1)
        nn.init.normal_(self.phase_tokens, mean=0.0, std=0.01)
        nn.init.normal_(self.horizon_queries, mean=0.0, std=0.01)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.output_proj.bias)
        self.horizon_gate = nn.Parameter(torch.full((self.pred_len,), float(gate_init)))
        self.strength = nn.Parameter(torch.tensor(float(strength)).clamp(min=0.0))

    def forward(self, latent, batch_size: int, nodes: int):
        if latent.size(0) != batch_size * nodes:
            raise ValueError("cross-phase decoder expects [B*N,time,d_model]")
        total = self.period * self.cycles
        if latent.size(1) < total:
            raise ValueError("cross-phase decoder needs at least period*cycles history tokens")
        # Use the most recent complete cycles so the route remains causal when
        # a caller supplies a longer history than the configured 7 cycles.
        history = latent[:, -total:].reshape(
            batch_size * nodes, self.cycles, self.period, self.d_model
        )
        tokens = history.mean(dim=1) + self.phase_tokens.unsqueeze(0)
        tokens = self.input_norm(tokens)
        attended, _ = self.phase_attention(tokens, tokens, tokens, need_weights=False)
        attended = tokens + attended
        queries = self.horizon_queries.unsqueeze(0).expand(latent.size(0), -1, -1)
        queried, _ = self.query_attention(queries, attended, attended, need_weights=False)
        decoded = self.output_proj(self.output_norm(queried)).squeeze(-1)
        gate = torch.sigmoid(self.horizon_gate).view(1, self.pred_len)
        residual = torch.clamp(self.strength, min=0.0) * gate * decoded
        return residual.reshape(batch_size, nodes, self.pred_len).permute(0, 2, 1)


class UngatedHarmonicPhaseDecoder(nn.Module):
    """Fixed-amplitude harmonic history decoder without a learned gate.

    The period basis is a structural prior; its amplitude is fixed by the
    configuration and the coefficient map is learned.  This variant is used
    for the no-gate CPR ablation so gains cannot come from a tuned sigmoid
    horizon valve or a learned branch strength.
    """
    def __init__(self, d_model: int, pred_len: int, periods=(24, 12),
                 strength: float = 0.03):
        super().__init__()
        self.d_model, self.pred_len = int(d_model), int(pred_len)
        self.periods = tuple(int(p) for p in periods if int(p) > 0)
        columns = []
        horizon = torch.arange(self.pred_len, dtype=torch.float32)
        for period in self.periods:
            phase = 2.0 * math.pi * horizon / float(period)
            columns.extend([torch.sin(phase), torch.cos(phase)])
        self.register_buffer("basis", torch.stack(columns, dim=1), persistent=False)
        self.coeff = nn.Linear(self.d_model, len(columns))
        nn.init.normal_(self.coeff.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.coeff.bias)
        self.register_buffer("fixed_strength", torch.tensor(float(strength)), persistent=False)

    def forward(self, latent, batch_size: int, nodes: int):
        if latent.size(0) != batch_size * nodes:
            raise ValueError("ungated harmonic decoder expects [B*N,time,d_model]")
        context = latent[:, -1].reshape(batch_size, nodes, self.d_model)
        coeff = self.coeff(context)
        basis = self.basis.to(device=latent.device, dtype=latent.dtype)
        raw = torch.einsum("bnk,hk->bhn", coeff, basis)
        return self.fixed_strength.to(dtype=raw.dtype) * raw


class HistoryConfidenceGate(nn.Module):
    """History-only sample/node/horizon reliability gate for residuals."""

    def __init__(self, d_model: int, pred_len: int, hidden: int = 16,
                 min_gate: float = 0.05, max_gate: float = 0.95):
        super().__init__()
        self.d_model, self.pred_len = int(d_model), int(pred_len)
        self.min_gate, self.max_gate = float(min_gate), float(max_gate)
        self.net = nn.Sequential(
            nn.Linear(3 * self.d_model, int(hidden)), nn.GELU(),
            nn.Linear(int(hidden), self.pred_len)
        )
        # Zero-init logits gives a neutral 0.5 reliability at step zero.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent, batch_size: int, nodes: int):
        if latent.size(0) != batch_size * nodes:
            raise ValueError("confidence gate expects [B*N,time,d_model]")
        view = latent.reshape(batch_size, nodes, latent.size(1), self.d_model)
        context = torch.cat((view[:, :, -1], view.mean(dim=2),
                             view.std(dim=2, unbiased=False)), dim=-1)
        raw = self.net(context)
        gate = torch.sigmoid(raw)
        return self.min_gate + (self.max_gate - self.min_gate) * gate


class HistoryPhaseGate(nn.Module):
    """History-conditioned gate for the cross-phase residual branch.

    The gate starts near one so the existing CPR-DX path is preserved at
    initialization.  It can only attenuate the phase residual and is driven
    by causal latent statistics; it never selects another model.
    """

    def __init__(self, d_model: int, pred_len: int, hidden: int = 16,
                 min_gate: float = 0.05, max_gate: float = 1.0,
                 init_gate: float = 0.9):
        super().__init__()
        self.d_model, self.pred_len = int(d_model), int(pred_len)
        self.min_gate, self.max_gate = float(min_gate), float(max_gate)
        self.net = nn.Sequential(
            nn.Linear(3 * self.d_model, int(hidden)), nn.GELU(),
            nn.Linear(int(hidden), self.pred_len)
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        target = min(max(float(init_gate), self.min_gate + 1e-4), self.max_gate - 1e-4)
        q = (target - self.min_gate) / (self.max_gate - self.min_gate)
        nn.init.constant_(self.net[-1].bias, math.log(q / (1.0 - q)))

    def forward(self, latent, batch_size: int, nodes: int):
        if latent.size(0) != batch_size * nodes:
            raise ValueError("history phase gate expects [B*N,time,d_model]")
        view = latent.reshape(batch_size, nodes, latent.size(1), self.d_model)
        context = torch.cat((view[:, :, -1], view.mean(dim=2),
                             view.std(dim=2, unbiased=False)), dim=-1)
        raw = self.net(context)
        gate = torch.sigmoid(raw)
        return self.min_gate + (self.max_gate - self.min_gate) * gate


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.raw_seq_len = getattr(configs, "raw_seq_len", configs.seq_len)
        self.patch_size = int(getattr(configs, "patch_size", 1))
        if self.patch_size > 1:
            self.patch_embedding = nn.Conv1d(
                configs.enc_in, configs.enc_in, kernel_size=self.patch_size,
                stride=self.patch_size, groups=configs.enc_in, bias=True
            )
            # Start from average patch pooling, then let training learn a
            # channel-wise local patch filter.
            with torch.no_grad():
                self.patch_embedding.weight.fill_(1.0 / self.patch_size)
                self.patch_embedding.bias.zero_()
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.use_precision_fusion = bool(getattr(configs, "precision_fusion", False))
        if self.use_precision_fusion:
            raw_w = getattr(configs, "precision_fusion_weights", None)
            raw_b = getattr(configs, "precision_fusion_bias", None)
            if raw_w is None or raw_b is None:
                raise ValueError("precision fusion requires fixed weights and bias")
            weights = torch.as_tensor(raw_w, dtype=torch.float32)
            bias = torch.as_tensor(raw_b, dtype=torch.float32)
            expected = (int(configs.pred_len), int(configs.down_sampling_layers) + 1)
            if tuple(weights.shape) != expected or tuple(bias.shape) != expected:
                raise ValueError(
                    f"precision fusion arrays must have shape {expected}, "
                    f"got {tuple(weights.shape)} and {tuple(bias.shape)}")
            self.register_buffer("precision_fusion_weights", weights)
            self.register_buffer("precision_fusion_bias", bias)
        self.use_reconciled_fmm = bool(getattr(configs, "reconciled_fmm", False))
        if self.use_reconciled_fmm:
            # Fixed block-mean operators for the native FMM scale forecasts.
            # The existing FMM heads still produce H hourly values; each
            # coarse stream is summarized at its native resolution before a
            # closed-form cross-resolution projection.  This keeps checkpoint
            # compatibility while imposing temporal coherence without any
            # sample-dependent selector.
            normal = torch.zeros(int(configs.pred_len), int(configs.pred_len))
            for index in range(int(configs.down_sampling_layers) + 1):
                factor = int(configs.down_sampling_window) ** index
                bins = max(1, int(configs.pred_len) // factor)
                matrix = torch.zeros(bins, int(configs.pred_len))
                for block in range(bins):
                    start = block * factor
                    end = min(int(configs.pred_len), (block + 1) * factor)
                    matrix[block, start:end] = 1.0 / float(max(1, end - start))
                normal = normal + matrix.transpose(0, 1).matmul(matrix)
                self.register_buffer(f"reconciled_aggregation_{index}", matrix)
            self.register_buffer("reconciled_normal_inverse", torch.linalg.inv(normal))
        self.paired_input_dim = int(getattr(configs, "paired_input_dim", 0))
        if self.paired_input_dim < 0:
            raise ValueError("paired_input_dim must be nonnegative")
        self.use_history_state_pdm = bool(
            getattr(configs, "use_history_state_pdm", False)
        )
        self.use_endpoint_bridge = bool(
            getattr(configs, "use_endpoint_bridge", False)
        )
        if self.use_endpoint_bridge:
            if self.channel_independence != 1:
                raise ValueError("endpoint bridge requires channel_independence=1")
            self.endpoint_bridge = FixedEndpointLevelDriftBridge(
                int(configs.d_model),
                int(configs.down_sampling_layers) + 1,
                rank=int(getattr(configs, "endpoint_bridge_rank", 4)),
                endpoint_width=int(getattr(configs, "endpoint_bridge_width", 24)),
            )
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])
        self.use_hcr = bool(getattr(configs, "use_hcr", False))
        if self.use_hcr:
            self.horizon_representation = HorizonConditionedRepresentation(
                int(configs.d_model), rank=int(getattr(configs, "hcr_rank", 8)),
                strength=float(getattr(configs, "hcr_strength", 0.1)),
            )
        self.use_scm = bool(getattr(configs, "use_scm", False))
        if self.use_scm:
            self.scale_consistent_mixer = ScaleConsistentMixer(
                int(configs.d_model), int(configs.down_sampling_layers) + 1,
                rank=int(getattr(configs, "scm_rank", 8)),
                strength=float(getattr(configs, "scm_strength", 0.1)),
            )
        self.use_mdmixer_lite = bool(getattr(configs, "use_mdmixer_lite", False))
        if self.use_mdmixer_lite:
            self.mdmixer_scale_gate = MDMixerScaleGate(
                int(configs.d_model), int(configs.down_sampling_layers) + 1,
                hidden=int(getattr(configs, "mdmixer_scale_gate_hidden", 16)),
                temperature=float(getattr(configs, "mdmixer_scale_temperature", 1.0)),
            )
        self.use_cpr = bool(getattr(configs, "use_cpr", False))
        if self.use_cpr:
            if self.channel_independence != 1:
                raise ValueError("CPR requires channel_independence=1")
            self.cpr = CPRTrendCrossSite(
                int(configs.d_model), int(configs.enc_in),
                rank=int(getattr(configs, "cpr_rank", 4)),
                strength=float(getattr(configs, "cpr_strength", 0.1)),
                alpha=float(getattr(configs, "cpr_ema_alpha", 0.25)),
            )
            self.cpr_horizon_gate = nn.Parameter(torch.zeros(int(configs.pred_len)))
        self.use_pori = bool(getattr(configs, "use_pori", False))
        if self.use_pori:
            if self.channel_independence != 1:
                raise ValueError("PORI requires channel_independence=1")
            self.pori = PORIPatchOrder(
                int(configs.d_model), int(configs.enc_in),
                rank=int(getattr(configs, "pori_rank", 4)),
                patch_len=int(getattr(configs, "pori_patch_len", 6)),
                strength=float(getattr(configs, "pori_strength", 0.1)),
            )
            segment_horizon = int(getattr(configs, "pori_segment_horizon", 6))
            segments = int(getattr(configs, "pori_segments", 4))
            self.pori_decoders = nn.ModuleList([
                PORISegmentDecoder(
                    int(configs.seq_len) // (configs.down_sampling_window ** index),
                    int(configs.d_model), segments=segments,
                    segment_horizon=segment_horizon,
                )
                for index in range(configs.down_sampling_layers + 1)
            ])
            self.pori_decoder_strength = nn.Parameter(torch.tensor(
                float(getattr(configs, "pori_decoder_strength", 0.1))
            ).clamp(min=0.0))
            self.pori_channel_router = PORISalientChannelRouter(
                int(configs.d_model), int(configs.enc_in),
                rank=int(getattr(configs, "pori_router_rank", 4)),
                strength=float(getattr(configs, "pori_router_strength", 0.1)),
            )
        self.use_pori_minimal = bool(getattr(configs, "use_pori_minimal", False))
        if self.use_pori_minimal:
            if self.channel_independence != 1:
                raise ValueError("minimal PORI requires channel_independence=1")
            token_count = int(getattr(configs, "pori_token_count", 21))
            scale_lengths = [
                int(configs.seq_len) // (int(configs.down_sampling_window) ** index)
                for index in range(int(configs.down_sampling_layers) + 1)
            ]
            self.pori_minimal = PORIEqualTokenRecursion(
                int(configs.d_model), scale_lengths, token_count=token_count
            )
            self.pori_minimal_segment_fmm = PORIEqualTokenSegmentFMM(
                int(configs.d_model), int(configs.pred_len),
                token_count=token_count,
                segments=int(getattr(configs, "pori_segments", 4)),
                segment_horizon=int(getattr(configs, "pori_segment_horizon", 6)),
            )
        self.use_city_adapter = bool(getattr(configs, "use_city_adapter", False))
        if self.use_city_adapter:
            count = max(1, int(getattr(configs, "city_adapter_count", 7)))
            rank = max(1, int(getattr(configs, "city_adapter_rank", 8)))
            self.city_adapters = nn.ModuleList(
                [CitySpecificAdapter(int(configs.d_model), rank) for _ in range(count)]
            )
            self.city_adapter_index = int(getattr(configs, "city_adapter_index", 0))
        self.use_timecap = bool(getattr(configs, "use_timecap", False))
        if self.use_timecap:
            if self.channel_independence != 1:
                raise ValueError("TimeCAP requires channel_independence=1")
            self.timecap_router = TimeCAPGroupRouter(
                int(configs.d_model), int(configs.enc_in),
                group_size=int(getattr(configs, "timecap_group_size", 32)),
                stride=int(getattr(configs, "timecap_group_stride", 16)),
                rank=int(getattr(configs, "timecap_router_rank", 8)),
                strength=float(getattr(configs, "timecap_router_strength", 0.1)),
            )
            self.timecap_ar_heads = nn.ModuleList([
                TimeCAPARHead(int(configs.d_model), int(getattr(configs, "timecap_ar_steps", 6)))
                for _ in range(configs.down_sampling_layers + 1)
            ])
            gate_init = float(getattr(configs, "timecap_gate_init", -6.0))
            self.timecap_horizon_gate = nn.Parameter(
                torch.full((int(configs.pred_len),), gate_init)
            )
        self.use_dynamic_cross_node = bool(getattr(configs, "use_dynamic_cross_node", False))
        if self.use_dynamic_cross_node:
            if self.channel_independence != 1:
                raise ValueError("dynamic cross-node route requires channel_independence=1")
            self.dynamic_cross_node = FactorizedDynamicCrossNode(
                int(configs.d_model), int(configs.enc_in),
                rank=int(getattr(configs, "dynamic_cross_node_rank", 8)),
                strength=float(getattr(configs, "dynamic_cross_node_strength", 0.05)),
                use_history_risk_gate=bool(getattr(configs, "use_history_risk_gate", False)),
                risk_gate_hidden=int(getattr(configs, "history_risk_gate_hidden", 16)),
                risk_gate_min=float(getattr(configs, "history_risk_gate_min", 0.05)),
                risk_gate_max=float(getattr(configs, "history_risk_gate_max", 0.95)),
                risk_gate_init=float(getattr(configs, "history_risk_gate_init", 0.25)),
            )
        self.use_regime_state = bool(getattr(configs, "use_regime_state", False))
        if self.use_regime_state:
            if self.channel_independence != 1:
                raise ValueError("causal regime state requires channel_independence=1")
            self.regime_state = CausalRegimeState(int(configs.d_model))
        self.use_post_regime_state = bool(getattr(configs, "use_post_regime_state", False))
        if self.use_post_regime_state:
            if self.channel_independence != 1:
                raise ValueError("post-PDM regime state requires channel_independence=1")
            self.post_regime_state = CausalRegimeState(int(configs.d_model))
        self.use_causal_innovation = bool(getattr(configs, "use_causal_innovation", False))
        if self.use_causal_innovation:
            if self.channel_independence != 1:
                raise ValueError("causal innovation requires channel_independence=1")
            self.innovation_lag = max(1, int(getattr(configs, "innovation_lag", 24)))
            self.innovation_projection = nn.Linear(2, int(configs.d_model))
        self.use_post_pdm_dynamic_cross_node = bool(
            getattr(configs, "use_post_pdm_dynamic_cross_node", False)
        )
        if self.use_post_pdm_dynamic_cross_node:
            if self.channel_independence != 1:
                raise ValueError("post-PDM dynamic cross-node requires channel_independence=1")
            self.post_pdm_dynamic_cross_node = FactorizedDynamicCrossNode(
                int(configs.d_model), int(configs.enc_in),
                rank=int(getattr(configs, "post_pdm_dynamic_cross_node_rank", 8)),
                strength=float(getattr(configs, "post_pdm_dynamic_cross_node_strength", 0.05)),
            )
        self.use_dynamic_phase_cross_node = bool(
            getattr(configs, "use_dynamic_phase_cross_node", False)
        )
        if self.use_dynamic_phase_cross_node:
            if self.channel_independence != 1:
                raise ValueError("dynamic phase cross-node requires channel_independence=1")
            self.dynamic_phase_cross_node = DynamicPhaseFactorizedCrossNode(
                int(configs.d_model), int(configs.enc_in),
                rank=int(getattr(configs, "dynamic_phase_cross_node_rank", 8)),
                node_strength=float(getattr(configs, "dynamic_phase_cross_node_strength", 0.05)),
                phase_strength=float(getattr(configs, "dynamic_phase_cross_node_phase_strength", 0.01)),
                base_period=int(getattr(configs, "dynamic_phase_cross_node_period", 24)),
            )
        self.use_harmonic_phase_decoder = bool(
            getattr(configs, "use_harmonic_phase_decoder", False)
        )
        if self.use_harmonic_phase_decoder:
            periods_text = str(getattr(configs, "harmonic_phase_periods", "24,12"))
            periods = tuple(int(item.strip()) for item in periods_text.split(",") if item.strip())
            if bool(getattr(configs, "use_ungated_harmonic", False)):
                self.harmonic_phase_decoder = UngatedHarmonicPhaseDecoder(
                    int(configs.d_model), int(configs.pred_len), periods=periods,
                    strength=float(getattr(configs, "harmonic_phase_strength", 0.03)),
                )
            else:
                self.harmonic_phase_decoder = HarmonicPhaseResidualDecoder(
                    int(configs.d_model), int(configs.pred_len), periods=periods,
                    strength=float(getattr(configs, "harmonic_phase_strength", 0.02)),
                    gate_init=float(getattr(configs, "harmonic_phase_gate_init", -4.0)),
                )
        self.use_cross_phase_attention_decoder = bool(
            getattr(configs, "use_cross_phase_attention_decoder", False)
        )
        if self.use_cross_phase_attention_decoder:
            self.cross_phase_attention_decoder = CrossPhaseAttentionResidualDecoder(
                int(configs.d_model), int(configs.pred_len),
                period=int(getattr(configs, "cross_phase_period", 24)),
                cycles=int(getattr(configs, "cross_phase_cycles", 7)),
                heads=int(getattr(configs, "cross_phase_heads", 4)),
                strength=float(getattr(configs, "cross_phase_strength", 0.02)),
                gate_init=float(getattr(configs, "cross_phase_gate_init", -4.0)),
            )
        self.use_history_cross_phase_gate = bool(
            getattr(configs, "use_history_cross_phase_gate", False)
        )
        if self.use_history_cross_phase_gate:
            self.history_cross_phase_gate = HistoryPhaseGate(
                int(configs.d_model), int(configs.pred_len),
                hidden=int(getattr(configs, "history_cross_phase_gate_hidden", 16)),
                min_gate=float(getattr(configs, "history_cross_phase_gate_min", 0.05)),
                max_gate=float(getattr(configs, "history_cross_phase_gate_max", 1.0)),
                init_gate=float(getattr(configs, "history_cross_phase_gate_init", 0.9)),
            )
        self.use_confidence_residual_gate = bool(
            getattr(configs, "use_confidence_residual_gate", False)
        )
        if self.use_confidence_residual_gate:
            self.confidence_residual_gate = HistoryConfidenceGate(
                int(configs.d_model), int(configs.pred_len),
                hidden=int(getattr(configs, "confidence_gate_hidden", 16)),
                min_gate=float(getattr(configs, "confidence_gate_min", 0.05)),
                max_gate=float(getattr(configs, "confidence_gate_max", 0.95)),
            )
        self.use_trend_dynamic_cross_node = bool(
            getattr(configs, "use_trend_dynamic_cross_node", False)
        )
        if self.use_trend_dynamic_cross_node:
            if self.channel_independence != 1:
                raise ValueError("trend dynamic cross-node route requires channel_independence=1")
            self.trend_dynamic_cross_node = TrendGatedFactorizedCrossNode(
                int(configs.d_model), int(configs.enc_in),
                rank=int(getattr(configs, "trend_dynamic_cross_node_rank", 8)),
                strength=float(getattr(configs, "trend_dynamic_cross_node_strength", 0.05)),
                trend_kernel=int(getattr(configs, "trend_dynamic_cross_node_kernel", 5)),
            )
        self.use_patch_token_cross_node = bool(
            getattr(configs, "use_patch_token_cross_node", False)
        )
        if self.use_patch_token_cross_node:
            if self.channel_independence != 1:
                raise ValueError("patch token cross-node route requires channel_independence=1")
            self.patch_token_cross_node = PatchTokenFactorizedCrossNode(
                int(configs.d_model), int(configs.enc_in),
                rank=int(getattr(configs, "patch_token_cross_node_rank", 8)),
                strength=float(getattr(configs, "patch_token_cross_node_strength", 0.02)),
                patch_hours=int(getattr(configs, "patch_token_cross_node_hours", 6)),
            )
        self.use_mrti = bool(getattr(configs, "use_mrti", False))
        if self.use_mrti:
            self.mrti = UrbanMRTI(configs)
        self.use_phase_mrti = bool(getattr(configs, "use_phase_mrti", False))
        if self.use_phase_mrti:
            if self.use_mrti:
                raise ValueError("use_mrti and use_phase_mrti are mutually exclusive")
            self.phase_mrti = PhaseAlignedUrbanMRTI(configs)
        self.use_event_mrti = bool(getattr(configs, "use_event_mrti", False))
        if self.use_event_mrti:
            if self.use_mrti or self.use_phase_mrti:
                raise ValueError("MRTI variants are mutually exclusive")
            self.event_mrti = EventAwareUrbanMRTI(configs)
        self.use_apid = bool(getattr(configs, "use_apid", False))
        if self.use_apid:
            if self.use_mrti:
                raise ValueError("use_mrti and use_apid are mutually exclusive")
            self.apid = UrbanAPID(configs)

        self.use_phase_cycle = bool(getattr(configs, "use_phase_cycle", False))
        if self.use_phase_cycle:
            if self.channel_independence != 1 or self.seq_len != 168:
                raise ValueError("phase-cycle operator requires CI and seq_len=168")
            self.phase_cycle = PhaseCycleTokenOperator(
                int(configs.d_model), cycles=7, phases=24,
                routers=int(getattr(configs, "phase_cycle_routers", 2)),
                heads=int(getattr(configs, "phase_cycle_heads", 4)),
                dropout=float(getattr(configs, "phase_cycle_dropout", configs.dropout)),
                use_pos_embed=bool(getattr(configs, "phase_cycle_pos_embed", False)),
            )

        self.use_phase_latent_decomp = bool(
            getattr(configs, "use_phase_latent_decomp", False)
        )
        if self.use_phase_latent_decomp:
            if self.channel_independence != 1 or self.seq_len != 168:
                raise ValueError(
                    "phase-latent decomposition requires CI and seq_len=168"
                )
            self.phase_latent_decomp = PhaseLatentDecomposition(
                int(configs.d_model),
                cycles=int(getattr(configs, "phase_latent_cycles", 7)),
                phases=int(getattr(configs, "phase_latent_phases", 24)),
            )

        self.use_joint_phase_region = bool(getattr(configs, "use_joint_phase_region", False))
        if self.use_joint_phase_region:
            if self.channel_independence != 1 or self.seq_len != 168:
                raise ValueError("joint phase-region operator requires CI and seq_len=168")
            self.joint_phase_region = JointPhaseRegionOperator(
                int(configs.d_model), int(configs.enc_in), cycles=7, phases=24,
                phase_routers=int(getattr(configs, "joint_phase_routers", 2)),
                region_routers=int(getattr(configs, "joint_region_routers", 4)),
                phase_heads=int(getattr(configs, "joint_phase_heads", 4)),
                region_heads=int(getattr(configs, "joint_region_heads", 4)),
                dropout=float(getattr(configs, "joint_phase_region_dropout", configs.dropout)),
            )

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature
        self.use_tariff_operator = bool(getattr(configs, "use_tariff_operator", False))
        if self.use_tariff_operator:
            self.tariff_feature_dim = int(getattr(configs, "tariff_feature_dim", 2))
            self.tariff_encoder = nn.Sequential(
                nn.Linear(self.tariff_feature_dim, configs.d_model),
                nn.GELU(), nn.LayerNorm(configs.d_model),
            )
            self.tariff_rank = int(getattr(configs, "tariff_rank", 8))
        self.use_tariff_coordinate = getattr(configs, "prediction_method", "linear") == "tariff_coordinate"
        if self.use_tariff_coordinate:
            if self.channel_independence != 1:
                raise ValueError("tariff coordinate route requires channel independence")
            self.tariff_elasticity = nn.Parameter(
                torch.zeros(configs.enc_in, int(getattr(configs, "tariff_feature_dim", 2)))
            )
        self.use_tariff_context = getattr(configs, "prediction_method", "linear") == "tariff_context"
        self.use_tariff_bilinear_context = getattr(configs, "prediction_method", "linear") == "tariff_bilinear"
        self.use_tariff_scale = getattr(configs, "prediction_method", "linear") == "tariff_scale"
        if self.use_tariff_scale:
            self.tariff_scale_transport = TariffScaleTransport(
                configs.down_sampling_layers + 1, configs.d_model,
                int(getattr(configs, "tariff_scale_rank", 4)),
            )

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(
                self.paired_input_dim if self.paired_input_dim else 1 + (self.tariff_feature_dim if self.use_tariff_context else
                     2 * self.tariff_feature_dim if self.use_tariff_bilinear_context else 0),
                configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers


        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True,
                         subtract_last=bool(getattr(configs, "subtract_last", 0)),
                         non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if getattr(configs, "prediction_method", "linear") == "horizon_basis":
                self.predict_layers = torch.nn.ModuleList(
                    [
                        HorizonBasisProjection(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                            getattr(configs, "basis_dim", 16),
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
            elif getattr(configs, "prediction_method", "linear") == "tariff_tensor":
                self.predict_layers = torch.nn.ModuleList(
                    [
                        TariffTensorProjection(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len, configs.d_model,
                            getattr(configs, "tariff_rank", 8),
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
            elif getattr(configs, "prediction_method", "linear") in {"tariff_fusion", "tariff_context", "tariff_bilinear"}:
                self.predict_layers = torch.nn.ModuleList(
                    [
                        TariffFusionProjection(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len, configs.d_model,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
            elif getattr(configs, "prediction_method", "linear") == "tariff_cross":
                self.predict_layers = torch.nn.ModuleList(
                    [
                        TariffCrossAttentionProjection(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len, configs.d_model,
                            heads=int(getattr(configs, "tariff_heads", 4)),
                            dropout=float(getattr(configs, "dropout", 0.1)),
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
            elif getattr(configs, "prediction_method", "linear") == "tariff_state":
                self.predict_layers = torch.nn.ModuleList(
                    [
                        TariffStateTransitionProjection(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len, configs.d_model,
                            getattr(configs, "tariff_state_rank", 8),
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
            elif getattr(configs, "prediction_method", "linear") == "tariff_flow":
                self.predict_layers = torch.nn.ModuleList(
                    [
                        TariffControlledFlowProjection(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len, configs.d_model,
                            getattr(configs, "tariff_state_rank", 8),
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
            else:
                self.predict_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model,
                    self.paired_input_dim if self.paired_input_dim else 1,
                    bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

        # Attach the research operator only after all upstream parameters have
        # been initialized. A paired seed then gives the same backbone weights
        # for CI and innovation models; only the new operator differs.
        if bool(getattr(configs, "use_innovation_exchange", False)):
            if self.channel_independence != 1:
                raise ValueError("innovation exchange requires channel_independence=1")
            from models.innovation_exchange import PhaseLagInnovationExchange
            rng_after_backbone = torch.get_rng_state()
            try:
                for block in self.pdm_blocks:
                    block.innovation_exchange = PhaseLagInnovationExchange(
                        configs.urban_adj, configs.d_model,
                        scales=configs.down_sampling_layers + 1,
                        topk=int(getattr(configs, "innovation_topk", 4)),
                        rank=int(getattr(configs, "innovation_rank", 8)),
                        lags=tuple(getattr(configs, "innovation_lags", (0, 1, 2))),
                        down_sampling_window=configs.down_sampling_window,
                    )
            finally:
                # Keep subsequent data-loader/dropout random streams paired
                # with an otherwise identical TimeMixer-CI construction.
                torch.set_rng_state(rng_after_backbone)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def _apply_city_adapters(self, enc_out_list, batch_size: int):
        """Apply source/target city residual adapters to CI latent streams."""
        if not self.use_city_adapter:
            return enc_out_list
        ids = getattr(self, "_city_adapter_ids", None)
        if ids is None:
            ids = torch.full((batch_size,), self.city_adapter_index,
                             device=enc_out_list[0].device, dtype=torch.long)
        else:
            ids = torch.as_tensor(ids, device=enc_out_list[0].device, dtype=torch.long)
            if ids.numel() != batch_size:
                raise ValueError("city adapter ids must have one index per batch sample")
        output = []
        for latent in enc_out_list:
            if latent.size(0) != batch_size * self.enc_in:
                raise ValueError("city adapter expects CI latent layout [B*N,time,d_model]")
            _, length, d_model = latent.shape
            view = latent.reshape(batch_size, self.enc_in, length, d_model)
            refined = []
            for sample in range(batch_size):
                index = int(ids[sample].item()) % len(self.city_adapters)
                refined.append(self.city_adapters[index](view[sample]))
            output.append(torch.stack(refined, dim=0).reshape_as(latent))
        return output

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.patch_size > 1:
            if x_enc.size(1) != self.raw_seq_len:
                raise ValueError(
                    f"patch TimeMixer expected raw length {self.raw_seq_len}, got {x_enc.size(1)}"
                )
            if self.raw_seq_len % self.patch_size != 0:
                raise ValueError("raw sequence length must be divisible by patch_size")
            x_enc = self.patch_embedding(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc[:, ::self.patch_size, :]
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        tariff_context_scales = None
        if self.use_tariff_coordinate:
            if x_mark_dec is None or x_mark_dec.ndim != 3:
                raise ValueError("tariff coordinate route requires packed history/future price features")
            B, T, N = x_enc.size()
            expected_length = self.raw_seq_len + self.pred_len
            if x_mark_dec.shape[:2] != (B, expected_length):
                raise ValueError("packed tariff coordinate history/horizon does not match input")
            expected = N * self.tariff_feature_dim
            if x_mark_dec.shape[2] != expected:
                raise ValueError("packed tariff coordinate channels do not match zone count")
            packed = x_mark_dec.reshape(B, expected_length, N, self.tariff_feature_dim)
            history_price = packed[:, :self.raw_seq_len]
            future_price = packed[:, self.raw_seq_len:]
            beta = 0.25 * torch.tanh(self.tariff_elasticity).view(1, 1, N, self.tariff_feature_dim)
            history_effect = (history_price * beta).sum(dim=-1)
            self.future_tariff_effect = (future_price * beta).sum(dim=-1)
            x_enc = x_enc - history_effect.to(dtype=x_enc.dtype)
        elif self.use_tariff_context or self.use_tariff_bilinear_context:
            if x_mark_dec is None or x_mark_dec.ndim != 3:
                raise ValueError("tariff context route requires packed history/future price features")
            B, T, N = x_enc.size()
            expected_length = self.raw_seq_len + self.pred_len
            if x_mark_dec.shape[:2] != (B, expected_length):
                raise ValueError("packed tariff context history/horizon does not match input")
            expected = N * self.tariff_feature_dim
            if x_mark_dec.shape[2] != expected:
                raise ValueError("packed tariff context channels do not match zone count")
            packed = x_mark_dec.reshape(B, expected_length, N, self.tariff_feature_dim)
            history_price = packed[:, :self.raw_seq_len]
            future_price = packed[:, self.raw_seq_len:]
            self.future_tariff_tokens = self.tariff_encoder(
                future_price.permute(0, 2, 1, 3).reshape(B * N, self.pred_len, self.tariff_feature_dim)
            )
            price_work = history_price.permute(0, 2, 3, 1).reshape(
                B, N * self.tariff_feature_dim, self.raw_seq_len
            )
            tariff_context_scales = [history_price]
            for _ in range(self.configs.down_sampling_layers):
                price_work = F.avg_pool1d(price_work, self.configs.down_sampling_window)
                tariff_context_scales.append(
                    price_work.reshape(B, N, self.tariff_feature_dim, price_work.size(-1))
                    .permute(0, 3, 1, 2)
                )
        elif self.use_tariff_operator:
            if x_mark_dec is None or x_mark_dec.ndim != 3:
                raise ValueError("tariff operator requires packed future price features")
            B, T, N = x_enc.size()
            if x_mark_dec.shape[0] != B or x_mark_dec.shape[1] != self.pred_len:
                raise ValueError("future tariff batch/horizon does not match input")
            expected = N * self.tariff_feature_dim
            if x_mark_dec.shape[2] != expected:
                raise ValueError("packed future tariff channels do not match zone count")
            packed = x_mark_dec.reshape(B, self.pred_len, N, self.tariff_feature_dim)
            self.future_tariff_tokens = self.tariff_encoder(
                packed.permute(0, 2, 1, 3).reshape(B * N, self.pred_len, self.tariff_feature_dim)
            )
        elif self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        regime_history = x_enc
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    if not self.paired_input_dim:
                        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    if tariff_context_scales is not None:
                        context = tariff_context_scales[i].permute(0, 2, 1, 3).reshape(
                            B * N, T, self.tariff_feature_dim
                        ).to(dtype=x.dtype)
                        x = torch.cat((x, context, x * context), dim=-1) if self.use_tariff_bilinear_context else torch.cat((x, context), dim=-1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    if not self.paired_input_dim:
                        x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    if tariff_context_scales is not None:
                        context = tariff_context_scales[i].permute(0, 2, 1, 3).reshape(
                            B * N, T, self.tariff_feature_dim
                        ).to(dtype=x.dtype)
                        x = torch.cat((x, context, x * context), dim=-1) if self.use_tariff_bilinear_context else torch.cat((x, context), dim=-1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Optional history-only latent coordinates supplied by an adapter.
        # They are added before PDM/FMM, after the official value embedding,
        # so multimodal context cannot rescale the observed volume input or
        # act as an output correction.  The adapter owns the list construction
        # and clears it after one forecast call to avoid state leakage.
        external_history = getattr(self, "_external_history_latents", None)
        history_state_context = (
            external_history if self.use_history_state_pdm else None
        )
        if external_history is not None:
            if len(external_history) != len(enc_out_list):
                raise ValueError("external history latent scale count mismatch")
            for index, (latent, external) in enumerate(zip(enc_out_list, external_history)):
                if external.shape != latent.shape:
                    raise ValueError(
                        f"external history latent shape mismatch at scale {index}: "
                        f"{tuple(external.shape)} vs {tuple(latent.shape)}"
                    )
                enc_out_list[index] = latent + external.to(dtype=latent.dtype)
            self._external_history_latents = None

        # Optional representation tap used by the Cedar-style domain
        # generalization adapter.  It is deliberately taken after the
        # official value embedding and before PDM, before any forecast head,
        # so a discrepancy penalty regularizes the shared representation
        # rather than adding an output correction or a gate.
        if bool(getattr(self.configs, "capture_shared_latent", False)):
            self.last_shared_latent = enc_out_list[0]

        if self.use_regime_state:
            # Shared causal coordinates are broadcast along each observed
            # scale before PDM.  The same projection is used at every scale;
            # this preserves the multi-resolution state identity without a
            # learned selector or post-hoc forecast correction.
            regime = self.regime_state(regime_history)
            regime = regime.reshape(B * self.enc_in, -1).unsqueeze(1)
            enc_out_list = [latent + regime for latent in enc_out_list]

        if self.use_causal_innovation:
            # History-only first and daily differences are encoded at each
            # resolution before PDM.  This changes the representation rather
            # than adding an output correction or a learned selector.
            innovation_latents = []
            for scale, (latent, raw_scale) in enumerate(zip(enc_out_list, x_list[0])):
                values = raw_scale[..., 0]
                lag = min(self.innovation_lag, max(1, values.size(1) - 1))
                first = torch.cat((values[:, :1], values[:, 1:] - values[:, :-1]), dim=1)
                daily = torch.zeros_like(values)
                daily[:, :lag] = first[:, :lag]
                daily[:, lag:] = values[:, lag:] - values[:, :-lag]
                features = torch.stack((first, daily), dim=-1)
                innovation_latents.append(latent + self.innovation_projection(features))
            enc_out_list = innovation_latents

        if self.use_mrti:
            enc_out_list = self.mrti(enc_out_list)
        if self.use_phase_mrti:
            enc_out_list = self.phase_mrti(enc_out_list)
        if self.use_event_mrti:
            enc_out_list = self.event_mrti(enc_out_list)
        if self.use_apid:
            enc_out_list = self.apid(enc_out_list)
        if self.use_phase_cycle:
            enc_out_list[0] = self.phase_cycle(enc_out_list[0])
        if self.use_phase_latent_decomp:
            enc_out_list[0] = self.phase_latent_decomp(enc_out_list[0])
        if self.use_joint_phase_region:
            enc_out_list[0] = self.joint_phase_region(enc_out_list[0])

        if self.use_pori:
            # Dual CI/CM salient-channel view is inserted before PDM.  The
            # final router projection is zero initialized, preserving the
            # official representation at step zero.
            enc_out_list = self.pori_channel_router(enc_out_list, B)

        if self.use_city_adapter:
            enc_out_list = self._apply_city_adapters(enc_out_list, B)

        if self.use_timecap:
            enc_out_list = self.timecap_router(enc_out_list, B)

        if self.use_dynamic_cross_node:
            # History-only, station-identity-preserving node mixing.  It is
            # deliberately placed immediately before PDM so the official
            # multi-resolution decoder remains unchanged.
            enc_out_list = self.dynamic_cross_node(enc_out_list, B)

        if self.use_trend_dynamic_cross_node:
            enc_out_list = self.trend_dynamic_cross_node(enc_out_list, B)

        if self.use_patch_token_cross_node:
            enc_out_list = self.patch_token_cross_node(enc_out_list, B)

        if self.use_dynamic_phase_cross_node:
            # Combined route stays before PDM: node factors and learned phase
            # tokens are both history-only residual representation operators.
            enc_out_list = self.dynamic_phase_cross_node(enc_out_list, B)

        if self.use_endpoint_bridge:
            # Inject a shared low-rank endpoint state into the latent streams
            # before PDM.  The bridge is causal and identity initialized; it
            # is not a forecast residual or sample-dependent selector.
            enc_out_list = self.endpoint_bridge(enc_out_list)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            # The causal history-state decomposition is intentionally applied
            # only to the first PDM block.  Later blocks consume the ordinary
            # mixed representation, so the external coordinate cannot be
            # subtracted a second time after nonlinear PDM mixing.
            pdm_history = history_state_context if i == 0 else None
            if pdm_history is None:
                # Compatibility with structural PDM wrappers used by the
                # research adapters (their forward accepts one argument).
                enc_out_list = self.pdm_blocks[i](enc_out_list)
            else:
                enc_out_list = self.pdm_blocks[i](enc_out_list, pdm_history)
        self._external_history_latents = None

        if self.use_post_regime_state:
            regime = self.post_regime_state(regime_history)
            regime = regime.reshape(B * self.enc_in, -1).unsqueeze(1)
            enc_out_list = [latent + regime for latent in enc_out_list]

        if self.use_post_pdm_dynamic_cross_node:
            # Position ablation: the same history-only node mixer is applied
            # after PDM, so PDM sees the official representation unchanged.
            enc_out_list = self.post_pdm_dynamic_cross_node(enc_out_list, B)

        if self.use_hcr:
            # HCR is a representation operator between PDM and the unchanged
            # FMM. It only sees observed-history latent states.
            enc_out_list = self.horizon_representation(enc_out_list)
        if self.use_scm:
            enc_out_list = self.scale_consistent_mixer(enc_out_list)

        cpr_trend_list = cpr_residual_list = None
        if self.use_cpr:
            # CPR operates after PDM, before FMM.  Keep the two components
            # separately so trend and residual use distinct decoders.
            enc_out_list, cpr_trend_list, cpr_residual_list = self.cpr(
                enc_out_list, B
            )

        pori_coarse_list = pori_detail_list = None
        pori_minimal_tokens = None
        if self.use_pori_minimal:
            enc_out_list, pori_minimal_tokens = self.pori_minimal(enc_out_list, B)
        if self.use_pori:
            enc_out_list, pori_coarse_list, pori_detail_list = self.pori(
                enc_out_list, B
            )

        # Expose the encoded finest-scale history for research objectives that
        # operate on the representation itself (for example masked
        # multi-resolution denoising).  The default forecast path never reads
        # this attribute, so retaining it is behavior-preserving for all
        # existing baselines.
        self._last_encoder_states = enc_out_list

        # Future Multipredictor Mixing as decoder for future
        if self.use_timecap:
            one_shot = self.future_multi_mixing(B, enc_out_list, x_list)
            gate = torch.sigmoid(self.timecap_horizon_gate).view(1, self.pred_len, 1)
            dec_out_list = []
            for index, (base_out, latent) in enumerate(zip(one_shot, enc_out_list)):
                ar_flat = self.timecap_ar_heads[index](latent)
                if ar_flat.size(1) < self.pred_len:
                    ar_flat = F.pad(ar_flat, (0, self.pred_len - ar_flat.size(1)))
                elif ar_flat.size(1) > self.pred_len:
                    ar_flat = ar_flat[:, :self.pred_len]
                ar_out = ar_flat.reshape(B, self.enc_in, self.pred_len).permute(0, 2, 1)
                dec_out_list.append(base_out + gate * (ar_out - base_out))
        elif self.use_pori_minimal:
            # The official FMM remains the reference path.  Each aligned
            # token scale contributes one ungated segment-wise FMM branch;
            # zero initialization makes this exactly the official prediction
            # before the structured branch learns from supervision.
            base_dec = self.future_multi_mixing(B, enc_out_list, x_list)
            segment_dec = self.pori_minimal_segment_fmm(
                pori_minimal_tokens, B, self.enc_in
            )
            dec_out_list = [base + segment for base, segment in zip(base_dec, segment_dec)]
        elif self.use_pori:
            # Keep the official FMM as the identity path and let the
            # four-by-six-hour decoder learn a coarse trend residual.  This
            # avoids the scale attenuation caused by replacing FMM outright.
            base_dec = self.future_multi_mixing(B, enc_out_list, x_list)
            dec_out_list = []
            for index, (coarse_latent, base_out) in enumerate(
                    zip(pori_coarse_list, base_dec)):
                coarse_flat = self.pori_decoders[index](coarse_latent)
                coarse_out = coarse_flat.reshape(B, self.enc_in, self.pred_len)
                coarse_out = coarse_out.permute(0, 2, 1).contiguous()
                dec_out_list.append(base_out + torch.clamp(
                    self.pori_decoder_strength, min=0.0
                ) * coarse_out)
        elif self.use_cpr:
            trend_dec = self.future_multi_mixing(B, cpr_trend_list, x_list)
            residual_dec = self.future_multi_mixing(B, cpr_residual_list, x_list)
            # Preserve the official FMM at initialization: because the
            # projection is linear, F(trend)+F(residual)=F(input).  The
            # horizon parameter only learns a bounded contrast between the
            # two decoded components instead of attenuating the baseline by
            # an arbitrary convex weight.
            contrast = torch.tanh(self.cpr_horizon_gate).view(1, self.pred_len, 1)
            dec_out_list = [trend + residual + contrast * (trend - residual)
                            for trend, residual in zip(trend_dec, residual_dec)]
        else:
            dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        # Research hook for causal multi-resolution objectives.  Keep the
        # individual PDM/FMM scale forecasts with gradients until the caller
        # finishes its loss computation; the ordinary inference path below
        # still sums the same list exactly as before.  This is deliberately a
        # fixed scale supervision hook, not a learned scale gate or selector.
        self.last_scale_forecasts = dec_out_list

        if self.use_reconciled_fmm:
            # Convert each native hourly scale forecast to its fixed block
            # means, then solve the least-squares reconciliation problem in
            # the hourly trajectory space.  The projection is global and
            # deterministic; it cannot select a scale per sample/node.
            rhs = dec_out_list[0].new_zeros(
                dec_out_list[0].shape[0], self.pred_len, dec_out_list[0].shape[2]
            )
            for index, values in enumerate(dec_out_list):
                matrix = getattr(self, f"reconciled_aggregation_{index}").to(
                    device=values.device, dtype=values.dtype
                )
                bins = int(matrix.shape[0])
                factor = int(self.down_sampling_window) ** index
                if factor > 1:
                    values = values[:, :bins * factor].reshape(
                        values.shape[0], bins, factor, values.shape[2]
                    ).mean(dim=2)
                else:
                    values = values[:, :bins]
                rhs = rhs + torch.einsum("bhn,hk->bkn", values, matrix)
            inverse = self.reconciled_normal_inverse.to(
                device=rhs.device, dtype=rhs.dtype
            )
            dec_out = torch.einsum("bhn,hk->bkn", rhs, inverse)
        elif self.use_precision_fusion:
            # Fixed BLUE/simplex combination of the scale forecasts.  The
            # weights and train-prefix bias are global across samples and
            # stations; this is an ungated linear projection inside FMM.
            stack = torch.stack(dec_out_list, dim=-1)  # [B,H,N,S]
            weights = self.precision_fusion_weights.to(
                device=stack.device, dtype=stack.dtype).view(1, self.pred_len, 1, -1)
            bias = self.precision_fusion_bias.to(
                device=stack.device, dtype=stack.dtype).view(1, self.pred_len, 1, -1)
            dec_out = ((stack - bias) * weights).sum(-1)
        elif self.use_mdmixer_lite and self.channel_independence == 1:
            # Dynamic, node-specific fusion of FMM scale forecasts.  The
            # weight is inferred from past latent states only; reshape is
            # explicit to keep [B,N] semantics under CI flattening.
            scale_weights = self.mdmixer_scale_gate(enc_out_list)  # [B*N,S]
            if scale_weights.size(0) != B * self.enc_in:
                raise RuntimeError("MDMixer gate batch does not match CI layout")
            stack = torch.stack(dec_out_list, dim=-1)  # [B,H,N,S]
            weights = scale_weights.reshape(B, self.enc_in, -1)
            weights = weights.unsqueeze(1).to(dtype=stack.dtype)
            dec_out = (stack * weights).sum(-1)
        else:
            dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        if self.use_harmonic_phase_decoder:
            # Add a bounded history-only harmonic correction after the
            # official FMM has produced all horizons.
            harmonic_residual = self.harmonic_phase_decoder(
                enc_out_list[0], B, self.enc_in
            )
            if self.use_confidence_residual_gate:
                reliability = self.confidence_residual_gate(
                    enc_out_list[0], B, self.enc_in
                ).permute(0, 2, 1).to(dtype=harmonic_residual.dtype)
                harmonic_residual = harmonic_residual * reliability
            dec_out = dec_out + harmonic_residual
        if self.use_cross_phase_attention_decoder:
            phase_residual = self.cross_phase_attention_decoder(
                enc_out_list[0], B, self.enc_in
            )
            if self.use_history_cross_phase_gate:
                reliability = self.history_cross_phase_gate(
                    enc_out_list[0], B, self.enc_in
                ).permute(0, 2, 1).to(dtype=phase_residual.dtype)
                phase_residual = phase_residual * reliability
            dec_out = dec_out + phase_residual
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        if self.use_tariff_coordinate:
            dec_out = dec_out + self.future_tariff_effect.to(dtype=dec_out.dtype)
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            if self.use_tariff_scale:
                latent_list = [
                    self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                    for i, enc_out in zip(range(len(x_list)), enc_out_list)
                ]
                latent_list = self.tariff_scale_transport(
                    latent_list, self.future_tariff_tokens
                )
                for dec_latent in latent_list:
                    dec_out = self.projection_layer(dec_latent)
                    dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(
                        0, 2, 1
                    ).contiguous()
                    dec_out_list.append(dec_out)
                return dec_out_list
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                if self.use_tariff_operator and not self.use_tariff_coordinate:
                    dec_out = self.predict_layers[i](
                        enc_out.permute(0, 2, 1), self.future_tariff_tokens
                    )
                else:
                    dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                        0, 2, 1)  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')
