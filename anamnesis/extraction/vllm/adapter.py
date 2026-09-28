"""The attention families of the fast lane's schema, from the engine's observables.

The adapter is the arithmetic bridge between what a capture holds (the first
pass's per-row, per-head online statistics and the second pass's reduced
products) and the attention coordinates of the fast lane's schema. It mirrors
:class:`anamnesis.extraction.fast.attention.AttentionReducer` construct for
construct: CPU-side arithmetic over observables that already exist, no kernels.

Every construct consumes the rounded rows the reducer itself would reduce: the
agreement entropy pair, the entropy series, the fp32 head-mean span, spectral
and decay rows, the per-row coverage and the per-head summaries, all taken from
the second pass's per-head probabilities rounded to the model dtype, which is the
rounding the fast lane's materialized attention carries. The arithmetic here is
the reducer's own line for line, including the reducer's always-masked final
column, so identical rows give bit-identical coordinates. The first pass's
statistics supply the head count.

The per-head kv-key-spread summaries are not emitted here: their substrate is the
captured keys, which :func:`anamnesis.extraction.vllm.readout.reduce_capture`
reduces directly.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from anamnesis.extraction.fast.ops import (
    FeatureCollector,
    corr,
    decay,
    slope,
    std,
)
from anamnesis.extraction.feature_families.attention_flow import (
    _attention_flow_names,
)
from anamnesis.extraction.feature_families.per_head import _per_head_names
from anamnesis.extraction.vllm.rows import RequestRowSchema
from anamnesis.extraction.vllm.stats_kernel import NUM_STATS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class AttentionFeatureAdapter:
    """One replayed request's attention coordinates from its capture.

    ``consume`` takes one layer's observables exactly as ``LaneCapture``
    assembles them; ``smoothness`` takes the position-corrected residuals
    the readout already computes per sampled layer, after that layer's
    ``consume``. ``assert_complete`` refuses a partially adapted request.
    """

    def __init__(self, collector: FeatureCollector, *,
                 schema: RequestRowSchema, num_layers: int,
                 sampled_layers, n_windows: int = 4,
                 include_stft: bool = True):
        _require(isinstance(schema, RequestRowSchema),
                 'a RequestRowSchema is required')
        _require(type(num_layers) is int and num_layers >= 1,
                 'num_layers must be a positive integer')
        sampled = tuple(sampled_layers)
        _require(len(set(sampled)) == len(sampled), 'duplicate sampled layer')
        _require(all(type(x) is int and 0 <= x < num_layers for x in sampled),
                 'sampled layers must be layer indices')
        _require(type(n_windows) is int and n_windows >= 1,
                 'n_windows must be a positive integer')
        self.out = collector
        self.schema = schema
        self.t = schema.steps
        self.c = schema.prompt_length
        self.num_layers = num_layers
        self.sampled = frozenset(sampled)
        self.n_windows = n_windows
        self.include_stft = include_stft
        self.seen: set[int] = set()
        self.spectral: dict[int, tuple[tuple[int, ...], Tensor]] = {}
        # Row lengths in the reducer's own form: float64 c+1 .. c+t.
        self.lengths = torch.arange(
            self.c + 1, self.c + self.t + 1, dtype=torch.float64)

    def _validate_families(self, sampled, entropy_rows, span_rows, head_ent,
                           head_sink, head_prompt, head_recency,
                           heads: int) -> None:
        width = self.schema.end - 1
        _require(isinstance(entropy_rows, Tensor)
                 and entropy_rows.dtype == torch.float32
                 and entropy_rows.shape == (len(self.schema.entropy), heads),
                 'entropy rows must be fp32 with one entropy row per '
                 'selection')
        pieces = [entropy_rows]
        if sampled:
            _require(isinstance(span_rows, Tensor)
                     and span_rows.dtype == torch.float32
                     and span_rows.shape == (self.t, width),
                     'span rows must be fp32 head-mean rows at the capture '
                     'width')
            for name, series in (('head_ent', head_ent),
                                 ('head_sink', head_sink),
                                 ('head_prompt', head_prompt),
                                 ('head_recency', head_recency)):
                _require(isinstance(series, Tensor)
                         and series.dtype == torch.float64
                         and series.shape == (self.t, heads),
                         f'{name} must be float64 with one span row each')
            pieces += [span_rows, head_ent, head_sink, head_prompt,
                       head_recency]
        else:
            _require(all(x is None for x in (
                span_rows, head_ent, head_sink, head_prompt, head_recency)),
                'per-head span summaries belong to sampled layers')
        for piece in pieces:
            _require(bool(torch.isfinite(piece).all()),
                     'attention observables must be finite')

    def _validate(self, layer, stats, h_mean, h_heads, coverage,
                  spectral_rows, decay_rows) -> bool:
        _require(type(layer) is int and 0 <= layer < self.num_layers,
                 'layer outside the declared model')
        if layer in self.seen:
            raise ValueError(f'layer {layer} adapted twice in one request')
        width = self.schema.end - 1
        _require(isinstance(stats, Tensor) and stats.dtype == torch.float32
                 and stats.shape == (self.t, stats.shape[1], NUM_STATS)
                 and stats.ndim == 3 and stats.shape[1] >= 1,
                 'stats must be fp32 [steps, heads, NUM_STATS]')
        n_agreement = len(self.schema.agreement)
        for name, series in (('h_mean', h_mean), ('h_heads', h_heads)):
            _require(isinstance(series, Tensor)
                     and series.dtype == torch.float64
                     and series.shape == (n_agreement,),
                     f'{name} must be float64 with one agreement row each')
        sampled = layer in self.sampled
        if not sampled:
            _require(coverage is None and spectral_rows is None
                     and decay_rows is None,
                     'second-pass span products belong to sampled layers')
        else:
            _require(isinstance(coverage, Tensor)
                     and coverage.dtype == torch.float64
                     and coverage.shape == (self.t,),
                     'coverage must be float64 with one span row each')
            _require(isinstance(spectral_rows, Tensor)
                     and spectral_rows.dtype == torch.float32
                     and spectral_rows.shape == (len(self.schema.spectral),
                                                 width),
                     'spectral rows must be fp32 at the capture width')
            _require(isinstance(decay_rows, Tensor)
                     and decay_rows.dtype == torch.float32
                     and decay_rows.shape == (len(self.schema.decay), width),
                     'decay rows must be fp32 at the capture width')
        pieces = [stats, h_mean, h_heads]
        if sampled:
            pieces += [coverage, spectral_rows, decay_rows]
        for piece in pieces:
            _require(bool(torch.isfinite(piece).all()),
                     'attention observables must be finite')
        return sampled

    def consume(self, layer: int, *, stats: Tensor, h_mean: Tensor,
                h_heads: Tensor, entropy_rows: Tensor,
                coverage: Tensor | None = None,
                spectral_rows: Tensor | None = None,
                decay_rows: Tensor | None = None,
                span_rows: Tensor | None = None,
                head_ent: Tensor | None = None,
                head_sink: Tensor | None = None,
                head_prompt: Tensor | None = None,
                head_recency: Tensor | None = None) -> None:
        """Emit one layer's attention coordinates.

        Every layer takes its statistics, agreement pair and entropy rows; a
        sampled layer also takes its coverage, spectral, decay and span rows
        and its per-head summaries, and no other layer may.
        """
        sampled = self._validate(layer, stats, h_mean, h_heads, coverage,
                                 spectral_rows, decay_rows)
        self._validate_families(sampled, entropy_rows, span_rows, head_ent,
                                head_sink, head_prompt, head_recency,
                                stats.shape[1])
        self.seen.add(layer)
        # The reducer reduces a strided view of its full [t, heads] series;
        # scatter the selected rows into a full-size buffer and take its exact
        # slice so the reduction blocking matches.
        ent = entropy_rows.new_zeros((self.t, entropy_rows.shape[1]))
        ent[list(self.schema.entropy)] = entropy_rows
        sample = ent[::max(1, self.t // 60)] if self.t > 60 else ent
        self.out.put(f'attn_entropy_mean_L{layer}', sample.mean())
        self.out.put(f'attn_entropy_std_L{layer}', std(sample))
        agreement = (
            1 - (h_mean - h_heads).clamp_min(0)
            / max(math.log(stats.shape[1]), 1e-12)
        ).float()
        self.out.put(f'head_agreement_mean_L{layer}', agreement.mean())
        self.out.put(f'head_agreement_std_L{layer}', std(agreement))
        if not sampled:
            return
        self._consume_sampled_rows(layer, coverage, spectral_rows, decay_rows,
                                   span_rows, head_ent, head_sink, head_prompt,
                                   head_recency)

    def _consume_sampled_rows(self, layer, coverage, spectral_rows,
                              decay_rows, span_rows, head_ent, head_sink,
                              head_prompt, head_recency) -> None:
        """A sampled layer's families from the rounded rows, line for line with
        the reducer.

        ``span_rows`` are the retained fp32 head-mean rows at the capture
        width ``c + t``; the reducer's materialized width is ``c + t + 1``
        with an always-masked final column, so that exact zero column is
        appended before promotion and every sum reduces in the reducer's
        own layout.
        """
        mean = torch.cat(
            [span_rows, span_rows.new_zeros(self.t, 1)], dim=-1).double()
        device = mean.device
        pos = torch.arange(mean.shape[1], device=device)
        ilengths = torch.arange(self.c + 1, self.c + self.t + 1,
                                device=device)
        total = mean.sum(dim=-1).clamp_min(1e-12)
        cutoff = (ilengths.double() * 0.8).long().clamp_min(1)
        rec = (mean * (pos[None, :] >= cutoff[:, None])).sum(dim=-1) / total
        sink = mean[:, 0]
        prompt = mean[:, :self.c].sum(dim=-1)
        generated = mean[:, self.c:].sum(dim=-1)
        self.out.put(f'cache_recency_bias_L{layer}', rec.mean())
        self.out.put(f'cache_sink_mass_L{layer}', sink.mean())
        self.out.put(f'cache_cache_coverage_L{layer}', coverage.mean())
        self.out.put(
            f'cache_lookback_ratio_L{layer}',
            prompt.sum() / generated.sum().clamp_min(1e-12),
        )
        self.out.put(f'cache_attn_decay_rate_L{layer}',
                     self._decay_fit(decay_rows))
        width = self.t // 4
        for i in range(4):
            series = (rec[i * width:(i + 1) * width if i < 3 else self.t]
                      if self.t >= 4 else rec)
            self.out.put(f'cache_recency_traj{i}_L{layer}', series.mean())
        self._spectral(layer, spectral_rows)
        self._flow_rows(layer, mean, pos, ilengths, rec, total, prompt,
                        head_prompt, head_recency)
        self._per_head_rows(layer, head_ent, head_sink)

    def _flow_rows(self, layer, mean, pos, ilengths, rec, total, prompt,
                   head_prompt, head_recency) -> None:
        prefix = f'attn_flow_L{layer}'
        if self.t < 2:
            for name in _attention_flow_names(layer, self.n_windows,
                                              self.include_stft):
                self.out.put(name, 0.0)
            return
        sys = prompt / total
        self.out.moments(prefix + '_prompt_mass', sys)
        self.out.put(prefix + '_prompt_decay_rate', decay(sys))
        c = self.c
        third = ((ilengths - c) // 3).clamp_min(1)
        regions = (
            sys,
            (mean * ((pos[None, :] >= c)
                     & (pos[None, :] < c + third[:, None]))).sum(dim=1)
            / total,
            (mean * ((pos[None, :] >= c + third[:, None])
                     & (pos[None, :] < c + 2 * third[:, None]))).sum(dim=1)
            / total,
            (mean * (pos[None, :] >= c + 2 * third[:, None])).sum(dim=1)
            / total,
        )
        for label, series in zip(
            ('prompt', 'early_gen', 'mid_gen', 'recent'), regions,
            strict=True,
        ):
            self.out.moments(prefix + '_region_' + label, series)
        self.out.put(prefix + '_head_diversity_prompt',
                     std(head_prompt, dim=1).mean())
        self.out.put(prefix + '_head_diversity_recency',
                     std(head_recency, dim=1).mean())
        self.out.operators(prefix + '_prompt_mass', sys, self.n_windows,
                           self.include_stft)
        self.out.operators(prefix + '_recency_bias', rec, self.n_windows,
                           self.include_stft)

    def _per_head_rows(self, layer, head_ent, head_sink) -> None:
        prefix = f'ph_L{layer}'
        if self.t < 4:
            for name in _per_head_names(layer):
                if '_kv_key_spread_' not in name:
                    self.out.put(name, 0.0)
            return
        means = head_ent.mean(dim=0)
        for name, value in (
            ('mean', means.mean()),
            ('std', std(means)),
            ('min', means.min()),
            ('max', means.max()),
        ):
            self.out.put(prefix + '_head_entropy_' + name, value)
        half = self.t // 2
        self.out.put(
            prefix + '_head_role_stability',
            corr(head_ent[:half].mean(dim=0), head_ent[half:].mean(dim=0),
                 min_n=2),
        )
        self.out.put(prefix + '_sink_head_std', std(head_sink.mean(dim=0)))

    def _decay_fit(self, decay_rows: Tensor) -> Tensor:
        if self.t < 10:
            return torch.zeros((), dtype=torch.float64)
        # The reducer fits over its materialized width c+t+1, whose final
        # column is masked on every row; append that exact column so the
        # flattened masked sums reduce in the reducer's own layout.
        rows = torch.cat(
            [decay_rows, decay_rows.new_zeros(len(decay_rows), 1)], dim=-1
        )[:, 1:].double()
        pos = torch.arange(1, rows.shape[1] + 1, dtype=torch.float64,
                           device=rows.device)
        row_lengths = self.lengths[list(self.schema.decay)].to(rows.device)
        distances = (row_lengths[:, None] - 1 - pos[None, :]).clamp_min(1)
        valid = pos[None, :] < row_lengths[:, None]
        mask = (rows > 1e-10) & valid
        return -slope(rows.clamp_min(1e-300).log().flatten(),
                      distances.flatten(), mask.flatten())

    def _spectral(self, layer: int, spectral_rows: Tensor) -> None:
        steps = self.schema.spectral
        matrix = spectral_rows[:, :self.schema.spectral_width].double()
        matrix = matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        graph = (matrix @ matrix.T).clamp_min(0)
        graph = (graph + graph.T) * 0.5
        graph.fill_diagonal_(0)
        degrees = graph.sum(dim=-1)
        lap = (
            torch.diag(degrees)
            - graph
            + torch.eye(len(steps), dtype=torch.float64,
                        device=matrix.device) * 1e-10
        )
        ev = torch.linalg.eigvalsh(lap).clamp_min(0)
        energy = ev.sum()
        radius = ev[-1]
        fiedler = (
            torch.where(
                radius > 1e-12, ev[1] / radius.clamp_min(1e-30),
                ev.new_zeros(())
            )
            if len(ev) > 1
            else ev.new_zeros(())
        )
        # torch.median selects the lower middle element; NumPy averages the
        # two, and the reducer follows NumPy.
        mid = len(ev) // 2
        median = ev[mid] if len(ev) % 2 else (ev[mid - 1] + ev[mid]) * 0.5
        hfer = ev[ev > median].sum() / energy.clamp_min(1e-30)
        p = ev / energy.clamp_min(1e-30) + 1e-12
        p = p / p.sum()
        spec = (
            -(p * p.log()).sum() / math.log(len(ev))
            if len(ev) > 1
            else ev.new_zeros(())
        )
        self.out.put(f'spectral_fiedler_L{layer}', fiedler)
        self.out.put(
            f'spectral_hfer_L{layer}',
            torch.where(energy > 1e-12, hfer, hfer.new_zeros(())),
        )
        self.out.put(
            f'spectral_spectral_entropy_L{layer}',
            torch.where(energy > 1e-12, spec, spec.new_zeros(())),
        )
        inv = degrees.clamp_min(1e-12).rsqrt()
        normalized = (
            torch.eye(len(steps), dtype=torch.float64, device=matrix.device)
            - inv[:, None] * graph * inv[None, :]
        )
        self.spectral[layer] = (steps, normalized)

    def smoothness(self, layer: int, corrected_hidden: Tensor) -> None:
        """The spectral smoothness of a sampled layer's corrected residual norms
        over the Laplacian its :meth:`consume` built."""
        if layer not in self.spectral:
            raise ValueError(
                f'no retained spectral Laplacian for layer {layer}')
        steps, lap = self.spectral.pop(layer)
        signal = corrected_hidden[list(steps)].double().norm(dim=-1)
        self.out.put(
            f'spectral_smoothness_L{layer}',
            (signal @ lap @ signal) / (signal @ signal).clamp_min(1e-12),
        )

    def assert_complete(self) -> None:
        """Refuse a request whose layers or smoothness were not all adapted."""
        if self.seen != set(range(self.num_layers)):
            missing = sorted(set(range(self.num_layers)) - self.seen)
            raise ValueError(f'layers never adapted: {missing}')
        if self.spectral:
            raise ValueError(
                f'smoothness never consumed: {sorted(self.spectral)}')
