"""One span through the fast lane, in process, against a model the caller holds.

A process that keeps one model resident — a server that generates and then reads
the signature of what it generated — calls :func:`harvest_loaded` once per span
against a :class:`~anamnesis.extraction.fast.runtime.PreparedLane`. Nothing is
parsed from a command line and nothing is written to disk; the result carries the
feature vector, the lane's per-span receipt, and, when asked, the per-position
logit series.

The logit series is read from the output of the same forward the lane reduces,
through a forward hook on the model that lives only for the call. It is not
computed by the lane: the files :class:`~anamnesis.extraction.fast.features.GpuFeatureLane`
hashes into its ``lane_id`` are untouched by it, so asking for the series changes
neither the feature vector nor the lane identity a row carries. Its arrays use
the key names and dtypes :mod:`anamnesis.extraction.raw_saver` banks logits under.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.fast.runtime import LaneSpan, span_is_supported
from anamnesis.extraction.state_extractor import ExtractionResult

if TYPE_CHECKING:
    from torch import Tensor

    from anamnesis.extraction.fast.runtime import PreparedLane

ROWS_PER_CHUNK = 16
"""Positions reduced at once. Each row is a full-vocabulary float64 copy, so the
series never holds more than this many of them."""


@dataclass(frozen=True)
class LogitSeries:
    """The next-token distribution at each generated position, reduced to its head.

    Row ``t`` is the distribution the model produced at absolute position
    ``prompt_length + t`` — the one that chose the token at ``prompt_length + t + 1``.
    A span of ``N`` generated tokens gives ``T = N - 1`` rows: the first generated
    token was chosen by the prompt's last position, which the lane does not replay.

    Attributes
    ----------
    top_k
        Columns kept per row: the requested ``k``, or the vocabulary if smaller.
    values
        ``[T, top_k]`` float32 logits, largest first.
    indices
        ``[T, top_k]`` int32 token ids of ``values``.
    entropy
        ``[T]`` float32 entropy of the full distribution in nats, computed in
        float64 as ``logsumexp(x) - sum(softmax(x) * x)``.
    chosen_ids
        ``[T]`` int32 token each row's distribution was followed by.
    """

    top_k: int
    values: NDArray[np.float32]
    indices: NDArray[np.int32]
    entropy: NDArray[np.float32]
    chosen_ids: NDArray[np.int32]

    def arrays(self) -> dict[str, np.ndarray]:
        """The series under the key names a raw bank stores logits under."""
        return {
            "logits_values": self.values,
            "logits_indices": self.indices,
            "logits_entropy": self.entropy,
            "chosen_ids": self.chosen_ids,
        }


@dataclass(frozen=True)
class HarvestResult:
    """One span's signature, as :func:`harvest_loaded` returns it.

    Attributes
    ----------
    features
        The named feature vector, float32.
    feature_names
        One name per entry of ``features``.
    family_slices
        Family name to ``(start, end)`` bounds in ``features``.
    mean_logprob
        Mean log-probability of the generated tokens after the first.
    knnlm_baseline
        Final hidden state at the last replayed position, or ``None`` when the
        battery does not bank it.
    receipt
        The lane's per-span metadata: ``lane_id``, the token digest, the
        calibration digest, and the transfer accounting.
    prompt_length
        Where the prompt ends in ``input_ids``.
    input_ids
        The whole replayed sequence, int32.
    logit_series
        The per-position series, or ``None`` when it was not asked for.
    """

    features: NDArray[np.float32]
    feature_names: tuple[str, ...]
    family_slices: dict[str, tuple[int, int]]
    mean_logprob: float
    knnlm_baseline: NDArray[np.float32] | None
    receipt: dict[str, Any]
    prompt_length: int
    input_ids: NDArray[np.int32]
    logit_series: LogitSeries | None

    @property
    def lane_id(self) -> str:
        """The lane identity this row carries; rows with different ones do not mix."""
        return str(self.receipt["lane_id"])

    def extraction_result(self) -> ExtractionResult:
        """The features in the shape :func:`anamnesis.extraction.feature_pipeline.save_features` writes."""
        return ExtractionResult(
            self.features,
            list(self.feature_names),
            dict(self.family_slices),
            self.knnlm_baseline,
        )

    def series_arrays(self) -> dict[str, np.ndarray]:
        """The logit series plus the sequence it was read from, ready for ``np.savez``.

        Raises
        ------
        ValueError
            When the harvest did not ask for a logit series.
        """
        if self.logit_series is None:
            raise ValueError("this harvest carries no logit series; pass logit_series_top_k")
        return {
            **self.logit_series.arrays(),
            "input_ids": self.input_ids,
            "prompt_length": np.array(self.prompt_length, dtype=np.int32),
        }


def reduce_logit_series(rows: Tensor, chosen_ids: Sequence[int], top_k: int) -> LogitSeries:
    """Top-k, entropy and chosen ids over ``rows``, a ``[T, vocab]`` logit block.

    Top-k is taken over the float32 logits and entropy over their float64 copy,
    :data:`ROWS_PER_CHUNK` rows at a time, on the device the rows are on.

    Raises
    ------
    ValueError
        When ``top_k`` is not positive or ``chosen_ids`` is not one id per row.
    """
    import torch

    if top_k < 1:
        raise ValueError(f"logit_series_top_k must be positive, got {top_k}")
    if len(chosen_ids) != rows.shape[0]:
        raise ValueError(
            f"{len(chosen_ids)} chosen ids for {rows.shape[0]} logit rows; one per row"
        )
    k = min(int(top_k), int(rows.shape[-1]))
    values, indices, entropy = [], [], []
    with torch.no_grad():
        for first in range(0, rows.shape[0], ROWS_PER_CHUNK):
            chunk = rows[first : first + ROWS_PER_CHUNK]
            top = torch.topk(chunk.float(), k, dim=-1)
            values.append(top.values)
            indices.append(top.indices)
            x = chunk.double()
            entropy.append(
                torch.logsumexp(x, dim=-1) - (torch.softmax(x, dim=-1) * x).sum(dim=-1)
            )
        return LogitSeries(
            top_k=k,
            values=torch.cat(values).cpu().numpy().astype(np.float32),
            indices=torch.cat(indices).cpu().numpy().astype(np.int32),
            entropy=torch.cat(entropy).cpu().numpy().astype(np.float32),
            chosen_ids=np.asarray(chosen_ids, dtype=np.int32),
        )


def harvest_loaded(
    prepared: PreparedLane,
    token_ids: Sequence[int],
    *,
    prompt_len: int,
    logit_series_top_k: int | None = None,
) -> HarvestResult:
    """Replay ``token_ids`` through the fast lane and return its signature.

    One no-cache forward over the whole sequence, reduced on the device, exactly
    the pass :mod:`anamnesis.scripts.run_gpu_replay` banks. The lane for the
    span's schema is built on first use and reused after.

    Parameters
    ----------
    prepared
        From :func:`anamnesis.extraction.fast.runtime.prepare_fast_lane`.
    token_ids
        Prompt and generation, as one sequence of token ids.
    prompt_len
        How many of ``token_ids`` are prompt. The rest are the generation.
    logit_series_top_k
        Also return the per-position logit series, keeping this many logits per
        position. ``None`` returns none and installs no hook.

    Raises
    ------
    ValueError
        When the span has no prompt, fewer than two generated tokens, or reaches
        past the calibrated positions; when ``logit_series_top_k`` is not positive;
        or from the lane's own checks on the model, the tokens and the arithmetic.
    RuntimeError
        From the lane, when a capture it needs is missing or a feature is not
        finite.
    """
    ids = [int(t) for t in token_ids]
    span = LaneSpan(gen_id=-1, input_ids=ids, prompt_length=int(prompt_len))
    if not span_is_supported(span, prepared.calibration.positions_calibrated):
        raise ValueError(
            f"span of {span.end} tokens with a {span.prompt_length}-token prompt is "
            "outside what the lane supports: it needs a prompt, at least two generated "
            f"tokens, and a last position below {prepared.calibration.positions_calibrated}"
        )
    if logit_series_top_k is not None and logit_series_top_k < 1:
        raise ValueError(f"logit_series_top_k must be positive, got {logit_series_top_k}")
    lane, schema = prepared.lane(span.n_steps)
    loaded = prepared.loaded
    captured: list[Tensor] = []
    handle = None
    if logit_series_top_k is not None:

        def keep_logits(_module: Any, _args: Any, output: Any) -> None:
            captured.append(output.logits)

        handle = loaded.model.register_forward_hook(keep_logits)
    try:
        result = lane.replay_span(loaded, ids, span.prompt_length, span.end)
    finally:
        if handle is not None:
            handle.remove()
    series = None
    if logit_series_top_k is not None:
        if len(captured) != 1 or captured[0].shape[:2] != (1, span.end):
            shapes = [tuple(t.shape) for t in captured]
            raise RuntimeError(
                f"expected one full-sequence forward of {span.end} positions, saw {shapes}"
            )
        rows = captured[0][0, span.prompt_length : span.end - 1]
        series = reduce_logit_series(
            rows, ids[span.prompt_length + 1 : span.end], logit_series_top_k
        )
        captured.clear()
    return HarvestResult(
        features=result.features,
        feature_names=result.feature_names,
        family_slices=dict(schema.family_slices),
        mean_logprob=result.mean_logprob,
        knnlm_baseline=result.knnlm_baseline,
        receipt=result.metadata,
        prompt_length=span.prompt_length,
        input_ids=np.asarray(ids, dtype=np.int32),
        logit_series=series,
    )
