"""The lens family: what a block-routed model would predict if its stack ended at a layer.

On a plain transformer a logit lens applies the final norm and the unembedding to a layer's
residual stream. A block attention-residual model has no residual stream to read: its final
read is a routing over the committed blocks and the last block's partial, with the final
routing query, then the final norm and the unembedding. "Stopping at layer ``l``" here is
that same read over the sources that exist once layer ``l`` has run: the blocks committed
before it, and layer ``l``'s post-MLP partial (``depth[:, l]``) as the partial. The routing
is recomputed with the model's own final query and key norm, so at the last layer the lens
is the model's own final read, and its distribution is the step's next-token distribution.

**The head.** :class:`LensHead` holds the tensors the read needs: the final routing query and
its key-norm gain (with that norm's epsilon), the final norm's gain (and epsilon), and the
unembedding ``[vocab, hidden]``. :meth:`LensHead.from_model` takes them from a loaded model
and binds the head to that model's ``weights-v1`` digest
(:func:`anamnesis.extraction.weights_digest.weights_digest`);
:meth:`LensHead.from_checkpoint` reads them from a checkpoint file, and the caller names the
weights digest the head belongs to. A head's own digest (:attr:`LensHead.sha256`) is part of
its identity, so rows read through two heads never share a lane.

**Features**, at each layer of ``layers`` (the lane's sampled layers), over the span's steps:

* ``lens_L{l}_entropy_mean`` / ``_std``: the entropy of the lens distribution, in nats;
* ``lens_L{l}_agree_final_mean``: how often its top-1 token is the final read's top-1;
* ``lens_L{l}_agree_chosen_mean``: how often its top-1 token is the token the step chose;
* ``lens_L{l}_kl_final_mean`` / ``_std``: KL(lens ‖ final), in nats;

and one row summary, ``lens_depth``: the first of ``layers`` from which the agreement with
the final read stays above one half for every later one, as a fraction of the stack's depth
(``l / (layers - 1)``; 1.0 when even the last of ``layers`` agrees half the time or less).

**Batchmates.** The steps of every row in a batch are flattened into positions and read in
chunks of :data:`CHUNK` positions, the last one padded with zeros, so every kernel launch has
the same shape whatever the batch holds. Each position's values depend on its own inputs
only, and the reductions over steps are the batchmate-invariant ones of
:mod:`anamnesis.extraction.fast.invariant_ops`, masked to each row's own steps. The
arithmetic is float32 with TF32 off for the read, whatever the process's matmul setting.

Cost: one ``[CHUNK, hidden] x [hidden, vocab]`` product per chunk per lens layer, plus one
for the final read. A 1.5K-wide, 49K-vocabulary model at 10 layers and 128 rows of 40 steps
is about 6 TFLOP per batch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from anamnesis.extraction.fast.invariant_ops import row_mean, row_std

LENS_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
"""This module's source digest, part of a lens's identity."""

LENS_VARIANT = "routed-v1"
"""The read this module computes: the final routing recomputed over the sources at a layer."""

CHUNK = 256
"""Positions per launch. Every launch has this many rows, so its kernels never depend on how
many positions a batch has."""

AGREE_THRESHOLD = 0.5
"""The agreement with the final read that ``lens_depth`` asks every later layer to exceed."""

QUANTITIES = ("entropy_mean", "entropy_std", "agree_final_mean", "agree_chosen_mean",
              "kl_final_mean", "kl_final_std")
"""The per-layer features, in order."""


class LensError(ValueError):
    """A head or an input the lens cannot read."""


def lens_names(layers: Sequence[int]) -> tuple[str, ...]:
    """The lens family's feature names for ``layers``, in order."""
    names = [f"lens_L{int(layer)}_{q}" for layer in layers for q in QUANTITIES]
    return tuple(names + ["lens_depth"])


def _tensor_digest(tensors: Mapping[str, Tensor], extra: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(tensors):
        t = tensors[key].detach().contiguous().cpu()
        h.update(key.encode() + b"\0" + str(t.dtype).encode() + b"\0"
                 + ",".join(map(str, t.shape)).encode() + b"\0")
        h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
    for key in sorted(extra):
        h.update(f"{key}={extra[key]!r}".encode() + b"\0")
    return h.hexdigest()


HEAD_KEYS = {"final_query": "final_res_query", "final_key_gain": "final_res_norm.weight",
             "norm_gain": "norm.weight", "unembedding": "lm_head.weight"}
"""Where a kotodama-family state dict keeps each head tensor. A tied model has no
``lm_head.weight``; its unembedding is ``embed_tokens.weight``."""

TIED_UNEMBEDDING = "embed_tokens.weight"


@dataclass(frozen=True)
class LensHead:
    """The tensors of a block-routed model's final read (see the module docstring), the
    ``weights-v1`` digest of the model they belong to (``weights``), and where they came from
    (``source``, provenance only)."""

    final_query: Tensor
    final_key_gain: Tensor
    final_key_eps: float
    norm_gain: Tensor
    norm_eps: float
    unembedding: Tensor
    weights: str
    source: str = ""
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        hidden = int(self.unembedding.shape[-1]) if self.unembedding.ndim == 2 else -1
        for name in ("final_query", "final_key_gain", "norm_gain"):
            t = getattr(self, name)
            if t.ndim != 1 or int(t.shape[0]) != hidden:
                raise LensError(f"{name} must be [{hidden}] to match the unembedding "
                                f"{tuple(self.unembedding.shape)}, got {tuple(t.shape)}")
        for name in ("final_key_eps", "norm_eps"):
            if not float(getattr(self, name)) > 0.0:
                raise LensError(f"{name} must be positive, not {getattr(self, name)}")
        if not self.weights:
            raise LensError("a lens head names the weights digest it belongs to")
        tensors = dict(final_query=self.final_query, final_key_gain=self.final_key_gain,
                       norm_gain=self.norm_gain, unembedding=self.unembedding)
        object.__setattr__(self, "sha256", _tensor_digest(
            tensors, dict(final_key_eps=float(self.final_key_eps), norm_eps=float(self.norm_eps))))

    @property
    def hidden(self) -> int:
        return int(self.unembedding.shape[1])

    @property
    def vocab(self) -> int:
        return int(self.unembedding.shape[0])

    def identity(self) -> dict[str, Any]:
        """What a lane id names of this lens."""
        return dict(variant=LENS_VARIANT, source_sha256=LENS_SOURCE_SHA256,
                    head_sha256=self.sha256, chunk=CHUNK)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Tensor], *, weights: str,
                        final_key_eps: float, norm_eps: float, source: str = "") -> "LensHead":
        """The head from a kotodama-family state dict (``_orig_mod.`` / ``module.`` prefixes
        are stripped). ``weights`` is the digest of the model the state dict loads into.

        Raises
        ------
        LensError
            When a head tensor is missing.
        """
        clean = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
        keys = dict(HEAD_KEYS)
        if keys["unembedding"] not in clean and TIED_UNEMBEDDING in clean:
            keys["unembedding"] = TIED_UNEMBEDDING
        missing = [k for k in keys.values() if k not in clean]
        if missing:
            raise LensError(f"the state dict has no {', '.join(missing)}; the lens reads a "
                            "block-routed model's final read")
        return cls(**{field: clean[key].detach() for field, key in keys.items()},
                   final_key_eps=float(final_key_eps), norm_eps=float(norm_eps),
                   weights=weights, source=source)

    @classmethod
    def from_model(cls, model: torch.nn.Module, *, weights: str | None = None) -> "LensHead":
        """The head of a loaded kotodama-family model, bound to its ``weights-v1`` digest
        (computed here unless ``weights`` gives it)."""
        from anamnesis.extraction.weights_digest import weights_digest

        raw = getattr(model, "_orig_mod", model)
        try:
            key_eps = float(raw.final_res_norm.eps)
            norm_eps = float(getattr(raw.norm, "eps", getattr(raw.norm, "variance_epsilon", 0.0)))
        except AttributeError as err:
            raise LensError(f"the model has no block-routed final read: {err}") from err
        return cls.from_state_dict(raw.state_dict(), weights=weights or weights_digest(raw),
                                   final_key_eps=key_eps, norm_eps=norm_eps,
                                   source=type(raw).__name__)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, weights: str, final_key_eps: float = 1e-5,
                        norm_eps: float = 1e-5, dtype: torch.dtype | None = torch.bfloat16
                        ) -> "LensHead":
        """The head from a training checkpoint (its ``model`` entry, or the file itself when
        it is a state dict), each tensor cast to ``dtype`` (the dtype the model is served in;
        ``None`` keeps the file's). ``weights`` is the digest of the model it is served as:
        the file does not carry it, so the caller names it."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
        state = blob.get("model", blob) if isinstance(blob, Mapping) else None
        if not isinstance(state, Mapping):
            raise LensError(f"{path} holds no state dict")
        head = {k: v for k, v in state.items()
                if k.replace("_orig_mod.", "").replace("module.", "")
                in set(HEAD_KEYS.values()) | {TIED_UNEMBEDDING}}
        if dtype is not None:
            head = {k: v.to(dtype) for k, v in head.items()}
        return cls.from_state_dict(head, weights=weights, final_key_eps=final_key_eps,
                                   norm_eps=norm_eps, source=str(path))

    def to(self, device: torch.device | str) -> "_DeviceHead":
        return _DeviceHead(
            qw=(self.final_query.float() * self.final_key_gain.float()).to(device),
            key_eps=float(self.final_key_eps), gain=self.norm_gain.float().to(device),
            eps=float(self.norm_eps), unembedding=self.unembedding.float().to(device).contiguous())


@dataclass(frozen=True)
class _DeviceHead:
    qw: Tensor
    key_eps: float
    gain: Tensor
    eps: float
    unembedding: Tensor


def committed_ends(route_sources: Sequence[tuple[int, int]]) -> list[int]:
    """The last layer of each committed block, in commit order: the embedding is ``-1``, and
    a block ends at the layer before each boundary layer past the first. A layer is a
    boundary when its MLP input routes over two sources more than the blocks committed
    before it (the block it commits and its new partial)."""
    ends: list[int] = []
    for layer, (n_attn, n_mlp) in enumerate(route_sources):
        before = n_attn - 1 if n_attn else 0
        if n_mlp == before + 2:
            ends.append(layer - 1)
    return ends


def visible_blocks(ends: Sequence[int], layer: int) -> list[int]:
    """The committed blocks the read at ``layer`` routes over: those that end before it."""
    return [c for c, end in enumerate(ends) if end < layer]


def _read(head: _DeviceHead, sources: Tensor) -> Tensor:
    """Log probabilities ``[positions, vocab]`` of the final read over ``sources
    [n_src, positions, hidden]`` (float32; the last source is the partial)."""
    rsqrt = torch.rsqrt(sources.pow(2).mean(-1) + head.key_eps)
    logits = (sources * head.qw).sum(-1) * rsqrt
    weights = torch.softmax(logits, dim=0)
    h = (weights.unsqueeze(-1) * sources).sum(0)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + head.eps) * head.gain
    return torch.log_softmax(h @ head.unembedding.T, dim=-1)


@dataclass(frozen=True)
class LensFeatures:
    """Named lens features of ``B`` rows (float64 ``values [B, len(names)]``) and, per row,
    the largest gap between the final read's entropy and the step's own (``final_check``),
    the lens's built-in check against the model's read."""

    names: tuple[str, ...]
    values: Tensor
    final_check: Tensor


def lens_features(head: LensHead | _DeviceHead, depth: Tensor, committed: Sequence[Tensor],
                  route_sources: Sequence[tuple[int, int]], layers: Sequence[int],
                  chosen: Tensor, step_entropy: Tensor, lengths: Tensor | None) -> LensFeatures:
    """The lens family of ``B`` rows.

    ``depth [B, steps, layers, hidden]`` is every layer's post-MLP partial; ``committed`` the
    committed blocks ``[B, steps, hidden]`` in commit order (the embedding first);
    ``route_sources`` the rows' shared routing plan; ``chosen [B, steps]`` the token each step
    chose; ``step_entropy [B, steps]`` the entropy of each step's own next-token distribution
    (``logit_stats``' first column), for the built-in check; ``lengths [B]`` each row's real
    steps (``None``: every row fills ``steps``).

    Raises
    ------
    LensError
        When the inputs disagree with each other or with the head.
    """
    if depth.ndim != 4:
        raise LensError(f"depth must be [rows, steps, layers, hidden], got {tuple(depth.shape)}")
    batch, steps, n_layers, hidden = (int(s) for s in depth.shape)
    device = depth.device
    dev = head.to(device) if isinstance(head, LensHead) else head
    if int(dev.unembedding.shape[1]) != hidden:
        raise LensError(f"the head reads width {dev.unembedding.shape[1]}, the rows {hidden}")
    if len(route_sources) != n_layers:
        raise LensError(f"{len(route_sources)} routing entries for {n_layers} layers")
    ends = committed_ends(route_sources)
    if len(ends) != len(committed):
        raise LensError(f"the routing plan commits {len(ends)} blocks, the rows carry "
                        f"{len(committed)}")
    layers = [int(layer) for layer in layers]
    if any(not 0 <= layer < n_layers for layer in layers):
        raise LensError(f"lens layers {layers} outside 0..{n_layers - 1}")
    if tuple(chosen.shape) != (batch, steps) or tuple(step_entropy.shape) != (batch, steps):
        raise LensError(f"chosen and step_entropy must be [{batch}, {steps}]")
    last = n_layers - 1
    positions = batch * steps
    n_chunks = max(1, -(-positions // CHUNK))
    padded = n_chunks * CHUNK

    def flat(x: Tensor) -> Tensor:
        x = x.reshape(positions, *x.shape[2:]).float()
        return torch.cat([x, x.new_zeros(padded - positions, *x.shape[1:])]) if padded > positions else x

    blocks = [flat(c.to(device)) for c in committed]
    depth_flat = flat(depth)  # [padded, layers, hidden]
    chosen_flat = torch.cat([chosen.reshape(-1).to(device).long(),
                             chosen.new_zeros(padded - positions, device=device).long()])
    per_pos = torch.zeros(len(layers), 4, padded, dtype=torch.float32, device=device)
    final_entropy = torch.zeros(padded, dtype=torch.float32, device=device)
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        for c in range(n_chunks):
            lo, hi = c * CHUNK, (c + 1) * CHUNK

            def read(layer: int) -> Tensor:
                srcs = [blocks[b][lo:hi] for b in visible_blocks(ends, layer)]
                return _read(dev, torch.stack(srcs + [depth_flat[lo:hi, layer]], dim=0))

            final = read(last)
            p_final = final.exp()
            final_entropy[lo:hi] = -(p_final * final).sum(-1)
            top_final = final.argmax(-1)
            for i, layer in enumerate(layers):
                lp = final if layer == last else read(layer)
                p = lp.exp()
                top = lp.argmax(-1)
                per_pos[i, 0, lo:hi] = -(p * lp).sum(-1)
                per_pos[i, 1, lo:hi] = (top == top_final).float()
                per_pos[i, 2, lo:hi] = (top == chosen_flat[lo:hi]).float()
                per_pos[i, 3, lo:hi] = (p * (lp - final)).sum(-1)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
    per_row = per_pos[:, :, :positions].reshape(len(layers), 4, batch, steps).double()
    columns: list[Tensor] = []
    agree = []
    for i in range(len(layers)):
        ent, agf, agc, kl = (per_row[i, j].contiguous() for j in range(4))
        a = row_mean(agf, lengths=lengths)
        agree.append(a)
        columns += [row_mean(ent, lengths=lengths), row_std(ent, lengths=lengths), a,
                    row_mean(agc, lengths=lengths), row_mean(kl, lengths=lengths),
                    row_std(kl, lengths=lengths)]
    depth_frac = torch.ones(batch, dtype=torch.float64, device=device)
    holds = torch.ones(batch, dtype=torch.bool, device=device)
    for i in sorted(range(len(layers)), key=lambda i: layers[i], reverse=True):
        holds = holds & (agree[i] > AGREE_THRESHOLD)
        depth_frac = torch.where(holds, torch.full_like(depth_frac, layers[i] / max(last, 1)),
                                 depth_frac)
    columns.append(depth_frac)
    gap = (final_entropy[:positions].reshape(batch, steps).double()
           - step_entropy.to(device).double()).abs()
    if lengths is not None:
        gap = torch.where(torch.arange(steps, device=device)[None, :] < lengths.to(device)[:, None],
                          gap, gap.new_zeros(()))
    values = torch.stack(columns, dim=1) if columns else torch.zeros(batch, 0, dtype=torch.float64,
                                                                     device=device)
    return LensFeatures(lens_names(layers), values, gap.amax(dim=1))


__all__ = (
    "AGREE_THRESHOLD",
    "CHUNK",
    "LENS_SOURCE_SHA256",
    "LENS_VARIANT",
    "QUANTITIES",
    "LensError",
    "LensFeatures",
    "LensHead",
    "committed_ends",
    "lens_features",
    "lens_names",
    "visible_blocks",
)
