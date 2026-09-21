"""The measurement discipline every signature audit runs under.

An audit asks whether a signature carries some structure. The five blocks here
are what stand between asking that and answering it wrongly, and they are one
module because each of them was copy-pasted across a suite of analysis scripts
until a fold split in one of them disagreed with the same fold split in another:

**Length control.** Generation length correlates with almost every feature and
with mode, so an uncontrolled accuracy is partly a length readout. The control
regresses features on ``[prompt_length, num_generated_tokens]`` and keeps the
residual — fit on TRAIN only, applied to both splits, because a control fit on
the test rows leaks the test rows.

**Leak-free folds.** The unit that must not straddle a fold boundary is the
*topic*, not the row: all repetitions of a topic share a prompt and a seed
lineage, so a row-wise split trains on one repetition of the very prompt it is
tested on. ``GroupKFold`` by topic holds out topics, prompts and seeds together,
and ``subsample_topics`` shrinks the training set by dropping whole topics, so
the learning curve is a curve in topics rather than in leaked neighbours.

**Signature-matrix IO.** One merged ``(n, P)`` matrix from banked ``gen_*.npz``
files across runs, with labels, topic indices and the length covariates the
control needs. Feature order is pinned to the first generation seen and missing
features fill with zero, which is what lets two runs extracted with slightly
different suite versions enter the same matrix without silently transposing
columns.

**Surface sampling.** The learned-floor protocol reads raw tensors, not
signatures, so each surface is turned into a fixed-dimension per-generation
vector: a constant number of evenly-spaced generation positions, flattened. The
attention surface is the exception — its rows grow with position — so it is
resampled region-aware, prompt part and generated part pooled into separate bin
counts, which keeps the prompt/generated split explicit for the length control.

**Fold preprocessing.** Residualize, standardize, then reduce to the training
row space by the Gram trick. The reduction is lossless for a linear readout
(an L2 solution lies in the train row space) and turns a ``P``-wide problem into
an ``n``-wide one, which is the difference between a floor that runs and a floor
that does not. It runs on whichever device it is handed.

The functions are extracted from the v3 audit suite, which stays whole in the
frozen repository as the record of what was run when. Its import sites there —
``vmb_s51_encoder_on_raw`` (all three donor modules), ``vmb_s51_resolver``,
``pathsig_constant_injection``, ``pathsig_read_e1``, ``pathsig_incremental``,
``pathsig_s51_regen`` and ``analyze_signature_richness`` — read from the donors,
not from here: this module is the living copy, and nothing edits the record.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import GroupKFold

F64 = NDArray[np.float64]

HARD: frozenset[str] = frozenset(
    {"linear", "socratic", "contrastive", "dialectical", "analogical"}
)
"""The five-way hard mode set — the classification target of record."""

#: Evenly-spaced generation positions sampled per surface vector.
N_POS = 5
#: Bin counts for the region-aware attention reduction (prompt part, generated part).
PROMPT_BINS, GEN_BINS = 8, 24
ATTN_BINS = PROMPT_BINS + GEN_BINS

#: Surfaces whose raw tensors are fixed-dimension per token: name -> raw npz key.
#: Their sampled positions flatten directly. ``attention`` is not among them.
SIMPLE: dict[str, str] = {
    "residual": "hidden_states",
    "keys": "pre_rope_keys",
    "values": "v_proj_values",
    "queries": "queries",
    "gate": "gate_activations",
}
ALL_SURFACES = list(SIMPLE) + ["attention"]


# ── Length control ────────────────────────────────────────────────────────────
def residualize(Ftr: F64, Fte: F64, Ctr: F64, Cte: F64) -> tuple[F64, F64]:
    """Regress features on covariates C (+intercept) fit on TRAIN, subtract from both."""
    A = np.hstack([Ctr, np.ones((len(Ctr), 1))]); B = np.hstack([Cte, np.ones((len(Cte), 1))])
    coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
    return Ftr - A @ coef, Fte - B @ coef


def residualize_all(F: F64, C: F64) -> F64:
    """Whole-matrix variant (no train/test split — for unsupervised structure)."""
    A = np.hstack([C, np.ones((len(C), 1))])
    coef, *_ = np.linalg.lstsq(A, F, rcond=None)
    return F - A @ coef


# ── Leak-free folds ───────────────────────────────────────────────────────────
def subsample_topics(
    tr: NDArray[np.int_], topic: NDArray[np.int_], frac: float, seed: int
) -> NDArray[np.int_]:
    """Subsample train TOPICS (not rows) to frac — the learning-curve knob."""
    if frac >= 1.0:
        return tr
    utop = np.unique(topic[tr])
    rng = np.random.default_rng(1000 + seed)
    keep = set(rng.choice(utop, max(2, int(round(frac * len(utop)))), replace=False).tolist())
    return tr[np.array([topic[i] in keep for i in tr])]


def leak_free_folds(
    X: NDArray[Any],
    y: NDArray[Any],
    topic: NDArray[np.int_],
    *,
    n_splits: int = 5,
    seeds: int = 1,
    frac: float = 1.0,
) -> Iterator[tuple[int, NDArray[np.int_], NDArray[np.int_]]]:
    """Yield ``(seed, train_idx, test_idx)`` for the protocol of record.

    The fold split itself is ``GroupKFold(n_splits)`` grouped by topic, which is
    deterministic — no seed enters it. The seed enters through
    :func:`subsample_topics`, so ``seeds > 1`` at ``frac < 1`` gives several
    draws of the same learning-curve point rather than several fold layouts, and
    at ``frac == 1.0`` every seed yields the identical folds.

    Parameters
    ----------
    X, y
        The feature matrix and labels, passed through to ``GroupKFold.split``.
    topic
        Per-row topic index: the grouping that must not straddle a fold.
    n_splits
        Folds per seed.
    seeds
        How many training subsamples to draw per fold.
    frac
        Fraction of TRAIN topics to keep, per :func:`subsample_topics`.

    Raises
    ------
    ValueError
        When ``n_splits`` exceeds the number of distinct topics, which would ask
        ``GroupKFold`` for more held-out groups than exist.
    """
    n_groups = int(np.unique(topic).size)
    if n_splits > n_groups:
        raise ValueError(
            f"leak_free_folds: {n_splits} folds over {n_groups} distinct topic(s); "
            "a fold cannot hold out a topic that is not there"
        )
    for seed in range(seeds):
        for tr, te in GroupKFold(n_splits).split(X, y, topic):
            yield seed, subsample_topics(tr, topic, frac, seed), te


# ── Signature-matrix IO ───────────────────────────────────────────────────────
def unwrap_generations(meta: Any) -> list[dict[str, Any]]:
    """metadata.json wraps the generation list under a "generations" key — unwrap it
    (tolerates the bare-list legacy format)."""
    return meta["generations"] if isinstance(meta, dict) and "generations" in meta else meta


def gen_metadata_by_id(metadata_path: Path | str) -> dict[int, dict[str, Any]]:
    """Load metadata.json -> {generation_id: record}."""
    with open(metadata_path) as f:
        meta = json.load(f)
    return {int(g["generation_id"]): g for g in unwrap_generations(meta)}


class SignatureMatrix(BaseModel):
    """Loaded signature feature matrix + labels/covariates/feature names."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    X: np.ndarray      # (n, P) float64, nan_to_num'd
    y: np.ndarray      # (n,) mode labels (str)
    topic: np.ndarray  # (n,) topic_idx (int)
    C: np.ndarray      # (n, 2) float64 [prompt_length, num_generated_tokens]
    names: np.ndarray  # (P,) feature names (str)


def load_signature_matrix(
    runs: Sequence[str],
    runs_root: Path | str,
    subdir: str = "signatures_v3",
    modes: Collection[str] = HARD,
) -> SignatureMatrix:
    """Load merged per-gen signature vectors across runs (feature order pinned to
    the first gen seen; missing features fill 0.0; matrix nan_to_num'd).

    Runs and directories that do not exist are skipped silently, which is what
    lets one call name a run and its extension corpus without knowing whether
    the extension was banked on this machine.
    """
    names: list[str] | None = None
    rows: list[list[float]] = []
    y: list[str] = []
    topic: list[int] = []
    C: list[list[float]] = []
    for run in runs:
        rd = Path(runs_root) / run
        sd = rd / subdir
        if not (rd / "metadata.json").exists() or not sd.exists():
            continue
        md = gen_metadata_by_id(rd / "metadata.json")
        for p in sorted(sd.glob("gen_*.npz"), key=lambda x: int(x.stem.split("_")[1])):
            g = int(p.stem.split("_")[1])
            if g not in md or md[g]["mode"] not in modes:
                continue
            z = np.load(p, allow_pickle=True)
            nm = [str(x) for x in z["feature_names"]]
            if names is None:
                names = nm
            d = {n: float(v) for n, v in zip(nm, z["features"])}
            rows.append([d.get(n, 0.0) for n in names])
            y.append(md[g]["mode"])
            topic.append(md[g]["topic_idx"])
            C.append([md[g]["prompt_length"], md[g]["num_generated_tokens"]])
    return SignatureMatrix(
        X=np.nan_to_num(np.array(rows, float)),
        y=np.array(y),
        topic=np.array(topic),
        C=np.array(C, float),
        names=np.array(names if names is not None else [], dtype=object),
    )


# ── Surface sampling ──────────────────────────────────────────────────────────
def sample_positions(T: int, n: int = N_POS) -> NDArray[np.int_]:
    """n evenly-spaced generation-step indices in [0, T). Fixed count regardless of T,
    so the flattened per-gen vector has constant dim across gens."""
    if T <= 0:
        return np.zeros(n, dtype=int)
    return np.linspace(0, T - 1, n).round().astype(int)


def _resample_rows(X: NDArray[Any], B: int) -> F64:
    """(R, n) nonneg rows -> (R, B). Mean-pool into B contiguous bins (n>=B), linear-interp
    up (n<B), passthrough (n==B). The basis for the region-aware attention reduction."""
    R, n = X.shape
    if n == 0:
        return np.zeros((R, B), dtype=np.float64)
    Xf = X.astype(np.float64)
    if n == B:
        return Xf
    if n < B:
        xp = np.linspace(0.0, 1.0, n)
        xq = np.linspace(0.0, 1.0, B)
        out = np.empty((R, B), dtype=np.float64)
        for r in range(R):
            out[r] = np.interp(xq, xp, Xf[r])
        return out
    idx = (np.arange(n) * B) // n                     # bin id per source column
    out = np.zeros((R, B), dtype=np.float64)
    np.add.at(out, (slice(None), idx), Xf)            # sum columns into their bin (all rows)
    cnt = np.zeros(B, dtype=np.float64)
    np.add.at(cnt, idx, 1.0)
    return out / np.maximum(cnt, 1.0)[None, :]


def attention_vector(z: np.lib.npyio.NpzFile, pos: NDArray[np.int_]) -> NDArray[Any] | None:
    """Region-aware resample of per-head attention rows at the sampled positions.
    Returns flat (N_POS * L * H * ATTN_BINS,) float16, or None if attention absent."""
    if "attentions" not in z.files or "actual_lengths" not in z.files:
        return None
    attn = z["attentions"]                            # (T, L, H, max_seq) f16
    if attn.size == 0:
        return None
    alen = z["actual_lengths"]
    plen = int(z["prompt_length"])
    _, L, H, _ = attn.shape
    blocks = []
    for t in pos:
        al = int(alen[t])
        if al <= 0:
            blocks.append(np.zeros((L * H, ATTN_BINS), dtype=np.float64))
            continue
        pe = min(max(plen, 0), al)                     # prompt-end clamped into the row
        rows = attn[t].reshape(L * H, -1)[:, :al]      # (L*H, al) valid (unpadded) attention
        prm = _resample_rows(rows[:, :pe], PROMPT_BINS)
        gen = _resample_rows(rows[:, pe:al], GEN_BINS)
        blocks.append(np.concatenate([prm, gen], axis=1))   # (L*H, ATTN_BINS)
    return np.stack(blocks).reshape(-1).astype(np.float16)   # (N_POS*L*H*ATTN_BINS,)


def surface_vector(
    z: np.lib.npyio.NpzFile, surface: str, pos: NDArray[np.int_], T: int
) -> NDArray[Any] | None:
    """Fixed-dim per-gen vector for one surface, or None to skip this gen."""
    if surface == "attention":
        return attention_vector(z, pos)
    key = SIMPLE[surface]
    if key not in z.files:
        return None
    arr = z[key]
    if arr.size == 0 or arr.shape[0] < T:
        return None
    return arr[pos].reshape(-1).astype(np.float16)


# ── Fold preprocessing ────────────────────────────────────────────────────────
def preprocess_fold_gpu(
    Xtr_np: NDArray[Any],
    Xte_np: NDArray[Any],
    Ctr: NDArray[Any],
    Cte: NDArray[Any],
    resid: bool,
    device: str,
    eps: float = 1e-8,
) -> tuple[F64, F64]:
    """ALL wide preprocessing on one device, computed ONCE per fold (shared by every arch):
    residualize on [prompt_len, gen_len] → per-feature standardize (= the raw floor) →
    lossless row-space reduce (Gram trick → exact sample coords; P~1e5 → ~n_tr). Returns the
    small reduced (Ztr, Zte) for the encoder.

    Doing it here rather than with a CPU residualize plus an sklearn StandardScaler avoids
    float64 churn and repeated host↔device transfers, which is what makes the floor
    latency-bound. Faithful to the CPU path up to float32 vs float64 (negligible); logit on
    Z is the raw floor (the L2 solution lies in the train row space). A single global scale
    conditions the deep arch WITHOUT per-PC whitening (whitening amplifies noise → hurts).

    ``device`` is passed to ``torch``, so ``"cpu"`` is a valid and exact choice; the name
    records that this path exists because a GPU makes the reduction cheap enough to run per
    fold.
    """
    import torch

    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=device)
    Xte = torch.tensor(Xte_np, dtype=torch.float32, device=device)
    if resid:
        A = torch.tensor(np.hstack([Ctr, np.ones((len(Ctr), 1))]), dtype=torch.float32, device=device)
        B = torch.tensor(np.hstack([Cte, np.ones((len(Cte), 1))]), dtype=torch.float32, device=device)
        coef = torch.linalg.lstsq(A, Xtr).solution
        Xtr = Xtr - A @ coef
        Xte = Xte - B @ coef
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-8)
    Xtr = (Xtr - mu) / sd                       # per-feature standardize (centers + scales)
    Xte = (Xte - mu) / sd
    Ktr = Xtr @ Xtr.T                           # (n_tr, n_tr) Gram on standardized features
    w, U = torch.linalg.eigh(Ktr)
    keep = w > eps * w.max().clamp_min(1e-30)
    w, U = w[keep], U[:, keep]
    s = torch.sqrt(w.clamp_min(1e-12))
    Ztr = U * s                                 # (n_tr, r) exact train coordinates
    Zte = (Xte @ Xtr.T @ U) / s                 # (n_te, r) test coords via cross-Gram
    g = Ztr.std().clamp_min(1e-8)
    return (Ztr / g).cpu().numpy(), (Zte / g).cpu().numpy()
