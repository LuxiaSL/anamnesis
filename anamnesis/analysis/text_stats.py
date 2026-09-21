"""Reading the generated text itself — the channel beside the signature.

A signature says how a forward pass ran. These readouts say what came out of it,
in the text, and they exist because a claim about computation needs the surface
checked too: an intervention that moves a signature and also collapses the
text's type-token ratio has not shown what it looks like it has shown, and one
whose text is indistinguishable is a stronger result than one whose text is not
examined. The three readouts are three different kinds of evidence:

**Coherence** (:func:`text_stats`). Mean length, type-token ratio and trigram
repetition over a bank's generations. This is the degeneracy check — it catches
the failure where a steered model produces fluent-looking loops, which every
other metric would score as a change rather than as a collapse.

**Distributional state** (:func:`entropy_and_nll_over_generation`). Per-position
next-token entropy and the surprisal of the token actually generated, measured by
forwarding once over the generated span. Entropy is the state of the
distribution the text was sampled from; NLL is how surprised the model is by its
own output. They answer different questions and are reported apart.

**Lexicon** (:func:`group_marker_rates`). Hedging versus definitive marker rates
per thousand words, over a curated lexicon. A negative result here means the
coordinate does not express *in this vocabulary* — never that there was no
behavioural expression, which entropy and diversity certify independently.

:func:`add_null_ratios` is the discipline that makes any of these citable: a
readout is reported against its matched nulls, and a ratio to a baseline that
sits near zero is suppressed in favour of a band readout rather than printed as
a large number that means nothing.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover — typing only; torch is imported at call time
    import torch

    from anamnesis.extraction.model_loader import ResidualWriteSpec


# ── Coherence ─────────────────────────────────────────────────────────────────
def text_stats(meta_path: Path | str) -> dict[str, float | int]:
    """Length, type-token ratio and trigram repetition over a bank's generations.

    Reads a ``metadata.json`` whose generation list may be wrapped under a
    ``"generations"`` key or be the whole document. Generations with no text are
    skipped rather than counted as zero-length, so ``n`` is the number of
    generations the statistics actually rest on.
    """
    md = json.loads(Path(meta_path).read_text())
    gens = md["generations"] if "generations" in md else md
    lens, ttrs, reps = [], [], []
    for g in gens:
        txt = g.get("generated_text", "")
        toks = txt.split()
        if not toks:
            continue
        lens.append(g.get("num_generated_tokens", len(toks)))
        ttrs.append(len(set(toks)) / len(toks))
        tri = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
        reps.append(1.0 - (len(set(tri)) / max(len(tri), 1)))
    return {"n": len(lens), "mean_len": float(np.mean(lens)) if lens else 0.0,
            "mean_ttr": float(np.mean(ttrs)) if ttrs else 0.0,
            "mean_trigram_rep": float(np.mean(reps)) if reps else 0.0}


# ── Distributional state ──────────────────────────────────────────────────────
def entropy_and_nll_over_generation(
    model: Any,
    ids: torch.Tensor,
    prompt_length: int,
    spec: ResidualWriteSpec | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-position next-token ENTROPY + NLL over the generated span.

    One forward pass over ``ids``, optionally with a residual-write intervention
    attached, recording only per-position scalars — no raw tensors, so this costs
    nothing to store. Entropy is the state of the distribution; NLL is
    ``-log p(token)`` for the token actually generated, the likelihood rung of the
    detector hierarchy.

    Parameters
    ----------
    model
        A loaded causal LM, called directly rather than through ``generate``.
    ids
        ``[1, T]`` token ids: the prompt followed by the generated span.
    prompt_length
        Where the generated span starts in ``ids``.
    spec
        A residual-write intervention to apply during the pass, or None to read
        the unsteered distribution over the same tokens.
    """
    import torch

    from anamnesis.extraction.model_loader import attach_residual_write

    P = prompt_length
    h = attach_residual_write(model, spec) if spec is not None else None
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0].float()   # [T, vocab]
    if h is not None:
        h.remove()
    lp = torch.log_softmax(logits, dim=-1)
    ent = -(lp.exp() * lp).sum(dim=-1)                            # [T]
    T = ids.shape[1]
    pos = torch.arange(P - 1, T - 1)                              # distributions that produced gen tokens
    tgt = ids[0, P:T]                                             # the actual generated tokens
    nll = -lp[pos, tgt]                                           # [len(pos)] surprisal
    return ent[pos].cpu().numpy(), nll.cpu().numpy()


# ── Lexicon ───────────────────────────────────────────────────────────────────
# Curated lexicons, word-boundary matched, case-insensitive. Multi-word phrases allowed.
HEDGE_MARKERS = [
    r"maybe", r"perhaps", r"might", r"could", r"possibly", r"potentially", r"presumably",
    r"probably", r"likely", r"arguably", r"seemingly", r"apparently", r"somewhat",
    r"roughly", r"sort of", r"kind of", r"i think", r"i believe", r"i guess", r"i suppose",
    r"it seems", r"it appears", r"tends? to", r"suggests?", r"may", r"can be",
    r"in some cases", r"to some extent", r"more or less", r"not necessarily", r"not sure",
    r"uncertain", r"unclear", r"can vary", r"depends? on", r"if anything",
]
DEFINITIVE_MARKERS = [
    r"definitely", r"certainly", r"clearly", r"obviously", r"undoubtedly", r"absolutely",
    r"surely", r"indeed", r"in fact", r"without doubt", r"no doubt", r"of course",
    r"always", r"never", r"must", r"will always", r"guaranteed", r"proven", r"undeniably",
    r"unquestionably", r"invariably", r"inevitably", r"precisely", r"exactly",
    r"is the", r"are the", r"the fact that", r"it is clear", r"there is no",
]
HEDGE_RE = re.compile(r"\b(?:" + "|".join(HEDGE_MARKERS) + r")\b", re.IGNORECASE)
DEF_RE = re.compile(r"\b(?:" + "|".join(DEFINITIVE_MARKERS) + r")\b", re.IGNORECASE)


def marker_rate(text: str, rx: re.Pattern) -> tuple[int, int]:
    """``(matches, words)`` for one text — the numerator and denominator, unmixed."""
    w = max(len(text.split()), 1)
    return len(rx.findall(text)), w


def texts_by_prompt_group(run_dir: Path | str, cell: str) -> dict[tuple, list[str]] | None:
    """Group a cell's generated texts by ``(topic_idx, mode_idx)``, or None if absent.

    The prompt group is the statistical unit: the k resamples of one prompt are
    one observation, not k. Returns None when the cell has no ``metadata.json``,
    so a missing cell is a skip rather than an exception.
    """
    md_path = Path(run_dir) / cell / "metadata.json"
    if not md_path.exists():
        return None
    md = json.loads(md_path.read_text())
    gens = md["generations"] if "generations" in md else md
    groups: dict[tuple, list[str]] = defaultdict(list)
    for g in gens:
        groups[(g.get("topic_idx"), g.get("mode_idx"))].append(g.get("generated_text", ""))
    return groups


def group_marker_rates(groups: dict[tuple, list[str]]) -> dict[str, Any]:
    """Per-group hedge/definitive rates per 1k words (pooled over the k resamples), + census.

    The census is reported because two cells are only comparable when they cover
    the same prompt groups; the histogram is what a reader checks rather than
    assumes.
    """
    hedge_pg, defn_pg = [], []
    mode_hist: dict = defaultdict(int)
    keys = sorted(groups.keys(), key=lambda t: (t[0] if t[0] is not None else -1,
                                                 t[1] if t[1] is not None else -1))
    for k in keys:
        texts = groups[k]
        mode_hist[k[1]] += 1
        h, dcount, wtot = 0, 0, 0
        for t in texts:
            hi, w = marker_rate(t, HEDGE_RE)
            di, _ = marker_rate(t, DEF_RE)
            h += hi; dcount += di; wtot += w
        wtot = max(wtot, 1)
        hedge_pg.append(1000.0 * h / wtot)
        defn_pg.append(1000.0 * dcount / wtot)
    return {"hedge_pg": hedge_pg, "def_pg": defn_pg,
            "hedge_per_1k": float(np.mean(hedge_pg)), "def_per_1k": float(np.mean(defn_pg)),
            "net_hedge_per_1k": float(np.mean(hedge_pg) - np.mean(defn_pg)),
            "n_groups": len(keys), "census_mode_hist": dict(sorted(mode_hist.items()))}


def placebo_marker_floor(
    groups: dict[tuple, list[str]], seed: int = 20260716
) -> dict[str, float | int]:
    """Disjoint unsteered-vs-unsteered split within each group: the noise floor.

    Any rate difference between two cells has to clear the difference this finds
    between two halves of the *same* cell. Without it a small marker shift is
    unreadable, because nobody knows what zero looks like.
    """
    rng = np.random.default_rng(seed)
    diffs_h, diffs_n = [], []
    for _, texts in groups.items():
        if len(texts) < 2:
            continue
        idx = rng.permutation(len(texts))
        half = len(texts) // 2
        a, b = [texts[i] for i in idx[:half]], [texts[i] for i in idx[half:2 * half]]

        def rate(group: list[str], rx: re.Pattern) -> float:
            h = sum(len(rx.findall(t)) for t in group)
            w = max(sum(len(t.split()) for t in group), 1)
            return 1000.0 * h / w

        diffs_h.append(rate(a, HEDGE_RE) - rate(b, HEDGE_RE))
        diffs_n.append((rate(a, HEDGE_RE) - rate(a, DEF_RE)) - (rate(b, HEDGE_RE) - rate(b, DEF_RE)))
    return {"placebo_hedge_abs_mean": round(float(np.mean(np.abs(diffs_h))), 3),
            "placebo_hedge_sd": round(float(np.std(diffs_h)), 3),
            "placebo_net_abs_mean": round(float(np.mean(np.abs(diffs_n))), 3),
            "n_placebo_groups": len(diffs_h)}


# ── Reading a readout against its nulls ───────────────────────────────────────
def add_null_ratios(
    rows: list[dict],
    null_prefixes: Sequence[str],
    keys: Sequence[str] = ("mean_entropy_steered", "entropy_rise", "base_model_nll"),
) -> None:
    """Attach matched-null (÷-Rc) columns with the ZERO-DENOMINATOR GUARD.

    A ratio to a signed near-zero baseline (e.g. ``entropy_rise``, whose nulls
    hover at ~0) is uninformative and explodes. Guard: suppress ``_over_Rc``
    (-> None) when the denominator's coefficient of variation is large; ALWAYS
    emit the band readout ``_vs_Rc_band`` (null min/max/mean/sd + z + outside-band
    flag), which is the correct specificity statistic for a
    difference-from-zero quantity. The same function serves a live replay and a
    device-free reaggregation, so there is one source of truth for the column.

    Rows are mutated in place. A row marked ``is_null`` is a null and is skipped;
    nulls are matched to their row by ``(site, alpha_frac)``. ``null_prefixes`` is
    accepted because callers name their null family when they call, but the
    selection is made by each row's own ``is_null`` flag, which is what the
    banked rows carry.
    """
    for r in rows:
        if r.get("is_null"):
            continue
        for key in keys:
            nulls = [nr[key] for nr in rows if nr.get("is_null")
                     and nr.get("site") == r.get("site") and nr.get("alpha_frac") == r.get("alpha_frac")
                     and key in nr]
            if not nulls or key not in r:
                r[f"{key}_over_Rc"] = None
                continue
            nm, nsd = float(np.mean(nulls)), float(np.std(nulls))
            nmin, nmax = float(np.min(nulls)), float(np.max(nulls))
            # A ratio to a signed near-zero baseline is unstable; suppress when the denominator's
            # coefficient of variation is large (CV>25% ⇒ a ratio-of-differences both near zero).
            cv = (nsd / abs(nm)) if nm else float("inf")
            ratio_meaningful = (nm != 0.0) and (cv <= 0.25)
            r[f"{key}_over_Rc"] = round(float(r[key] / nm), 3) if ratio_meaningful else None
            r[f"{key}_vs_Rc_band"] = {
                "null_mean": round(nm, 4), "null_sd": round(nsd, 4),
                "null_min": round(nmin, 4), "null_max": round(nmax, 4),
                "z_vs_null": round((r[key] - nm) / nsd, 3) if nsd > 0 else None,
                "outside_null_band": bool(r[key] < nmin or r[key] > nmax),
                "ratio_suppressed_zero_denom": not ratio_meaningful,
            }
