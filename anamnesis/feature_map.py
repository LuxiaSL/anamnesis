"""Feature map — the (SOURCE x METHOD x DEPTH) taxonomy, the vocabulary of record for
describing what a feature reads.

A feature's cell here is what says which substrate it reads. The contiguous blocks a
signature vector is addressed in — `extract_norms_and_output_stats`..`extract_residual_pca`
in `anamnesis/extraction/state_extractor.py` build them, and
`anamnesis/analysis/gauntlet/signature_io.py` reads them back — are addresses into the
vector and nothing more. Three of the four span several sources at once, so a block's
accuracy is not a reading of any single substrate and no claim is stated per block. Address a
stored artifact by its block; describe a feature by its cell.

This module tags every signature feature on four axes:

  SOURCE   = which substrate is read (`Source`).
  METHOD   = the base operator: magnitude / distributional / geometry / spectral /
             iterated_integral / learned (`Method`).
  DYNAMIC  = the temporal wrapper: static (a *_mean / snapshot) vs dynamic (*_std / slope /
             trajectory / window / drift / novelty).
  DEPTH    = layer, and the band it falls in — a band is a fraction of the network, so the
             model's layer count is part of every depth label.

This is the reusable substrate for the discover→decompose→distill workflow:
  raw -> encoder (per-model discovery, all sources) -> decompose by CELL (this map) -> which (source,method,
  depth) cells carry THIS task -> instantiate theory-motivated features for those cells -> portable,
  lightweight signature. Redundancy is task-specific, so you re-decompose per task; the cells are the unit.

Design: pure-numpy/pydantic (no torch/sklearn — importable anywhere, like state_extractor).
Classification is a parser over banked feature names, so it can be wrong and says where:
`FeatureMap.unclassified()` lists the names it could not place and `.summary()` exposes every
call it did make, so a tag is auditable and overridable rather than taken on trust.

  >>> fm = FeatureMap(feature_names, n_layers=32)
  >>> X_attn_mid = X[:, fm.mask(source=Source.attention, band=Band.mid)]   # slice a cell
  >>> for cell, idx in fm.cells("source", "band").items(): acc[cell] = lda(X[:, idx], y)   # decompose
  >>> lean = fm.select(source=Source.attention)            # every attention-source feature

Run as a script to validate the taxonomy + coverage on a real run:
    ANAMNESIS_RUNS=/models/anamnesis-extract/runs python -m anamnesis.feature_map 8b_fat_01
A run whose model is not in `MODEL_LAYERS` takes its layer count as a second
argument, because depth bands are fractions of the network and no default is right
for an unnamed model.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

# Model layer counts (for depth bands). Extend per model/architecture.
MODEL_LAYERS = {"3b": 28, "8b": 32, "kotodama_3b": 28,
                "olmo2-7b": 32, "gemma3-27b": 62, "qwen-7b": 28,
                "dsv2": 27, "dsv2_lite": 27, "dsv2-lite": 27}  # three spellings of DeepSeek-V2-Lite


class UnknownDepthError(ValueError):
    """A run's layer count is not known, so its depth bands cannot be assigned."""


def layers_for_run(run: str) -> int:
    """The layer count of the model a run name identifies, by longest prefix.

    A band is a fraction of the network, so the layer count is the denominator of
    every early/mid/late label. Getting it wrong does not fail: it relabels the
    bands and reports a plausible map of where signal lives, over the wrong depths.
    There is no count that is safe to assume for an unnamed model, so an unmatched
    run refuses and the caller states the depth instead.

    Longest prefix rather than first match, because the keys nest: a run beginning
    ``dsv2_lite`` also begins ``dsv2``, and a shorter key that happened to be
    ordered first would answer for a model it does not name.

    Raises
    ------
    UnknownDepthError
        When no key of :data:`MODEL_LAYERS` prefixes ``run``. Add the model there,
        or pass the count.
    """
    matches = [k for k in MODEL_LAYERS if run.startswith(k)]
    if not matches:
        raise UnknownDepthError(
            f"no layer count known for run {run!r}; add its model to MODEL_LAYERS "
            f"in anamnesis/feature_map.py or state the layer count at the call site "
            f"(known prefixes: {sorted(MODEL_LAYERS)})"
        )
    return MODEL_LAYERS[max(matches, key=len)]


class Source(str, Enum):
    output = "output"          # logit / token-distribution statistics
    residual = "residual"      # the residual stream itself (norms, trajectory, x-layer delta, PCA)
    attention = "attention"    # attention-weight reads: entropy, head-agreement, flow/region/recency, cache mass
    keys = "keys"              # pre-RoPE key-vector geometry (spread/drift/novelty/eff_dim) + epoch detection
    gate = "gate"              # SwiGLU gate activations
    values = "values"          # v_proj value-vector geometry, and value-vs-key coupling
    qk = "qk"                  # pre-RoPE query geometry and q·k content alignment (position-free)
    routing = "routing"        # cross-block attention-residual routing weights (block-routing architectures)
    expert_routing = "expert_routing"  # MoE router: which expert a token was sent to. Not
                               # `routing` above, which is how much of its own history a block
                               # attended over. A dense model has no expert allocation to read.
    unknown = "unknown"        # flagged: classifier did not match (audit these)


class Method(str, Enum):
    magnitude = "magnitude"            # norms / means of magnitudes
    distributional = "distributional"  # entropy, JSD/agreement, top-k mass, coverage, mass fractions, sparsity
    geometry = "geometry"              # cosine / spread / drift / novelty / participation-ratio / PCA projection
    spectral = "spectral"              # graph-spectral (Fiedler, HFER, spectral entropy, smoothness)
    learned = "learned"                # contrastive-projection / encoder dimensions. No naming rule
                                       # returns it: `cp_` names carry no operator mark, so the
                                       # classifier leaves them for `unclassified()` to report.
    iterated_integral = "iterated_integral"  # log-signature of a time-augmented trajectory: level-1
                                       # displacement, level-2 Levy areas. The other methods each read
                                       # one series alone; this reads two coordinates' joint order.
    unknown = "unknown"


class Band(str, Enum):
    early = "early"
    mid = "mid"
    late = "late"


class FeatureTag(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    source: Source
    method: Method
    dynamic: Optional[bool]      # True=dynamic, False=static, None=ambiguous
    layer: Optional[int]
    band: Optional[Band]
    family: str                  # the family label banked analyses key their per-family
                                 # numbers by; `FAMILY_LABELS` is the whole label set


# ---------------------------------------------------------------------------- classification rules

# FROZEN WIRE VOCABULARY — do not change these strings. `anamnesis/analysis/battery/floors.py`
# keys floor results by `family:<label>`, so a relabelled family stops lining up with the numbers
# banked under it. The constant names say what each label groups; the labels themselves follow the
# stored block layout and describe nothing.
STORED_FAMILY_CACHE_AND_KEYS = "T2.5"
STORED_FAMILY_ATTENTION_SPECTRAL = "T2_spectral"
STORED_FAMILY_ATTENTION_OTHER = "T2_other"
STORED_FAMILY_RESIDUAL_PCA = "T3"
STORED_FAMILY_NORMS_AND_OUTPUT_STATS = "T1"

# The families that also have a block of their own in a banked analysis result, named
# here because `anamnesis/analysis/complementarity.py` keys its block translation on
# them and a literal repeated in two modules is two vocabularies waiting to diverge.
FAMILY_ATTENTION_FLOW = "attention_flow"
FAMILY_GATE = "gate"
FAMILY_RESIDUAL_TRAJECTORY = "residual_traj"
FAMILY_TEMPORAL_DYNAMICS = "temporal_dynamics"
FAMILY_CONTRASTIVE_PROJECTION = "contrastive_projection"


class FamilyRule(BaseModel):
    """One naming rule: the family a feature name is grouped under when it matches.

    A rule matches on a name's prefix, which is how every family but one marks its
    features. ``contains`` is for the family whose mark is not a prefix, and it is
    checked in the same pass so that rule order stays the whole story.
    """

    model_config = ConfigDict(frozen=True)
    family: str
    prefixes: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()

    def matches(self, name: str) -> bool:
        return name.startswith(self.prefixes) or any(mark in name for mark in self.contains)


# The rules, in the order they are tried: the first match wins, so a narrower mark
# comes before a wider one that would swallow it. Two spellings of one family sit in one
# rule, because banks disagree on how the engineered families spell their columns —
# `af_`/`gf_`/`rt_` in some, `attn_flow_`/`gate_`/`res_traj_` in others — and a family is
# one family however its columns were named.
FAMILY_RULES: tuple[FamilyRule, ...] = (
    FamilyRule(family="value_geometry", prefixes=("value_",)),
    FamilyRule(family="qk_geometry", prefixes=("qk_", "q_")),
    FamilyRule(family="kv_cka", contains=("cka",)),   # before the kv_ rule, which would take it
    FamilyRule(family="per_head", prefixes=("ph_",)),
    FamilyRule(family=FAMILY_ATTENTION_FLOW, prefixes=("attn_flow_", "af_")),
    FamilyRule(family=FAMILY_GATE, prefixes=("gate_", "gf_")),
    FamilyRule(family="path_signature", prefixes=("res_sig",)),           # level-2 log-signature
    FamilyRule(family="path_signature_output", prefixes=("out_sig",)),    # output-statistics path
    FamilyRule(family="path_signature_attention", prefixes=("attn_sig",)),  # attention-region path
    FamilyRule(family=FAMILY_RESIDUAL_TRAJECTORY, prefixes=("res_traj", "rt_")),
    FamilyRule(family=STORED_FAMILY_CACHE_AND_KEYS, prefixes=("cache_", "kv_", "epoch_")),
    FamilyRule(family=STORED_FAMILY_ATTENTION_SPECTRAL, prefixes=("spectral_",)),
    FamilyRule(
        family=STORED_FAMILY_ATTENTION_OTHER,
        prefixes=("attn_entropy_", "head_agreement_", "delta_"),
    ),
    FamilyRule(family="attn_res", prefixes=("attnres_",)),
    FamilyRule(family="expert_routing", prefixes=("xrt_", "expert_routing_")),  # MoE router
    FamilyRule(family=FAMILY_TEMPORAL_DYNAMICS, prefixes=("td_",)),
    FamilyRule(family=FAMILY_CONTRASTIVE_PROJECTION, prefixes=("cp_",)),
    FamilyRule(family=STORED_FAMILY_RESIDUAL_PCA, prefixes=("pca_",)),
    # Last, because this family's block is also where the stored layout puts a name
    # nothing else claims: the rule states the marks it really holds, so that a name
    # matching none of them is reported as unplaced instead of credited here.
    FamilyRule(
        family=STORED_FAMILY_NORMS_AND_OUTPUT_STATS,
        prefixes=("activation_norm", "logit_", "top1_", "top5_", "surprise", "mean_", "std_"),
    ),
)

FAMILY_LABELS: frozenset[str] = frozenset(rule.family for rule in FAMILY_RULES)
"""Every family label the rules can return. A reader that translates a label into its
own vocabulary checks itself against this set, so a family added here cannot be silently
dropped by a table that does not know it."""


def named_family(n: str) -> str | None:
    """The family a rule names for this feature, or None when no rule does.

    :func:`stored_family` answers for every string, because the block an unrecognised
    name is stored in is also a family. A caller that has to tell a match from that
    fallback asks here — attributing a number to a block that never held the feature is
    worse than reporting that the name could not be placed.
    """
    for rule in FAMILY_RULES:
        if rule.matches(n):
            return rule.family
    return None


def stored_family(n: str) -> str:
    """The family label a feature is grouped under in banked per-family results.

    Five labels follow the stored block layout and are held as the STORED_FAMILY_*
    constants above; the rest name their family directly. A name no rule matches takes
    the norms-and-output-stats label, which is the block the stored layout ends with.
    What a feature *reads* is answered by `classify()` and its (source, method, depth)
    cell, never by this.
    """
    return named_family(n) or STORED_FAMILY_NORMS_AND_OUTPUT_STATS


def _source(n: str) -> Source:
    # Attention-residual routing FIRST — its names contain "entropy"/"top" which would else mis-hit output.
    if n.startswith("attnres_committed"): return Source.residual   # committed residual-block snapshots (geometry)
    if n.startswith("attnres_"): return Source.routing             # block-routing softmax = cross-block allocation
    # MoE expert routing before the output rules: router names contain entropy/top-k.
    if n.startswith(("xrt_", "expert_routing_")): return Source.expert_routing
    # The path-signature siblings. Their prefixes are disjoint from every other family's names,
    # which is what keeps these two branches from reclassifying anything already banked.
    if n.startswith("out_sig"): return Source.output       # output-statistics path (entropy/margin/eos/varentropy)
    if n.startswith("attn_sig"): return Source.attention   # attention-region path (prompt/early/mid/recent[/sink] mass)
    # Checked before the keyword scan below, because these prefixes are specific and it is not.
    if n.startswith("value_"): return Source.values            # incl. value↔key coupling: a value property
    if n.startswith(("qk_", "q_")): return Source.qk           # query / q·k content geometry
    if n.startswith("kv_value"): return Source.values          # cross-layer value CKA (before the kv_→keys rule)
    if n.startswith("gate_"): return Source.gate
    if n.startswith("kv_") or n.startswith("epoch_"): return Source.keys      # key-vector geometry / key-centroid epochs
    if n.startswith(("cache_", "attn_flow_", "attn_entropy_", "head_agreement_", "ph_", "spectral_")):
        return Source.attention   # attention-weight reads, spectral_* among them: the similarity
        # graph `_extract_spectral_features` builds comes from attention distributions, not from
        # hidden states. smoothness is hybrid (attention graph × residual-norm signal) and is read
        # here, with its graph.
    if n.startswith(("activation_norm", "res_traj", "res_sig", "delta_", "pca_")):
        return Source.residual   # residual stream. res_sig_* takes iterated integrals OF the
        # residual trajectory, so the substrate it reads is the residual stream exactly as
        # res_traj_* does; only the operator differs.
    # output / token-distribution stats
    if any(k in n for k in ("logit", "surpris", "entropy", "token", "chosen", "top",
                            "perplex", "ppl", "prob", "rank")):
        return Source.output
    return Source.unknown


_GEOM = ("key_spread", "key_drift", "key_novelty", "eff_dim", "spread", "drift", "novelty",
         "cosine", "committed_cos", "res_traj", "participation", "curvature", "velocity", "align", "cka")
_DIST = ("entropy", "agreement", "coverage", "sink", "recency", "prompt_mass", "region",
         "diversity", "lookback", "sparsity", "top", "surpris", "mass", "decay", "anchor",
         "transition", "regularity", "stability", "role")


def _method(n: str, source: Source) -> Method:
    # The path-signature names carry no operator keyword the generic scan below would catch, so
    # they are matched first; falling through would flag the whole family as unplaced.
    if n.startswith(("res_sig_", "out_sig_", "attn_sig_")): return Method.iterated_integral
    if source == Source.routing: return Method.distributional      # routing summaries (entropy/top1/anchor/recency/eff_src)
    if source == Source.expert_routing:                            # MoE router allocation reads
        # Ahead of the generic _GEOM/_DIST scan, which "top"/"mass"/"drift"/"norm" would mis-route,
        # and geometry first within the branch so shared_routed_cos does not read as shared_mass.
        if any(k in n for k in ("cka", "cos", "eff_experts")): return Method.geometry
        if any(k in n for k in ("margin", "mass", "logit_norm")): return Method.magnitude
        if any(k in n for k in ("spectral", "period")): return Method.spectral
        return Method.distributional
    if n.startswith("pca_"): return Method.geometry
    if "spectral" in n or "fiedler" in n or "hfer" in n: return Method.spectral
    if "_norm" in n: return Method.magnitude                       # activation_norm / delta_norm — before distributional
    if any(k in n for k in _GEOM): return Method.geometry
    if any(k in n for k in _DIST): return Method.distributional
    if source == Source.output: return Method.distributional       # logit/token-distribution stats
    return Method.unknown


# Static (a level / snapshot) vs Dynamic (a change/dispersion over generation time). Token-matched so it
# is robust to the two naming patterns ({stat}_L{n} and L{n}_..._{stat}); bare measures = static levels.
_STATIC_TOK = {"mean", "traj0", "lvl1"}
_DYNAMIC_TOK = {"std", "slope", "traj1", "traj2", "traj3", "traj4", "drift", "switch",
                "churn", "flatness", "period", "lvl2"}
# A token belongs in these sets only when one family's names use it: a token shared with a second
# family silently reclassifies that family. "spectral" is the deliberate omission — the STFT features
# spell it too, and read it as a static snapshot. The path-signature tokens sit where they do because
# a level-1 log-signature term is the path's net displacement, invariant under permuting the
# increments, while the level-2 Levy areas exist only because of order and that shuffle destroys them.


def _dynamic(n: str) -> Optional[bool]:
    if n.startswith("pca_"): return False                          # per-feature PCA projection = snapshot
    toks = set(n.split("_"))
    if toks & _DYNAMIC_TOK: return True
    if toks & _STATIC_TOK: return False
    return False                                                   # bare level (recency_bias, sink_mass, key_spread, ...)


def _layer(n: str) -> Optional[int]:
    m = re.search(r"_L(\d+)", n)
    return int(m.group(1)) if m else None


def _band(layer: Optional[int], n_layers: int) -> Optional[Band]:
    if layer is None: return None
    if layer < n_layers / 3: return Band.early
    if layer < 2 * n_layers / 3: return Band.mid
    return Band.late


def classify(name: str, n_layers: int) -> FeatureTag:
    """Tag one feature name. ``n_layers`` is the denominator of its depth band."""
    if n_layers <= 0:
        raise UnknownDepthError(
            f"n_layers={n_layers}: a band is a fraction of the network, and a "
            "non-positive layer count would label every layer 'late'"
        )
    L = _layer(name)
    src = _source(name)
    return FeatureTag(name=name, source=src, method=_method(name, src), dynamic=_dynamic(name),
                      layer=L, band=_band(L, n_layers), family=stored_family(name))


# ---------------------------------------------------------------------------- the map

class FeatureMap:
    """Tags a fixed feature-name list and slices it by any axis or (source,method,depth) cell."""

    def __init__(self, names: list[str], n_layers: int):
        if n_layers <= 0:
            raise UnknownDepthError(
                f"n_layers={n_layers}: state the model's layer count, which is what "
                "the early/mid/late bands are cut on"
            )
        self.names = list(names)
        self.n_layers = n_layers
        self.tags = [classify(n, n_layers) for n in self.names]

    def __len__(self) -> int:
        return len(self.names)

    def _match(self, t: FeatureTag, source=None, method=None, dynamic=None, band=None, layer=None) -> bool:
        return ((source is None or t.source == source)
                and (method is None or t.method == method)
                and (dynamic is None or t.dynamic == dynamic)
                and (band is None or t.band == band)
                and (layer is None or t.layer == layer))

    def mask(self, **criteria) -> np.ndarray:
        """Boolean mask over features matching ALL given criteria (source/method/dynamic/band/layer)."""
        return np.array([self._match(t, **criteria) for t in self.tags], dtype=bool)

    def select(self, **criteria) -> list[str]:
        return [t.name for t in self.tags if self._match(t, **criteria)]

    def cells(self, *axes: str) -> dict[tuple, list[int]]:
        """Group feature INDICES by the given axes, e.g. cells('source','band') -> {(Source, Band): [idx...]}.
        The unit of the decompose-by-cell workflow: iterate, classify each cell's accuracy for a task."""
        out: dict[tuple, list[int]] = {}
        for i, t in enumerate(self.tags):
            key = tuple(getattr(t, a) for a in axes)
            out.setdefault(key, []).append(i)
        return dict(sorted(out.items(), key=lambda kv: (-len(kv[1]),)))

    def summary(self) -> dict:
        def counts(attr):
            c: dict = {}
            for t in self.tags:
                c[getattr(t, attr)] = c.get(getattr(t, attr), 0) + 1
            return dict(sorted(c.items(), key=lambda kv: -kv[1]))
        return {"n": len(self), "source": counts("source"), "method": counts("method"),
                "dynamic": counts("dynamic"), "band": counts("band")}

    def unclassified(self) -> list[str]:
        """Coverage check — features the classifier could not place (source/method unknown or dynamic None)."""
        return [t.name for t in self.tags
                if t.source == Source.unknown or t.method == Method.unknown or t.dynamic is None]


# ---------------------------------------------------------------------------- validation CLI

def _load_names(run: str, n_layers: int | None = None) -> tuple[list[str], int]:
    """One run's banked feature names, with the layer count its bands are cut on.

    ``n_layers`` is what a caller states for a model the map does not know; left
    out, :func:`layers_for_run` answers or refuses.
    """
    import os
    from pathlib import Path
    runs = Path(os.environ.get("ANAMNESIS_RUNS", "outputs/runs"))
    sd = runs / run / "signatures_v3"
    p = sorted(sd.glob("gen_*.npz"))[0]
    z = np.load(p, allow_pickle=True)
    names = [str(x) for x in z["feature_names"]]
    return names, layers_for_run(run) if n_layers is None else n_layers


def main():
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else "8b_fat_01"
    stated = int(sys.argv[2]) if len(sys.argv) > 2 else None
    names, nl = _load_names(run, stated)
    fm = FeatureMap(names, nl)
    s = fm.summary()
    print(f"=== {run}  n={s['n']}  n_layers={nl} ===")
    for axis in ("source", "method", "dynamic", "band"):
        print(f"  {axis:8s}: " + "  ".join(f"{getattr(k,'value',k)}={v}" for k, v in s[axis].items()))
    unc = fm.unclassified()
    print(f"  coverage: {len(names) - len(unc)}/{len(names)} classified; {len(unc)} flagged")
    if unc:
        print("  flagged (sample): " + ", ".join(unc[:15]))
    print("  source x band cells (count):")
    for (src, band), idx in fm.cells("source", "band").items():
        print(f"    {getattr(src,'value',src):10s} x {getattr(band,'value',band) if band else 'none':5s}: {len(idx)}")


if __name__ == "__main__":
    main()
