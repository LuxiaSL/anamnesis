"""Feature map — the (SOURCE x METHOD x DEPTH) taxonomy, the vocabulary of record for
describing what a feature reads.

A feature's cell here is what says which substrate it reads. The contiguous blocks a
signature vector is addressed in — `extract_norms_and_output_stats`..`extract_residual_pca`
in `anamnesis/extraction/state_extractor.py` build them, and
`anamnesis/analysis/gauntlet/signature_io.py` reads them back — are addresses into the
vector and nothing more. Three of the four span several sources at once, so a block's
accuracy is not a reading of any single substrate and no claim is stated per block. Address a
stored artifact by its block; describe a feature by its cell.

This module tags every signature feature by the axes that ARE interpretable and were
empirically validated on the merged v3 corpus (2026-06-14):

  SOURCE   = which substrate is read.   Ranked (LDA, model-stable): attention >> residual > gate > keys > output.
  METHOD   = the base operator (magnitude / distributional / geometry / spectral / learned /
             iterated_integral [added 2026-09-11 for the path-signature family]).
  DYNAMIC  = the temporal wrapper: static (a *_mean / snapshot) vs dynamic (*_std / slope / trajectory /
             window / drift / novelty). [modes ≈ average level (static) ≥ dynamics, at n≈900.]
  DEPTH    = layer + band (early/mid/late). [mode signal concentrates at MID layers.]

This is the reusable substrate for the discover→decompose→distill workflow:
  raw -> encoder (per-model discovery, all sources) -> decompose by CELL (this map) -> which (source,method,
  depth) cells carry THIS task -> instantiate theory-motivated features for those cells -> portable,
  lightweight signature. Redundancy is task-specific, so you re-decompose per task; the cells are the unit.

Design: pure-numpy/pydantic (no torch/sklearn — importable anywhere, like state_extractor). Classification
is name-based and TRANSPARENT — `FeatureMap.unclassified()` and `.summary()` expose every call so it can be
audited/overridden. (Long-term ideal: tag at generation time; this is the pragmatic post-hoc parser
over the frozen v3 names.)

  >>> fm = FeatureMap(feature_names, n_layers=32)
  >>> X_attn_mid = X[:, fm.mask(source=Source.attention, band=Band.mid)]   # slice a cell
  >>> for cell, idx in fm.cells("source", "band").items(): acc[cell] = lda(X[:, idx], y)   # decompose
  >>> lean = fm.select(source=Source.attention)            # the ~600-780-feat lean mode/taste signature

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
                "dsv2": 27, "dsv2_lite": 27, "dsv2-lite": 27}  # M6 DeepSeek-V2-Lite (27 layers)


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
    values = "values"          # v_proj (future — banked, not yet featurized)
    qk = "qk"                  # post-RoPE QK geometry (future)
    routing = "routing"        # cross-block attention-residual routing weights (block-routing architectures)
    expert_routing = "expert_routing"  # MoE router: expert-allocation reads. Distinct from
                               # `routing` above, which is cross-block attention-residual routing:
                               # one reads which expert a token was sent to, the other how much of
                               # its own history a block attended over. Mixture-of-experts models
                               # only; a dense model has no expert allocation to read.
    unknown = "unknown"        # flagged: classifier did not match (audit these)


class Method(str, Enum):
    magnitude = "magnitude"            # norms / means of magnitudes
    distributional = "distributional"  # entropy, JSD/agreement, top-k mass, coverage, mass fractions, sparsity
    geometry = "geometry"              # cosine / spread / drift / novelty / participation-ratio / PCA projection
    spectral = "spectral"              # graph-spectral (Fiedler, HFER, spectral entropy, smoothness)
    learned = "learned"                # contrastive projection / encoder (future in-signature)
    iterated_integral = "iterated_integral"  # rough-path log-signature: level-1 displacement +
                                       # level-2 Levy areas of a (projected or natively low-dim),
                                       # time-augmented trajectory. NEW axis value (SPEC-path-
                                       # signature-family-2026-09-11 section 5): the other methods
                                       # are all MARGINAL reads of one series at a time; this one is
                                       # the JOINT order of two coordinates (did i move before j).
                                       # Emitted by `res_sig_` (residual path, projected), and by its
                                       # two no-projection siblings `out_sig_` (output-statistics
                                       # path) / `attn_sig_` (attention-region path), per
                                       # SPEC-path-receptacles-and-span-coverage-2026-09-11 §1a/§1b.
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
                                 # numbers by; see `stored_family()` for the label set


# ---------------------------------------------------------------------------- classification rules

# The family labels that follow the stored block layout rather than naming a family.
# They are a wire vocabulary, not a description: `anamnesis/analysis/battery/floors.py`
# keys its floor results by `family:<label>`, so a label that changed would stop lining
# up with the numbers already banked under it. The constants carry what each one groups;
# the strings stay where they belong, on disk.
STORED_FAMILY_CACHE_AND_KEYS = "T2.5"
STORED_FAMILY_ATTENTION_SPECTRAL = "T2_spectral"
STORED_FAMILY_ATTENTION_OTHER = "T2_other"
STORED_FAMILY_RESIDUAL_PCA = "T3"
STORED_FAMILY_NORMS_AND_OUTPUT_STATS = "T1"


def stored_family(n: str) -> str:
    """The family label a feature is grouped under in banked per-family results.

    Five labels follow the stored block layout and are held as the STORED_FAMILY_*
    constants above; the rest name their family directly. What a feature reads is
    answered by `classify()` and its (source, method, depth) cell, never by this.
    """
    if n.startswith("value_"): return "value_geometry"
    if n.startswith(("qk_", "q_")): return "qk_geometry"
    if "cka" in n: return "kv_cka"
    if n.startswith("ph_"): return "per_head"
    if n.startswith("attn_flow_"): return "attention_flow"
    if n.startswith("gate_"): return "gate"
    if n.startswith("res_sig"): return "path_signature"   # level-2 log-signature (new; no legacy analog)
    if n.startswith("out_sig"): return "path_signature_output"      # output-stats path source (2026-09-11 §1a)
    if n.startswith("attn_sig"): return "path_signature_attention"  # attention-region path source (2026-09-11 §1b)
    if n.startswith("res_traj"): return "residual_traj"
    if n.startswith(("cache_", "kv_", "epoch_")): return STORED_FAMILY_CACHE_AND_KEYS
    if n.startswith("spectral_"): return STORED_FAMILY_ATTENTION_SPECTRAL
    if n.startswith(("attn_entropy_", "head_agreement_", "delta_")): return STORED_FAMILY_ATTENTION_OTHER
    if n.startswith("attnres_"): return "attn_res"
    if n.startswith(("xrt_", "expert_routing_")): return "expert_routing"  # M6 MoE router (new; no legacy analog)
    if n.startswith("pca_"): return STORED_FAMILY_RESIDUAL_PCA
    return STORED_FAMILY_NORMS_AND_OUTPUT_STATS


def _source(n: str) -> Source:
    # Attention-residual routing FIRST — its names contain "entropy"/"top" which would else mis-hit output.
    if n.startswith("attnres_committed"): return Source.residual   # committed residual-block snapshots (geometry)
    if n.startswith("attnres_"): return Source.routing             # block-routing softmax = cross-block allocation
    # MoE expert routing (vmb arm A7) — before output rules: router names will contain entropy/top-k.
    if n.startswith(("xrt_", "expert_routing_")): return Source.expert_routing  # reserved prefix for A7 router features
    # Path-signature SIBLING sources (SPEC-path-receptacles-and-span-coverage-2026-09-11 §1a/§1b):
    # out_sig_ / attn_sig_ are new prefixes disjoint from every existing family's names, so adding
    # these two branches cannot reclassify anything in the frozen corpus (regression bar, spec §3).
    if n.startswith("out_sig"): return Source.output       # output-statistics path (entropy/margin/eos/varentropy)
    if n.startswith("attn_sig"): return Source.attention   # attention-region path (prompt/early/mid/recent[/sink] mass)
    # v3 hand-suite (checked first; these prefixes are specific). value_* incl value↔key corr = a value prop.
    if n.startswith("value_"): return Source.values            # v_proj value-vector geometry
    if n.startswith(("qk_", "q_")): return Source.qk           # query / q·k content geometry
    if n.startswith("kv_value"): return Source.values          # cross-layer value CKA (before the kv_→keys rule)
    if n.startswith("gate_"): return Source.gate
    if n.startswith("kv_") or n.startswith("epoch_"): return Source.keys      # key-vector geometry / key-centroid epochs
    if n.startswith(("cache_", "attn_flow_", "attn_entropy_", "head_agreement_", "ph_", "spectral_")):
        return Source.attention   # attention-weight reads, spectral_* among them:
        # (addendum 2026-07-12b): the similarity graph is built from ATTENTION distributions
        # (_extract_spectral_features), not hidden states — the old comment sided with a stale
        # docstring; the code's behavior says attention. smoothness is hybrid (attention graph
        # × residual-norm signal) and rides with its graph. Pre-retag analyses counted these
        # 66 features under residual — cross-date family-mass comparisons carry that asterisk.
    if n.startswith(("activation_norm", "res_traj", "res_sig", "delta_", "pca_")):
        return Source.residual   # residual stream. res_sig_* = the path-signature family:
        # iterated integrals OF the residual trajectory — the substrate read is the residual
        # stream, exactly as res_traj_*; only the operator is new.
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
    # Path-signature family (+ its two sibling sources) FIRST — none of these names
    # carry an operator keyword the generic scan would catch, and letting them fall through to
    # Method.unknown would flag the whole family. New prefixes only (regression-safe, see _source).
    if n.startswith(("res_sig_", "out_sig_", "attn_sig_")): return Method.iterated_integral
    if source == Source.routing: return Method.distributional      # routing summaries (entropy/top1/anchor/recency/eff_src)
    if source == Source.expert_routing:                            # MoE router allocation reads (arm A7, M6). Spec §3 + v2.1:
        # Placed BEFORE the generic _GEOM/_DIST scan (else "top"/"mass"/"drift"/"norm" mis-route). All four
        # non-learned methods now present (v2.1 48870af4): geometry (cka/cos/eff_experts), magnitude
        # (margin/mass/logit_norm), spectral (spectral/period), else distributional (entropy/KL/JSD/coverage/
        # switch/churn/topk_weight). Order matters: geometry first so shared_routed_cos ≠ shared_mass.
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
# "lvl1"/"lvl2" (path-signature family): level-1 log-signature terms are the path's
# NET DISPLACEMENT — a level, and literally invariant under the increment-permutation null, so
# static is the correct reading. Level-2 Levy areas exist only because of order and are destroyed
# by that same shuffle — dynamic, and the sharpest example of it in the suite. Both tokens are
# res_sig-UNIQUE across the frozen v3 name corpus, so the sets stay safe.
# "switch" (xrt_switch_rate), and v2.1 (48870af4): "churn" (xrt_set_churn_rate), "flatness"
# (xrt_alloc_entropy_spectral_flatness), "period" (xrt_switch_dominant_period) — all per-step temporal
# reads. Each token is xrt-UNIQUE (deliberately NOT "spectral", which the corpus-wide stft features use
# as a static snapshot — adding it globally would reclassify 3B/8B spectral features), so the set stays safe.


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
