"""Column C: the FLOOR vector of one declared position (its depth path, read-profile and routing).

Column C reads a signature at a single token position, so it exists for spans of any length, one
token included. At a declared position it reads

* **depth**: the position's path through every layer (index 0 is the embedding or the first
  layer's output, then one point per layer, in the model's hidden-state convention);
* **read**: how that position's attention row reads the context, per layer: head entropy,
  head-mean entropy, sink mass, recency mass and prompt mass;
* **route** (AttnRes models only): the routing softmax at every routing site, split by relative
  source (emb, prev, older, self), with its lookback and normalised entropy.

Everything here is pure numpy. The reduction is testable without a model, and the same function
reduces a path whether a native, an HF or a vLLM capture produced it. The schema is fixed by the
model geometry alone (layer count, head count, routing layout), never by the row's length: a long
row computes exactly the names a one-token row does. Its digest names the schema.

Two schema versions share these reductions:

* **v0** (``floor_names``, ``floor_digest``) is the full set, and the one an extractor computes.
* **v1** (``floor_v1_names``, ``floor_v1_digest``) is v0 without the ``depth.band*`` and
  ``depth.area.*`` families, read at the last answer token captured in a one-token step
  (``FLOOR_V1_POSITION``). ``to_v1`` selects v1's columns from v0 rows, so a v0 bank holds v1.
  Capturing every position at a fixed shape keeps chunk-length rounding out of the vector: a
  position computed inside a chunk as long as the answer carries the answer's length in its last
  digits.

Feature families:

* ``depth.lognorm.P{i}``: log norm of path point i.
* ``depth.dlognorm.P{i}``: log norm of the step from point i-1 to i.
* ``depth.cos.P{i}``: cosine between points i-1 and i.
* ``depth.bend.P{i}``: cosine between the steps into and out of point i (curvature).
* ``depth.cosfinal.P{i}``: cosine between point i and the last point (alignment to the
  readout state).
* ``depth.peak.P{i}``: log of max|x| / rms(x) at point i, a channel-agnostic massive-activation
  index (it never names a channel).
* ``depth.band{k}`` (v0 only): share of the step sequence's power in band k of its DFT over the
  layer axis (band 0 is the DC term).
* ``depth.area.{f}_{g}`` (v0 only): signed (Levy) area between two scalar depth curves, the order
  in which they change along depth.
* ``read.{ent,agree,sink,sinkspread,recency,prompt}.L{l}``: per layer, mean head entropy (nats),
  head agreement (entropy of the head-mean row minus mean head entropy, the generalised JSD),
  head-mean sink mass, its spread over heads, head-mean recency mass (columns at or past
  floor(0.8 len)), and head-mean mass on the prompt (columns before the answer start).
* ``route.{site}.{emb,prev,older,self,lookback,ent}``: per routing site with two or more sources.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

FLOOR_VERSION = "column-c-floor-v0"
# v1 is a projection of v0: the same reductions at the last answer token captured in a one-token
# step, without the depth.band* and depth.area.* families. Extractors compute v0, the superset.
FLOOR_V1_VERSION = "column-c-floor-v1"
FLOOR_V1_POSITION = "alast_step"
V1_DROPPED_PREFIXES = ("depth.band", "depth.area.")
EPS = 1e-12
N_BANDS = 6
AREA_CURVES = ("lognorm", "cosfinal", "peak", "arclen")
READ_STATS = ("ent", "agree", "sink", "sinkspread", "recency", "prompt")
ROUTE_REL = ("emb", "prev", "older", "self")


@dataclass(frozen=True)
class RouteSite:
    """One routing site's layout: its name and the relative label of each source index."""

    name: str                      # e.g. "pre_attn.L12", "pre_mlp.L12", "final"
    rel: tuple[str, ...]           # relative label per source (ROUTE_REL)
    back: tuple[float, ...]        # layers back each source ends (self = 0)


@dataclass(frozen=True)
class Geometry:
    """What fixes the FLOOR schema: path length, layer count for the read-profile, routing sites."""

    model: str
    n_points: int                  # depth path points (e.g. 73 = embed + 72 layers)
    n_read_layers: int             # 0 when the capture carries no attention rows
    route_sites: tuple[RouteSite, ...] = field(default_factory=tuple)

    def to_json(self) -> dict:
        return dict(model=self.model, n_points=self.n_points, n_read_layers=self.n_read_layers,
                    route_sites=[[s.name, list(s.rel)] for s in self.route_sites])


# ── names and digest ───────────────────────────────────────────────────────────

def depth_names(n_points: int) -> list[str]:
    if n_points < 4:
        raise ValueError(f"a depth path needs at least 4 points, got {n_points}")
    p = range(n_points)
    names = [f"depth.lognorm.P{i}" for i in p]
    names += [f"depth.dlognorm.P{i}" for i in range(1, n_points)]
    names += [f"depth.cos.P{i}" for i in range(1, n_points)]
    names += [f"depth.bend.P{i}" for i in range(1, n_points - 1)]
    names += [f"depth.cosfinal.P{i}" for i in range(n_points - 1)]
    names += [f"depth.peak.P{i}" for i in p]
    names += [f"depth.band{k}" for k in range(N_BANDS)]
    names += [f"depth.area.{AREA_CURVES[a]}_{AREA_CURVES[b]}"
              for a in range(len(AREA_CURVES)) for b in range(a + 1, len(AREA_CURVES))]
    return names


def read_names(n_layers: int) -> list[str]:
    return [f"read.{s}.L{l}" for s in READ_STATS for l in range(n_layers)]


def route_names(sites: Sequence[RouteSite]) -> list[str]:
    names = []
    for s in sites:
        names += [f"route.{s.name}.{r}" for r in ROUTE_REL if r in s.rel]
        names += [f"route.{s.name}.lookback", f"route.{s.name}.ent"]
    return names


def floor_names(geo: Geometry) -> list[str]:
    return depth_names(geo.n_points) + read_names(geo.n_read_layers) + route_names(geo.route_sites)


def floor_digest(geo: Geometry) -> str:
    """sha256 over the version, the geometry and the ordered names: the FLOOR schema's identity."""
    blob = json.dumps(dict(version=FLOOR_VERSION, geometry=geo.to_json(), names=floor_names(geo)),
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def floor_v1_names(geo: Geometry) -> list[str]:
    return [n for n in floor_names(geo) if not n.startswith(V1_DROPPED_PREFIXES)]


def floor_v1_digest(geo: Geometry) -> str:
    """sha256 over v1's version, position, geometry and ordered names: the frozen schema's identity."""
    blob = json.dumps(dict(version=FLOOR_V1_VERSION, position=FLOOR_V1_POSITION, geometry=geo.to_json(),
                           names=floor_v1_names(geo)), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def to_v1(x: np.ndarray, v0_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """v1's columns of v0 rows (``[..., len(v0_names)]``), in v1 order; refuses names that are not v0's."""
    v0 = list(v0_names)
    if not any(n.startswith("depth.lognorm") for n in v0):
        raise ValueError("not a v0 FLOOR name list")
    keep = [i for i, n in enumerate(v0) if not n.startswith(V1_DROPPED_PREFIXES)]
    return np.asarray(x)[..., keep], [v0[i] for i in keep]


# ── the reductions ─────────────────────────────────────────────────────────────

def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), EPS)


def _levy_area(f: np.ndarray, g: np.ndarray) -> float:
    """Signed area of the planar path (f, g) against its chord start: 0.5 * sum(f df' - g df)."""
    f0, g0 = f - f[0], g - g[0]
    df, dg = np.diff(f), np.diff(g)
    return float(0.5 * (f0[:-1] * dg - g0[:-1] * df).sum())


def depth_curves(path: np.ndarray) -> dict[str, np.ndarray]:
    """Per-point scalar curves of a path [points, d] (float64 inside)."""
    x = np.asarray(path, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f"depth path must be a finite [points, d] array, got {x.shape}")
    norms = np.linalg.norm(x, axis=1)
    steps = np.diff(x, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    rms = norms / np.sqrt(x.shape[1])
    arclen = np.concatenate([[0.0], np.cumsum(step_norms)])
    return dict(
        lognorm=np.log(norms + EPS),
        dlognorm=np.log(step_norms + EPS),
        cos=_cos(x[1:], x[:-1]),
        bend=_cos(steps[1:], steps[:-1]),
        cosfinal=np.concatenate([_cos(x[:-1], x[-1:]), [1.0]]),
        peak=np.log(np.abs(x).max(axis=1) / np.maximum(rms, EPS) + EPS),
        arclen=arclen / max(arclen[-1], EPS),
        steps=steps,
    )


def band_energies(steps: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
    """Share of the step sequence's DFT power (over the layer axis, summed over dims) per band."""
    spec = np.abs(np.fft.rfft(steps, axis=0)) ** 2           # [freqs, d]
    power = spec.sum(axis=1)
    total = max(power.sum(), EPS)
    out = np.zeros(n_bands)
    out[0] = power[0] / total
    rest = np.array_split(np.arange(1, len(power)), n_bands - 1)
    for k, idx in enumerate(rest, start=1):
        out[k] = power[idx].sum() / total if len(idx) else 0.0
    return out


def depth_features(path: np.ndarray) -> np.ndarray:
    """The depth family of one position, in ``depth_names(len(path))`` order."""
    c = depth_curves(path)
    areas = [_levy_area(c[AREA_CURVES[a]], c[AREA_CURVES[b]])
             for a in range(len(AREA_CURVES)) for b in range(a + 1, len(AREA_CURVES))]
    return np.concatenate([c["lognorm"], c["dlognorm"], c["cos"], c["bend"], c["cosfinal"][:-1],
                           c["peak"], band_energies(c["steps"]), areas])


def read_features(head_ent: np.ndarray, mean_ent: np.ndarray, sink: np.ndarray,
                  recency: np.ndarray, prompt: np.ndarray) -> np.ndarray:
    """The read-profile of one position, in ``read_names(L)`` order.

    ``head_ent, sink, recency, prompt``: [L, H]; ``mean_ent``: [L] (entropy of the head-mean row).
    """
    he = np.asarray(head_ent, dtype=np.float64)
    if he.ndim != 2:
        raise ValueError(f"head_ent must be [L, H], got {he.shape}")
    mh = he.mean(axis=1)
    sk = np.asarray(sink, dtype=np.float64)
    return np.concatenate([mh, np.asarray(mean_ent, dtype=np.float64) - mh, sk.mean(axis=1),
                           sk.std(axis=1), np.asarray(recency, np.float64).mean(axis=1),
                           np.asarray(prompt, np.float64).mean(axis=1)])


def route_features(weights: Sequence[np.ndarray], sites: Sequence[RouteSite]) -> np.ndarray:
    """Routing at one position: ``weights[k]`` is site k's softmax over its sources [n_src]."""
    out = []
    for w, s in zip(weights, sites, strict=True):
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (len(s.rel),):
            raise ValueError(f"site {s.name}: {w.shape} weights for {len(s.rel)} sources")
        out += [float(w[[j for j, r in enumerate(s.rel) if r == rel]].sum())
                for rel in ROUTE_REL if rel in s.rel]
        out.append(float((w * np.asarray(s.back)).sum()))
        out.append(float(-(w * np.log(w + 1e-10)).sum() / np.log(len(w))))
    return np.asarray(out)


def floor_vector(geo: Geometry, path: np.ndarray, read: dict | None = None,
                 route: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """One position's FLOOR vector in ``floor_names(geo)`` order; refuses a mismatched geometry."""
    if path.shape[0] != geo.n_points:
        raise ValueError(f"path has {path.shape[0]} points, geometry says {geo.n_points}")
    parts = [depth_features(path)]
    if geo.n_read_layers:
        if read is None:
            raise ValueError("geometry declares a read-profile; none given")
        parts.append(read_features(**read))
    if geo.route_sites:
        if route is None:
            raise ValueError("geometry declares routing sites; none given")
        parts.append(route_features(route, geo.route_sites))
    vec = np.concatenate(parts)
    if len(vec) != len(floor_names(geo)):
        raise AssertionError(f"vector {len(vec)} != schema {len(floor_names(geo))}")
    return vec
