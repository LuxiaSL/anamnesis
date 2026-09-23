"""The mode registry: runnable sets, banked label vocabularies, and the maps between.

:data:`MODE_SETS_FILE` is the registry, a JSON file beside this module, and
:data:`MODE_SETS_ENV` names further files merged over it. So a researcher runs an
extraction over their own modes by writing rows, and nothing on that path is a
code change.

Three kinds of row, because a corpus can carry a label this package cannot produce:

**A mode set** is runnable. Each mode carries an instruction and an index, and the
set carries the format constraint appended to every one of its prompts — flowing
prose, no lists, no headers. That clause is what makes a set an instrument rather
than a style sampler: without it a classifier separates the modes on surface
layout alone and the question of whether the *computation* differs never gets
asked. A set may ``extend`` another, which takes the parent's modes and indices
unchanged and adds to them; the parent's rows are not restated, so the shared
prompts are byte-identical and the shared indices cannot drift.

**A label vocabulary** is names and glosses with no prompts: what a banked corpus
labels its generations with when the protocol that produced it is not shipped
here. It exists so a stored label and a transfer prediction resolve to something a
reader can read instead of to nothing.

**A mode mapping** predicts where each label of one vocabulary lands among
another's. The reverse direction is the inverse of the pairs, and each side's
unpaired label is its wildcard, so neither is written down twice.

Two things are frozen and the loader treats them as such:

* **A mode's index is part of the reproducibility contract.** It reaches
  :func:`anamnesis.extraction.generation_runner.make_seed`, so the generation a
  coordinate names depends on it. An index is therefore declared in the data and
  never derived from a dict's order or a sort.
* **A set's name reaches the seed too**, through the prompt-set stamp a run is
  built with, and a mode's prompt is what every stored label of that name means.

Which is why merging is additive and refuses collisions: a file adds sets,
vocabularies and mappings, and cannot redefine one already here. A variant of a
shipped set is a new set. A file named in the environment that cannot be read is
an error rather than a fall-through, because a silent skip would resolve the
shipped name and generate under the wrong prompt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

MODE_SETS_FILE: Path = Path(__file__).resolve().parent / "mode_sets.json"
"""The registry file this package ships."""

MODE_SETS_ENV = "ANAMNESIS_MODE_SETS"
"""Environment variable naming further registry files, separated like ``PATH``."""

CORE_MODE_SET = "run4"
"""The five format-controlled modes: the set of the core protocol."""

EXTENDED_MODE_SET = "mixed"
"""The five plus the three format-controlled additions."""

DEFAULT_MODE_MAPPING = "process_to_format"
"""The mapping a cross-vocabulary readout reads when its caller names none."""


class ModeRegistryError(RuntimeError):
    """A registry file is missing, unreadable, or not what this module expects."""


class UnknownModeSetError(KeyError):
    """A mode set, label vocabulary or mapping name the registry does not hold."""


class Mode(BaseModel):
    """One mode: its label, its position in the label order, and what it asks for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="The label a generation of this mode is stored under")
    index: int = Field(
        ge=0,
        description=(
            "Position in the label order; part of the generation seed, so it is "
            "declared here rather than derived from this file's order"
        ),
    )
    instruction: str = Field(description="The system prompt, before the format constraint")

    @field_validator("name", "instruction")
    @classmethod
    def _not_blank(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("a mode's name and instruction are both non-empty")
        return text


class ModeSetSpec(BaseModel):
    """One mode-set row as a registry file holds it, before any parent is resolved."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Registry key for this set, and the stamp a run carries")
    description: str = Field(default="", description="What this set is, and what it is for")
    extends: str | None = Field(
        default=None, description="A set whose modes and indices this one continues"
    )
    hard: bool = Field(
        default=False,
        description="Whether a pair drawn from this set's own modes is a hard pair",
    )
    format_constraint: str | None = Field(
        default=None,
        description=(
            "Appended to every prompt of this set; inherited from the parent when a set "
            "extends one, and declaring a different clause there is a different protocol"
        ),
    )
    modes: tuple[Mode, ...] = Field(description="This set's own modes, parent's excluded")

    @model_validator(mode="after")
    def _own_modes_are_coherent(self) -> Self:
        if not self.modes:
            raise ValueError(f"mode set {self.name!r} declares no modes")
        names = [mode.name for mode in self.modes]
        if len(set(names)) != len(names):
            raise ValueError(f"mode set {self.name!r} names a mode twice: {names}")
        indices = [mode.index for mode in self.modes]
        if len(set(indices)) != len(indices):
            raise ValueError(f"mode set {self.name!r} uses an index twice: {indices}")
        if self.extends is None and self.format_constraint is None:
            raise ValueError(
                f"mode set {self.name!r} extends nothing and declares no format_constraint; "
                "state the clause appended to its prompts, or the empty string to append none"
            )
        if self.extends is not None and self.format_constraint is not None:
            raise ValueError(
                f"mode set {self.name!r} extends {self.extends!r} and declares its own "
                "format_constraint; an extending set inherits the clause, and a different "
                "clause is a different protocol and so its own set"
            )
        return self


class ModeSet(BaseModel):
    """A mode set with its parent's modes folded in: the whole protocol, in label order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    hard: bool
    format_constraint: str
    modes: tuple[Mode, ...] = Field(description="Every mode of this set, ordered by index")
    extends: str | None = None

    @model_validator(mode="after")
    def _indices_are_the_label_order(self) -> Self:
        indices = [mode.index for mode in self.modes]
        if indices != list(range(len(indices))):
            raise ValueError(
                f"mode set {self.name!r} has indices {indices}; a label order is "
                f"0..{len(indices) - 1} with no gaps, because an index is a position "
                "in a label vector as well as a seed coordinate"
            )
        return self

    def names(self) -> tuple[str, ...]:
        """Every mode name, in label order."""
        return tuple(mode.name for mode in self.modes)

    def prompts(self) -> dict[str, str]:
        """Mode name to system prompt, in label order, format constraint appended."""
        return {mode.name: mode.instruction + self.format_constraint for mode in self.modes}

    def indices(self) -> dict[str, int]:
        """Mode name to label index, in label order."""
        return {mode.name: mode.index for mode in self.modes}

    def prompt(self, mode: str) -> str:
        """One mode's system prompt.

        Raises
        ------
        UnknownModeSetError
            When this set has no mode of that name.
        """
        try:
            return self.prompts()[mode]
        except KeyError:
            raise UnknownModeSetError(
                f"mode set {self.name!r} has no mode {mode!r}; it holds {self.names()}"
            ) from None


class Label(BaseModel):
    """One label of a vocabulary this package cannot run: the name, and what it names."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    gloss: str = Field(description="What the label names, at the resolution this package can state")

    @field_validator("name", "gloss")
    @classmethod
    def _not_blank(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("a label's name and gloss are both non-empty")
        return text


class LabelVocabulary(BaseModel):
    """Labels a banked corpus carries whose prompts are not in this package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""
    runnable_note: str = Field(
        default="",
        description="Why an extraction cannot produce this vocabulary, for a reader who tries",
    )
    labels: tuple[Label, ...]

    @model_validator(mode="after")
    def _labels_are_distinct(self) -> Self:
        names = [label.name for label in self.labels]
        if not names:
            raise ValueError(f"label vocabulary {self.name!r} holds no labels")
        if len(set(names)) != len(names):
            raise ValueError(f"label vocabulary {self.name!r} names a label twice: {names}")
        return self

    def names(self) -> tuple[str, ...]:
        """Every label, in declaration order."""
        return tuple(label.name for label in self.labels)

    def gloss(self, label: str) -> str:
        """What one label names.

        Raises
        ------
        UnknownModeSetError
            When this vocabulary has no label of that name.
        """
        for row in self.labels:
            if row.name == label:
                return row.gloss
        raise UnknownModeSetError(
            f"vocabulary {self.name!r} has no label {label!r}; it holds {self.names()}"
        )


class MappingReference(BaseModel):
    """A measured pass this mapping's later passes are read beside."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(description="The model the comparison was measured on")
    description: str = ""
    forward_pair: str = Field(description="The source label whose forward rate is quoted")
    forward_pair_accuracy: float = Field(ge=0.0, le=1.0)
    reverse_pair: str = Field(description="The target label whose reverse rate is quoted")
    reverse_pair_accuracy: float = Field(ge=0.0, le=1.0)
    projected_silhouette: float = Field(
        description="Silhouette of the projected run inside the fitted run's discriminant space"
    )
    fitted_silhouette: float = Field(description="The fitted run's silhouette in its own space")


class ModeMapping(BaseModel):
    """Where each label of one vocabulary is predicted to land in another's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""
    source: str = Field(description="The vocabulary the left-hand labels come from")
    target: str = Field(description="The vocabulary the right-hand labels come from")
    pairs: dict[str, str] = Field(description="Source label to its predicted target label")
    reference: MappingReference | None = None

    @model_validator(mode="after")
    def _pairs_are_injective(self) -> Self:
        if not self.pairs:
            raise ValueError(f"mapping {self.name!r} declares no pairs")
        targets = list(self.pairs.values())
        if len(set(targets)) != len(targets):
            raise ValueError(
                f"mapping {self.name!r} sends two source labels to {sorted(targets)}; "
                "the reverse direction is this mapping inverted, so the pairs are one to one"
            )
        if self.source == self.target:
            raise ValueError(f"mapping {self.name!r} maps {self.source!r} onto itself")
        return self

    def reverse_pairs(self) -> dict[str, str]:
        """The mapping inverted: target label to its predicted source label."""
        return {target: source for source, target in self.pairs.items()}


class ModeRegistryFile(BaseModel):
    """The shape of one registry file."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    mode_sets: dict[str, ModeSetSpec] = Field(default_factory=dict)
    label_vocabularies: dict[str, LabelVocabulary] = Field(default_factory=dict)
    mode_mappings: dict[str, ModeMapping] = Field(default_factory=dict)


class ModeRegistry(BaseModel):
    """Every registry file, merged, with each set's parent folded in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode_sets: dict[str, ModeSet]
    label_vocabularies: dict[str, LabelVocabulary]
    mode_mappings: dict[str, ModeMapping]
    sources: tuple[Path, ...] = Field(description="The files this was read from, in merge order")

    def set_names(self) -> tuple[str, ...]:
        """Every mode-set name, in merge order."""
        return tuple(self.mode_sets)

    def mode_set(self, name: str) -> ModeSet:
        """One mode set, parent folded in.

        Raises
        ------
        UnknownModeSetError
            Naming every set the registry holds, and the files it read.
        """
        try:
            return self.mode_sets[name]
        except KeyError:
            raise UnknownModeSetError(
                f"unknown mode set {name!r}; sets: {', '.join(self.mode_sets)}; "
                f"read from: {', '.join(str(path) for path in self.sources)}; "
                f"add a set and name its file in {MODE_SETS_ENV}"
            ) from None

    def vocabulary(self, name: str) -> LabelVocabulary:
        """One label vocabulary, which a mode set also is.

        A mode set answers here as a vocabulary of its own mode names, so a mapping
        can name either kind on either side.

        Raises
        ------
        UnknownModeSetError
            Naming every vocabulary and set the registry holds.
        """
        if name in self.label_vocabularies:
            return self.label_vocabularies[name]
        if name in self.mode_sets:
            row = self.mode_sets[name]
            return LabelVocabulary(
                name=row.name,
                description=row.description,
                labels=tuple(
                    Label(name=mode.name, gloss=mode.instruction) for mode in row.modes
                ),
            )
        raise UnknownModeSetError(
            f"unknown label vocabulary {name!r}; "
            f"vocabularies: {', '.join(self.label_vocabularies)}; "
            f"mode sets: {', '.join(self.mode_sets)}"
        )

    def mapping(self, name: str) -> ModeMapping:
        """One mode mapping.

        Raises
        ------
        UnknownModeSetError
            Naming every mapping the registry holds.
        """
        try:
            return self.mode_mappings[name]
        except KeyError:
            raise UnknownModeSetError(
                f"unknown mode mapping {name!r}; mappings: {', '.join(self.mode_mappings)}"
            ) from None

    def unpaired(self, name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """One mapping's labels with no partner, source side then target side.

        A label with no predicted partner is the wildcard of its direction: where it
        lands is a finding rather than an error, so it is reported and never scored.
        """
        mapping = self.mapping(name)
        source = self.vocabulary(mapping.source).names()
        target = self.vocabulary(mapping.target).names()
        return (
            tuple(label for label in source if label not in mapping.pairs),
            tuple(label for label in target if label not in mapping.reverse_pairs()),
        )

    def hard_modes(self) -> frozenset[str]:
        """Every mode of a set declared hard: a pair drawn from these is a hard pair."""
        return frozenset(
            name for row in self.mode_sets.values() if row.hard for name in row.names()
        )

    def easy_modes(self) -> frozenset[str]:
        """Every mode of a set not declared hard, and not in one that is."""
        everything = frozenset(
            name for row in self.mode_sets.values() for name in row.names()
        )
        return everything - self.hard_modes()


def registry_paths() -> tuple[Path, ...]:
    """The shipped registry file, then every file named in :data:`MODE_SETS_ENV`.

    The environment value is read at call time rather than at import, so a process
    can point at another registry without reloading the package.
    """
    paths = [MODE_SETS_FILE]
    for entry in os.environ.get(MODE_SETS_ENV, "").split(os.pathsep):
        text = entry.strip()
        if text:
            paths.append(Path(text).expanduser())
    return tuple(paths)


def _read_file(path: Path) -> ModeRegistryFile:
    """One registry file, parsed and validated.

    Raises
    ------
    ModeRegistryError
        When the file is absent, is not readable as JSON, does not match the
        registry shape, or holds a row whose ``name`` disagrees with its key.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModeRegistryError(f"mode registry unreadable: {path} ({exc})") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModeRegistryError(f"{path}: invalid JSON at line {exc.lineno} ({exc.msg})") from exc
    try:
        parsed = ModeRegistryFile.model_validate(payload)
    except ValidationError as exc:
        raise ModeRegistryError(f"{path}: not a mode registry ({exc})") from exc
    for kind, rows in (
        ("mode set", parsed.mode_sets),
        ("label vocabulary", parsed.label_vocabularies),
        ("mode mapping", parsed.mode_mappings),
    ):
        for key, row in rows.items():
            if row.name != key:
                raise ModeRegistryError(
                    f"{path}: {kind} {key!r} carries name {row.name!r}; "
                    "the key and the name are one thing"
                )
    return parsed


def _resolve_set(
    name: str, specs: dict[str, ModeSetSpec], seen: tuple[str, ...] = ()
) -> ModeSet:
    """One spec with its parent chain folded in, indices and names checked across it.

    Raises
    ------
    ModeRegistryError
        When a parent is missing or the chain is a cycle, or when a child restates a
        parent's mode name or reuses one of its indices.
    """
    if name in seen:
        raise ModeRegistryError(
            f"mode set {name!r} extends itself through {' -> '.join([*seen, name])}"
        )
    spec = specs[name]
    if spec.extends is None:
        return ModeSet(
            name=spec.name,
            description=spec.description,
            hard=spec.hard,
            format_constraint=spec.format_constraint or "",
            modes=tuple(sorted(spec.modes, key=lambda mode: mode.index)),
            extends=None,
        )
    if spec.extends not in specs:
        raise ModeRegistryError(
            f"mode set {name!r} extends {spec.extends!r}, which no registry file defines; "
            f"sets: {', '.join(sorted(specs))}"
        )
    parent = _resolve_set(spec.extends, specs, (*seen, name))
    clashing_names = sorted(set(parent.names()) & {mode.name for mode in spec.modes})
    if clashing_names:
        raise ModeRegistryError(
            f"mode set {name!r} restates {clashing_names} from {parent.name!r}; an extending "
            "set adds modes, because a restated prompt is a second text one label could mean"
        )
    clashing_indices = sorted(
        set(parent.indices().values()) & {mode.index for mode in spec.modes}
    )
    if clashing_indices:
        raise ModeRegistryError(
            f"mode set {name!r} reuses indices {clashing_indices} from {parent.name!r}; "
            "an index is a seed coordinate, so the parent's keep their meaning"
        )
    return ModeSet(
        name=spec.name,
        description=spec.description,
        hard=spec.hard,
        format_constraint=parent.format_constraint,
        modes=tuple(sorted([*parent.modes, *spec.modes], key=lambda mode: mode.index)),
        extends=spec.extends,
    )


def _merge(files: list[tuple[Path, ModeRegistryFile]]) -> ModeRegistry:
    """Merge registry files in order, refusing any name two of them claim.

    Raises
    ------
    ModeRegistryError
        On a collision, naming the row and both files; on a vocabulary shadowing a
        mode set; on a mapping naming a vocabulary that is not there, or pairing a
        label neither side holds; on a reference quoting an unpaired label.
    """
    specs: dict[str, ModeSetSpec] = {}
    vocabularies: dict[str, LabelVocabulary] = {}
    mappings: dict[str, ModeMapping] = {}
    owner: dict[tuple[str, str], Path] = {}

    def claim(kind: str, key: str, path: Path) -> None:
        held = owner.get((kind, key))
        if held is not None:
            raise ModeRegistryError(
                f"{path}: {kind} {key!r} is already defined in {held}; a registry file adds "
                f"rows and never redefines them, because a mode's prompt and its index are "
                f"what banked labels already mean"
            )
        owner[(kind, key)] = path

    for path, parsed in files:
        for key, spec in parsed.mode_sets.items():
            claim("mode set", key, path)
            specs[key] = spec
        for key, vocabulary in parsed.label_vocabularies.items():
            claim("label vocabulary", key, path)
            vocabularies[key] = vocabulary
        for key, mapping in parsed.mode_mappings.items():
            claim("mode mapping", key, path)
            mappings[key] = mapping

    shadowed = sorted(set(specs) & set(vocabularies))
    if shadowed:
        raise ModeRegistryError(
            f"{shadowed} are declared as both a mode set and a label vocabulary; a mode set "
            "already answers as the vocabulary of its own mode names"
        )

    resolved: dict[str, ModeSet] = {}
    for name in specs:
        try:
            resolved[name] = _resolve_set(name, specs)
        except ValidationError as exc:
            raise ModeRegistryError(
                f"{owner[('mode set', name)]}: mode set {name!r} does not resolve ({exc})"
            ) from exc
    registry = ModeRegistry(
        mode_sets=resolved,
        label_vocabularies=vocabularies,
        mode_mappings=mappings,
        sources=tuple(path for path, _ in files),
    )

    for mapping in mappings.values():
        source = registry.vocabulary(mapping.source)
        target = registry.vocabulary(mapping.target)
        unknown_sources = sorted(set(mapping.pairs) - set(source.names()))
        if unknown_sources:
            raise ModeRegistryError(
                f"mapping {mapping.name!r} pairs {unknown_sources}, which {source.name!r} "
                f"does not hold; it holds {source.names()}"
            )
        unknown_targets = sorted(set(mapping.pairs.values()) - set(target.names()))
        if unknown_targets:
            raise ModeRegistryError(
                f"mapping {mapping.name!r} predicts {unknown_targets}, which {target.name!r} "
                f"does not hold; it holds {target.names()}"
            )
        if mapping.reference is not None:
            if mapping.reference.forward_pair not in mapping.pairs:
                raise ModeRegistryError(
                    f"mapping {mapping.name!r} quotes a forward rate for "
                    f"{mapping.reference.forward_pair!r}, which it does not pair"
                )
            if mapping.reference.reverse_pair not in mapping.reverse_pairs():
                raise ModeRegistryError(
                    f"mapping {mapping.name!r} quotes a reverse rate for "
                    f"{mapping.reference.reverse_pair!r}, which it does not pair"
                )

    return registry


_CACHE: dict[tuple[tuple[str, int, int], ...], ModeRegistry] = {}


def load_mode_registry(paths: tuple[Path, ...] | None = None) -> ModeRegistry:
    """Every registry file, merged and validated.

    The result is cached against each file's path, size and modification time, so
    repeated lookups do not re-read the files while an edit to one of them is seen.

    Raises
    ------
    ModeRegistryError
        When a file is missing or malformed, or two files claim one name.
    """
    sources = registry_paths() if paths is None else tuple(paths)
    stamps: list[tuple[str, int, int]] = []
    for path in sources:
        try:
            stat = path.stat()
        except OSError as exc:
            raise ModeRegistryError(f"mode registry unreadable: {path} ({exc})") from exc
        stamps.append((str(path), stat.st_size, stat.st_mtime_ns))
    key = tuple(stamps)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    registry = _merge([(path, _read_file(path)) for path in sources])
    _CACHE[key] = registry
    return registry


def mode_set_names() -> tuple[str, ...]:
    """Every mode set an extraction can be run over, in merge order."""
    return load_mode_registry().set_names()


def mode_set(name: str = CORE_MODE_SET) -> ModeSet:
    """One mode set, its parent's modes folded in."""
    return load_mode_registry().mode_set(name)


def mode_prompts(name: str = CORE_MODE_SET) -> dict[str, str]:
    """Mode name to system prompt for one set, in label order."""
    return mode_set(name).prompts()


def mode_indices(name: str = CORE_MODE_SET) -> dict[str, int]:
    """Mode name to label index for one set, in label order."""
    return mode_set(name).indices()


def format_constraint(name: str = CORE_MODE_SET) -> str:
    """The clause appended to every prompt of one set."""
    return mode_set(name).format_constraint


def label_vocabulary(name: str) -> LabelVocabulary:
    """One label vocabulary; a mode set answers as the vocabulary of its mode names."""
    return load_mode_registry().vocabulary(name)


def mode_mapping(name: str) -> ModeMapping:
    """One predicted mapping between two vocabularies."""
    return load_mode_registry().mapping(name)


def mapping_wildcards(name: str) -> tuple[str | None, str | None]:
    """One mapping's wildcard on each side, forward then reverse.

    A side with no unpaired label, or with more than one, has no wildcard: the
    readout reports a single unpartnered label's landing place, and there is nothing
    for it to report otherwise.
    """
    forward, reverse = load_mode_registry().unpaired(name)
    return (
        forward[0] if len(forward) == 1 else None,
        reverse[0] if len(reverse) == 1 else None,
    )


def hard_modes() -> frozenset[str]:
    """Every mode of a set declared hard: a pair drawn from these is a hard pair."""
    return load_mode_registry().hard_modes()


def easy_modes() -> frozenset[str]:
    """Every mode of a set not declared hard, and not in one that is."""
    return load_mode_registry().easy_modes()
