# PORT MAP

Where a name from the extraction repository lives in this one. The instrument was
refounded as an allow-list: a capability is here because it was carried over
deliberately, and everything else stays in the frozen repository
`anamnesis-pl`, which remains the citation target for every path it holds. The
research record cites files by their old paths, so this table is how a reader
holding a citation finds the living code — or learns that the code they are
holding is the record's own copy and has no living successor.

One row per name. **Old path** is relative to `pipeline/` in `anamnesis-pl`;
**new home** is relative to this repository's root. A row marked *record* is not
a gap: it is a decision that the name belongs to the history of the program
rather than to its instrument.

## Configuration

`config.py` held two descriptions of a model — a preset registry and a loader
config — and no bridge between them, so callers projected one onto the other
field by field. The port keeps both roles and adds the bridge: every
configuration class is built by a `from_preset` constructor over one registry
row, so an architecture, a layer plan, a decode policy and a calibration path
cannot disagree about which model they describe.

| old path | new home | notes |
|---|---|---|
| `anamnesis/config.py` | `anamnesis/config/` | Split four ways; the sections below say which part went where. |
| `config.py` · `ModelPreset`, `MODEL_PRESETS` | `anamnesis/config/models.py` | The registry of per-model facts, with the decode policy and the calibration location now part of the row. |
| `config.py` · `ModelConfig` | `anamnesis/config/models.py` | The loader contract, built by `ModelConfig.from_preset()`. Architecture fields are required, so a config cannot describe a model other than the one being loaded. |
| `config.py` · `GenerationConfig`, `ExtractionConfig`, `FeaturePipelineConfig`, `CalibrationConfig`, `GenerationSpec`, `ExperimentConfig`, `ProcessingMode` | `anamnesis/config/experiment.py` | Each class carries `from_preset`; `ExperimentConfig.from_preset()` derives all four section configs together. |
| `config.py` · `RUNS`, `RunSpec` | `anamnesis/config/runs.py` + `anamnesis/config/runs.json` | The registry is data beside the module that reads it. A row is a root token plus relative paths; `resolve_run()` turns it into directories on this machine. |
| `config.py` · `PROJECT_ROOT`, `OUTPUTS_BASE`, `LEGACY_DATA_ROOT`, `CALIBRATION_DIR`, `OUTPUTS_DIR`, `SIGNATURES_DIR`, `FIGURES_DIR`, `PROMPTS_PATH`, `RUN_NAME` | `anamnesis/config/paths.py` | The data roots, read at call time rather than at import. `ANAMNESIS_OUTPUTS` names the outputs root and `ANAMNESIS_LEGACY_DATA` the Phase-0 tree; both the preset registry and the run registry resolve against them, which is why the roots are their own module rather than the property of either. |
| `config.py` · `PROCESSING_MODES`, `MODE_INDEX` | `anamnesis/modes/run4_modes.py` · `RUN4_MODES`, `RUN4_MODE_INDEX` | The re-export is gone: configuration does not import the mode prompts. A caller that needs the prompts names the modes package. |

## Modes and prompts

The mode prompts are the protocol: a label in banked data means the exact text in
this package. They port unchanged, and the tests pin their digests.

| old path | new home | notes |
|---|---|---|
| `anamnesis/modes/run4_modes.py` | `anamnesis/modes/run4_modes.py` | The five format-controlled modes, byte-identical. The format constraint is public here, since the eight-mode set shares it rather than restating it. |
| `anamnesis/modes/extended_modes.py` | `anamnesis/modes/extended_modes.py` | The eight-mode set, byte-identical. |
| `anamnesis/modes/prompt_swap.py` | `anamnesis/modes/prompt_swap.py` | The three swap pairs, byte-identical. |
| `anamnesis/modes/__init__.py` | `anamnesis/modes/__init__.py` | Reworked: it exports the three ported sets and nothing else. |
| `anamnesis/modes/run3_original_modes.py` | *record* | Not ported, by ruling: a frozen Phase-0 historical record, with its one script consumer staying beside it in `anamnesis-pl`. The old package re-exported it from `modes/__init__.py` and `config.py` reached that `__init__` on import, which made a record-only module a hard dependency of the whole package; the reworked `__init__` and a configuration package that imports no modes are what let the ruling hold. Tests assert the module is absent and that neither coupling has returned. |
| `anamnesis/prompts/prompt_sets.json` | `anamnesis/prompts/prompt_sets.json` | Byte-identical; digest pinned in the tests. |
| `anamnesis/prompts/prompt_sets_narrative.json` | `anamnesis/prompts/prompt_sets_narrative.json` | Byte-identical; digest pinned in the tests. |
