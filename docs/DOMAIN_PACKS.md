# Domain Packs

Domain packs make domain-specific metadata declarative. The pipeline still detects
scenes, generates keyframe metadata, creates embeddings, and writes JSON files.
Domain packs define the vocabulary around that engine: prompts, fields,
taxonomies, filters, and export mapping.

Implemented files:

- `config/domains/archive.yaml`
- `config/domains/media_broadcast.yaml`
- `data/eval/media_broadcast_queries.yaml`
- `src/kuaa/domain.py`
- `src/kuaa/models/describer/domain_prompts.py`
- `tests/test_domain_packs.py`

## Goals

- Keep the existing archive demo behavior as the default.
- Let another domain change prompt wording and output schema without editing
  pipeline code.
- Give reviewers concrete evidence that the project is domain-adaptable rather
  than hardcoded for one film archive example.

## Non-Goals

Domain packs do not add a full domain-specific UI renderer. Search, Scenes, and
Annotate use the existing common scene metadata fields. Structured exports and
future UI work consume the domain export mapping and filter definitions added
here.

## Config Selection

The default config selects the archive pack:

```yaml
domain:
  pack: "archive"
  packs_dir: "config/domains"
```

Selection rules:

1. If `domain.path` is set, load that YAML file.
2. Otherwise, load `<domain.packs_dir>/<domain.pack>.yaml`.
3. If no domain section is present, fall back to `archive`.

Relative paths resolve from the project root used to load the application
config.

Both Moondream scene-describer backends read the selected pack and iterate its
`prompt_templates`. The UI renders the shared metadata fields, while extra
domain responses remain available in `_raw_responses` and through
`export_mapping`.

## Pack Schema

Required top-level fields:

- `id`: stable machine id such as `archive`.
- `label`: human-facing name.
- `metadata_fields`: ordered list of field definitions.
- `prompt_templates`: mapping from prompt key to prompt text and token budget.
- `export_mapping`: mapping from export field name to metadata path.

Optional top-level fields:

- `description`
- `taxonomy`
- `filters`
- `sample_outputs`
- `evaluation`

Example:

```yaml
id: archive
label: Film archive
metadata_fields:
  - name: description
    label: Description
    type: text
    required: true
prompt_templates:
  description:
    prompt: "Describe this film scene in one or two sentences."
    max_new_tokens: 80
export_mapping:
  scene_id: scene_id
  description: description
  tags: tags
```

## Prompt Templates

Each prompt template has:

- `prompt`: text sent to the scene describer.
- `max_new_tokens`: positive integer token budget.

The archive pack mirrors the current prompt set:

- `description`
- `location`
- `setting`
- `time_of_day`
- `people_and_action`
- `objects`

Additional domain packs may add keys. Metadata assembly still parses the
standard keys into the common fields used by the UI. Extra prompt responses are
preserved under `_raw_responses` and can be exported through `export_mapping`.

## Export Mapping

`export_mapping` maps output field names to metadata paths using dot notation.
For example:

```yaml
export_mapping:
  scene_id: scene_id
  shot_type: _raw_responses.shot_type
  licensing_notes: _raw_responses.licensing_notes
```

This lets a domain pack define a structured export shape independent of the
export endpoint's own implementation.

## Archive vs media_broadcast: side-by-side

The same pipeline drives both packs; only the declarative vocabulary changes.
This is the generalization claim made concrete.

| Aspect | `archive` (`config/domains/archive.yaml`) | `media_broadcast` (`config/domains/media_broadcast.yaml`) |
|---|---|---|
| `label` | Film archive | Media broadcast |
| Intent | Catalog historical/archival scenes for search, review, preservation | Catalog news/documentary/production-library footage for asset reuse + licensing |
| Metadata fields | description, location (interior/exterior), setting, time_of_day, num_people, people_action, objects, tags | description, shot_type, visible_people, location_type, action, objects, logos_or_text, licensing_notes, reusable_broll_score |
| Prompt keys | description, location, setting, time_of_day, people_and_action, objects | description, shot_type, visible_people, location_type, action, objects, logos_or_text, licensing_notes, reusable_broll_score |
| Taxonomy axes | location · time_of_day · common_tags | shot_type · location_type · rights_flags |
| Filters | location (enum) · time_of_day (enum) · tags (multi-select) | shot_type · location_type · rights_flags (text_contains) · reusable_broll_score (numeric_range) |
| Export shape | flat scene fields (description/location/setting/time_of_day/objects/tags) | editorial fields incl. `_raw_responses.shot_type`, `_raw_responses.licensing_notes`, `_raw_responses.reusable_broll_score` |
| Domain-specific signal | day/night + interior/exterior heuristics for archival browse | rights/licensing flags + b-roll reuse score for editorial triage |
| Eval query set | `data/eval/archive_demo_queries.yaml` | `data/eval/media_broadcast_queries.yaml` |

What stays identical across both: scene detection, keyframe extraction, visual
embeddings, the search/hybrid/rhymes/rerank stack, and the JSON/CSV export
machinery. Only prompts, fields, taxonomy, filters, and export mapping move into
the pack YAML — no pipeline code changes (proven by `tests/test_domain_packs.py`).

## Adding A New Domain Pack

1. Create `config/domains/<id>.yaml`.
2. Define metadata fields and prompt templates.
3. Add taxonomy terms and filters that are useful for search/browse.
4. Add an export mapping for downstream users.
5. Add a small evaluation query file under `data/eval/`.
6. Run `uv run pytest tests/test_domain_packs.py -q`.

## Design record: archive and media_broadcast packs

### Archive domain (default)

Tasks:

- Define the domain-pack YAML schema.
- Add `config/domains/archive.yaml` matching current metadata behavior.
- Add `src/kuaa/domain.py` with loading, validation, and export helpers.
- Add unit tests for valid and invalid packs.
- Wire the selected pack into scene-describer prompts.

Acceptance:

- Archive domain loads by default.
- Archive prompt keys match the current prompt set.
- Invalid packs fail with clear messages.
- Existing demo and tests still pass.

### Media broadcast domain

Tasks:

- Add `config/domains/media_broadcast.yaml`.
- Define fields for shot type, action, visible people, location type,
  logos/text, licensing notes, and b-roll reuse.
- Add a small media-broadcast query set.
- Add sample output records in the domain pack.
- Document how media-broadcast differs from archive cataloging.

Acceptance:

- Selecting `domain.pack: media_broadcast` changes prompt keys and export shape.
- The sample pack is credible enough to discuss with a media-asset reviewer.
- Tests prove the pack loads and maps exports without pipeline edits.

## Limitations

Domain packs are configuration, not magic model fine-tuning. A weak prompt can
still produce weak metadata. Evaluation remains required when changing domains,
and public quality claims should always cite the domain, dataset, model, and
artifact bundle used.
