# Cookbook Migration Guide

Unified migration guide for the notebook refresh after the `jinko-sdk` upgrade.
This merges the former migration inventory and migration plan into one document.

This guide is aligned with the SDK currently installed in the repo venv:
`jinko-sdk 1.3.1`, re-verified on 2026-07-02 (originally written against 1.2.0
on 2026-06-19).

**Naming convention (2026-07-03):** migrated notebooks now keep their original
base name (the earlier `_v2` suffix has been dropped). The pre-migration
`jinko_helpers` version, if any, is kept alongside it as
`<name>_deprecated.ipynb` for reference. Notebooks that still use
`jinko_helpers` and have no migrated counterpart yet (e.g.
`combine_vpop_design_deprecated.ipynb`, `subsampling_deprecated.ipynb`,
`vpop_generation_with_deep_learning_deprecated.ipynb`,
`GP_saem_jinko_deprecated.ipynb`) were also renamed with the `_deprecated`
suffix and flagged as pending migration, even though there's no name
collision to resolve. Everywhere below, a bare `<name>.ipynb` reference means
the migrated (or never-needing-migration) notebook; check the actual
`cookbooks/` tree for whether a `_deprecated` sibling currently exists.

## Reference

- Keep [`cookbooks/basics/run_a_trial.ipynb`](/home/jean-baptiste.gourlet/git/jinko/api/jinko-api-cookbook/cookbooks/basics/run_a_trial.ipynb)
  unchanged and use it as the reference for notebook style, API level, and
  create-and-run flow.
- Maintain [`docs/jinko_sdk_v2_cheatsheet.md`](/home/jean-baptiste.gourlet/git/jinko/api/jinko-api-cookbook/docs/jinko_sdk_v2_cheatsheet.md)
  as the pattern map from legacy `jinko_helpers` usage to the installed SDK
  surface.

## Working Rules

1. Migrate away from `jinko_helpers`; migrated notebooks should import from
   `jinko`, not from `jinko_helpers`.
2. Prefer typed resource methods over raw transport calls.
3. Prefer SDK convenience helpers over manual payload assembly when the SDK
   already exposes the operation.
4. Keep low-level `client.raw_request(...)` or resource-level raw result
   payloads as a last resort for API gaps only.
5. Do not introduce notebook-local HTTP wrappers if the installed `jinko` SDK
   already exposes the same capability.
6. When a notebook creates a resource from another resource, prefer
   object-oriented flows such as `model.create_trial(...)`,
   `model.create_protocol_design(...)`,
   `trial.create_subsampling_design(...)`, or
   `vpop_design.generate_vpop(...)`.
7. When a notebook imports CSV-backed resources, prefer file helpers such as
   `client.create_vpop_from_csv(...)` and
   `client.create_data_table_from_csv(...)`.
8. When a notebook works with data tables after creation, prefer typed item
   methods such as `data_table.export(...)`, `data_table.validate()`,
   `data_table.summary()`, and `data_table.update_mappings(...)`.
9. If a data-table notebook must replace raw content while keeping the same
   project item lineage, use the advanced `data_table._update_raw(...)` path
   with version headers instead of service-private helpers.
10. When a notebook reads results, prefer resource methods such as
    `trial.output_ids()`, `trial.results.summary()`,
    `trial.results.scalars(...)`, `trial.results.timeseries(...)`,
    `calibration.results.sorted_patients(...)`, and
    `calibration.results.errors(...)`.
11. Migrate notebooks in place under their original name by adapting each
    notebook cell directly to the new SDK style, without a compatibility
    wrapper; before editing, rename the pre-migration original to
    `<name>_deprecated.ipynb` and add a deprecation notice cell pointing at
    the migrated `<name>.ipynb`.
12. Keep the notebooks runnable as notebooks, but do not execute them during
    migration.
13. Preserve the existing notebook intent and structure as much as possible
    while updating imports, client setup, item creation, and result access.
14. Prefer the simplest readable SDK flow that matches the notebook intent; do
    not add extra helper objects or setup steps unless the notebook actually
    needs them.
15. Keep the Jinkō initialization cell minimal: `from jinko import
    JinkoClient`, then `client = JinkoClient()` and `client.auth_check()`.
    Move notebook-specific imports and constants into the cookbook cell.
16. If a manual migration proves a better pattern than an existing script,
    update the migration script and cheatsheet before converting more notebooks.
17. If the local SDK version changes, verify the installed API surface again
    before migrating more notebooks.

## Migration Order

Migrate notebook families in this order so we reuse patterns instead of
inventing new ones per file.

1. Trial result reader notebooks in
   `cookbooks/trial_simulation_and_analytics/`.
2. Trial design and execution notebooks that create items from a model or
   trial.
3. Calibration notebooks in `cookbooks/modeling/` and related R&D folders.
4. Knowledge or document notebooks.
5. Data table workflows.

## Family Inventory

### Family 1: Trial Result Readers

These notebooks mostly read an existing trial and transform results into pandas
dataframes or plots:

- `cookbooks/trial_simulation_and_analytics/visualizing_timeseries.ipynb`
- `cookbooks/trial_simulation_and_analytics/visualizing_scalar_results.ipynb`
- `cookbooks/trial_simulation_and_analytics/visualizing_survival.ipynb`
- `cookbooks/trial_simulation_and_analytics/basic_stat_analysis.ipynb`
- `cookbooks/trial_simulation_and_analytics/producing_data_summary.ipynb`
- `cookbooks/trial_simulation_and_analytics/quantifying_uncertainty.ipynb`
- `cookbooks/trial_simulation_and_analytics/data_privacy.ipynb`
- `cookbooks/trial_simulation_and_analytics/sensitivity_analysis.ipynb`
- `cookbooks/trial_simulation_and_analytics/shap_contribution_analysis.ipynb`
- `cookbooks/trial_simulation_and_analytics/trial_comparison.ipynb`
- `cookbooks/trial_simulation_and_analytics/filtering_and_grouping_trial_results.ipynb`
- `cookbooks/r_and_d/trial_design_optimization/time_to_event_outcome.ipynb`

Common legacy helper usage:

- `jinko.get_latest_trial_with_status(...)`
- `jinko.get_trial_scalars_summary(...)`
- `jinko.get_trial_scalars_as_dataframe(...)`
- `jinko.get_timeseries_as_dataframe(...)`
- `jinko.get_trial_scalars_with_filter_and_groups_as_dataframe(...)`
- `jinko.make_request(...)` for `output_ids`, `timeseries/download`,
  `scalars/download`, or grouped scalar queries

Preferred migration target:

- Load the trial with `client.get_trial(trial_sid, revision=...)`
- Validate run state with `trial.status()`
- Read metadata with `trial.results.summary()` and `trial.output_ids()`
- Download tabular data with `trial.results.timeseries(...).to_dataframe()` and
  `trial.results.scalars(...).to_dataframe()`
- For grouped or filtered scalar endpoints, prefer
  `trial.results.scalars_per_population(...)` and
  `trial.results.patients_matching_filters(...)`

Known caveat:

- There is no high-level replacement for
  `get_latest_trial_with_status(...)` in the installed SDK. Prefer an explicit
  `revision` when the notebook intends to analyze a stable completed trial. If
  status-based revision lookup is still necessary, document it as an SDK gap
  before using `client.raw_request(...)`.

Reusable status-check idiom (used across every Family 1 migration so far):

```python
trial = client.get_trial(trial_sid)
status = trial.status()
if status.get("status") != "completed":
    raise Exception(
        f"Trial {trial_sid} latest revision is not completed (status: {status.get('status')})"
    )
```

`trial.status()` returns a dict shaped like `{"status": "completed" | "not_launched" | ..., "isRunning": bool, ...}` —
confirmed by reading the SDK's own `wait_until_completed()` implementation, which
branches on the same `status["status"]` key. Prefer this exact key over
guessing at alternative shapes.

Migrating `jinko.get_trial_scalars_summary(...)` / `get_trial_scalars_as_dataframe(...)`:
`trial.results.summary()` and `trial.results.scalars(scalar_ids).to_dataframe()` are drop-in
replacements — they hit the same `results_summary` / `scalars/download` endpoints and return the
same `arms`/`scalars`/`categoricals` / `armId`/`patientId`/`scalarId`/`value` shapes, so downstream
pandas code (`.pivot(...)`, `.groupby(...)`, etc.) does not need to change.
Likewise `jinko.make_request(.../output_ids)` -> `trial.output_ids()` and the manual
zip/`io.BytesIO` timeseries-download dance -> `trial.results.timeseries(mapping).to_dataframe()`
are drop-in; `.to_dataframe()` already handles the zip-or-plain-CSV response internally, so the
notebook's own `import io` / `import zipfile` and manual `zipfile.ZipFile(...)` extraction can be
deleted entirely, not just rewritten.

**Known caveat — stale demo trial data (unrelated to the SDK):** while migrating a batch of
Family 1 notebooks on 2026-07-02, most of the hardcoded demo `trial_sid`/`trialId` constants in
`cookbooks/trial_simulation_and_analytics/` (e.g. `tr-OxkW-mB8I`, `tr-9Bid-BL1I`, `tr-Fzt9-uO98`,
`tr-pPqG-BqG2`, `tr-0qYI-s3rF`) returned `status: completed` from `trial.status()` but a generic
HTTP 500 (`"Something went wrong"`) from `trial.results.summary()`, `trial.output_ids()`, and
`trial.results.scalars(...)`. This was confirmed to be server-side and unrelated to the SDK/client
by reproducing the same 500 via `client.raw_request(...)` against the raw endpoint. Only
`tr-HLRF-b0zW`, `tr-f9PT-uDkz`, `tr-dI79-x7V2`, and `tr-OJvV-CPhT` were confirmed live and returning
real results as of that date. When migrating a Family 1 notebook, verify its hardcoded trial id
still returns results (`client.get_trial(sid).results.summary()`) *before* investing time wiring up
the rest of the notebook — if it 500s, the migration itself may still be correct, but you won't be
able to execute the notebook end-to-end until the demo data is refreshed. Do not swap in a
different trial id without checking the user first, since the notebook's narrative and
constants (e.g. specific scenario override names, biomarker ids) are usually tied to one specific
reference model.

Update 2026-07-02 (same day): the user fixed `tr-OxkW-mB8I`, `tr-9Bid-BL1I`, and `tr-pPqG-BqG2` on
the Jinkō side after being told about them; `data_privacy.ipynb`, `visualizing_timeseries.ipynb`,
`sensitivity_analysis.ipynb`, and `filtering_and_grouping_trial_results.ipynb` then ran
end-to-end. `tr-gLnd-8yYx` (used by `trial_comparison.ipynb`/`visualizing_survival.ipynb`) was also
fixed — it had a different failure mode (`status: not_launched`, i.e. the trial had never been run,
not a 500) and both notebooks now run end-to-end too. Still unconfirmed as of this writing:
`tr-Fzt9-uO98` and `tr-0qYI-s3rF` (used by `visualizing_scalar_results.ipynb` and
`shap_contribution_analysis.ipynb`, neither migrated yet). Lesson: report the exact trial id and
failure mode (500 vs `not_launched`) when a Family 1 migration is blocked this way — these have
consistently turned out to be real, fixable issues on the Jinkō side rather than something to work
around in the notebook.

**Second gotcha confirmed live 2026-07-02 — `trial.results.scalars_per_population(...)` returns raw
nested JSON, not a dataframe:** unlike `.scalars(...)`/`.timeseries(...)` (which return a
`TabularDownload` with `.to_dataframe()`), `scalars_per_population(...)` is the direct typed
replacement for the removed `jinko.get_trial_scalars_with_filter_and_groups_as_dataframe(...)`
helper but returns the raw `per_population` endpoint payload directly:
`{"scalars": [[<group tokens>, {<arm>: {<scalarId>: {"unit": ..., "values": [...]}}}]]}`. There is
no typed flattening helper, so notebooks must convert this themselves — see
`cookbooks/trial_simulation_and_analytics/filtering_and_grouping_trial_results.ipynb` for a
worked `scalars_per_population_as_dataframe(...)` example, now fully verified end-to-end (including
the grouped-bucket case) once `tr-pPqG-BqG2` was fixed on 2026-07-02.
**Gotcha within the gotcha:** each `group_key` token's `contents.tag` alone (e.g. `"Scalar"`) is the
same for every bucket — the actual bucket identity lives in `contents.contents.descriptorId` plus
`contents.contents.parameters` (a `{"lowBound", "highBound"}` span for `mode: "Buckets"`). Flattening
on `tag` alone silently collapses every distinct bucket into one indistinguishable label; build the
group label from `contents` instead (see `_describe_group_token(...)` in the same notebook).
`trial.results.timeseries(...)` also has no server-side patient-id filter (unlike the legacy
`get_timeseries_as_dataframe(..., patient_ids_to_keep)`) — download the full timeseries and filter
the resulting dataframe client-side with `.isin(patient_ids_to_keep)` instead.

**Gotcha confirmed live 2026-07-02 — `create_subsampling_design`/`edit` use a
newer field schema than the legacy raw `vpop_generator` payload:**

- `numeric_filters`/`categorical_filters` entries need `id` (not the legacy
  `descriptorId`) plus an explicit `isActive: True`.
- `target_survivals` entries need `id` (not the legacy `timeToEventScalarId`),
  plus `isActive: True` and `conditions: []`.
- `target_summary_statistics` entries need `isActive: True` and `conditions: []`
  added (arm/id were already correct).
- `target_correlations` entries are restructured: the legacy `correlateX`/
  `correlateY` pair becomes a top-level `id`/`arm` (for X) plus a nested
  `correlatedVariable: {id, arm}` (for Y); also needs `isActive: True` and
  `conditions: []`.
- `target_marginals`/`target_categoricals` did not need changes (their `id`/`arm`
  shape already matched, and `isActive`/`conditions` are optional there).
- Confirmed by pydantic `ValidationError`s from `jinko.openapi_types`, cross-checked
  against an existing subsampling design's persisted `content()` and a fresh raw
  `generate_vpop` fitness response.
- **Non-obvious asymmetry:** the *response* `subsamplingFitness.correlations` payload
  still nests `correlateX`/`correlateY` regardless of which request schema was used
  to create the design — the request and response shapes are not mirror images of
  each other. Any dataframe-flattening code (replacing the removed
  `jinko.subsampling_goodness_of_fit_as_dataframe(...)` helper) must read
  `correlateX`/`correlateY` from the response even though it *writes*
  `correlatedVariable` in the request.
- `subsampling_design.generate_vpop(...)` does not surface the endpoint's
  `subsamplingFitness` payload at all (SDK gap) — use `client.raw_request(...)` for
  that specific call when the goodness-of-fit data is needed, and note that its
  JSON response carries only `coreItemId`/`snapshotId` (no SID), so there is no way
  to wrap the result into a typed `Vpop`/`get_vpop(...)` call without a separate
  SID lookup that the SDK does not expose either.
- Vpop CSV export (`GET /core/v2/vpop_manager/vpop/{id}` with `accept=text/csv`) has
  no typed equivalent either (`Vpop.content()` returns the JSON patient-list shape,
  not a flat table) — `client.raw_request(..., accept="text/csv")` is the correct
  fallback.

### Family 2: Trial And Vpop Creators

These notebooks create project items from files, descriptors, or an existing
model or trial:

- `cookbooks/basics/run_a_trial.ipynb`
- `cookbooks/modeling/combine_vpop_design.ipynb`
- `cookbooks/modeling/subsampling.ipynb`
- `cookbooks/r_and_d/vpop_generation_inn/vpop_generation_with_deep_learning.ipynb`

Preferred migration target:

- `client.create_model_from_json(...)` when the notebook starts from model JSON
  resources
- `model.get_baseline_descriptors()`
- `model.create_vpop_design_from_design(...)`
- `vpop_design.generate_vpop(...)`
- `trial.create_subsampling_design(...)` then `subsampling_design.generate_vpop(...)`
- `client.create_vpop_from_csv(...)`
- `model.create_protocol_design(...)`
- `client.create_data_table_from_csv(...)`
- `model.create_trial(...)`
- Prefer a minimal init cell and the shortest readable create-flow that
  preserves the notebook narrative

For model-download style notebooks in this family:

- Prefer `client.iter_models(folder=folder_id)` directly when you only need the
  folder filter; do not resolve a folder object first unless the notebook needs
  folder metadata.
- If the notebook wants the legacy JSON payload shape
  `{ "model": ..., "solvingOptions": ... }`, factor that serialization into one
  local helper and reuse it across cells.
- Use `model.download_as_zip(...)` only when the cookbook explicitly wants a
  zip export bundle rather than the legacy JSON file output.

### Family 3: Calibration Readers

These notebooks work from an existing calibration and inspect or reuse its
results:

- `cookbooks/modeling/pre_calibration_uq.ipynb`
- `cookbooks/modeling/post_calibration_uq.ipynb`
- `cookbooks/modeling/calib_debugging.ipynb`
- `cookbooks/r_and_d/vpop_calibration_saem/GP_saem_jinko.ipynb`

Preferred migration target:

- `client.get_calibration(calibration_sid, revision=...)`
- `calibration.status()`
- `calibration.objective_weights()`
- `calibration.results_summary()`
- `calibration.results.sorted_patients(...)`
- `calibration.results.errors(...)`
- `calibration.results.augment_data_tables(...)`
- `calibration.results.timeseries_per_patient(...)`
- `calibration.results.scalars_per_patient(...)`

Known caveat:

- Several calibration result methods still expose endpoint-shaped payloads. That
  is acceptable as long as the notebook calls them through the typed
  `calibration.results.*` interface rather than rebuilding raw HTTP calls.

### Family 4: Knowledge And Document Workflows

As of `jinko-sdk` 1.3.x these notebooks are now covered by typed resource
helpers; they no longer need `client.raw_request(...)`:

- `cookbooks/knowledge/create_reference_from_pdf.ipynb` ->
  `client.create_reference_from_pdf(pdf_file_path=..., name=...)`
- `cookbooks/knowledge/create_document_from_markdown.ipynb` ->
  `client.create_document_from_markdown(markdown_content=..., name=...)`
- `cookbooks/knowledge/export_document_to_latex.ipynb` ->
  `client.get_document(document_sid).download_latex_zip()`, then write the
  returned bytes to the output file

Migration rule:

- Check the installed SDK first for a typed resource helper.
- If none exists, use `client.raw_request(...)` directly in the notebook and
  add a short inline note that the call is an SDK-gap workaround.

### Family 5: Data Table Workflows

These notebooks upload, transform, validate, or re-export data tables:

- `cookbooks/ai_agents/data_table_transformation.ipynb`

Preferred migration target:

- `client.create_data_table_from_csv(...)`
- `client.create_data_table_from_dataframe(...)`
- `client.get_data_table(data_table_sid, revision=...)`
- `data_table.export(accept="text/csv")` for pandas loading
- `data_table.validate()` and `data_table.summary()`
- `data_table.update_mappings(...)` when the notebook changes mappings only
- `data_table._update_raw(...)` with version headers when the notebook must
  push a new version of the same data table instead of creating a new object

Known caveat:

- The installed SDK exposes strong create/read/mapping helpers for data tables,
  but does not yet provide a dedicated high-level notebook-friendly helper for
  replacing a data table's raw dataframe content while preserving the same item
  lineage. When that exact behavior is required, prefer the advanced
  `data_table._update_raw(...)` compatibility path over reaching into
  `client._data_tables_service...` from notebooks.

## Preparation Status

The current preparation work identifies these recurring migration families:

- Trial readers: existing helper calls can usually be replaced with
  `client.get_trial(...)`, `trial.status()`, `trial.results.summary()`,
  `trial.output_ids()`, `trial.results.timeseries(...)`, and
  `trial.results.scalars(...)`.
- Calibration readers: existing helper calls can usually be replaced with
  `client.get_calibration(...)`, `calibration.results_summary()`,
  `calibration.objective_weights()`, and `calibration.results.*`.
- Item creators: existing helper calls should be replaced with rich resource
  methods such as `model.create_trial(...)`,
  `model.create_protocol_design(...)`, `client.create_vpop_from_csv(...)`, and
  `client.create_data_table_from_csv(...)`.
- Data table workflows: prefer typed `DataTable` methods for export,
  validation, and mapping updates; if the notebook needs to transform the raw
  dataframe content while preserving the same item lineage, use the advanced
  `data_table._update_raw(...)` path instead of calling service-private helpers
  from notebook code.
- The current `run_a_trial` reference also shows two style decisions worth
  reusing broadly: keep the initialization cell minimal, and prefer the
  shortest readable create-flow that the SDK supports instead of adding
  intermediate helpers unless they are needed for the notebook narrative.

## Current Caveats

- The initial modeling notebook `_v2` migration batch was started against the
  wrong SDK assumption and should not be treated as a reference. The disabled
  script at
  [`scripts/rewrite_modeling_v2.py`](/home/jean-baptiste.gourlet/git/jinko/api/jinko-api-cookbook/scripts/rewrite_modeling_v2.py)
  exists specifically to prevent regenerating those invalid conversions.

## Process

- Migrate per notebook cell, not by broad search-and-replace.
- Reuse the notebook narrative and variable naming where reasonable.
- Do not add compatibility wrappers around `jinko_helpers`.
- Do not keep `make_request(...)` code when the SDK already exposes a richer
  `client.*`, `model.*`, `trial.*`, or `calibration.*` method.
- Re-run the audit script when needed:
  [`scripts/audit_cookbook_sdk_usage.py`](/home/jean-baptiste.gourlet/git/jinko/api/jinko-api-cookbook/scripts/audit_cookbook_sdk_usage.py)
