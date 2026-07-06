# Jinko SDK Cheatsheet

Preferred migration patterns for notebook refreshes.

## Core Principle

This cheatsheet is based on the SDK currently installed in the repo venv:
`jinko-sdk 1.3.1` (re-verified 2026-07-02 against the installed package and the
[online API tutorials](https://doc.jinko.ai/docs/category/api---tutorials)).

Migrated notebooks keep their original base name (as of 2026-07-03 the earlier
`_v2` suffix was dropped); pre-migration `jinko_helpers` versions, and
notebooks not migrated yet, live alongside them as `<name>_deprecated.ipynb`.
See the naming-convention note in `docs/cookbook_migration_plan.md` for
details.

Prefer the highest-level `jinko` API that keeps the notebook readable. The `run_a_trial` cookbook is the current reference implementation for model/trial creation flow.

## Preferred Mappings

- `import jinko_helpers as jinko` -> `from jinko import JinkoClient`
- `jinko.initialize(...)` -> `client = JinkoClient()` followed by `client.auth_check()`
- `jinko.get_project_item(...)` -> `client.get_project_item(...)`, `client.get_trial(...)`, `client.get_calibration(...)`, `client.get_model(...)`, etc., depending on the item type
- `jinko.get_latest_trial_with_status(...)` -> no direct high-level equivalent in the installed SDK; prefer `client.get_trial(trial_sid, revision=...)` plus `trial.status()` and treat status-based revision discovery as an API gap if still required
- `jinko.get_project_item_url_from_sid(sid)` -> `client.get_* (sid).url` when you already know the type; otherwise use `client.get_project_item(sid).url`
- `jinko.get_project_item_info_from_response(response)` -> typed project item returned directly by `client.create_*` methods, or use the resulting object's `.core_id`, `.snapshot_id`, `.sid`, and `.url`
- manual typed data table lookup by generic project-item helper -> `client.get_data_table(...)` when the notebook already knows it is working with a data table
- raw model creation via collection internals -> `client.create_model_from_json(...)`; when the notebook already has separate model and solving-options JSON, prefer assembling a small `payload = {"model": ..., "solvingOptions": ...}` first
- legacy `jinko.download_model(...)` JSON export -> when the notebook needs the legacy `{ "model": ..., "solvingOptions": ... }` file shape, build it once from `model.content()` and `model.get_solving_options()` in a small local helper; use `model.download_as_zip(...)` only when the notebook actually wants an export bundle
- baseline descriptor fetch by manual request -> `model.get_baseline_descriptors()`
- raw vpop generator creation from design payload -> `client.create_vpop_design_from_design(...)` or `model.create_vpop_design_from_design(...)`; prefer a simple descriptor-id-to-distribution mapping when that is all the notebook needs
- vpop design creation with no marginals/correlations yet (design to be filled in afterwards) -> `model.create_vpop_design_from_model(...)`, then set descriptors on the returned `VpopDesign`
- raw subsampling generator creation payload -> `trial.create_subsampling_design(...)` (not
  `create_vpop_generator_from_subsampling_generator`, which does not exist on the installed SDK);
  its filter/target dict schemas differ from the legacy raw payload (`id` instead of
  `descriptorId`/`timeToEventScalarId`, explicit `isActive`, and `target_correlations` uses a
  top-level `id`/`arm` plus nested `correlatedVariable` instead of `correlateX`/`correlateY`) —
  see the "Gotcha confirmed live" note in `docs/cookbook_migration_plan.md` Family 2 before
  reusing this call, including the request/response asymmetry on `correlations`
- raw vpop generation request -> `vpop_design.generate_vpop(...)`
- raw subsampled vpop generation request -> `subsampling_design.generate_vpop(...)` (not
  `generate_vpop_by_subsampling`; verified against the installed 1.3.1 signature)
- CSV vpop upload by manual payload -> `client.create_vpop_from_csv(...)`
- raw protocol upload -> `client.create_protocol_design_from_json(...)`, `client.create_protocol_design(...)`, or `model.create_protocol_design(...)`
- CSV data table upload via manual SQLite/base64 conversion -> `client.create_data_table_from_csv(...)`
- pandas dataframe data table upload via manual SQLite/base64 conversion -> `client.create_data_table_from_dataframe(...)`
- data table export endpoint call -> `data_table.export(accept="text/csv")` for notebook pandas loading
- data table validation endpoint call -> `data_table.validate()`
- data table summary endpoint call -> `data_table.summary()`
- raw data table replacement on the same item lineage -> advanced fallback: `data_table._update_raw(...)` with `build_project_item_headers(version=...)` when the notebook specifically needs a new version on the same item
- raw trial creation -> `client.create_trial(model, ...)` or `model.create_trial(...)`; add `simple_output_set=...` only when the notebook needs explicit output selection
- trial run endpoint call -> `trial.run()`
- polling helper by IDs -> `trial.wait_until_completed()`
- trial status endpoint call -> `trial.status()`
- output IDs request by manual endpoint call -> `trial.output_ids()`
- trial results summary request -> `trial.results.summary()`
- timeseries download by manual endpoint call -> `trial.results.timeseries(...)`
- scalar summary helper -> `trial.results.summary()`
- scalar download by manual endpoint call -> `trial.results.scalars(...)`
- grouped or filtered scalar result request -> `trial.results.scalars_per_population(scalars=..., filter_tokens=..., group_tokens=...)`;
  unlike `.scalars(...)`/`.timeseries(...)` this returns the raw nested `per_population` JSON
  payload, not a `TabularDownload` — there is no `.to_dataframe()`, so flatten it yourself (see
  the worked example in `filtering_and_grouping_trial_results.ipynb` and the matching note in
  `docs/cookbook_migration_plan.md`)
- patient filtering request -> `trial.results.patients_matching_filters(filter_tokens=...)`
- calibration status request -> `calibration.status()`
- calibration objective weights request -> `calibration.objective_weights()`
- calibration results summary request -> `calibration.results_summary()`
- calibration sorted patients request -> `calibration.results.sorted_patients(...)`
- calibration per-patient errors request -> `calibration.results.errors(...)`
- calibration augmented data tables request -> `calibration.results.augment_data_tables(...)`
- calibration per-patient timeseries request -> `calibration.results.timeseries_per_patient(...)`
- calibration per-patient scalar request -> `calibration.results.scalars_per_patient(...)`
- raw PDF-to-reference upload (`/app/v1/reference/file`) -> `client.create_reference_from_pdf(pdf_file_path=..., name=...)`
- raw markdown-to-document upload (`/app/v1/document/markdown`) -> `client.create_document_from_markdown(markdown_content=..., name=...)`
- raw markdown document update -> `document.update_markdown_from_file(...)` (or re-create with `create_document_from_markdown` for a new item)
- raw LaTeX export (`/app/v1/document/{id}/export/latex`) -> `document.download_latex_zip()`, then write the returned bytes to a file
- raw reference-extract creation -> `reference.create_extract(anchors=..., text_content=..., name=...)`; list existing extracts with `reference.iter_extracts(...)` or `reference.list_extracts(...)`

## Notebook Family Patterns

- Trial reader notebooks should start from `client.get_trial(...)` and then use
  `trial.status()`, `trial.results.summary()`, `trial.output_ids()`,
  `trial.results.timeseries(...)`, and `trial.results.scalars(...)`. Where the
  legacy notebook used `jinko.get_latest_trial_with_status(...)`, replace it
  with `client.get_trial(sid)` followed by an explicit status check:
  `status = trial.status(); if status.get("status") != "completed": raise ...`
  (`status()` returns a dict with a `"status"` key — confirmed against the
  SDK's own `wait_until_completed()` implementation).
- Trial creation notebooks should follow the `run_a_trial` style:
  keep the Jinkō init cell minimal, move notebook-specific imports and constants
  into the cookbook cell, create rich items from `model` or `trial` objects
  whenever possible, and use file helpers on `client` for CSV-backed imports.
- Model download notebooks should prefer iterating existing `Model` objects from
  `client.iter_models(...)`, reuse one local save helper instead of duplicating
  export code per cell, and avoid relying on mutable state from earlier cells
  when listing versions or downloading a specific revision.
- For trial creation, default to the shortest working call such as
  `model.create_trial(...)`; create a simple output set explicitly only when the
  notebook needs custom output selection or wants to teach that concept.
- Calibration notebooks should start from `client.get_calibration(...)` and use
  `calibration.results.*` before considering any raw transport call.
- Data table notebooks should start from `client.get_data_table(...)` or a
  `client.create_data_table_from_*` helper, then use `data_table.export(...)`,
  `data_table.validate()`, `data_table.summary()`, `data_table.raw()`, and
  `data_table.update_mappings(...)` before considering any lower-level path.
  If the notebook must replace the raw content while keeping the same data table
  object, `data_table._update_raw(...)` is the acceptable advanced fallback.
- Knowledge or document notebooks should use the typed `client.create_reference_from_pdf(...)`,
  `client.create_document_from_markdown(...)`, `document.update_markdown_from_file(...)`,
  `document.download_latex_zip(...)`, and `reference.create_extract(...)` /
  `reference.iter_extracts(...)` methods added in the 1.3.x line; `client.raw_request(...)`
  is no longer needed for these flows and should only be used for an operation that still
  has no typed equivalent.

## Avoid By Default

- `import jinko_helpers as jinko`
- notebook-local request wrappers that duplicate `JinkoClient`
- `client._transport.request(...)` from notebooks
- service-private helpers such as `client._data_tables_service...`
- collection-private helpers such as `client.models._create_raw_model(...)`
- generic raw creation calls when a typed helper exists
- over-structuring notebook cells with extra SDK helpers when a shorter typed call is already clear
- manual encoding or file format conversion done only to satisfy an older API pattern

## Allowed Exceptions

Use low-level requests only when both conditions hold:

1. The installed `jinko-sdk` does not expose the needed operation through `JinkoClient` or the relevant typed resource object.
2. The low-level code uses `client.raw_request(...)` and is documented inline as an API-gap workaround.

## Notes

- `trial.results.timeseries(...)` and `trial.results.scalars(...)` return a tabular download object; convert it with `.to_dataframe()` in notebooks when you want pandas output. `.to_dataframe()` already detects and unzips a zipped CSV payload internally, so a notebook's manual `zipfile.ZipFile(io.BytesIO(...))` extraction dance from the legacy raw-request era can be deleted outright rather than translated.
- `trial.results.summary()` and `trial.results.scalars(...).to_dataframe()` hit the same endpoints and return the same shapes (`arms`/`scalars`/`categoricals`, `armId`/`patientId`/`scalarId`/`value`) as the legacy `jinko.get_trial_scalars_summary(...)` / `get_trial_scalars_as_dataframe(...)` helpers, so downstream pandas code (pivots, groupbys, merges) does not need to change.
- `trial.results.scalars(...)` accepts either a raw scalar-to-arms mapping or a
  plain list of scalar ids; for the list form, the SDK infers all trial arms and
  adds `crossArms`.
- Some calibration and result-manager flows are still intentionally low-level in the SDK API shape, but they are now reachable through typed resource helpers such as `calibration.results.*` rather than through `jinko_helpers`.
- Several hardcoded demo trial ids in `cookbooks/trial_simulation_and_analytics/` currently 500 on `results.summary()`/`output_ids()`/`results.scalars()` server-side (confirmed unrelated to the SDK via `raw_request`). See the "stale demo trial data" caveat in `docs/cookbook_migration_plan.md` before assuming a migrated Family 1 notebook is broken — verify the trial id independently first.
