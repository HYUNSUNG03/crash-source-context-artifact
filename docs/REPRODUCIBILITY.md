# Reproducibility scope

## Tested here

The saved-vector route uses Python 3.12 and NumPy 2.3.5. It checks byte hashes before scoring. Original hashes are retained in provenance; the release SHA256 manifest covers the actual packaged files. Source metadata was reduced by an allowlist and historical workstation paths were replaced, so these derivative files are not claimed to have the original hashes.

## Original full run: completed and verified; fresh-package rerun needs external inputs

`historical/earlier/bprime_location_v1.py` protects basename:line[:column] coordinates during the original hygiene pass. `historical/earlier/l3_development_v1.py` and `historical/latest/full152-stage4-v1/build_full152_stage4_inputs.py` build the source-frame contexts and four L conditions. `historical/latest/d1/extract_d1.py` extracts bounded candidates from a saved Joern graph, `select_d_endpoints.py` selects endpoints, `full152-d4-v1/run_full152_d4.py` materializes report-specific inputs, and `full152-d4-v2/build_full152_d4_v2.py` applies the final operand-role/numeric-literal corrections. The latter is the D version used for scoring.

The original full pipeline ran with the recorded source documents, graphs, parser outputs, model snapshot, and development layout. Its frozen vectors, candidates, selections, scores, matrix hashes and phase manifests were checked before publication of this package. The historical scripts depend on those additional inputs and are not portable one-command entry points. `/RESEARCH_WORKSPACES`, `/RESEARCH_HOME`, and `/RESEARCH_OUTPUT` mark paths that must be configured. Consequently, copying only this public package does not reconstruct the full original run on a new machine.

The frozen numerical procedures are also extracted into `src/numeric.py`: consecutive chunks of 510 tokens, normalized chunk mean, equal B/L/D fusion, HDBSCAN candidate generation and singleton noise labels. Use the environment in `requirements-full.txt` and model `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. Original CPU inference used batch 8, threads 4, seed 0, float32 storage and float64 fusion. The source-text inventory hash is recorded in `docs/provenance.json`; the raw source-derived text inventory and model weights are not redistributed here. Supplying other text is a new experiment, not reproduction of the frozen matrices.

## Not completed

- Same-encoder D-status-only fusion ablation.
- Independent held-out evaluation of the final D4/L configuration.
- Same-encoder 64-dimensional control.
- A clean-machine, from-scratch rerun using only this public package.

No raw crashes are executed by the public saved-vector reproduction entry point. Matching original outputs is an additional consistency check; it does not replace the already completed original execution, nor does it prove source/binary correspondence.
