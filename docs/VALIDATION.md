# Release validation

Validated on 2026-09-09 using Python 3.12 and NumPy 2.3.5:

- 48/48 reconstructed nominal matrix hashes match the recorded experiment.
- All 48 AP values match their saved values within 1e-10.
- 96/96 S0/Smax choices match both evaluation index and partition hash.
- Selected and grid partitions match F at the original evaluator's five-decimal storage precision, ARI within 1e-10, and exact lost/over/under counts.
- All 960 grid rows and 320 macro summaries pass the checks.
- All six deterministic baseline rows and the FreeType ranking/status counts match draft v6.
- The three B+L/S0 and B+L+D/S0 partitions are identical after ignoring cluster-label numbering.

`scripts/reproduce.py` exits with an error on any failed check. It writes the full check list to the chosen output directory's `validation.json`. A successful public-package run does not execute an encoder, HDBSCAN fit, source graph construction or a crash-triggering input; it validates the results of the already completed original execution from frozen artifacts. The synthetic/older controls and D-depth contracts are reported from their saved results rather than re-executed by this release check.

GitHub Actions completed the clean Linux saved-vector check successfully for this release. As with the local release check, this CI run validates frozen artifacts only; it does not rerun source extraction, inference, clustering, or crash inputs.
