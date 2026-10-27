# Release validation (27 October 2026)

Eight invariant tests passed: dropout mass conservation; expected-weight normalization; absent-branch invariance; singleton preservation; training-only preprocessing; normalized fusion and exact zero-correction bypass; participant-based support; and frozen aggregate table values.

A separate CPU-only orchestration smoke run completed all three grouped folds and three seeds for the `joint_drop` arm, including fitted LightGBM baselines, neural optimization, checkpoint saving/reloading and aligned local predictions. Inputs were generated synthetic examples (180 visits, 90 fictitious groups), not ADNI data. No smoke-run predictions or checkpoints are distributed. This verifies software execution on a tiny synthetic case, not the archived scientific findings or clinical performance.

Observed local runtime: Python 3.13, PyTorch 2.9.1+cu128 (CPU execution for these checks), LightGBM 4.7.0. The declared minimum-version combinations and every supported platform have not all been tested. Dependency versions can change numerical behavior; use and document an appropriate environment for a new study.

The paper compiled to nine pages with fifteen resolved numbered references. The title page, optional-correction equation and bibliography were rendered for visual review. Existing CAS title-block/empty-anchor warnings remain; no equation overflow or undefined citation was reported. Highlights are separate.

An explicit public-file allowlist and content scan checked aggregate CSV schemas, common ADNI participant-identifier patterns, credential patterns, private absolute paths and prohibited checkpoint/data extensions. This is a practical release check, not a formal privacy proof. No ADNI participant records or original test inference were involved in release validation.
