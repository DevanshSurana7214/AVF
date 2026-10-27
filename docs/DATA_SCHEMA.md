# Private input schema

Do not commit data conforming to this participant-level schema. The training driver reads an already prepared **development-only** cohort; it does not reconstruct the full ADNI cohort or verify upstream visit alignment/provenance.

`episodes.csv`: one unique `episode_id` per diagnosis visit; `participant_id` groups repeated visits; `y` is CN=0, MCI=1, dementia=2; binary `m_mri`, `m_fdg`, `m_amy`, `m_tau` encode available ROI blocks. At least one modality must be present. Sort/mapping must agree with the original cohort to replay historical folds. Participant identifiers remain private.

`mri.csv`, `fdg.csv`, `amy.csv`, `tau.csv`: unique `episode_id` plus numeric regional-feature columns. Empty feature values represent missing entries. Use the archived feature specifications and original cohort-construction rules for a genuine historical replay. The public release does not distribute those private processed tables.

Training outputs are restricted to the ignored `local_outputs/` tree. They include private predictions and learned preprocessing statistics and must not be uploaded. Keep original outer validation/test subjects outside the supplied development files. This separation is a caller responsibility, not something inferable from arbitrary input CSVs.

Private prediction replay requires `episode_id`, `participant_id` (or historical `RID`), `y`, `pattern` (or historical `effective_pattern`), `fold`, `seed`, `p0`, `p1`, `p2`. Supply one method per CSV, natural availability, all three historical seeds, identical episode ordering/metadata and the verified historical development cohort. No prediction CSV is included in this repository.

Public `results/*.csv` files are aggregate summaries, not this schema.
