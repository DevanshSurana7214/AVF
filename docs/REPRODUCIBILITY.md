# Reproduction scope and limitations

The paper's tables are frozen aggregate results. Original participant-level predictions, checkpoints and ADNI feature tables remain private. The public extracted components preserve the executed function bodies exactly, as shown by their SHA-256 provenance. Release wrappers, the training driver and synthetic smoke demonstration are new orchestration. Passing their tests does not establish that the historical cohort or complete training run has been reproduced.

The training driver accepts locally prepared regional blocks. Upstream cohort construction, 180-day alignment, feature specifications, modality-presence decisions and the original development-pool exclusion must be verified with authorized local artifacts. Without these, a new run is a new experiment and need not match the paper.

Grouped three-fold validation keeps a base model's evaluation participants out of its fitting set. It does not make the study fully nested: its validation fold selected the checkpoint, and other-fold fusion-tuning predictions came from models fitted partly on target-fold subjects. The pool also underwent repeated exploration. The bootstrap is conditional on this fitted/selected procedure.

The historical dropout function's supplied generator controls the Bernoulli decision, but not the separate random choice of which modality to remove. Global torch seeds were used during fitting. This preserved behavior contributed to the archived cross-family stress-mask mismatch. Natural-availability findings are primary; cross-family stress superiority claims are excluded. No corrected stress rerun is claimed.

TIMM, Flex-MoE and UniMIX rows are simplified regional-input adaptations, not official reproductions. This release focuses on AVF and factorial controls; it does not provide or claim faithful reproduction of those published systems. Joint weighting is not universally superior, several supported cells worsen, sparse cells are omitted from SCR, and no clinical validity or worst-cell guarantee is established.

The private replay utility verifies alignment and computes metrics for fixed archived alphas. It does not refit models, recreate checkpoint selection, or remove selection bias. The public aggregate bootstrap file cannot be used to rerun participant resampling without the private predictions.
