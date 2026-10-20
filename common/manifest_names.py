"""File names of the two JSON records a checkpoint directory may carry."""

# Structural description of an expanded checkpoint (which experts are the
# removable bank). Loaders read it; it travels with every expanded descendant.
EXPANSION_MANIFEST = "expert_expansion_manifest.json"

# Provenance record written by every step; never read back for validation.
RUN_RECORD = "run.json"

# Record files written by earlier code versions. Existing checkpoints still
# contain them; they must not be inherited into new outputs.
LEGACY_RECORD_FILES = frozenset({
    "ablation_manifest.json",
    "stage2_manifest.json",
    "recombination_manifest.json",
    "rmu_pipeline_manifest.json",
    "checkpoint_adapter_manifest.json",
    "passthrough_manifest.json",
    "precision_preservation_manifest.json",
    "training_progress.json",
    "sft_checkpoint.json",
    "archive_manifest.json",
    "weights_removed.json",
    "reload_verification.json",
    "reload_diagnosis.json",
})
