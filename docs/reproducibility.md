# Reproducibility notes

The public package separates three kinds of files:

1. submission inference code at the repository root;
2. final training code and locked training artifacts under `training/`;
3. external challenge data and public pretrained model caches under
   `/data/focus` and `/cache`.

The final trainer uses the path-portable manifests in
`training/artifacts/final_manifests/`. Their frame indices, timestamps,
exposure order, and hashes are retained from the final training preflight;
only the host-specific video prefix is represented as `${FOCUS_DATA_ROOT}`.

The final Qwen revision, prompt/timestamp design, optimizer settings, stream
sizes, and checkpoint cadence are recorded in
`training/configs/general_final_training_config.json` and
`training/configs/aggregation_final_training_config.json`.

The Qwen base weights and SigLIP encoder are public pretrained resources and
are not duplicated in this repository. The final task-specific union packs
are included under `training/weights/` and their SHA256 values are recorded in
`training/weights/README.md`.

No training code is imported by the Docker submission entry point. Conversely,
the training package does not alter the official submission interface.
